#import data
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from attack_eval import parse_args_and_config
from purify import get_classifier
import time
from utils import *
from purify import *
import scipy.stats as stats
import numpy as np
from scipy.stats import norm
import math

def predict_logits(args, clf, X, y, reps, requires_grad=False):
    X = X.cuda()
    y = y.cuda()
    
    if requires_grad:
        if args.classifier_name == 'wideresnet':
            logits = clf(X)
        else:
            logits = clf((X + 1) * 0.5)
    else:
        with torch.no_grad():
            if args.classifier_name == 'wideresnet':
                logits = clf(X.data)
            else:
                logits = clf((X.data + 1) * 0.5)

    # finite-sample approximation of stochastic classifier prediction
    if reps>1:
        # print(f'logits shape: {logits.shape}')
        logits_pred = logits.view([reps, int(logits.shape[0]/reps), logits.shape[1]]).mean(0)
    else:
        logits_pred = logits
    _, y_pred = torch.max(logits_pred, 1)
    correct = torch.eq(y_pred, y)

    return correct, logits_pred

def purify_and_eval(args, clf, X, y, model, scheduler, reps):
    X = X.cuda()
    y = y.cuda()
    # parallel states for either EOT attack grad or EOT defense with large-sample evaluation of stochastic classifier
    X_repeat = X.repeat([reps, 1, 1, 1])
    if reps > 50:
        n = reps // 50  # Number of subgroups for the memory limit of 50 reps for diffpure
        X_purified_list = []
        for i in range(n):
            X_subgroup = X.repeat([50, 1, 1, 1])
            X_purified_subgroup, x_pure_list, noi_pure_list, curr_ts, next_ts = purify(args, model, scheduler, X_subgroup)
            X_purified_list.append(X_purified_subgroup)
            # If reps is not exactly divisible by 50, handle the remainder
        remainder = reps % 50
        if remainder > 0:
            X_subgroup = X.repeat([remainder, 1, 1, 1])
            X_purified_subgroup, x_pure_list, noi_pure_list, curr_ts, next_ts = purify(args, model, scheduler, X_subgroup)
            X_purified_list.append(X_purified_subgroup)
            # Concatenate all purified subgroups
        X_purified = torch.cat(X_purified_list, dim=0)
    else:
        X_repeat = X.repeat([reps, 1, 1, 1])
        X_purified, x_pure_list, noi_pure_list, curr_ts, next_ts = purify(args, model, scheduler, X_repeat)

    correct, logits_pred = predict_logits(args, clf, X_purified, y, reps)
    return correct, logits_pred

def purify_times(X, y, args, clf, model, scheduler, times):
    X.cuda()
    y.cuda()
    correct_sum = 0  # Total count of correctly classified images
    logits_list = [] # List to record all logits

    for i in range(times):  # Classify `times` more times
        corr, logits_pred = purify_and_eval(args, clf, X, y, model, scheduler, 1)
        correct_sum += corr.long().sum()
        logits_list.append(logits_pred.cpu())  # Append logits_pred to the list
    
    # Stack along the new dimension (times, 1, 10) and average
    logits_stacked = torch.stack(logits_list, dim=0)
    logits_avg = logits_stacked.mean(dim=0)  # Shape [1, 10]
    # Get correct logit
    # correct_class = y.item()
    # correct_logit = logits_avg[0, correct_class]
    # Get second highest incorrect logit
    # mask = torch.ones(10, dtype=torch.bool, device=logits_avg.device)
    # mask[correct_class] = False
    # incorrect_logits = logits_avg[0][mask]
    # sorted_incorrect = torch.sort(incorrect_logits, descending=True).values
    # logit2 = sorted_incorrect[1] if len(sorted_incorrect) >= 2 else sorted_incorrect[0]

    avg_acc = 100*correct_sum/times
    _, y_pred = torch.max(logits_avg, 1)
    updated_correct = torch.eq(y_pred, y) 
    
    return avg_acc, updated_correct, y_pred

def calculate_sample_size(list1, list2, alpha=0.05, power=0.8):
    """
    Calculate the required sample size per group for a two-sample independent t-test.
    
    Parameters:
    - list1, list2 (array-like): Input lists of numerical values (pilot data).
    - alpha (float): Significance level (default: 0.05).
    - power (float): Desired statistical power (default: 0.8).
    
    Returns:
    - n (int): Sample size per group (rounded up to the nearest integer).
    
    Raises:
    - ValueError: If the effect size (Δ) is zero.
    """
    # Convert inputs to NumPy arrays if they are not already
    list1 = np.array(list1) if not isinstance(list1, np.ndarray) else list1
    list2 = np.array(list2) if not isinstance(list2, np.ndarray) else list2
    # Calculate means and variances (using unbiased sample variance)
    mu1, mu2 = np.mean(list1), np.mean(list2)
    var1, var2 = np.var(list1, ddof=1), np.var(list2, ddof=1)
    
    # Effect size (Δ) and check for validity
    delta = abs(mu1 - mu2)
    if delta == 0:
        raise ValueError("Effect size Δ is zero; the means are identical.")
    
    # Z-scores for significance level and power
    z_alpha = norm.ppf(1 - alpha / 2)  # Two-tailed test
    z_beta = norm.ppf(power)
    
    # Sample size formula (per group)
    n = ((z_alpha + z_beta) ** 2 * (var1 + var2)) / (delta ** 2)
    
    # Round up to ensure sufficient power
    return mu1, mu2, var1, var2, math.ceil(n)

def plot(args, logit1_record, logit2_record, logit1_mean, logit2_mean, logit1_std, logit2_std, sample_size, times, avg_acc, e):
    # plotting first histogram
    plt.hist(logit1_record.cpu().numpy(), bins=20, color='blue', alpha=0.5, label='correct')
    plt.hist(logit2_record.cpu().numpy(), bins=20, color='gold', alpha=0.5, label='2nd choice')
    # Add dotted lines for mean values
    plt.axvline(logit1_mean, color='blue', linestyle='dotted', linewidth=2, label='Mean logit')
    plt.axvline(logit2_mean, color='gold', linestyle='dotted', linewidth=2, label='2nd Mean logit')
    # Add text annotations for mean and standard deviation
    # Add text annotations for mean and standard deviation
    plt.text(logit1_mean, plt.ylim()[1] * 0.6, f'Mean: {logit1_mean:.2f}\nStd: {logit1_std:.2f}\n H_def estimate:{int(sample_size)}', 
         color='blue', bbox=dict(facecolor='white', alpha=0.5))
    plt.text(logit2_mean, plt.ylim()[1] * 0.4, f'Mean: {logit2_mean:.2f}\nStd: {logit2_std:.2f}', 
         color='brown', bbox=dict(facecolor='white', alpha=0.5))

    plt.xlabel('Logits')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.title(f'Stochastic Logits for Unstable Classification on 1 image \n purify {times} times, acc {avg_acc:.2f}%')
    plt.savefig(args.exp_dir + f'/log/wrong_unstable_logits_purify{times}_ind{e}.png')
    plt.close()


def class_times(X, y, args, clf, model, scheduler, times1, times2, reps_list, index, logger):

    
    avg_acc, correct, y_pred = purify_times(X, y, args, clf, model, scheduler, times1)
    logger.log(f'avg_acc over {times1} times: {avg_acc:.2f}')
    logger.log(f'Final prediction correct true or not false: {correct}')
    
    for reps in reps_list:
        agree_sum = 0  # Total count of correctly classified images
        for i in range(times2): 
            correct_reps, logit1  = purify_and_eval(args, clf, X, y, model, scheduler, reps)
            _, y_pred_reps = torch.max(logit1, 1)
            agree = torch.eq(y_pred_reps.cpu(), y_pred.cpu())
            agree_sum +=agree
        avg_agree = 100*agree_sum.sum()/times2
        logger.log(f'probablity to agree with the stable label prediction running {times2} times:{avg_agree:.2f}% for reps {reps} for image {index}')

    
    return avg_agree

def wrong_img(X, y, class_path, args, clf, model, scheduler, reps):
    adv_class = torch.zeros(0).long()
    ims_adv_list = []  # Use a list to dynamically store the adversarial images
    ims_label = torch.zeros(0).long()

    for ind in range(class_path.shape[1]):
        if class_path[-1, ind] == 0:
            correct, _ = purify_and_eval(args, clf, X[ind].unsqueeze(0), y[ind].unsqueeze(0), model, scheduler, reps)
            print(f'correct item {correct.item()}')
            adv_class = torch.cat((adv_class, correct.detach().cpu()), dim=0)  # Record correctness
            if correct.item():  # save if doesn't match with the class with adversarial images
                ims_adv_list.append(X[ind].cpu())  # Append the image to the list
                ims_label = torch.cat((ims_label, y[ind].unsqueeze(0).cpu()), dim=0)  # Append the label

    if ims_adv_list:
        ims_adv_batch = torch.stack(ims_adv_list)  # Stack the list into a tensor
    else:
        ims_adv_batch = torch.zeros(0, X.shape[1], X.shape[2], X.shape[3])  # Handle the case where no images are saved

    return adv_class, ims_adv_batch, ims_label

def correct_img(X, y, class_path, args, clf, model, scheduler, reps):
    adv_class = torch.zeros(0).long()
    ims_adv_list = []  # Use a list to dynamically store the adversarial images
    ims_label = torch.zeros(0).long()

    for ind in range(class_path.shape[1]):
        if class_path[-1, ind] == 1:
            correct, _ = purify_and_eval(args, clf, X[ind].unsqueeze(0), y[ind].unsqueeze(0), model, scheduler, reps)
            print(f'correct item {correct.item()}')
            adv_class = torch.cat((adv_class, correct.detach().cpu()), dim=0)  # Record correctness
            if not correct.item():  # save if doesn't match with the class with adversarial images
                ims_adv_list.append(X[ind].cpu())  # Append the image to the list
                ims_label = torch.cat((ims_label, y[ind].unsqueeze(0).cpu()), dim=0)  # Append the label

    if ims_adv_list:
        ims_adv_batch = torch.stack(ims_adv_list)  # Stack the list into a tensor
    else:
        ims_adv_batch = torch.zeros(0, X.shape[1], X.shape[2], X.shape[3])  # Handle the case where no images are saved

    return adv_class, ims_adv_batch, ims_label

def reload_eval(args, config): 
    # set up distribution
    # setup_dist(rank, world_size)
    args, config = parse_args_and_config()
    args.batch_size = 1
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # results1 = '/home/yuandu/MEGA/data/cifar10_pgdattack_defense_reps1_60'
    # # Load saved tensors
    # saved_data1 = torch.load(results1)
    # # for key, value in saved_data.items():  # Iterate over key-value pairs
    # #     print(f'key: {key}')
    
    # ims_orig = saved_data1['ims_orig']
    # ims_adv = saved_data1['x_final_adv']
    # labels = saved_data1['labs']
    # class_path = saved_data1['class_path']
    
    # record of original images, adversarial images, and labels
    class Logger():
        def __init__(self, log_path):
            self.log_path = log_path
        
        def log(self, str_to_log):
            print(str_to_log)
            if not self.log_path is None:
                with open(self.log_path, 'a') as f:
                    f.write(str_to_log + '\n')
                    f.flush()
    
    # Create Output Directory (add timestamp)
    args.exp_dir = os.path.join(args.exp_dir, f'{args.model_data}',time.strftime("%Y_%m_%d_%H_%M", time.localtime()))
    # setup seed, make folders and save code
    setup_exp(args.exp_dir, args.seed, folder_list=['log'], code_file_list=['defense_reps_estimate_update.py'], use_careful_setup=False)
    log_path=f'{args.exp_dir}/log_reps_estimate.txt'
    logger = Logger(log_path)

    # Evaluate original and adversarial images
    config.classifier_name = "cifar10-wideresnet-28-10"
    rank=0
    clf = get_classifier(args, rank, device).to(device)

    #load purify model
    model, scheduler = load_purify_model(rank, args, config, device)
    model.eval()
    
    #PartI
    # save the wrong predicted but later correct images in the validation
    # reps = 50
    # adv_class, ims_adv_batch, ims_label = correct_img(ims_adv, labels, class_path, args, clf, model, scheduler, reps)
    # torch.save({'ims_adv_batch': ims_adv_batch, 'ims_label': ims_label},
    # args.exp_dir + f'/log/{args.model_data}_correct_diff_reps{reps}.pth')
    # logger.log(f'ims_adv_batch shape:{ims_adv_batch.shape}')
    # logger.log(f'ims_label shape:{ims_label.shape}')
    
    #PartII
    results2 = '/home/yuandu/MEGA/data/cifar10_wrong_diff_reps50_1st.pth'
    # # Load saved tensors
    saved_data2 = torch.load(results2)
    # for key, value in saved_data2.items():  # Iterate over key-value pairs
    #     print(f'key: {key}')

    ims_adv = saved_data2['ims_adv_batch']
    labels = saved_data2['ims_label']
    print(f'labels shape: {labels.shape}') #53 images

    # purify a stable base image, compare correct logit and 2nd choice logit
    #find the images where mu1>mu2:
    # for index in range(len(labels)):
    # for index in range(1):
    #     times = 30
    #     X = ims_adv[index].unsqueeze(0)
    #     y = labels[index]
    #     avg_acc, logit1_record, logit2_record = purify_times(X, y, args, clf, model, scheduler, times)
    #     for 
    #     logit1_mean, logit2_mean, logit1_std, logit2_std, sample_size = calculate_sample_size(logit1_record, logit2_record)
    #     if logit1_mean>logit2_mean:
    #         print(f'logit1_mean>logit2_mean: image index {index}')
    #     plot(args, logit1_record, logit2_record, logit1_mean, logit2_mean, logit1_std, logit2_std, sample_size, times, avg_acc, index)


    #PartIII    
    for index in range(4,len(labels)):
        reps_list =[1,20,50,100]
        times1 = 1000 #the stable reference
        times2 = 100 #the validation times
        X = ims_adv[index].unsqueeze(0)
        y = labels[index]
        avg_agree = class_times(X, y, args, clf, model, scheduler, times1, times2, reps_list, index, logger)
        # if dist.get_rank() == 0:
            # print(f'probablity to agree with the stable label prediction running {times2} times:{avg_agree:.2f}% for reps {reps} for image {index}')
        # logger.log(f'probablity to agree with the stable label prediction running {times2} times:{avg_agree:.2f}% for reps {times1} for image {index}')

    # dist.barrier()

    # # close dist
    # cleanup_dist()
        


if __name__ == '__main__':
    # distribution params
    n_gpus = torch.cuda.device_count()
    print('Running experiment with {} GPUS'.format(n_gpus))
    # world_size = n_gpus

    args, config = parse_args_and_config()
    reload_eval(args, config)
    # mp.spawn(reload_eval, args=(args, config, world_size), nprocs=world_size, join=True)