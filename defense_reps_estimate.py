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
    # parallel states for either EOT attack grad or EOT defense with large-sample evaluation of stochastic classifier
    X_repeat = X.repeat([reps, 1, 1, 1])
    if requires_grad:
        if args.classifier_name == 'wideresnet':
            logits = clf(X_repeat)
        else:
            logits = clf((X_repeat + 1) * 0.5)
    else:
        with torch.no_grad():
            if args.classifier_name == 'wideresnet':
                logits = clf(X_repeat.data)
            else:
                logits = clf((X_repeat.data + 1) * 0.5)

    # finite-sample approximation of stochastic classifier prediction
    if reps>1:
        # print(f'logits shape: {logits.shape}')
        logits_pred = logits.view([reps, int(logits.shape[0]/reps), logits.shape[1]]).mean(0)
    else:
        logits_pred = logits
    _, y_pred = torch.max(logits_pred, 1)
    correct = torch.eq(y_pred, y)

    logit_correct = logits_pred[torch.arange(logits_pred.size(0)), y]  # Get the correct logits
    logit12, _ = torch.topk(logits_pred, 2, dim=1)  # Get top-1 and top-2 logits
    logit2 = logit12[:, 1]  # 2nd choice, Default to top-2 logit 
    # If logit1 equals logit_correct, keep logit2 as logit1
    logit2[logit12[:, 0] == logit_correct] = logit12[:, 0][logit12[:, 0] == logit_correct]  
    
    # Get top-1 
    logit1, _ = torch.topk(logits_pred, 1, dim=1)

    return correct, logit1, logit_correct, logit2

def purify_and_eval(args, clf, X, y, model, scheduler, reps):
    X = X.cuda()
    y = y.cuda()
    X_purified, x_pure_list, noi_pure_list, curr_ts, next_ts = purify(args, model, scheduler, X)
    correct, logit1, logit_correct, logit2 = predict_logits(args, clf, X_purified, y, reps)
    return correct, logit1, logit_correct, logit2

def purify_times(X, y, args, clf, model, scheduler, times):
    X.cuda()
    y.cuda()
    correct_sum = 0  # Total count of correctly classified images
    logit1_record = torch.zeros(0)  # List to record top-1 logits
    logit2_record =  torch.zeros(0)  # List to record top-2 logits

    for i in range(times):  # Classify `times` more times
        corr, logit1, logit_correct, logit2 = purify_and_eval(args, clf, X, y, model, scheduler, 1)
        correct_sum += corr.long().sum()
        logit1_record = torch.cat((logit1_record,logit_correct.cpu()), dim=0)
        logit2_record = torch.cat((logit2_record,logit2.cpu()), dim=0)

    avg_acc = 100*correct_sum/times
    
    return avg_acc, logit1_record, logit2_record

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


def class_times(X, y, args, clf, model, scheduler, times1, times2, reps):

    agree_sum = 0  # Total count of correctly classified images
    logit1_record = torch.zeros(0)  # List to record top-1 logits

    for i in range(times1):  
        corr, logit1, _, _  = predict_logits(args, clf, X, y, 1)
        logit1_record = torch.cat((logit1_record,logit1.cpu()), dim=0)
    
    # Calculate mean logit1 and prediction labels
    mean_logit1 = logit1_record.view(times1, -1, logit1_record.shape[1]).mean(dim=0)
    _, y_pred = torch.max(mean_logit1, 1)
    correct = torch.eq(y_pred, y)
    if dist.get_rank() == 0:
        print(f'prediction correct true or not false: {correct}')
    
    for i in range(times2): 
        correct_reps, logit1, _, _  = purify_and_eval(args, clf, X, y, model, scheduler, reps)
        _, y_pred_reps = torch.max(logit1, 1)
        agree = torch.eq(y_pred_reps.cpu(), y_pred.cpu())
        agree_sum +=agree

    avg_agree = 100*agree_sum.sum()/times2
    
    return avg_agree

def wrong_img(X, y, class_path, args, clf, model, scheduler, reps):
    adv_class = torch.zeros(0).long()
    ims_adv_list = []  # Use a list to dynamically store the adversarial images
    ims_label = torch.zeros(0).long()

    for ind in range(class_path.shape[1]):
        if class_path[-1, ind] == 0:
            correct, _, _, _ = purify_and_eval(args, clf, X[ind].unsqueeze(0), y[ind].unsqueeze(0), model, scheduler, reps)
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

def reload_eval(rank, args, config, world_size): 
    # set up distribution
    setup_dist(rank, world_size)
    args, config = parse_args_and_config()
    args.batch_size = 1
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # results1 = '/home/yuandu/DREBM/Result/diffpure/log/results_defense_reps50_6_10.pth'
    # # Load saved tensors
    # saved_data1 = torch.load(results1)
    # # for key, value in saved_data.items():  # Iterate over key-value pairs
    # #     print(f'key: {key}')
    
    # ims_orig = saved_data1['ims_orig']
    # ims_adv = saved_data1['ims_adv']
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
    setup_exp(args.exp_dir, args.seed, folder_list=['log'], code_file_list=['defense_reps_estimate.py'], use_careful_setup=False)
    log_path=f'{args.exp_dir}/log_reps_estimate.txt'
    logger = Logger(log_path)

    # Evaluate original and adversarial images
    config.classifier_name = "cifar10-wideresnet-28-10"
    clf = get_classifier(args, rank, device).to(device)

    #load purify model
    model, scheduler = load_purify_model(rank, args, config, device)
    model.eval()
    
    #PartI
    # save the wrong predicted but later correct images in the validation
    # reps = 150
    # adv_class, ims_adv_batch, ims_label = wrong_img(ims_adv, labels, class_path, args, clf, model, scheduler, reps)
    # torch.save({'ims_adv_batch': ims_adv_batch, 'ims_label': ims_label},
    # args.exp_dir + f'/log/{args.model_data}_wrong_diff_reps{reps}.pth')
    # logger.log(f'ims_adv_batch shape:{ims_adv_batch.shape}')
    # logger.log(f'ims_label shape:{ims_label.shape}')
    
    #PartII
    results2 = '/home/yuandu/DREBM/Result/log/borderline_reps50_diff.pth'
    # # Load saved tensors
    saved_data2 = torch.load(results2)
    for key, value in saved_data.items():  # Iterate over key-value pairs
        print(f'key: {key}')

    # ims_adv = saved_data2['ims_adv_batch']
    # labels = saved_data2['ims_label']

    # # purify a stable base image, compare correct logit and 2nd choice logit
    # index = 2
    # times = 10000
    # X = ims_adv[index].unsqueeze(0)
    # y = labels[index]
    # avg_acc, logit1_record, logit2_record = purify_times(X, y, args, clf, model, scheduler, times)
    # logit1_mean, logit2_mean, logit1_std, logit2_std, sample_size = calculate_sample_size(logit1_record, logit2_record)
    # plot(args, logit1_record, logit2_record, logit1_mean, logit2_mean, logit1_std, logit2_std, sample_size, times, avg_acc, index)


    #PartIII    
    # index = 1 
    # reps_list =[1,5,10,50,100]
    # times1 = 10000 #the stable reference
    # times2 = 100 #the validation times
    # X = ims_adv[index].unsqueeze(0)
    # y = labels[index]
    # for reps in reps_list:
    #     avg_agree = class_times(X, y, args, clf, model, scheduler, times1, times2, reps)
    #     if dist.get_rank() == 0:
    #         print(f'probablity to agree with the stable label prediction:{avg_agree:.2f}% for reps {reps} for image {index}')
    #     logger.log(f'probablity to agree with the stable label prediction:{avg_agree:.2f}% for reps {reps} for image {index}')

    dist.barrier()

    # close dist
    cleanup_dist()
        


if __name__ == '__main__':
    # distribution params
    n_gpus = torch.cuda.device_count()
    print('Running experiment with {} GPUS'.format(n_gpus))
    world_size = n_gpus

    args, config = parse_args_and_config()
    mp.spawn(reload_eval, args=(args, config, world_size), nprocs=world_size, join=True)