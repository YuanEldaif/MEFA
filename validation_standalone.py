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

#check adv and orignal images distance
def distance(x1, x2):
    elementwise_difference = torch.abs(x1 - x2)
    max_diff = torch.max(elementwise_difference)
    # Define the epsilon value (e.g., 8/255)
    epsilon = 2*8 / 255

    if max_diff <= epsilon:
        print("The maximum difference between the tensors is within the norm epsilon.")
    else:
        print("The maximum difference between the tensors is outside the norm epsilon.")
    return max_diff, epsilon

def classify_and_evaluate_all(args, clf, images, labels, model, scheduler, rank, world_size, device, reps):
    dataset = TensorDataset(images, labels)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset,num_replicas=world_size,rank=rank)
    loader = DataLoader(dataset,batch_size=args.batch_size,num_workers=0,pin_memory=True,sampler=sampler,drop_last=False,shuffle=False)
    # Set end_batch based on the number of batches
    num_batches = len(loader)
    args.end_batch = num_batches
    total = num_batches*args.batch_size*world_size
    if rank == 0:
        print(f"Number of batches: {num_batches}")
        print(f"end_batch set to: {args.end_batch}")
        print(f"batch_size: {args.batch_size}")
        print(f"total: {total}")

    correct_adv_sum = torch.zeros(0).to(device)
    for batch, (X_batch, y_batch) in enumerate(loader):
        if (batch + 1) < args.start_batch:
            continue
        elif (batch + 1) > args.end_batch:
            break
        else:
            X = X_batch.to(device)
            batch_labels = y_batch.to(device)
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

  
            correct_adv, _, _= predict_logits(args, clf, X_purified, batch_labels, requires_grad=True, reps=reps, eot_defense_ave='logits', eot_attack_ave='logits')
            correct_adv_sum = torch.cat((correct_adv_sum, correct_adv.to(device)), dim=0)
            correct_adv_cpu = gather_on_cpu(correct_adv_sum)

    accuracy_adv = 100 * correct_adv_cpu.sum() / total
    
    return accuracy_adv, correct_adv_cpu.sum(), total

def reload_eval(rank, args, config, world_size): 
    # set up distribution
    setup_dist(rank, world_size)
    args, config = parse_args_and_config()
    args.batch_size = 1
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    results = '/home/yuandu/MEGA/Result/diffpure/cifar10/2025_03_19_16_09/log/cifar10_pgdattack_defense_reps1.pth'

    # Load saved tensors
    saved_data = torch.load(results)
    ims_orig = saved_data['ims_orig']
    ims_adv = saved_data['x_final_adv']
    labels = saved_data['labs']

    if rank == 0:
        print('ims_adv shape:',ims_adv.shape)
        print('labels shape:',labels.shape)
        print(f'min and max value: original min:{ims_orig.min()} max:{ims_orig.max()} and adv images:min:{ims_adv.min()} max:{ims_adv.max()}' )
        print(f'distance between original and adv image: {distance(ims_orig,ims_adv)}')


    # Evaluate original and adversarial images
    config.classifier_name = "cifar10-wideresnet-28-10"
    clf = get_classifier(args, rank, device)

    #load purify model
    model, scheduler = load_purify_model(rank, args, config, device)
    model.eval()

    
    start_time_diff = time.time()
    accuracies = {}
    accuracies_adv = {}
    for reps in args.reps_list:
        accuracy_nat, _, total = classify_and_evaluate_all(args, clf, ims_orig, labels, model, scheduler, rank, world_size, device, reps=reps)
        accuracy_adv, _, _ = classify_and_evaluate_all(args, clf, ims_adv, labels, model, scheduler, rank, world_size, device, reps=reps)
        accuracies[reps] = accuracy_nat  # Store accuracy with reps as the key
        accuracies_adv[reps] = accuracy_adv

    average_accuracy = np.mean(list(accuracies.values()))
    std_deviation = np.std(list(accuracies.values()))
    average_accuracy_adv = np.mean(list(accuracies_adv.values()))
    std_deviation_adv = np.std(list(accuracies_adv.values()))

    if dist.get_rank() == 0:
        minutes, seconds = divmod(time.time() - start_time_diff, 60)
        print(f'{total} image attack eval total time: {int(minutes):0>2}:{seconds:05.2f}')
        for reps, accuracy in accuracies.items():
            print(f"Average Natural Accuracy for reps={reps}: {accuracy:.2f}%")
        for reps, accuracy_adv in accuracies_adv.items():
            print(f"Average Adversarial Accuracy for reps={reps}: {accuracy_adv:.2f}%")
        print(f"Average Natural Accuracy: {average_accuracy:.2f}%; Adversarial Accuracy: {average_accuracy_adv:.2f}%")
        print(f"Standard Deviation of Natural Accuracy: {std_deviation:.2f}%; Adversarial Accuracy: {std_deviation_adv:.2f}%")

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




