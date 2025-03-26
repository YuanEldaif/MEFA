# -*- coding: utf-8 -*-
"""Attack Eval Diffpure sdeint_ajointorch.ipynb

Original file is located at
    https://colab.research.google.com/drive/1SENd02GSuWx2NlSF5onXms0ojC43PwKB
"""

import torch
import torchsde
import torch.nn.functional as F
import torch.nn as nn
import torchvision.utils as tvu
from torch.utils.data import DataLoader, TensorDataset
import torchvision as tv
import torch.multiprocessing as mp
import torch.distributed as dist
import matplotlib.pyplot as plt

import os
import random
import numpy as np
import yaml
import json
import datetime
import time
import argparse

import trampoline
from score_sde.models.ncsnpp import NCSNpp
from score_sde.losses import get_optimizer
from score_sde.models import utils as mutils
from score_sde.models.ema import ExponentialMovingAverage
from score_sde import sde_lib
from utils import *
from purify import *
import logging


def robustness_eval(rank, args, config, world_size):
    
    # Set the device for this process
    device = torch.device(f'cuda:{rank}')
    # set up distribution
    setup_dist(rank, world_size)
    # Create Output Directory (add timestamp)
    args.exp_dir = os.path.join(args.exp_dir, f'{args.model_data}',time.strftime("%Y_%m_%d_%H_%M", time.localtime()))

    # setup seed, make folders and save code
    setup_exp(args.exp_dir, args.seed, folder_list=['log'], code_file_list=['attack_eval.py', 'purify.py', 'utils.py'], use_careful_setup=False)
    # save config
    with open(os.path.join(args.exp_dir, 'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')

    if dist.get_rank() == 0:
        print('Loading data and classifier nets.')
    
    dist.barrier()
    
    # data loader load the first 510 for 3 gpus.
    logger = Logger1(file_name=f'{args.exp_dir}/log.txt', file_mode="w+", should_flush=True)
    
    if not args.use_saved_data:
        attack_loader = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        data_type=args.data_type,
        rank=dist.get_rank(),
        world_size=world_size,
        class_cond=True,
        random_crop=False,
        random_flip=False
        )
    else:
        results = '/home/yuandu/MEGA/data/data_eval.pth'
        # Load saved tensors
        saved_data = torch.load(results)
        ims_orig = saved_data['ims_orig'][:args.n_images]
        labels = saved_data['labs'][:args.n_images]
        dataset = TensorDataset(ims_orig, labels)
        attack_loader = DataLoader(dataset, batch_size=args.n_images, pin_memory=True, num_workers=0, shuffle=False)

    clf = get_classifier(args, rank, device)

    #original accuracy:
    orig_acc, n = get_accuracy(args, attack_loader, clf, rank, world_size, device)
    if rank == 0:
        print('original accuracy: {:.2%} on {} images'.format(orig_acc, n))
    
    #load purify model
    model, scheduler = load_purify_model(rank, args, config, device)

    ########################################
    # ## ATTACK CLASSIFIER AND PURIFIER ## #
    ########################################

    # defense record for over attacks
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

    log_path=f'{args.exp_dir}/log_adv.txt'
    logger = Logger(log_path)

    labs_device = torch.zeros(0).long().to(device)
    ims_orig_device = torch.zeros(0).to(device)
    x_final_adv_device = torch.zeros(0).to(device)
    acc_device = torch.zeros(0).to(device)
    nat_acc_device = torch.zeros(0).to(device)
    grad_device = torch.zeros(0).to(device)
 
    # run adversarial attacks on samples from image bank in small batches
    if rank == 0:
        print('\nAttack has begun.\n----------')
        start_time_total = time.time()
    for batch, (X_batch, y_batch) in enumerate(attack_loader):
        if (batch + 1) < args.start_batch:
            continue
        elif (batch + 1) > args.end_batch:
            break
        else:
            # load to device
            X_batch = X_batch.to(device)
            if not args.use_saved_data:
                y_batch = y_batch["y"].to(device)
            else:
                y_batch = y_batch.to(device)

            # record original states and labels
            ims_orig_device = torch.cat((ims_orig_device, X_batch), dim=0)
            labs_device = torch.cat((labs_device, y_batch), dim=0)

            # attack images using setting in config
            start_time_diff = time.time()
            if args.APGD:
                grad_batch, nat_acc, acc, ims_adv_final = attack_batch_auto(args, model, scheduler, clf, X_batch, y_batch, batch, device)
            else:
                grad_batch, nat_acc, acc, ims_adv_final = attack_batch(args, model, scheduler, clf, X_batch, y_batch, batch, device)

            if rank == 0:
                minutes, seconds = divmod(time.time() - start_time_diff, 60)
                print("1 batch attack eval time: {:0>2}:{:05.2f}".format(int(minutes), seconds))
                logger.log(f'{args.batch_size} batch attack eval total time: {int(minutes):0>2}:{seconds:05.2f}')

            dist.barrier()

            # update defense records
            grad_device = torch.cat((grad_device, grad_batch.to(device)), dim=0)
            x_final_adv_device = torch.cat((x_final_adv_device, ims_adv_final.to(device)), dim=0)
            acc_device = torch.cat((acc_device, acc.to(device)), dim=0)
            nat_acc_device = torch.cat((nat_acc_device, nat_acc.to(device)), dim=0)

            # gather results across devices
            labs = gather_on_cpu(labs_device.float()).long()
            ims_orig = gather_on_cpu(ims_orig_device)
            x_final_adv = gather_on_cpu(x_final_adv_device)
            acc = gather_on_cpu(acc_device)
            nat_acc = gather_on_cpu(nat_acc_device)
            grad = gather_on_cpu(grad_device.float())

            if rank == 0:
                print('Attack concluded on Batch {} of {}. Natural secured images {} of {}, Accuray {:.2f} \n------ Total Secure Images: {} of {}, Accuracy {:.4f} \n-----------'.
                    format(batch - args.start_batch + 2, args.end_batch - args.start_batch + 1,
                        nat_acc.sum(), nat_acc.shape[0], nat_acc.sum()/nat_acc.shape[0],
                        acc.sum(), acc.shape[0], acc.sum()/acc.shape[0] ))
                logger.log(f'Attack concluded on Batch {batch - args.start_batch + 2} of {args.end_batch - args.start_batch + 1}. Natural secured images {nat_acc.sum()} of {nat_acc.shape[0]}, \
                Accuray {(nat_acc.sum()/nat_acc.shape[0]):.4f} \n------ Total Secure Images: {acc.sum()} of {acc.shape[0]}, Accuracy {(acc.sum()/acc.shape[0]):.4f} \n-----------')
                # save attack info
                if args.bpda_only:
                    torch.save({'ims_orig': ims_orig, 'labs': labs, 'nat_acc': nat_acc, 'acc': acc, 'x_final_adv': x_final_adv, 'grad':grad},
                     args.exp_dir + f'/log/{args.model_data}_bpdattack_defense_reps{args.eot_defense_reps}.pth')
                else: 
                    torch.save({'ims_orig': ims_orig, 'labs': labs, 'nat_acc': nat_acc, 'acc': acc, 'x_final_adv': x_final_adv, 'grad':grad},
                     args.exp_dir + f'/log/{args.model_data}_pgdattack_defense_reps{args.eot_defense_reps}.pth')
            dist.barrier()
            
    def classify_and_evaluate_all(args, clf, images, labels, model, scheduler, device, reps):
        dataset = TensorDataset(images, labels)
        # sampler = torch.utils.data.distributed.DistributedSampler(dataset,num_replicas=world_size,rank=rank)
        loader = DataLoader(dataset,batch_size=args.batch_size,num_workers=0,pin_memory=True,drop_last=False,shuffle=False)
        # Set end_batch based on the number of batches
        num_batches = len(loader)
        args.end_batch = num_batches
        total = num_batches*args.batch_size
        print(f"Number of batches: {num_batches}")
        print(f"end_batch set to: {args.end_batch}")
        print(f"batch_size: {args.batch_size}")
        print(f"total: {total}")

        correct_adv_sum = 0
    
        for batch, (X_batch, y_batch) in enumerate(loader):
            if (batch + 1) < args.start_batch:
                continue
            elif (batch + 1) > args.end_batch:
                break
            else:
                batch_images = X_batch.to(device)
                batch_labels = y_batch.to(device)
                X_repeat = batch_images.repeat([reps, 1, 1, 1])
                args.pytorch == False
                X_repeat_purified, x_pure_list, noi_pure_list, curr_ts, next_ts = purify(args, model, scheduler, X_repeat)
  
                correct_adv, _, _= predict_logits(args, clf, X_repeat_purified, batch_labels, requires_grad=False, reps=reps, eot_defense_ave='logits', eot_attack_ave='logits')
                correct_adv_sum += correct_adv.sum().item()    

        accuracy_adv = 100 * correct_adv_sum / total
        return accuracy_adv

    def classify_and_evaluate_all_multigpu(args, clf, images, labels, model, scheduler, rank, world_size, device, reps):
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
    
        return accuracy_adv

    if args.bpda_only:
        results = args.exp_dir + f'/log/{args.model_data}_bpdattack_defense_reps{args.eot_defense_reps}.pth'
    else: 
        results = args.exp_dir + f'/log/{args.model_data}_pgdattack_defense_reps{args.eot_defense_reps}.pth'
    # Load saved tensors
    saved_data = torch.load(results)
    ims_orig = saved_data['ims_orig']
    ims_adv = saved_data['x_final_adv']
    labels = saved_data['labs']
    
    accuracies = {}
    accuracies_adv = {}
    for reps in args.reps_list:
        accuracy_nat = classify_and_evaluate_all_multigpu(args, clf, ims_orig, labels, model, scheduler, rank, world_size, device, reps=reps)
        accuracy_adv = classify_and_evaluate_all_multigpu(args, clf, ims_adv, labels, model, scheduler, rank, world_size, device, reps=reps)
        accuracies[reps] = accuracy_nat  # Store accuracy with reps as the key
        accuracies_adv[reps] = accuracy_adv

    average_accuracy = np.mean(list(accuracies.values()))
    std_deviation = np.std(list(accuracies.values()))
    average_accuracy_adv = np.mean(list(accuracies_adv.values()))
    std_deviation_adv = np.std(list(accuracies_adv.values()))

    if dist.get_rank() == 0:
        minutes, seconds = divmod(time.time() - start_time_total, 60)
        logger.log(f'{n} image attack eval total time: {int(minutes):0>2}:{seconds:05.2f}')
        for reps, accuracy in accuracies.items():
            logger.log(f"Average Natural Accuracy for reps={reps}: {accuracy:.2f}%")
        for reps, accuracy_adv in accuracies_adv.items():
            logger.log(f"Average Adversarial Accuracy for reps={reps}: {accuracy_adv:.2f}%")
        logger.log(f"Average Natural Accuracy: {average_accuracy:.2f}%; Adversarial Accuracy: {average_accuracy_adv:.2f}%")
        logger.log(f"Standard Deviation of Natural Accuracy: {std_deviation:.2f}%; Adversarial Accuracy: {std_deviation_adv:.2f}%")

    dist.barrier()

    # close dist
    cleanup_dist()

    
def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    ### Setup Arguments ###
    # parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='seed for reproducibility')
    parser.add_argument('--data_dir', default='/home/yuandu/MEGA/data/cifar10_pt_custom/val/', type=str, help='path to the data directory')
    
    parser.add_argument('--data_type', type=str, default='cifar10', help='dataset to use')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')

    parser.add_argument('--n_images', type=int, default=60, help='total saved images for eval')
    parser.add_argument('--image_size', type=int, default=32, help='image size')
    parser.add_argument('--start_batch', type=int, default=1, help='start batch number')
    parser.add_argument('--end_batch', type=int, default=20, help='end batch number')
    parser.add_argument('--exp_dir', type=str, default='./Result/', help='path to the save the experiment')
    parser.add_argument('--exp_name', type=str, default='ebm', choices=['ebm','hugging_face','diffpure','hf_DDPM','robust_diff'], help='path to the save the experiment')
    parser.add_argument('--model_data', type=str, default='cifar10', choices=['cifar10','food','cinic10'], help='dataset that the model pre-trained on')

    # EBM Arguments 
    parser.add_argument('--langevin_steps', type=int, default=1500, help='Sampling langegin steps')
    parser.add_argument('--mcmc_temp', type=float, default=1e-4, help='Sampling temperature')
    parser.add_argument('--langevin_eps', type=float, default=1e-2, help='Sampling epsilon')
    parser.add_argument('--langevin_init_noise', type=float, default=1.5e-2, help='Sampling epsilon') 
    parser.add_argument('--ebm_weight_path', default='./weights/smooth_small_ebm_softrelu.pth', type=str, help='path to the model weights directory')
    parser.add_argument('--pytorch', default=False, action='store_true', help='use pytorch attack')

    # Diffusion Arguments
    # diffpure
    parser.add_argument('--diffusion_type', type=str, default='ddpm', choices=['ddpm', 'sde'], help='choose diffusion type')
    parser.add_argument('--score_type', type=str, default='score_sde', choices=['guided_diffusion', 'score_sde'], help='choose score type')
    parser.add_argument('--model_dir', type=str, default='./weights/diffpure/checkpoint_8.pth')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--t', type=int, default=100, help='Sampling noise scale')
    parser.add_argument('--t_delta', type=int, default=0, help='Perturbation range of sampling noise scale')
    parser.add_argument('--rand_t', default=False, action='store_true', help='Decide if randomize sampling noise scale')
    parser.add_argument('--use_bm', default=False, action='store_true', help='whether to use brownian motion')
    # hugging_face
    parser.add_argument('--purify_t', type=int, default=100, help='Sampling noise scale') 
    parser.add_argument('--num_t_steps', type=int, default=1000, help='training t steps') 
    parser.add_argument('--diff_output', type=str, default='epsilon', choices=['epsilon', 'x_start'], help='choose diffusion model prediction')
    parser.add_argument('--t_schedule', type=str, default='linear', choices=['linear', 'cosine'], help='choose scheduler')
    parser.add_argument('--eta', type=int, default=1, help='ddim 0 or ddpm 1') 
    parser.add_argument('--unet_channels', type=int, default=(128, 128, 256, 256, 512, 512), nargs='+', help='Number of channels in each block of the UNet')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='Number of residual blocks')
    parser.add_argument('--unet_weight_path', default='./weights/cinic10_DDPM[250].pt', type=str, help='path to the model weights directory')

    # Adv Arguments
    parser.add_argument('--use_saved_data', default=True, action='store_true', help='using saved data')
    parser.add_argument('--APGD', default=False, action='store_true', help='use APGD attack')
    parser.add_argument('--bpda_only', default=False, action='store_true', help='use BPDA only attack')
    parser.add_argument('--classifier_name', type=str, default='cifar10-wideresnet-28-10', choices=['wideresnet', 'cifar10-wideresnet-28-10'], help='which classifier to use')
    parser.add_argument('--clf_weight_path', type=str, default='./weights/clf.pth')
    parser.add_argument('--eot_defense_ave', type=str, default='logits', choices=['logits', 'softmax', 'logsoftmax'], help='choose defense logit prediction type')
    parser.add_argument('--eot_attack_ave', type=str, default='logits', choices=['logits', 'softmax', 'logsoftmax'], help='choose attack ogit prediction type')
    parser.add_argument('--adv_steps', type=int, default=100, help='number of attack steps')
    parser.add_argument('--adv_rand_start', default=False, action='store_true', help='attack random start')
    parser.add_argument('--adv_norm', type=str, default='Linf', choices=['Linf', 'L2'], help='attack norm')
    parser.add_argument('--adv_eps', type=float, default=8*2/255, help='perturbation size linf 8*2/255 0.5*2')
    parser.add_argument('--adv_eta', type=float, default=2*2/255, help='perturbation step size')
    parser.add_argument('--eot_defense_reps', type=int, default=1, help='number of eot replicates for defense')
    parser.add_argument('--eot_attack_reps', type=int, default=20, help='number of eot replicates for attack')
    parser.add_argument('--grad_ckpt',  default=True, action='store_true',help='use gradiant check point for diffusion model')
    parser.add_argument('--log_freq', type=int, default=20, help='frequency to print the eval result')

    # Validation
    parser.add_argument('--reps_list', nargs='+', type=int, default=[1, 1, 1], help='List of replicas')

    # Parse the arguments
    args = parser.parse_args()
    with open('./configs/cifar10.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
        new_config = dict2namespace(config)
    
    # Print args into txt file in output directory
    args.exp_dir = os.path.join(args.exp_dir, args.exp_name)
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))
    
    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    return args, new_config

#################
# ## EXECUTE ## #
#################

if __name__ == '__main__':
    # distribution params
    n_gpus = torch.cuda.device_count()
    print('Running experiment with {} GPUS'.format(n_gpus))
    world_size = n_gpus

    args, config = parse_args_and_config()
    mp.spawn(robustness_eval, args=(args, config, world_size), nprocs=world_size, join=True)
