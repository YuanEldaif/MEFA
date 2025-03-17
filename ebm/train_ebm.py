#######################################
# ## TRAIN EBM USING IMAGE DATASET ## #
#######################################

import os
import time
import argparse
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from tqdm import tqdm

from nets import create_net

from utils_ebm import (
    load_data,
    set_seed,
    plot_ims,
    plot_checkpoint,
    setup_dist,
    setup,
    cleanup_dist,
    gather_on_cpu,
    sample_ebm,
    sample_data,
    update_ebm,
)
from fid_pt import run_fid_training


def train_ebm(rank, args, world_size):

    ########################
    # ## TRAINING SETUP # ##
    ########################

    # Set the device for this process
    device = torch.device(f'cuda:{rank}')
    set_seed(args.seed)
    # set up distribution
    setup_dist(rank, world_size)
    
    setup(args.exp_dir, ['shortrun','checkpoints', 'fid'])

    if dist.get_rank() == 0:
        print('Setting up network and optimizer...')
    ebm = create_net(args, args.ebm_type, args.image_dims[1], args.ebm_nf).to(device)    
    ebm = DDP(ebm, device_ids=[rank])
    # Create optimizer
    ebm_optim = torch.optim.Adam(ebm.parameters(), lr=args.lr)

    # Create learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(ebm_optim, milestones=args.lr_decay_milestones, gamma=args.lr_decay_factor)
 

    if dist.get_rank() == 0:
        print('Processing data...')
    # get loader from training data
    train_iterator = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_dims[1],
        data_type=args.data_type,
        rank=dist.get_rank(),
        world_size=world_size,
        random_crop=False,
        random_flip=False,
        num_workers=0
    )
    data_samples = next(train_iterator)[0]
    if rank == 0:
        plot_ims(os.path.join(args.exp_dir, 'shortrun/data.pdf'), data_samples)
    dist.barrier()

    # get separate iterator for fid (no random crop or random flip)
    if args.check_fid:
        fid_iterator = load_data(
            data_dir=args.data_dir_val,
            batch_size=args.batch_size,
            image_size=args.image_dims[1],
            data_type=args.data_type,
            rank=dist.get_rank(),
            world_size=world_size,
            random_crop=False,
            random_flip=False,
            num_workers=0
        )

    
    ############################
    # ## TRAINING FUNCTION  ## #
    ############################
    def train_loop_fn(iterator):
        ebm_loss_epoch = 0
        gard_norms_epoch = 0

        # obtain positive samples from data
        images_data = sample_data(iterator, args.data_epsilon, device)

        # obtain negative samples from model
        images_samp, grad_norm = sample_ebm(args, ebm, images_data)

        # update weights of ebm
        loss = update_ebm(args, ebm, ebm_optim, images_data, images_samp)

        # Update Loss and Grad Norm
        ebm_loss_epoch += loss.item()
        gard_norms_epoch += grad_norm.item()

            # Set description on master device only
        if rank == 0:
            pbar.set_description(f'Epoch {epoch}/{args.epochs} | Loss: {ebm_loss_epoch:.3e}| Grad Norm: {gard_norms_epoch:.3e}')
                # pbar.set_description(f'Epoch {epoch}/{args.epochs} Iter {batch+1}/{len(train_loader)} | Loss: {ebm_loss_epoch/(batch+1):.3e}| Grad Norm: {gard_norms_epoch/(batch+1):.3e}')
        
         # Update tqdm on master device only
        if rank == 0:
            pbar.update(1)

        # Update learning rate
        lr_scheduler.step()

        return ebm_loss_epoch, gard_norms_epoch, images_samp
    
    #######################
    # ## TRAINING LOOP ## #
    #######################
    # Initialize tqdm on master device only
    if rank == 0:
        print('Training has started.')
        pbar = tqdm(total=args.epochs,ncols=100)
        # start timer
        time_check = time.time()
    dist.barrier()

    # containers for diagnostic records
    loss_record = []#torch.zeros(args.epochs, device=rank)  # energy difference between positive and negative samples
    grad_norm_record = []#torch.zeros(args.epochs, device=rank)  # average image gradient magnitude along Langevin path
    rec_names = ['EBM Loss', 'Langevin Gradient Norm']
    fid_rec = torch.zeros([0])

    for epoch in range(1, args.epochs+1):
        loss, grad_norm, images_samp = train_loop_fn(train_iterator)
         # record diagnostics
        loss_record.append(loss)
        grad_norm_record.append(grad_norm)
        
        # Save checkpoints and final model
        if (epoch % args.checkpoint_freq == 0 or epoch in [1,5,args.epochs]) and rank == 0:
            # save network weights
            torch.save(ebm.module.state_dict(), os.path.join(args.exp_dir, 'checkpoints/ebm_{:>06d}.pth'.format(epoch+1)))
            # save optim weights
            torch.save(ebm_optim.state_dict(), os.path.join(args.exp_dir, 'checkpoints/ebm_optim_{:>06d}.pth'.format(epoch+1)))

            plot_checkpoint(loss_record, grad_norm_record, images_samp.detach().cpu().numpy(), epoch, os.path.join(args.exp_dir,f'epoch_{epoch}.png'))
            if epoch > 0:
                print('Time per batch: {:.3f}'.format((time.time() - time_check) / args.checkpoint_freq))
                time_check = time.time()
        dist.barrier()
            

        # calculate unofficial fid and save record
        if args.check_fid and ( epoch == 1 or (epoch + 1) % args.fid_freq == 0 ) :
            fid_rec = run_fid_training(args, fid_iterator, fid_rec, epoch, ebm, device, os.path.join(args.exp_dir,f'samples_{epoch+1}.pdf'))

    if rank == 0:
        print('Training complete.')
    # close dist
    cleanup_dist()

def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    ### Setup Arguments ###
    # parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='seed for reproducibility')
    parser.add_argument('--data_dir', default='/home/yuandu/DREBM/data/cifar10_pt_custom/train/', type=str, help='path to the data directory')
    parser.add_argument('--data_dir_val', default='/home/yuandu/DREBM/data/cifar10_pt_custom/val/', type=str, help='path to the data directory')
    
    # Training
    parser.add_argument('--epochs', type=int, default=33333, metavar='N')
    parser.add_argument('--data_type', type=str, default='cifar10', help='dataset to use')
    parser.add_argument('--image_dims', default=[3,32,32], nargs='+', type=int, metavar='N')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, metavar='N')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='N')
    parser.add_argument('--lr_decay_milestones', nargs='+', type=int, default=[0, 10000, 20000], help='List of epoch indices to decrease learning rate')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1, help='Factor by which learning rate is decreased at each milestone')
    parser.add_argument('--checkpoint_freq', type=int, default=1666, help='how often to save models')
 
    parser.add_argument('--exp_dir', type=str, default='./Result/', help='path to the save the experiment')
    parser.add_argument('--exp_name', type=str, default='ebm', choices=['ebm','hugging_face','diffpure','robust_diff'], help='path to the save the experiment')
    
    
    # EBM Arguments 
    parser.add_argument('--data_epsilon', type=float, default=1.5e-2, metavar='N')
    parser.add_argument('--mcmc_steps', type=int, default=1500, metavar='N')
    parser.add_argument('--ebm_type', type=str, default='ebm_small', choices=['small_ebm','ebm_sngan'], help='ebm type')
    parser.add_argument('--ebm_nf', type=int, default=128, help='channel size')
    parser.add_argument('--langevin_steps', type=int, default=1500, help='Sampling langegin steps')
    parser.add_argument('--mcmc_temp', type=float, default=4e-4, help='Sampling temperature')
    parser.add_argument('--langevin_eps', type=float, default=1e-2, help='Sampling epsilon')
    parser.add_argument('--langevin_init_noise', type=float, default=0, help='Sampling epsilon') 
    parser.add_argument('--ebm_weight_path', default='./weights/small_ebm.pth', type=str, help='path to the model weights directory')

    # FID
    parser.add_argument('--check_fid', default=True, action='store_true',  help='check fid score during training')
    parser.add_argument('--official_fid', default=True, action='store_true',  help='calculate offcial fid score')
    parser.add_argument('--fid_freq', type=int, default=1666, help='how often to calculate fid score')
    parser.add_argument('--num_fid_rounds', type=int, default=210, help='how often to calculate fid score')
    

    # Parse the arguments
    args = parser.parse_args()

    # Create Output Directory (add timestamp)
    args.exp_dir = os.path.join(args.exp_dir, args.exp_name, f'{args.data_type}',time.strftime("%Y_%m_%d_%H_%M", time.localtime()))

    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    
    # Print args into txt file in output directory
    with open(os.path.join(args.exp_dir,'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')

    return args
#################
# ## EXECUTE ## #
#################
if __name__ == '__main__':
    
    # distribution params
    n_gpus = torch.cuda.device_count()
    print('Running training with {} GPUS'.format(n_gpus))
    world_size = n_gpus
    args = parse_args()
    # run distributed training
    mp.spawn(train_ebm, args=(args, world_size), nprocs=world_size, join=True)
