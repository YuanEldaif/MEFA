# either calculate unofficial pt fid, or save samples for official tf1 fid

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.distributed as dist

from utils_ebm import gather_on_cpu, sample_ebm, sample_data

from fid_orig import calculate_fid, calculate_frechet_distance


##########################
# ## HELPER FUNCTIONS ## #
##########################


def prepare_images_for_fid(images):

    # convert to numpy and prepare in uint8 format
    images_numpy = np.clip(images.numpy(), -1, 1).transpose(0, 2, 3, 1)
    images_uint8 = np.rint(255 * (images_numpy + 1) / 2).astype(np.uint8)

    return images_uint8


###########################
# ## FID SAMPLING LOOP ## #
###########################

# save images from to npy files to use original fid code, or calculate unofficial FID score
def run_fid(args, fid_iterator, epoch, ebm, device, save_str='samples.pdf'):
    INCEPTION_LOCAL_WEIGHT_PATH = '/home/notebook/data/personal/US000240/nets_pt/inception_pt/inceptionv3_pt.pth'

    # location to save results
    exp_dir = os.path.join(args.exp_dir, args.exp_name)

    if args.official_fid:
        # containers for used to calculate fid score
        images_1 = torch.zeros([0] + args.image_dims)
        images_2 = torch.zeros([0] + args.image_dims)
    else:
        # load pytorch inception net
        inception = torchvision.models.inception.Inception3()
        inception.load_state_dict(torch.load(INCEPTION_LOCAL_WEIGHT_PATH))
        inception.eval().to(dist.get_rank())
        # empty initial activations
        act1 = torch.zeros((0, 2048))
        act2 = torch.zeros((0, 2048))

    # loop over batches to get samples and/or calculate activations
    for i in range(args.num_fid_rounds):
        # data images
        images_data = sample_data(fid_iterator, args.data_epsilon, device)
        images_sample = sample_ebm(args, ebm, images_data)[0]
        
        # visualize initial/final samples, energy path and gradient path for first batch
        if i == 0 and dist.get_rank() == 0:
            plot_images_data = np.block([[images_data[i*4+j,:,:,:].detach().cpu().numpy() for j in range(4)] for i in range(4)])
            plot_images_sample = np.block([[images_sample[i*4+j,:,:,:].detach().cpu().numpy() for j in range(4)] for i in range(4)])


        if args.official_fid:
            images_data_cpu = gather_on_cpu(images_data)
            images_sample_cpu = gather_on_cpu(images_sample)
            if dist.get_rank() == 0:
                # store batch of samples in banks
                images_1 = torch.cat((images_1, images_data_cpu), 0)
                images_2 = torch.cat((images_2, images_sample_cpu), 0)
        else:
            # get activations with unofficial inception net
            acts_data = gather_on_cpu(inception_acts(inception, images_data))
            acts_samples  = gather_on_cpu(inception_acts(inception, images_sample))
            if dist.get_rank() == 0:
                # add to activation record
                act1 = torch.cat((act1, acts_data), 0)
                act2 = torch.cat((act2, acts_samples), 0)
            

    if dist.get_rank() == 0:
        print('Calculating FID score.')
        if args.official_fid:
            print('Scores will use {} data samples and {} model samples'.
                  format(images_1.shape[0], images_2.shape[0]))
            # official fid score on single GPU
            images_1_uint8 = prepare_images_for_fid(images_1)
            images_2_uint8 = prepare_images_for_fid(images_2)
            fid_score = calculate_fid(images_1_uint8, images_2_uint8, None)
        else:
            print('Scores will use {} data samples and {} model samples'.
                  format(act1.shape[0], act2.shape[0]))
            # calculate unofficial fid score from all activations
            mu1, sigma1 = act1.numpy().mean(axis=0), np.cov(act1.numpy(), rowvar=False)
            mu2, sigma2 = act2.numpy().mean(axis=0), np.cov(act2.numpy(), rowvar=False)
            fid_score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        
        print('FID score: ', fid_score)
        fid_score_pt = torch.tensor([fid_score])
        # Create a figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        plot_images_data = (np.clip(plot_images_data, -1., 1.) + 1) / 2   
        axs[0].imshow(plot_images_data.transpose(1, 2, 0))
        axs[0].axis('off')
        axs[0].set_title(f'Data Image Samples', fontsize=16, fontweight='bold')

        plot_images_sample = (np.clip(plot_images_sample, -1., 1.) + 1) / 2   
        axs[1].imshow(plot_images_sample.transpose(1, 2, 0))
        axs[1].axis('off')
        axs[1].set_title(f'Generated Image Samples', fontsize=16, fontweight='bold')
        
        if args.exp_name =='ebm':
            fig.suptitle(f'Epoch {epoch} Steps {args.langevin_steps} | FID Score: {fid_score:.2f}', fontsize=16, fontweight='bold')

        # Adjust the spacing between subplots
        plt.tight_layout()

        # Save figure
        plt.savefig(save_str)
    else:
        fid_score_pt = None

    dist.barrier()

    return fid_score_pt


# run fid check during model training
def run_fid_training(args, fid_iterator, fid_rec, epoch, ebm, device, save_str):
    if dist.get_rank() == 0:
        print('Getting FID score.')
    # get fid score for current ebm/gen
    fid_score = run_fid(args, fid_iterator, epoch, ebm, device, save_str=save_str)
    if dist.get_rank() == 0:
        # update record of fid values
        fid_rec = torch.cat((fid_rec, fid_score), 0)
        # plot
        step_count = args.fid_freq * (1 + torch.arange(fid_rec.shape[0]))
        exp_folder = os.path.join(args.exp_dir)
        plt.plot(step_count, fid_rec)
        plt.savefig(os.path.join(exp_folder, 'fid', 'fid_plot.png'))
        plt.close()
        np.savetxt(os.path.join(exp_folder, 'fid', 'fid.txt'), fid_rec.numpy())
    dist.barrier()

    return fid_rec


