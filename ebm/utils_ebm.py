import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
import math
import random
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import os
from tqdm import tqdm
from datetime import timedelta
from mpl_toolkits.axes_grid1 import ImageGrid

def gather_on_gpu(tensor):
    tensor_list_gather = [torch.zeros(tensor.shape, device=dist.get_rank())
                          for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list_gather, tensor)
    return tensor_list_gather
    
def gather_on_cpu_nccl(tensor, batch_size=16):
    tensor_list_full = [torch.zeros([0] + list(tensor.shape[1:])) for _ in range(dist.get_world_size())]
    for i in range(int(np.ceil(tensor.shape[0] / batch_size))):
        start_ind = i * batch_size
        end_ind = min((i + 1) * batch_size, tensor.shape[0])
        tensor_list_gather = gather_on_gpu(tensor[start_ind:end_ind])
        tensor_list_full = [torch.cat((tensor_list_full[j], tensor_list_gather[j].cpu()), 0)
                            for j in range(len(tensor_list_full))]
    tensor_gather = torch.cat(tensor_list_full, 0)
    return tensor_gather

def gather_on_cpu_gloo(tensor):
    tensor_gather = [torch.zeros(tensor.shape) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_gather, tensor.cpu())
    tensor_gather = torch.cat(tensor_gather, 0)
    return tensor_gather

def gather_on_cpu(tensor, batch_size=16, dist_type='nccl'):
    if dist_type == 'nccl':
        return gather_on_cpu_nccl(tensor, batch_size)
    elif dist_type == 'gloo':
        return gather_on_cpu_gloo(tensor)
    else:
        raise ValueError('Invalid "dist_type", should be either "nccl" or "gloo".')

# clip image gradient separately for each batch dimension
def clip_grad_norm(grad, clip_val, clip_dims=(1, 2, 3)):
    grad_norm = torch.linalg.vector_norm(grad, ord=2, dim=clip_dims, keepdim=True)
    grad_clipped = (grad_norm < clip_val) * grad + (grad_norm >= clip_val) * clip_val * grad / grad_norm
    xm.mark_step()
    return grad_clipped

# sample positive images from dataset distribution (add noise to ensure min data sd is at least langevin noise sd)
def sample_data(iterator, data_epsilon, device):
    data_samples = next(iterator)[0].to(device)
    return data_samples + data_epsilon * torch.randn_like(data_samples)


def sample_ebm(args, ebm, images_init):

    # iterative langevin updates of MCMC samples
    images_samp = torch.autograd.Variable(images_init.clone(), requires_grad=True)
    grad_norm = torch.zeros(1).to(images_init.device)
    for ell in range(args.mcmc_steps):
        # gradient of ebm
        grad = torch.autograd.grad(ebm(images_samp).sum() / args.mcmc_temp, [images_samp])[0]

        # langevin update
        images_samp.data -= ((args.langevin_eps ** 2) / 2) * grad
        images_samp.data += args.langevin_eps * torch.randn_like(images_samp)
        # update grad norm record
        grad_norm += ((args.langevin_eps ** 2) / 2) * grad.view(grad.shape[0], -1).norm(dim=1).mean()

    return images_samp.detach(), grad_norm.squeeze() / args.mcmc_steps

def update_ebm(args, ebm, optim, images_data, images_samp):
    optim.zero_grad()
    # calculate ML computational loss, do backward pass
    loss = (ebm(images_data).mean() - ebm(images_samp).mean()) / args.mcmc_temp
    loss.backward()

    # update weights
    optim.step()

    return loss.detach().data

def plot_checkpoint(ebm_loss, grad_norm, image_samples, epoch, save_path, channels=3):
    """
    Plots the EBM loss, gradient norm, and image samples at a given epoch

    Args:
        ebm_loss (torch.tensor): EBM loss at each epoch
        grad_norm (torch.tensor): Gradient norm at each epoch
        image_samples (torch.tensor): Image samples at each epoch
        epoch (int): Epoch to plot
        save_dir (str): Directory to save the plot
        channels (int): Number of channels in the image samples
    """
    # Left plot with loss and grad norm
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    ax1 = axs[0]
    if grad_norm is not None:
        ax2 = ax1.twinx()

    line1 = ax1.plot(ebm_loss[0:epoch], 'o-', label='Train Loss', color='g')
    if grad_norm is not None:
        line2 = ax2.plot(grad_norm[0:epoch], 'o-', label='Grad Norm', color='b')

    ax1.set_xlabel('Epoch', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Train Loss', fontsize=16, fontweight='bold')
    if grad_norm is not None:
        ax2.set_ylabel('Grad Norm', fontsize=16, fontweight='bold')
        lines = line1 + line2
    else:
        lines = line1
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc=0)

    if channels == 1:
        all_images = np.block([[image_samples[i*4+j,:,:] for j in range(4)] for i in range(4)])
        all_images = (np.clip(all_images, -1., 1.) + 1) / 2  
        axs[1].imshow(all_images, cmap='gray') 
    elif channels == 3:
        all_images = np.block([[image_samples[i*4+j,:,:,:] for j in range(4)] for i in range(4)]) 
        all_images = (np.clip(all_images, -1., 1.) + 1) / 2
        axs[1].imshow(all_images.transpose(1, 2, 0))
        


    axs[1].axis('off')
    axs[1].set_title(f'Shortrun Image Samples', fontsize=16, fontweight='bold')

    fig.suptitle(f'Epoch {epoch} Info', fontsize=20, fontweight='bold')
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path)

####################
# ## PLOT TOOLS ## #
####################
# save images to file
# def plot_ims(p, x):
#     tv.utils.save_image(torch.clamp(x, -1., 1.), p, normalize=True, nrow=int(args.batch_size ** 0.5))
# visualize images with pixels in range [-1, 1]
def plot_ims(path, ims): 
    if torch.is_tensor(ims):
        ims = ims.permute(0, 2, 3, 1)
        ims = ims.numpy()
    ims = (np.clip(ims, -1., 1.) + 1) / 2

    # dimensions of image grid
    nrows = int(np.ceil(ims.shape[0] ** 0.5))
    ncols = int(np.ceil(ims.shape[0] ** 0.5))

    fig = plt.figure(figsize=(nrows, ncols))
    grid = ImageGrid(
        fig, 111,  # similar to subplot(111)
        nrows_ncols=(nrows, ncols),
        axes_pad=0.05,  # pad between axes in inch.
    )

    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    for ax, im in zip(grid, ims.tolist()):
        im = np.array(im)
        if im.shape[2] == 1:
            im = np.tile(im, (1, 1, 3))
        ax.imshow(im)
        ax.axis("off")
    plt.savefig(path, format="pdf", dpi=2000)
    plt.close()

def load_data(
    data_dir,
    batch_size,
    image_size,
    data_type='cifar10', 
    rank=0,
    world_size=1,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    use_train=True,
    num_workers=0
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the folder containing the file.
        class_names = [os.path.basename(os.path.dirname(path)) for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]

    if data_type == 'cifar10':
        dataset = Cifar10Dataset(
            image_size,
            all_files,
            classes=classes,
            shard=rank,
            num_shards=world_size,
            random_crop=random_crop,
            random_flip=random_flip,
        )
        sampler_kwargs = {'shuffle': not deterministic}
    else:
        raise ValueError('Invalid "data_type".')
    
    # return dataset
    # note: num_workers=0 will be slower, but jobs will not be interrupted by code changes in distributed training.
    # when using num_workers > 0, execution is faster but changes to code can cause a running job to crash.
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, 
                        drop_last=True, **sampler_kwargs)

    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif os.path.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results



class Cifar10Dataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        pil_image = Image.open(path)
        pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=1.0, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

######################
# Argument Functions #
######################

# make folders, save config and code
def setup(exp_dir, folder_list=['log']):
    # make directory for saving results
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    for folder in folder_list:
        if not os.path.exists(os.path.join(exp_dir, folder)):
            os.mkdir(os.path.join(exp_dir, folder))
            
def set_seed(seed):

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
def setup_dist(rank, world_size, dist_type='nccl', master_port='1234', timeout=6000):
    os.environ['MASTER_ADDR'] = 'localhost'
    if master_port is not None:
        os.environ['MASTER_PORT'] = master_port
    # initialize the process group
    dist.init_process_group(dist_type, rank=rank, world_size=world_size, 
                            timeout=timedelta(seconds=timeout))
    # Set the device for each process
    torch.cuda.set_device(rank)

def cleanup_dist():
    dist.destroy_process_group()