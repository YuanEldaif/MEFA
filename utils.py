# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import sys
import argparse
from typing import Any
from PIL import Image
import numpy as np
import math
import random
import os
from datetime import timedelta


import torch
import torch.nn as nn
import torchvision.models as models
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms

from robustbench import load_model
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


class Logger1(object):
    """
    Redirect stderr to stdout, optionally print stdout to a file,
    and optionally force flushing on both stdout and the file.
    """

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: str) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def update_state_dict(state_dict, idx_start=9):

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[idx_start:]  # remove 'module.0.' of dataparallel
        new_state_dict[name]=v

    return new_state_dict


# ------------------------------------------------------------------------
def get_accuracy(args, data_loader, model, rank, world_size, device):

    acc = torch.zeros(0).to(device)
    for batch, (x_orig, y_orig) in enumerate(data_loader):
        if (batch + 1) < args.start_batch:
            continue
        elif (batch + 1) > args.end_batch:
            break
        else:
            if batch == 0:
                if rank == 0:
                    plot_ims(os.path.join(args.exp_dir, 'sample.pdf'), x_orig)
            x = x_orig.to(device)
            if not args.use_saved_data:
                y = y_orig["y"].to(device)
            else:
                y = y_orig.to(device)
            
            with torch.no_grad():
                if args.classifier_name=='cifar10-wideresnet-28-10':
                    logits = model((x.data + 1) * 0.5)
                else:
                    logits = model(x.data)
            correct = torch.eq(logits.max(1)[1], y)
            acc = torch.cat((acc, correct), dim=0)
            acc_cpu = gather_on_cpu(acc).sum()
    
    n = acc.shape[0]*world_size
    rate = acc_cpu/n

    return rate, n


def get_image_classifier(classifier_name):
    class _Wrapper_ResNet(nn.Module):
        def __init__(self, resnet):
            super().__init__()
            self.resnet = resnet
            self.mu = torch.Tensor([0.485, 0.456, 0.406]).float().view(3, 1, 1)
            self.sigma = torch.Tensor([0.229, 0.224, 0.225]).float().view(3, 1, 1)

        def forward(self, x):
            x = (x - self.mu.to(x.device)) / self.sigma.to(x.device)
            return self.resnet(x)

    if 'imagenet' in classifier_name:
        if 'resnet18' in classifier_name:
            print('using imagenet resnet18...')
            model = models.resnet18(pretrained=True).eval()
        elif 'resnet50' in classifier_name:
            print('using imagenet resnet50...')
            model = models.resnet50(pretrained=True).eval()
        elif 'resnet101' in classifier_name:
            print('using imagenet resnet101...')
            model = models.resnet101(pretrained=True).eval()
        elif 'wideresnet-50-2' in classifier_name:
            print('using imagenet wideresnet-50-2...')
            model = models.wide_resnet50_2(pretrained=True).eval()
        elif 'deit-s' in classifier_name:
            print('using imagenet deit-s...')
            model = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True).eval()
        else:
            raise NotImplementedError(f'unknown {classifier_name}')

        wrapper_resnet = _Wrapper_ResNet(model)

    elif 'cifar10' in classifier_name:
        if 'wideresnet-28-10' in classifier_name:
            print('using cifar10 wideresnet-28-10...')
            model = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf')  # pixel in [0, 1]

        elif 'wrn-28-10-at0' in classifier_name:
            print('using cifar10 wrn-28-10-at0...')
            model = load_model(model_name='Gowal2021Improving_28_10_ddpm_100m', dataset='cifar10',
                               threat_model='Linf')  # pixel in [0, 1]

        elif 'wrn-28-10-at1' in classifier_name:
            print('using cifar10 wrn-28-10-at1...')
            model = load_model(model_name='Gowal2020Uncovering_28_10_extra', dataset='cifar10',
                               threat_model='Linf')  # pixel in [0, 1]

        elif 'wrn-70-16-at0' in classifier_name:
            print('using cifar10 wrn-70-16-at0...')
            model = load_model(model_name='Gowal2021Improving_70_16_ddpm_100m', dataset='cifar10',
                               threat_model='Linf')  # pixel in [0, 1]

        elif 'wrn-70-16-at1' in classifier_name:
            print('using cifar10 wrn-70-16-at1...')
            model = load_model(model_name='Rebuffi2021Fixing_70_16_cutmix_extra', dataset='cifar10',
                               threat_model='Linf')  # pixel in [0, 1]

        elif 'wrn-70-16-L2-at1' in classifier_name:
            print('using cifar10 wrn-70-16-L2-at1...')
            model = load_model(model_name='Rebuffi2021Fixing_70_16_cutmix_extra', dataset='cifar10',
                               threat_model='L2')  # pixel in [0, 1]

        elif 'wideresnet-70-16' in classifier_name:
            print('using cifar10 wideresnet-70-16 (dm_wrn-70-16)...')
            from robustbench.model_zoo.architectures.dm_wide_resnet import DMWideResNet, Swish
            model = DMWideResNet(num_classes=10, depth=70, width=16, activation_fn=Swish)  # pixel in [0, 1]

            model_path = 'pretrained/cifar10/wresnet-76-10/weights-best.pt'
            print(f"=> loading wideresnet-70-16 checkpoint '{model_path}'")
            model.load_state_dict(update_state_dict(torch.load(model_path)['model_state_dict']))
            model.eval()
            print(f"=> loaded wideresnet-70-16 checkpoint")

        elif 'resnet-50' in classifier_name:
            print('using cifar10 resnet-50...')
            from classifiers.cifar10_resnet import ResNet50
            model = ResNet50()  # pixel in [0, 1]

            model_path = 'pretrained/cifar10/resnet-50/weights.pt'
            print(f"=> loading resnet-50 checkpoint '{model_path}'")
            model.load_state_dict(update_state_dict(torch.load(model_path), idx_start=7))
            model.eval()
            print(f"=> loaded resnet-50 checkpoint")

        elif 'wrn-70-16-dropout' in classifier_name:
            print('using cifar10 wrn-70-16-dropout (standard wrn-70-16-dropout)...')
            from classifiers.cifar10_resnet import WideResNet_70_16_dropout
            model = WideResNet_70_16_dropout()  # pixel in [0, 1]

            model_path = 'pretrained/cifar10/wrn-70-16-dropout/weights.pt'
            print(f"=> loading wrn-70-16-dropout checkpoint '{model_path}'")
            model.load_state_dict(update_state_dict(torch.load(model_path), idx_start=7))
            model.eval()
            print(f"=> loaded wrn-70-16-dropout checkpoint")

        else:
            raise NotImplementedError(f'unknown {classifier_name}')

        wrapper_resnet = model

    elif 'celebahq' in classifier_name:
        attribute = classifier_name.split('__')[-1]  # `celebahq__Smiling`
        ckpt_path = f'pretrained/celebahq/{attribute}/net_best.pth'
        from classifiers.attribute_classifier import ClassifierWrapper
        model = ClassifierWrapper(attribute, ckpt_path=ckpt_path)
        wrapper_resnet = model
    else:
        raise NotImplementedError(f'unknown {classifier_name}')

    return wrapper_resnet

def load_data(
    data_dir,
    batch_size,
    image_size,
    data_type='cifar10', 
    rank=0,
    world_size=1,
    class_cond=True,
    deterministic=True,
    random_crop=False,
    random_flip=False,
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
    
    # note: num_workers=0 will be slower, but jobs will not be interrupted by code changes in distributed training.
    # when using num_workers > 0, execution is faster but changes to code can cause a running job to crash.
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler, pin_memory=True, 
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

def set_seed(seed):

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.benchmark = True


#####################
# ## SETUP TOOLS ## #
#####################

# set seed, make folders, save config and code
def setup_exp(exp_dir, seed, folder_list, code_file_list=[], use_careful_setup=True):
    # make directory for saving results
    if os.path.exists(exp_dir) and use_careful_setup:
        # prevents overwriting old experiment folders by accident
        raise RuntimeError('Folder "{}" already exists. Please use a different "EXP_DIR".'.format(exp_dir))
    elif not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    for folder in ['code'] + folder_list:
        folder_path = os.path.join(exp_dir, folder)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

    # save copy of code in the experiment folder
    def save_code():
        def save_file(file_name):
            file_in = open('./' + file_name, 'r')
            file_out = open(exp_dir + '/code/' + os.path.basename(file_name), 'w')
            for line in file_in:
                file_out.write(line)
        for file in code_file_list:
            save_file(file)
    save_code()

    # set seed for cpu and CUDA, get device
    set_seed(seed)
   


####################
# ## PLOT TOOLS ## #
####################
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


####################
# ## DIST TOOLS ## #
####################

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

def distribute_to_device(tensor):
    rank = dist.get_rank()
    states_per_device = tensor.shape[0] // dist.get_world_size()
    return tensor[(rank*states_per_device):((rank+1)*states_per_device)].to(rank)

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

#hugging face unet2dmodel
def create_hf_unet(unet_channels,num_res_blocks,im_sz,channels):

    # Create Diffusion Model
    diff_model = UNet2DModel(
        sample_size=im_sz,  # the target image resolution
        in_channels=channels,  # the number of input channels, 3 for RGB images
        out_channels=channels,  # the number of output channels
        layers_per_block=num_res_blocks,  # how many ResNet layers to use per UNet block
        # block_out_channels=(64, 64, 128, 128, 256, 256),  # the number of output channels for each UNet block
        block_out_channels=unet_channels,  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    return diff_model