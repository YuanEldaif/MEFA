# this code will save cifar10 images to local disk in png format.

# for dataloader that doesn't use epochs, there is a bad shuffling bug in torch distributed training:
# https://github.com/pytorch/pytorch/issues/31771
# get it together pytorch... smh...

# to get around this, we save the datasets to disk and use the openai dataloader from guided diffusion:
# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py
# we modify the openai dataloader so the class is given by the folder instead of the first part of file name

# each class is stored in its own folder

# IMPORTANT: use the original files for all official FID eval, bc shuffling doesn't matter (all data are used)


from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10

import os
import numpy as np
from PIL import Image

#define the path to save the dataset
OUTPUT_DIR = './data/cifar10_pt_custom/'
CLASS_MAP = [
    'airplane',  # 0
    'automobile',  # 1
    'bird',  # 2
    'cat',  # 3
    'deer',  # 4
    'dog',  # 5
    'frog',  # 6
    'horse',  # 7
    'ship',  # 8
    'truck',  # 9
]

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        root,
        train=True, 
        download=True,
    ):
        super().__init__()

        self.dataset = CIFAR10(root=root, train=train, download=download)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset.__getitem__(idx)

        x = x.convert("RGB")
        x = np.array(x)

        y = np.array(y, dtype=np.int64)

        return x, y

dataset_train = CIFAR10Dataset(
    root='./data/cifar10_pt/', 
    train=True,
    download=True,
)
train_loader = DataLoader(dataset_train, batch_size=1, num_workers=1, drop_last=False)

dataset_val = CIFAR10Dataset(
    root='./data/cifar10_pt/', 
    train=False,
    download=True,
)
val_loader = DataLoader(dataset_val, batch_size=1, num_workers=1, drop_last=False)

# make folders for storing results
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(os.path.join(OUTPUT_DIR, 'train')):
    os.makedirs(os.path.join(OUTPUT_DIR, 'train'))
if not os.path.exists(os.path.join(OUTPUT_DIR, 'val')):
    os.makedirs(os.path.join(OUTPUT_DIR, 'val'))
for class_name in CLASS_MAP:
    if not os.path.exists(os.path.join(OUTPUT_DIR, 'train', class_name)):
        os.makedirs(os.path.join(OUTPUT_DIR, 'train', class_name))
    if not os.path.exists(os.path.join(OUTPUT_DIR, 'val', class_name)):
        os.makedirs(os.path.join(OUTPUT_DIR, 'val', class_name))

# save train data
for im_num, (x, y) in enumerate(train_loader):
    class_name = CLASS_MAP[y]
    im = Image.fromarray(x[0].numpy())
    im.save(os.path.join(OUTPUT_DIR, 'train', class_name, str(im_num) + '.png'))
# save val data
for im_num, (x, y) in enumerate(val_loader):
    class_name = CLASS_MAP[y]
    im = Image.fromarray(x[0].numpy())
    im.save(os.path.join(OUTPUT_DIR, 'val', class_name, str(im_num) + '.png'))
