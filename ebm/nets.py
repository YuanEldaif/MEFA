import torch.nn as nn
import torch.nn.functional as F
import torch


#########################
# ## LIGHTWEIGHT EBM ## #
#########################
class NonSmooth_EBM(nn.Module):
    def __init__(self, n_c=3, n_f=32):
        super(NonSmooth_EBM, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(n_f * 8, 1, 4, 1, 0))

    def forward(self, x):
        return self.f(x).squeeze()

def soft_lrelu(x):
    # Reducing to ReLU when a=0.5 and e=0
    # Here, we set a-->0.5 from left and e-->0 from right,
    # where adding eps is to make the derivatives have better rounding behavior around 0.
    a = 0.49
    e = 0.01
    return (1-a)*x + a*torch.sqrt(x*x + e*e) - a*e

class Act(nn.Module):
    def _init_(self):
        super()._init_()
    def forward(self, x):
        return soft_lrelu(x)
    
class EBM(nn.Module):
    def __init__(self, n_c=3, n_f=32, leak=0.05):
        super(EBM, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),
            Act(),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),
            Act(),
            nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1),
            Act(),
            nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1),
            Act(),
            nn.Conv2d(n_f * 8, 1, 4, 1, 0))

    def forward(self, x):
        return self.f(x).squeeze()


################################
# ## WIDE RESNET CLASSIFIER # ##
################################

# Implementation from https://github.com/meliketoy/wide-resnet.pytorch/ with very minor changes
# Original Version: Copyright (c) Bumsoo Kim 2018 under MIT License

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=2**0.5)
        nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = nn.Identity()
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes=10, dropout_rate=0.0):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

# Define the create_net function
class EBMSNGAN32(nn.Module):
    r"""
    ResNet backbone discriminator for SNGAN.

    Attributes:
        nf (int): Variable controlling discriminator feature map sizes.
    """
    def __init__(self, nf=128, patch_size=1):
        super().__init__()
        self.nf = nf
        self.patch_size = patch_size

        # Build layers
        self.block1 = DBlockOptimized(3 * (self.patch_size ** 2), self.nf)
        self.block2 = DBlock(self.nf, self.nf, downsample=True)
        self.block3 = DBlock(self.nf, self.nf, downsample=False)
        self.block4 = DBlock(self.nf, self.nf, downsample=False)
        self.l5 = nn.Linear(self.nf, 1, bias=False)
        self.activation = Act() #nn.ReLU()

        # match tf2 weight init
        torch.nn.init.xavier_uniform_(self.l5.weight)

    def forward(self, x):
        r"""
        Feedforwards a batch of real/fake images and produces a batch of GAN logits.

        Args:
            x (Tensor): A batch of images of shape (N, C, H, W).

        Returns:
            Tensor: A batch of GAN logits of shape (N, 1).
        """
        h = x

        if self.patch_size > 1:
            h = F.pixel_unshuffle(h, self.patch_size)

        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)

        # Global average pooling
        # NOTE: unlike original repo, this uses average pooling, not sum pooling
        h = torch.mean(h, dim=(2, 3))
        output = self.l5(h)

        return output

class EBMSNGAN128(nn.Module):
    r"""
    ResNet backbone discriminator for SNGAN.

    Attributes:
        nf (int): Variable controlling discriminator feature map sizes.
    """
    def __init__(self, nf=1024, patch_size=1):
        super().__init__()
        self.nf = nf
        self.patch_size = patch_size

        # Build layers
        self.block1 = DBlockOptimized(3 * (self.patch_size ** 2), self.nf >> 4)
        self.block2 = DBlock(self.nf >> 4, self.nf >> 3, downsample=True)
        self.block3 = DBlock(self.nf >> 3, self.nf >> 2, downsample=True)
        self.block4 = DBlock(self.nf >> 2, self.nf >> 1, downsample=True)
        self.block5 = DBlock(self.nf >> 1, self.nf, downsample=True)
        self.block6 = DBlock(self.nf, self.nf, downsample=False)
        self.l7 = nn.Linear(self.nf, 1, bias=False)
        self.activation = Act()

        # match tf2 weight init
        torch.nn.init.xavier_uniform_(self.l7.weight)

    def forward(self, x):
        r"""
        Feedforwards a batch of real/fake images and produces a batch of GAN logits.

        Args:
            x (Tensor): A batch of images of shape (N, C, H, W).

        Returns:
            Tensor: A batch of GAN logits of shape (N, 1).
        """
        h = x

        if self.patch_size > 1:
            h = F.pixel_unshuffle(h, self.patch_size)

        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.activation(h)

        # Global average pooling
        # NOTE: unlike original repo, this uses average pooling, not sum pooling
        h = torch.mean(h, dim=(2, 3))
        output = self.l7(h)

        return output

class EBMSNGAN256(nn.Module):
    r"""
    ResNet backbone discriminator for SNGAN.

    Attributes:
        nf (int): Variable controlling discriminator feature map sizes.
    """
    def __init__(self, nf=1024, patch_size=1):
        super().__init__()
        self.nf = nf
        self.patch_size = patch_size

        # Build layers
        self.block1 = DBlockOptimized(3 * (self.patch_size ** 2), self.nf >> 4)
        self.block2 = DBlock(self.nf >> 4, self.nf >> 3, downsample=True)
        self.block3 = DBlock(self.nf >> 3, self.nf >> 2, downsample=True)
        self.block4 = DBlock(self.nf >> 2, self.nf >> 1, downsample=True)
        self.block5 = DBlock(self.nf >> 1, self.nf, downsample=True)
        self.block6 = DBlock(self.nf, self.nf, downsample=True)
        self.block7 = DBlock(self.nf, self.nf, downsample=False)
        self.l8 = nn.Linear(self.nf, 1, bias=False)
        self.activation = Act()

        # match tf2 weight init
        torch.nn.init.xavier_uniform_(self.l8.weight)

    def forward(self, x):
        r"""
        Feedforwards a batch of real/fake images and produces a batch of GAN logits.

        Args:
            x (Tensor): A batch of images of shape (N, C, H, W).

        Returns:
            Tensor: A batch of GAN logits of shape (N, 1).
        """
        h = x

        if self.patch_size > 1:
            h = F.pixel_unshuffle(h, self.patch_size)

        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.block7(h)
        h = self.activation(h)

        # Global average pooling
        # NOTE: unlike original repo, this uses average pooling, not sum pooling
        h = torch.mean(h, dim=(2, 3))
        output = self.l8(h)

        return output

def create_net(args, net_type, im_sz, nf, dummy_batch_size=4):
    if net_type == 'ebm_sngan':
        patch_size = args.sngan_patch_size if 'sngan_patch_size' in args.keys() else 1
        if im_sz // patch_size == 32:
            net = EBMSNGAN32(nf=nf, patch_size=patch_size)
            input_size = [dummy_batch_size, 3, 32, 32]
        elif im_sz // patch_size == 128:
            net = EBMSNGAN128(nf=nf, patch_size=patch_size)
            input_size = [dummy_batch_size, 3, 128, 128]
        elif im_sz // patch_size == 256:
            net = EBMSNGAN256(nf=nf, patch_size=patch_size)
            input_size = [dummy_batch_size, 3, 256, 256]
    elif net_type == 'ebm_small':
        net = EBM()
        input_size = [dummy_batch_size, 3, 32, 32]

    return net
