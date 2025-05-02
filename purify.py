import torch
import torchsde
import torch.nn.functional as F
import torch.nn as nn
import torchvision.utils as tvu
from torch.utils.data import DataLoader
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
from utils import get_image_classifier, dict2namespace, set_seed, load_data, setup_dist

from guided_diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    get_named_beta_schedule,
    ModelMeanType,
    ModelVarType,
    LossType
)

import ebm
from ebm.nets import WideResNet, EBM, NonSmooth_EBM

from autoattack import AutoAttack
from utils import create_hf_unet

###########################
# ##   diffpure model  ## #
###########################
def restore_checkpoint(ckpt_dir, state, device):
    loaded_state = torch.load(ckpt_dir, map_location=torch.device('cpu'))
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']

class RevVPSDE(torch.nn.Module):
    def __init__(self, model, score_type='score_sde', beta_min=0.1, beta_max=20, N=1000,
                 img_shape=(3, 32, 32), model_kwargs=None):
        """Construct a Variance Preserving SDE.

        Args:
          model: diffusion model
          score_type: [guided_diffusion, score_sde, ddpm]
          beta_min: value of beta(0)
          beta_max: value of beta(1)
        """
        super().__init__()
        self.model = model
        self.score_type = score_type
        self.model_kwargs = model_kwargs
        self.img_shape = img_shape

        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.alphas_cumprod_cont = lambda t: torch.exp(-0.5 * (beta_max - beta_min) * t**2 - beta_min * t)
        self.sqrt_1m_alphas_cumprod_neg_recip_cont = lambda t: -1. / torch.sqrt(1. - self.alphas_cumprod_cont(t))

        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.counter = torch.zeros(1, dtype=torch.int)

    def _scale_timesteps(self, t):
        assert torch.all(t <= 1) and torch.all(t >= 0), f't has to be in [0, 1], but get {t} with shape {t.shape}'
        return (t.float() * self.N).long()

    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int)

    def vpsde_fn(self, t, x):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        beta_t = beta_t
        drift = -0.5 * beta_t[:, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def rvpsde_fn(self, t, x, return_type='drift'):
        """Create the drift and diffusion functions for the reverse SDE"""
        drift, diffusion = self.vpsde_fn(t, x)

        if return_type == 'drift':
            x_img = x.view(-1, *self.img_shape)  # Ensure x_img is on the same device as x

            if self.score_type == 'score_sde':
                sde = sde_lib.VPSDE(beta_min=self.beta_0, beta_max=self.beta_1, N=self.N)
                score_fn = mutils.get_score_fn(sde, self.model, train=False, continuous=True)
                score = score_fn(x_img, t) 
                assert x_img.shape == score.shape, f'{x_img.shape}, {score.shape}'
                score = score.view(x.shape[0], -1)

            else:
                raise NotImplementedError(f'Unknown score type in RevVPSDE: {self.score_type}!')

            drift = drift - diffusion[:, None] ** 2 * score
            return drift

        else:
            return diffusion

    def f(self, t, x):
        """Create the drift function -f(x, 1-t) (by t' = 1 - t)
            sdeint only support a 2D tensor (batch_size, c*h*w)
        """
        t = t.expand(x.shape[0])  # (batch_size, )
        drift = self.rvpsde_fn(1 - t, x, return_type='drift')
        assert drift.shape == x.shape
        return -drift

    def g(self, t, x):
        """Create the diffusion function g(1-t) (by t' = 1 - t)
            sdeint only support a 2D tensor (batch_size, c*h*w)
        """
        t = t.expand(x.shape[0])  # (batch_size, )
        diffusion = self.rvpsde_fn(1 - t, x, return_type='diffusion')
        assert diffusion.shape == (x.shape[0], )
        return diffusion[:, None].expand(x.shape)

class RevGuidedDiffusion(torch.nn.Module):
    def __init__(self, args, config, device=None):
        super().__init__()
        self.config = config
        self.args = args
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        if config.data.dataset == 'CIFAR10':
            img_shape = (3, 32, 32)
            model_dir = args.model_dir 
            # print(f'model_config: {config}')
            model = mutils.create_model(config)
            optimizer = get_optimizer(config, model.parameters())
            ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
            state = dict(step=0, optimizer=optimizer, model=model, ema=ema)
            restore_checkpoint(model_dir, state, self.device)
            ema.copy_to(model.parameters())

        else:
            raise NotImplementedError(f'Unknown dataset {config.data.dataset}!')
        
        model.eval().to(self.device)

        self.model = model
        self.rev_vpsde = RevVPSDE(model=model, score_type=args.score_type, img_shape=img_shape,
                                  model_kwargs=None).to(self.device)
        self.betas = self.rev_vpsde.discrete_betas.float().to(self.device)
    
    def noise_add(self, img):
        assert isinstance(img, torch.Tensor)
        batch_size = img.shape[0]
        e = torch.randn_like(img).to(img.device)
        total_noise_levels = self.args.t
        a = (1 - self.betas).cumprod(dim=0).to(img.device)
        x = img * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()
        return x

    def ts_func(self, args):
        epsilon_dt0, epsilon_dt1 = 0, 1e-5
        t0, t1 = 1 - args.t * 1. / 1000 + epsilon_dt0, 1 - epsilon_dt1
        t_size = 2
        ts = torch.linspace(t0, t1, t_size).to(self.device)
        return ts

    def forward_purify(self, X):
        img_ = X.view(X.shape[0], -1)  # (batch_size, state_size)
        x_noisy = self.noise_add(img_)
        x_ = torch.autograd.Variable(x_noisy, requires_grad=True) #require grad after adding noise on the (batch_size, state_size)

        ts = self.ts_func(self.args)
        xs_, noises, curr_ts, next_ts = torchsde.sdeint(self.rev_vpsde, x_, ts, method='euler')
        x0 = xs_[-1].view(X.shape).to(self.device)
        return x0, xs_, noises, curr_ts, next_ts
    
    def forward_purify_ajoint(self, X):
        img_ = X.view(X.shape[0], -1)  # (batch_size, state_size)
        x_noisy = self.noise_add(img_)
        x_ = torch.autograd.Variable(x_noisy, requires_grad=True) #require grad after adding noise on the (batch_size, state_size)

        ts = self.ts_func(self.args)
        xs_ = torchsde.sdeint_adjoint(self.rev_vpsde, x_, ts, method='euler')
        x0 = xs_[-1].view(X.shape).to(self.device)
        return x0
    
    def forward(self, X):
        img_ = X.view(X.shape[0], -1)  # (batch_size, state_size)
        x_noisy = self.noise_add(img_)
        x_ = torch.autograd.Variable(x_noisy, requires_grad=True) #require grad after adding noise on the (batch_size, state_size)

        ts = self.ts_func(self.args)
        xs_, noises, curr_ts, next_ts = torchsde.sdeint(self.rev_vpsde, x_, ts, method='euler')
        x0 = xs_[-1].view(X.shape).to(self.device)
        return x0

    def grad_back_prop(self, x_diffpure_list, noi_diffpure_list, curr_ts, next_ts, net_grads):
        net_grads = net_grads.view(net_grads.shape[0], -1)  # (batch_size, state_size)
        ts = self.ts_func(self.args)
        grad_ckpt = torchsde.sdeint_grad(self.rev_vpsde, x_diffpure_list, ts, noi_diffpure_list, curr_ts, next_ts, net_grads, method='euler_grad')
        return grad_ckpt

def grad_back_diffpure(model, x_diffpure_list, noi_diffpure_list, curr_ts, next_ts, net_grads):
    grad_ckpt = model.grad_back_prop(x_diffpure_list, noi_diffpure_list, curr_ts, next_ts, net_grads)
    return grad_ckpt
    

######################################
# ## hugging_face/diffusion model ## #
######################################
def init_diff(args):
    betas = get_named_beta_schedule(args.t_schedule, args.num_t_steps)   
    scheduler = GaussianDiffusion(
                        betas=betas,
                        model_mean_type=(
                            ModelMeanType.EPSILON if args.diff_output == 'epsilon'
                            else ModelMeanType.START_X
                        ),
                        model_var_type=ModelVarType.FIXED_LARGE,
                        loss_type=LossType.MSE
                    )   
    return scheduler

def grad_ckpt_diff(args, model, scheduler, x_samples, model_kwargs={}):
    # x_samples = torch.autograd.Variable(x_samples.clone(), requires_grad=True)
    # assert x_samples.requires_grad
    if args.purify_t>0:
        n = args.purify_t
        indices = list(range(args.purify_t))[::-1]
        t_start = torch.tensor(x_samples.shape[0] * [args.purify_t]).long().to(x_samples.device)
        x_t = scheduler.q_sample(x_samples, t_start)
        X = x_t.detach()
    else:
        n = args.num_t_steps
        indices = list(range(args.num_t_steps))[::-1]
        noise_start = torch.randn(*x_samples.shape, device=x_samples.device)
        X = noise_start.detach()

    def forward(X):
        # A list to store intermediate activations
        hs = [torch.zeros(X.shape, dtype=torch.float32) for _ in range(n)]
        noise_list = [torch.zeros(X.shape, dtype=torch.float32) for _ in range(n)]
        for i in indices:
            t = torch.tensor([i] * X.shape[0], device=X.device)
            hs[i] = X.detach().cpu()
            with torch.no_grad():
                out = scheduler.ddim_sample(
                model,
                X, 
                t,
                clip_denoised=True,
                model_kwargs=model_kwargs,
                eta = args.eta
                )
      
            X = out["sample"]
            noise_out = out["noise"]
            noise_list[i] = noise_out.detach().cpu()
  
        return X, hs, noise_list
    net_out, net_out_list, noise_list = forward(X)

    return net_out, net_out_list, noise_list

def grad_back_diff(args, model, scheduler, X, x_diff_list, noi_diff_list, net_grads, model_kwargs={}):
    shape = X.shape
    if args.purify_t>0:
        n = args.purify_t
    else:
        n = args.num_t_steps
        
    for k in range(n):
        net_out = x_diff_list[n - k - 1]
        net_out = torch.autograd.Variable(net_out, requires_grad=True).to(model.device)
        noi_out = noi_diff_list[n - k - 1].to(model.device)

        t = torch.tensor([n - k - 1] * shape[0], device=X.device)
        next_layer = scheduler.ddim_sample_grad(
            model,
            net_out,
            t,
            noi_out,
            model_kwargs=model_kwargs,
            eta = args.eta
            )
        next_layer_out = next_layer["sample"]
        net_grads = torch.autograd.grad(next_layer_out, [net_out], grad_outputs=net_grads)[0]

    return net_grads

######################
# ##      EBM     ## #
######################
def ebm_update(args, ebm, X, noise_list, requires_grad=True):
    X = torch.autograd.Variable(X.clone(), requires_grad=True)
    assert X.requires_grad
    n = args.langevin_steps
    
    X = X + args.langevin_init_noise * torch.randn_like(X)
    for i in range(n):

        energy = ebm(X).sum() / args.mcmc_temp
        grad = torch.autograd.grad(energy, [X], create_graph=True)[0]
        # noise = torch.randn_like(X)

        X = X - ((args.langevin_eps ** 2) / 2) * grad
        X = X + args.langevin_eps * noise_list[i]

    return X

def ebm_fwd_only(args, ebm, X):
    assert not X.requires_grad
    noise_list = torch.randn((args.langevin_steps,) + X.shape, device=X.device) #faster with double memory
    X = X.detach().clone() + args.langevin_init_noise * torch.randn_like(X)
    out_list = [X]
    for i in range(args.langevin_steps):
        X_autograd = torch.autograd.Variable(X.clone(), requires_grad=True)
        energy = ebm(X_autograd).sum() / args.mcmc_temp
        grad = torch.autograd.grad(energy, [X_autograd], create_graph=False)[0]

        X = X - ((args.langevin_eps ** 2) / 2) * grad + args.langevin_eps * noise_list[i]
        out_list.append(X)

    return X, out_list, noise_list

def grad_back_ebm(args, ebm, saved_states, noise_list, grad_output):
    current_grad = grad_output

    # Traverse steps in reverse
    for step_idx in reversed(range(len(saved_states) - 1)):
        X_prev = saved_states[step_idx].requires_grad_(True).to(current_grad.device)
        noise = noise_list[step_idx].to(current_grad.device)

        # Recompute forward step with gradients
        energy = ebm(X_prev).sum() / args.mcmc_temp
        grad = torch.autograd.grad(energy, X_prev, create_graph=True)[0]

        X_next = X_prev - ((args.langevin_eps ** 2) / 2) * grad + args.langevin_eps * noise

        # Backprop through this step
        grad_params = torch.autograd.grad(
            outputs=X_next,
            inputs=X_prev,
            grad_outputs=current_grad
        )

        current_grad = grad_params[0]

    return current_grad

##################################################################################
######################
# ## PURIFICATION ## #
######################

def get_classifier(args, rank, device):
    if args.classifier_name == 'wideresnet':
        clf = WideResNet(num_classes=10)
        clf.load_state_dict(torch.load(args.clf_weight_path, map_location=torch.device('cpu'), weights_only=True))
        clf.to(device)
        clf.eval()
        if args.verbose:
            if rank == 0:
                num_params_clf = sum(p.numel() for p in clf.parameters())
                print('Creating wideresnet from EBM folder for image size {} with {} parameters'.format(args.image_size, num_params_clf))
        dist.barrier()
    else:
        clf = get_image_classifier(args.classifier_name)
        clf.to(device)
        clf.eval()
        if args.verbose:
            if rank == 0:
                num_params = sum(p.numel() for p in clf.parameters())
                print('Creating widerenet from robustbench for image size {} with {} parameters'.format(args.image_size, num_params))
            dist.barrier()
    return clf

## load purify model
def load_purify_model(rank, args, config, device):
    if args.exp_name=='hugging_face':
        from diffusers import UNet2DModel
        model = create_hf_unet(args.unet_channels,num_res_blocks=args.num_res_blocks,im_sz=32,channels=3)
        model.load_state_dict(torch.load(args.unet_weight_path, map_location=torch.device('cpu'), weights_only=True))
        model.to(device)
        model.eval()
        # diffusion scheduler setup
        scheduler = init_diff(args)
    elif args.exp_name=='hf_DDPM':
        from diffusers import UNet2DModel
        model_id = "google/ddpm-cifar10-32"
        model = UNet2DModel.from_pretrained(model_id)
        model.to(device)
        model.eval()
        # diffusion scheduler setup
        scheduler = init_diff(args)
    elif args.exp_name =='diffpure':
        #load diffpure model
        model = RevGuidedDiffusion(args, config, device).to(device)
        model.eval()
        scheduler = None
    elif args.exp_name=='ebm':
        model = EBM()               
        model.load_state_dict(torch.load(args.ebm_weight_path, map_location=torch.device('cpu'), weights_only=True))
        model.to(device)
        model.eval()
        scheduler = None
    elif args.exp_name=='nonsmooth_ebm':
        model = NonSmooth_EBM()               
        model.load_state_dict(torch.load(args.ebm_weight_path, map_location=torch.device('cpu'), weights_only=True))
        model.to(device)
        model.eval()
        scheduler = None

    if args.verbose:
        if rank == 0:
            print('check if model is on the device:', next(model.parameters()).device)
            num_params_diff = sum(p.numel() for p in model.parameters())
            print('Creating purification model for image size {} with {} parameters'.format(32, num_params_diff))

    return model, scheduler

def purify(args, model, scheduler, X):
    if args.exp_name =='diffpure':
        x0, out_list, noise_list, curr_ts, next_ts = model.forward_purify(X)

    elif args.exp_name in ('hugging_face', 'hf_DDPM'):
        x0, out_list, noise_list = grad_ckpt_diff(args, model, scheduler, X, model_kwargs={})
        curr_ts, next_ts = None, None

    elif args.exp_name in ('ebm','ebm_smooth'):
        
        if args.pytorch:
            noise_list = torch.randn((args.langevin_steps,) + X.shape, device=X.device)
            x0 = ebm_update(args, model, X, noise_list)
            out_list, noise_list = None, None
        else:
            x0, out_list, noise_list = ebm_fwd_only(args, model, X)
        curr_ts, next_ts = None, None
    else: 
        raise RuntimeError('Invalid model type evaluation')
    
    return x0, out_list, noise_list, curr_ts, next_ts
    
def eot_defense_prediction(logits, reps=1, eot_defense_ave=None):
    # finite-sample approximation of stochastic classifier for EOT defense averaging different methods
    # for deterministic logits with reps=1, this is just standard prediction for any eot_defense_ave
    if eot_defense_ave == 'logits':
        logits_pred = logits.view([reps, int(logits.shape[0]/reps), logits.shape[1]]).mean(0)
    elif eot_defense_ave == 'softmax':
        logits_pred = F.softmax(logits, dim=1).view([reps, int(logits.shape[0]/reps), logits.shape[1]]).mean(0)
    elif eot_defense_ave == 'logsoftmax':
        logits_pred = F.log_softmax(logits, dim=1).view([reps, int(logits.shape[0] / reps), logits.shape[1]]).mean(0)
    elif reps == 1:
        logits_pred = logits
    else:
        raise RuntimeError('Invalid ave_method_pred (use "logits" or "softmax" or "logsoftmax")')
    # finite sample approximation of stochastic classifier prediction
    _, y_pred = torch.max(logits_pred, 1)
    return y_pred, logits_pred

def eot_attack_loss(logits, y, reps=1, eot_attack_ave='loss', criterion = torch.nn.CrossEntropyLoss()):
    # finite-sample approximation of stochastic classifier loss for different EOT attack averaging methods
    # for deterministic logits with reps=1 this is just standard cross-entropy loss for any eot_attack_ave
    if eot_attack_ave == 'logits':
        logits_loss = logits.view([reps, int(logits.shape[0] / reps), logits.shape[1]]).mean(0)
        y_loss = y
    elif eot_attack_ave == 'softmax':
        logits_loss = torch.log(F.softmax(logits, dim=1).view([reps, int(logits.shape[0] / reps), logits.shape[1]]).mean(0))
        y_loss = y
    elif eot_attack_ave == 'logsoftmax':
        logits_loss = F.log_softmax(logits, dim=1).view([reps, int(logits.shape[0] / reps), logits.shape[1]]).mean(0)
        y_loss = y
    elif eot_attack_ave == 'loss':
        logits_loss = logits
        y_loss = y.repeat(reps)
    else:
        raise RuntimeError('Invalid ave_method_eot ("logits", "softmax", "logsoftmax", "loss")')
    # final cross-entropy loss to generate attack grad
    # cross-entropy loss function to generate attack gradients
    loss = criterion(logits_loss, y_loss)
    return loss


def predict_logits(args, clf, X, y, requires_grad=True, reps=1, eot_defense_ave=None, eot_attack_ave='loss'):
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
    y_pred, logits_pred = eot_defense_prediction(logits.detach(), reps, eot_defense_ave)
    correct = torch.eq(y_pred, y)
    # loss for specified EOT attack averaging method
    loss = eot_attack_loss(logits, y, reps, eot_attack_ave)
    
    # We use torch.arange to create indices for each batch and gather logits
    batch_size = logits_pred.size(0)
    
    # Create a tensor of indices for the batch
    indices = torch.arange(batch_size, device=X.device)
    
    # Extract the logits corresponding to the correct labels
    correct_logits = logits[indices, y]
    # Get top-1 
    logit1, _ = torch.topk(logits_pred, 1, dim=1)
    logitdiff = correct_logits - logit1
    
    return correct.detach(), loss, logitdiff

def rand_init_l_p(X_adv, adv_norm, adv_eps):
    # random initialization in l_inf or l_2 ball
    if adv_norm == 'Linf':
        X_adv.data = torch.clamp(X_adv.data + adv_eps * (2 * torch.rand_like(X_adv) - 1), min=-1, max=1)
    elif adv_norm == 'l_2':
        r = torch.randn_like(X_adv)
        r_unit = r / r.view(r.shape[0], -1).norm(p=2, dim=1).view(r.shape[0], 1, 1, 1)
        X_adv.data += adv_eps * torch.rand(X_adv.shape[0], 1, 1, 1).cuda() * r_unit
    else:
        raise RuntimeError('Invalid adv_norm ("l_inf" or "l_2"')
    return X_adv

def check_oscillation(x, j, k, y5, k3=0.75):
    t = torch.zeros(x.shape[1]).to(x.device)
    for counter5 in range(k):
        t += (x[j - counter5] > x[j - counter5 - 1]).float()

    return (t <= k * k3 * torch.ones_like(t)).float()

def pgd_update(X_adv, grad, X, adv_norm, adv_eps, adv_eta, eps=1e-10):
    if adv_norm == 'Linf':
        # l_inf steepest ascent update
        X_adv.data += adv_eta * torch.sign(grad)
        # project to l_inf ball
        X_adv = torch.clamp(torch.min(X + adv_eps, torch.max(X - adv_eps, X_adv)), min=-1, max=1)
    elif adv_norm == 'L2':
        # l_2 steepest ascent update
        X_adv.data += adv_eta * grad / grad.view(X.shape[0], -1).norm(p=2, dim=1).view(X.shape[0], 1, 1, 1)
        # project to l_2 ball
        dists = (X_adv - X).view(X.shape[0], -1).norm(dim=1, p=2).view(X.shape[0], 1, 1, 1)
        X_adv = torch.clamp(X + torch.min(dists, adv_eps*torch.ones_like(dists))*(X_adv-X)/(dists+eps), min=-1, max=1)
    else:
        raise RuntimeError('Invalid adv_norm ("Linf" or "l_2"')
    return X_adv

def normalize(args, x):
    if args.adv_norm == 'Linf':
        t = x.abs().view(x.shape[0], -1).max(1)[0]

    elif args.adv_norm == 'L2':
        t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()

    elif args.adv_norm == 'L1':
        try:
            t = x.abs().view(x.shape[0], -1).sum(dim=-1)
        except:
            t = x.abs().reshape([x.shape[0], -1]).sum(dim=-1)

    return x / (t.view(-1, *([1] * len(list(x.shape[1:])))) + 1e-12)

def L2_norm(x, keepdim=False):
    z = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
    if keepdim:
        z = z.view(-1, *[1]*(len(x.shape) - 1))
    return z

def pgd_update_autoattack(args, step, x_adv, x_adv_old, grad, X, adv_eps, step_size, eps=1e-10):
    with torch.no_grad():
        x_adv = x_adv.detach()
        grad2 = x_adv - x_adv_old
        x_adv_old = x_adv.clone()
        a = 0.75 if step > 0 else 1.0

        if args.adv_norm == 'Linf':  
            # l_inf steepest ascent update with adaptive step size
            x_adv_1 = x_adv + step_size * torch.sign(grad)
            # Project to l_inf ball
            x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, X - adv_eps), X + adv_eps), -1.0, 1.0)
            x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a), X - adv_eps), X + adv_eps), -1.0, 1.0)

        elif args.adv_norm == 'L2':  
            # l_2 steepest ascent update        
            x_adv_1 = x_adv + step_size * normalize(args, grad) 
            # project to l_2 ball
            x_adv_1 = torch.clamp(X + normalize(args, x_adv_1 - X) * torch.min(eps * torch.ones_like(X).detach(),L2_norm(x_adv_1 - X, keepdim=True)), -1.0, 1.0)
            x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
            x_adv_1 = torch.clamp(X + normalize(args, x_adv_1 - X) * torch.min(eps * torch.ones_like(X).detach(),L2_norm(x_adv_1 - X, keepdim=True)), -1.0, 1.0)
        else:  
            raise ValueError('Invalid adv_norm; choose "Linf" or "L2".')  
    
        x_adv = x_adv_1 + 0.
    return x_adv, step_size, x_adv_old

def purify_and_predict(args, model, scheduler, clf, X, y, purify_reps=1, requires_grad=True):
    # parallel states for either EOT attack grad or EOT defense with large-sample evaluation of stochastic classifier
    X_repeat = X.repeat([purify_reps, 1, 1, 1])

    if args.t > 0 or args.purify_t> 0 or args.langevin_steps >0:
        # parallel purification of replicated states
        X_repeat_purified, x_pure_list, noi_pure_list, curr_ts, next_ts = purify(args, model, scheduler, X_repeat)
    else:
        X_repeat_purified = X_repeat

    if args.grad_ckpt:
        X_repeat_purified = torch.autograd.Variable(X_repeat_purified, requires_grad=True)
    correct, loss, logitdiff = predict_logits(args, clf, X_repeat_purified, y, requires_grad, purify_reps,
                        args.eot_defense_ave, args.eot_attack_ave)                             

    if requires_grad:
        # get BPDA gradients with respect to purified states
        net_grads = torch.autograd.grad(loss, [X_repeat_purified])[0]
        
        if not args.bpda_only:
        # start_time_backprop = time.time()
            if args.exp_name =='diffpure':
                net_grads_ = grad_back_diffpure(model, x_pure_list, noi_pure_list, curr_ts, next_ts, net_grads) 
                X_grads = net_grads_.view(X_repeat_purified.shape)
            elif args.exp_name in ('hugging_face', 'hf_DDPM'): 
                X_grads = grad_back_diff(args, model, scheduler, X_repeat_purified, x_pure_list, noi_pure_list, net_grads)
            elif args.exp_name =='ebm': 
                if args.pytorch:
                    X_grads = net_grads
                else:
                    X_grads = grad_back_ebm(args, model, x_pure_list, noi_pure_list, net_grads)
        else:
            X_grads = net_grads

        # average gradients over parallel samples for EOT attack
        attack_grad = X_grads.view([purify_reps]+list(X.shape)).mean(dim=0)

        return correct, attack_grad, loss, logitdiff
    else:
        return correct, None, loss, logitdiff


def eval_and_bpda_eot_grad(args, model, scheduler, clf, X_adv, y, requires_grad=True):
    # forward pass to identify candidates for breaks (and backward pass to get BPDA + EOT grad if requires_grad==True)
    defended, attack_grad, loss, logitdiff = purify_and_predict(args, model, scheduler, clf, X_adv, y, args.eot_attack_reps, requires_grad)
    return defended, attack_grad, loss, logitdiff 


def attack_batch_auto(args, model, scheduler, clf, X, y, batch_num, device):
    # Reset the memory tracker
    torch.cuda.reset_peak_memory_stats()
    # get baseline accuracy for natural images
    defended, grad, loss, logitdiff = eval_and_bpda_eot_grad(args, model, scheduler, clf, X, y, True)

    if dist.get_rank() == 0:
        print('Batch {} of {} Baseline: {} of {}'.
            format(batch_num - args.start_batch + 2, args.end_batch - args.start_batch+ 1,
                defended.sum(), len(defended)))
    dist.barrier()
    
    # record of grad over attacks
    grad_batch = torch.zeros([args.adv_steps + 2, X.shape[0], X.shape[1], X.shape[2], X.shape[3]])
    grad_batch[0] = grad.cpu()

    class_batch = torch.zeros([args.adv_steps + 2, X.shape[0]]).bool()
    class_batch[0] = defended.cpu()

    # record for final adversarial images 
    ims_adv_batch = torch.zeros(X.shape)
    for ind in range(defended.nelement()):
        if defended[ind] == 0:
            # record mis-classified natural images as adversarial states
            ims_adv_batch[ind] = X[ind].cpu()
    
    # initialize adversarial image as natural image
    X_adv = X.clone()
    
    # start in random location of l_p ball
    if args.adv_rand_start:
        X_adv = rand_init_l_p(X_adv, args.adv_norm, args.adv_eps)
    
    nat_acc = defended.clone()

    x_best = X_adv.clone()
    x_best_adv = X_adv.clone()
    loss_steps = torch.zeros([args.adv_steps+1, X.shape[0]])
    loss_best_steps = torch.zeros([args.adv_steps + 2, X.shape[0]])
    
    acc = defended
    loss_best = loss.detach().clone()
    grad_best = grad.clone()
    counter3 = 0
    size_decr = max(int(0.03 * args.adv_steps), 1)
    n_iter_min = max(int(0.06 * args.adv_steps), 1)
    n_iter_2 = max(int(0.22 * args.adv_steps), 1)
    k =  n_iter_2 + 0

    alpha = 2. if args.adv_norm in ['Linf', 'L2'] else 1. if args.adv_norm in ['L1'] else 2e-2
    # step_size = alpha * args.adv_eps* torch.ones([X.shape[0], *([1] * 3)]).to(device).detach()
    step_size = alpha * args.adv_eps* torch.ones([X.shape[0], *([1] * 3)]).to(device).detach()
    # step_size_batch[0] = step_size
    x_adv_old = X_adv.clone()

    loss_best_last_check = loss_best.clone()
    reduced_last_check = torch.ones_like(loss_best)
    n_reduced = 0
    
    for step in range(args.adv_steps+1):
        # get attack gradient and update defense record
        defended, grad, loss, logitdiff = eval_and_bpda_eot_grad(args, model, scheduler, clf, X_adv, y, True)
        # update step-by-step defense record
        class_batch[step+1] = defended.cpu()
        
        acc = torch.min(acc, defended)
        
        # update step-by-step defense record
        grad_batch[step+1] = grad.cpu()

        X_adv, step_size, x_adv_old = pgd_update_autoattack(args, step, X_adv, x_adv_old, grad, X, args.adv_eps, step_size)
        # add adversarial images for newly broken images to list
        for ind in range(defended.nelement()):
            if class_batch[step, ind] == 1 and defended[ind] == 0:
                ims_adv_batch[ind] = X_adv[ind].cpu()
        if step == args.adv_steps:
            X_adv_final_batch = X_adv.detach().clone()

        ind_pred = (defended == 0).nonzero().squeeze()
        x_best_adv[ind_pred] = X_adv[ind_pred] + 0.
        ### check step size
        y1 = loss.detach().clone()
        loss_steps[step] = y1 + 0
        ind = (y1 > loss_best) #.nonzero().squeeze()
        x_best[ind] = X_adv[ind].clone()
        grad_best[ind] = grad[ind].clone()
        loss_best[ind] = y1[ind] + 0
        loss_best_steps[step + 1] = loss_best + 0

        counter3 += 1

        if counter3 == k:
            if args.adv_norm in ['Linf', 'L2']:
                fl_oscillation = check_oscillation(loss_steps, step, k, loss_best, k3=0.75).to(device)
                fl_reduce_no_impr = (1. - reduced_last_check) * (loss_best_last_check >= loss_best).float()
                fl_oscillation = torch.max(fl_oscillation, fl_reduce_no_impr)
                reduced_last_check = fl_oscillation.clone()
                loss_best_last_check = loss_best.clone()
    
                if fl_oscillation.sum() > 0:
                    ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                    step_size[ind_fl_osc] /= 2.0
                    n_reduced = fl_oscillation.sum()
    
                    X_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                    grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                k = max(k - size_decr, n_iter_min)

            counter3 = 0

        if dist.get_rank()==0 and (step == 1 or step % args.log_freq == 0 or step == args.adv_steps):
            # print attack info
            print('Batch {} of {}, Attack {} of {}   Batch defended: {} of {}, acc {:.4f}'.
                  format(batch_num - args.start_batch + 2, args.end_batch - args.start_batch + 1,
                         step, args.adv_steps, int(torch.sum(defended).cpu().numpy()), X_adv.shape[0], int(torch.sum(acc).cpu().numpy())/X_adv.shape[0] ))
        dist.barrier()

        # record final adversarial image for unbroken states
        for ind in range(defended.nelement()):
            if defended[ind] == 1:
                ims_adv_batch[ind] = X_adv[ind].cpu()

    return grad_batch, class_batch, loss_best_steps, nat_acc, acc, ims_adv_batch, x_best_adv, X_adv_final_batch

def attack_batch(args, model, scheduler, clf, X, y, batch_num, device):
    # Reset the memory tracker
    torch.cuda.reset_peak_memory_stats()
    # get baseline accuracy for natural images
    defended, grad, loss, logitdiff  = eval_and_bpda_eot_grad(args, model, scheduler, clf, X, y, False)

    nat_acc = defended.clone()
    acc = defended
    
    if dist.get_rank() == 0:
        print('Batch {} of {} Baseline: {} of {}'.
            format(batch_num - args.start_batch + 2, args.end_batch - args.start_batch+ 1,
                defended.sum(), len(defended)))
    dist.barrier()
    # record of defense over attacks
    class_batch = torch.zeros([args.adv_steps + 2, X.shape[0]]).bool()
    class_batch[0] = defended.cpu()
    loss_batch =  torch.empty([args.adv_steps + 2, X.shape[0]])
    loss_batch[0] = loss 
    # record of grad over attacks
    grad_batch = torch.zeros([args.adv_steps + 2, X.shape[0], X.shape[1], X.shape[2], X.shape[3]])
    if grad == None:
        grad_batch[0] = 0
    else:
        grad_batch[0] = grad.cpu()

    # record for final adversarial images 
    ims_adv_batch = torch.zeros(X.shape)
    for ind in range(defended.nelement()):
        if defended[ind] == 0:
            # record mis-classified natural images as adversarial states
            ims_adv_batch[ind] = X[ind].cpu()
    # initialize adversarial image as natural image
    X_adv = X.clone()
    # start in random location of l_p ball
    if args.adv_rand_start:
        X_adv = rand_init_l_p(X_adv, args.adv_norm, args.adv_eps)
    
    for step in range(args.adv_steps+1):
        # get attack gradient and update defense record
        defended, attack_grad, loss, logitdiff = eval_and_bpda_eot_grad(args, model, scheduler, clf, X_adv, y, True)

        loss_batch[step+1] = loss
        # update step-by-step defense record
        class_batch[step+1] = defended.cpu()
        
        if step < args.adv_steps:
            X_adv = pgd_update(X_adv, attack_grad, X, args.adv_norm, args.adv_eps, args.adv_eta)
        # add adversarial images for newly broken images to list
        for ind in range(defended.nelement()):
            if class_batch[step, ind] == 1 and defended[ind] == 0:
                ims_adv_batch[ind] = X_adv[ind].cpu()
        if step == args.adv_steps:
            X_adv_final_batch = X_adv.detach().clone()
        
        #update the accuracy for any brocken state
        acc = torch.min(acc, defended)

        if dist.get_rank()==0 and (step == 1 or step % args.log_freq == 0 or step == args.adv_steps):
            # print attack info
            print('Batch {} of {}, Attack {} of {}   Batch defended: {} of {}, acc {:.4f}'.
                  format(batch_num - args.start_batch + 2, args.end_batch - args.start_batch + 1,
                         step, args.adv_steps, int(torch.sum(defended).cpu().numpy()), X_adv.shape[0], int(torch.sum(defended).cpu().numpy())/X_adv.shape[0] ))
        dist.barrier()
        
        # record final adversarial image for unbroken states
        for ind in range(defended.nelement()):
            if defended[ind] == 1:
                ims_adv_batch[ind] = X_adv[ind].cpu()
    return grad_batch, class_batch, loss_batch, nat_acc, acc, ims_adv_batch, X_adv_final_batch



