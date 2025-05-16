# Code for **Memory Efficient Full-gradient Attacks (MEFA) Framework for Adversarial Defense Evaluations**
This repository contains PyTorch implementations of main experiments from our paper.

We propose MEFA Framework to evaluate iterative stochastic purification defenses.

**NOTE**: All configs and arguments use the pixel range [0, 255] for adversarial perturbation adv_eps and attack step size adv_eta. However, all experiments scale images so that the pixels range is [-1, 1]. Adversarial parameters are scaled accordingly during execution. The Langevin step size langevin_eps in the arguments uses the pixel range [-1, 1].
## Environment and Pretrained models
Please refer to ``requirement.txt`` for the required packages of running the codes in the repo.

Put model pre-trained models under ``weights/diffpure`` for score SDE-based defenses, ``weights/`` for EBM defenses and DDPM-based OOD defenses. 

A pre-trained DDPM ```ddpm_cinic10.pth```, ```ddpm_food.pth``` for CINIC10 and FOOD respectively, and smooth EBM ```ebm.pth``` for CIFAR-10 are provided in the ```release``` section of the repository.

The sampled 510 CIFAR-10 data for all core experiments is also provided in the ```release``` section of the repository.
## Attack Process Code
MEFA attack in Algorithm 1 of our paper is implemented by ``attack_eval.py``.

MEFA framework PGD+EOT20 attack against score SED-based defense on CIFAR-10 with WideResNet-28-10 under Linf attack:
```python
python3 attack_eval.py --exp_name 'diffpure'  
```
MEFA framework PGD+EOT20 attack against DDPM-based defense on CIFAR-10 with WideResNet-28-10 under Linf attack:
```python
python3 attack_eval.py --exp_name 'hf_DDPM'
```
MEFA framework PGD+EOT20 attack against EBM-based defense on CIFAR-10 with WideResNet-28-10 under Linf attack:
```python
python3 attack_eval.py --exp_name 'ebm'
```
## Validation Process Code
MEFA validation of our paper is implemented by ``validation_standalone.py``. Update the --results_dir before the execution. 

MEFA framework validation against score SED-based defense on CIFAR-10 with WideResNet-28-10:
```python
python3 validation_standalone.py --exp_name 'diffpure'  
```
MEFA framework validation against DDPM-based defense on CIFAR-10 with WideResNet-28-10:
```python
python3 validation_standalone.py --exp_name 'hf_DDPM'  
```
MEFA framework validation against EBM-based defense on CIFAR-10 with WideResNet-28-10:
```python
python3 validation_standalone.py --exp_name 'ebm'  
```
### Acknowledgement
The code base is built upon [AutoAttack](https://github.com/fra31/auto-attack), [DiffPure](https://github.com/NVlabs/DiffPure) and [Stochastic Security](https://github.com/point0bar1/ebm-defense.git).


