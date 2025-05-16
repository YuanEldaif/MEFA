# Code for **Memory Efficient Full-gradient Attacks (MEFA) Framework for Adversarial Defense Evaluations**
This repository contains PyTorch implementations of the evaluation experiments from our paper.
We provide the following scripts for reproducing the results.
NOTE: All configs and arguments use the pixel range [0, 255] for adversarial perturbation adv_eps and attack step size adv_eta. However, all experiments scale images so that the pixels range is [-1, 1]. Adversarial parameters are scaled accordingly during execution. The Langevin step size langevin_eps in the arguments uses the pixel range [-1, 1].
## Environment and Pretrained models
Please refer to ``requirement.txt`` for the required packages of running the codes in the repo.
Put model pre-trained models under ``weights/diffpure`` for score SDE-based defenses, ``weights/`` for EBM defenses. 
A pre-trained smooth EBM ```ebm.pth``` for CIFAR-10 is provided in the ```release``` section of the repository.
## Attack Process Code
MEFA attack in Algorithm 1 of our paper is implemented by ``attack_eval.py``.
## Validation Process Code
MEFA validation of our paper is implemented by ``validation_standalone.py``.
### Acknowledgement
The code base is built upon [AutoAttack](https://github.com/fra31/auto-attack), [DiffPure](https://github.com/NVlabs/DiffPure), [Stochastic Security](https://github.com/point0bar1/ebm-defense.git).


