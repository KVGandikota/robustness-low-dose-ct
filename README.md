# Evaluating Adversarial Robustness of Low dose CT Recovery 

Low dose computer tomography (CT) acquisition using reduced radiation or sparse angle
measurements is recommended to decrease the harmful effects of X-ray radiation. Recent
works successfully apply deep networks to the problem of low dose CT recovery on benchmark
datasets. However, their robustness needs a thorough evaluation before use in clinical
settings. In this work, we evaluate the robustness of different deep learning approaches
and classical methods for CT recovery. We show that deep networks, including model
based networks encouraging data consistency are more susceptible to untargeted attacks.
Surprisingly, we observe that data consistency is not heavily affected even for these poor
quality reconstructions, motivating the need for better regularization for the networks.
We demonstrate the feasibility of universal attacks and study attack transferability across
different methods. We analyze robustness to attacks causing localized changes in clinically
relevant regions. Both classical approaches and deep networks are affected by such attacks
leading to change in visual appearance of localized lesions, for extremely small perturbations.
As the resulting reconstructions have high data consistency with original measurements,
these localized attacks can be used to explore the solution space of CT recovery problem.

This repository is forked from [here](https://github.com/oterobaguer/dip-ct-benchmark).
We use the CT reconstruction networks from the [CT benchmark](https://github.com/oterobaguer/dip-ct-benchmark) for robustness evaluation.
For localized attacks on CT reconstruction we use an adversarially trained classifier from [here](https://github.com/drgHannah/Explorable_CT_Reconstruction) which uses a robust nodule classifier to explore the solution space of CT reconstruction.
If you find this work useful for your research please consider citing our work, the CT benchmark paper.  

# Getting started
Follow the [instructions](https://github.com/oterobaguer/dip-ct-benchmark/blob/master/instructions.txt) in the original repository to install astra toolbox, odl and dival



````
@inproceedings{
gandikota2023evaluating,
title={Evaluating Adversarial Robustness of Low dose {CT} Recovery},
author={Kanchana Vaishnavi Gandikota and Paramanand Chandramouli and Hannah Dr{\"o}ge and Michael Moeller},
booktitle={Medical Imaging with Deep Learning},
year={2023},
url={https://openreview.net/forum?id=L-N1uAxfQk1}
}
```

```
@article{Baguer_2020,
	doi = {10.1088/1361-6420/aba415},
	url = {https://doi.org/10.1088%2F1361-6420%2Faba415},
	year = 2020,
	month = {sep},
	publisher = {{IOP} Publishing},
	volume = {36},
	number = {9},
	pages = {094004},
	author = {Daniel Otero Baguer and Johannes Leuschner and Maximilian Schmidt},
	title = {Computed tomography reconstruction using deep image prior and learned reconstruction methods},
	journal = {Inverse Problems}
}
```
