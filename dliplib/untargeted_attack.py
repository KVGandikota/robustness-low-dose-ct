import argparse
import os
import sys
sys.path.append('../')
import torch
from dival import DataPairs
from dliplib.reconstructors import get_reconstructor
from dliplib.utils.helper import load_standard_dataset
from odl import Operator
from odl.contrib.torch import OperatorModule
import numpy as np
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import logging
import importlib
try:
    from skimage.metrics import structural_similarity
except ImportError:
    # fallback for scikit-image <= 0.15.0
    from skimage.measure import compare_ssim as structural_similarity

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing-untargeted attacks')
    parser.add_argument('--dataset', type=str, default='lodopab', help='data set for testing lodopab, lodopab_200')
    parser.add_argument('--eps', type=float, default=0.01, help='adversarial noise strength. option 0.01, 0.02, 0.05')
    parser.add_argument('--size_part', type=float, default=1.0, help ='fraction of data samples used in training')
    parser.add_argument('--log_dir', type=str, default=None, help='Experiment root')
    parser.add_argument('--method', type=str, default='fbpunet', help='method to test. Options are fbp, fbpunet, learnedpd, learnedgd, iradonmap')
    parser.add_argument('--niter', type=int, default=20, help='number of iterations for adversarial attack')
    parser.add_argument('--count', type=int, default=1, help='number of samples for testing')
    parser.add_argument('--restarts', type=int, default=1, help='number ofrandom restarts')

    return parser.parse_args()
    

'''define forward pass for all methods'''       
def forward_pass(reconstructor, method, fbp_op_mod, observation):
    if method == 'fbp':        
        pred = fbp_op_mod(observation)
    else:
        reconstructor.model.eval()
        if method == 'fbpunet': 
            pred = reconstructor.model(fbp_op_mod(observation))
        elif 'learned' in method:
            pred = reconstructor.model(observation/reconstructor.opnorm)
        elif method == 'iradonmap':
            pred = reconstructor.model(observation)
    return pred
    
def psnr(recon, gt):
    gt = gt.cpu().detach().numpy()
    recon = recon.cpu().detach().numpy()
    mse = np.mean((recon - gt)**2)
    if mse == 0.:
        return float('inf')
    data_range = (np.max(gt) - np.min(gt))
    return 20*np.log10(data_range) - 10*np.log10(mse)
        
def ssim(recon, gt):
    gt = gt.cpu().detach().squeeze(0).squeeze(0).numpy()
    recon = recon.cpu().squeeze(0).squeeze(0).detach().numpy()
    data_range = (np.max(gt) - np.min(gt))
    return structural_similarity(recon, gt, data_range=data_range)

def untargeted_attack_restarts(num_restarts,eps,niter,criterion,obs,clean_pred,reconstructor,method,fbp_op_mod):
    losses = []
    advrecons = []
    adv_noises = []
    for i in range(num_restarts):    
        #initialize adversarial noise
        epsilon = (eps)*(torch.max(obs)-torch.min(obs))        
        with torch.no_grad():#
            delta = torch.randn_like(obs).clamp_(-epsilon, epsilon)
            pred = forward_pass(reconstructor, method, fbp_op_mod, obs+delta)
            init_loss = -criterion(pred, clean_pred).detach().item()
            del_max = delta.detach()
        
        delta.requires_grad=True
        opt = optim.Adam([delta], lr=1e-3)
        
        #untargeted attack --generate adversarial noise
        for t in range(niter):
            pred = forward_pass(reconstructor, method, fbp_op_mod, obs+delta)
            loss = -criterion(pred, clean_pred)
            opt.zero_grad()
            loss.backward()
            if loss.item()<init_loss:
                init_loss=loss.item()
                del_max.data = delta.data
            opt.step()
            delta.data.clamp_(-epsilon, epsilon)
            
        #Adversarial_output  
        with torch.no_grad():
            advpred = forward_pass(reconstructor, method, fbp_op_mod, obs+delta)
            loss = -criterion(pred, clean_pred)
            losses.append(loss)
            advrecons.append(advpred)
            adv_noises.append(del_max)
    best_loc = losses.index(min(losses))    
    return advrecons[best_loc],adv_noises[best_loc]
    
def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''CREATE DIR'''
    if args.log_dir is None:
        args.log_dir = f'{args.dataset}_{args.method}_niter{args.niter}_eps{args.eps}'
    experiment_dir = 'log/' + args.log_dir
    if not os.path.isdir(experiment_dir): 
        os.mkdir(experiment_dir)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    
    '''Dataset'''
    dataset = load_standard_dataset(args.dataset)
    test_data = dataset.get_data_pairs('test', args.count)

    obs = list(y for y, x in test_data)
    gt = list(x for y, x in test_data)
    count = args.count
    if count is None:
        count = len(test_data)
    test_data = DataPairs(obs[0 : count], gt[0 : count], name='test')
    
    '''Reconstructor'''
    reconstructor = get_reconstructor(args.method, args.dataset, args.size_part) 

    if args.method in ['fbp','fbpunet']:
        fbp_op_mod = OperatorModule(reconstructor.fbp_op)
    else:
        fbp_op_mod = None  
         
    
    clean_psnr=[]
    clean_ssim=[]
    attack_psnr=[]
    attack_ssim=[]
    
    idx=0
    for obs, gt in zip(test_data.observations, test_data.ground_truth):
        obs = torch.from_numpy(np.asarray(obs)[None, None])
        gt = torch.from_numpy(np.asarray(gt)[None, None])
        
        #criterion
        criterion = nn.MSELoss()
        if hasattr(reconstructor, 'device'):
            obs = obs.to(reconstructor.device)
            gt = gt.to(reconstructor.device)
            criterion = criterion.to(reconstructor.device)
        
        #clean recon
        clean_recon = forward_pass(reconstructor, args.method, fbp_op_mod, obs).detach()
        
        '''Adversarial attack'''
        #Attack params
        eps = args.eps
        niter = args.niter
        num_restarts = args.restarts
        
        advpred,delta = untargeted_attack_restarts(num_restarts, eps, niter, criterion, obs, clean_recon, reconstructor, args.method, fbp_op_mod)
        #torch.save(delta.detach().cpu(),f'{experiment_dir}/adv{idx}.pt')

        #calculate and save metrics
        clean_psnr.append(psnr(clean_recon, gt))
        clean_ssim.append(ssim(clean_recon, gt))
        attack_psnr.append(psnr(advpred, gt))
        attack_ssim.append(ssim(advpred, gt))
        idx += 1
        
        
    clean_psnr = np.array(clean_psnr, dtype=np.float32).mean()
    clean_ssim = np.array(clean_ssim, dtype=np.float32).mean()
    attack_psnr = np.array(attack_psnr, dtype=np.float32).mean()
    attack_ssim = np.array(attack_ssim, dtype=np.float32).mean()
               
    log_string('End of testing... ')
    log_string(f'clean psnr: {clean_psnr} \n'
               f'clean_ssim: {clean_ssim}  \n'
               f'attack psnr: {attack_psnr}  \n'
               f'attack_ssim: {attack_ssim}  \n')
    
        
if __name__ == '__main__':
    args = parse_args()
    main(args)
