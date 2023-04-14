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
    parser.add_argument('--count', type=int, default=100, help='number of samples for crafting universal noise ')
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
    ray_trafo = OperatorModule(dataset.ray_trafo)

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
    clean_recon_clean_obs_psnr=[]
    
    attack_psnr=[]
    attack_ssim=[]
    attack_recon_clean_obs_psnr=[]
    attack_recon_adv_obs_psnr=[]
    
    nnorm_delta=[]
    normsquare_delta=[]
    lipschitz_lb=[]
    
    n_epochs=args.niter
    bs = 100
    criterion = nn.MSELoss()
    #Attack params
    eps = args.eps
    obs=torch.from_numpy(np.asarray(obs)[None, None])
    epsilon =(eps)*(obs.max()-obs.min())
    print(epsilon)
    
    
    delta = torch.randn(1, 1, 1000, 513).clamp_(-epsilon, epsilon).cuda()
    del_max = delta.detach()
    delta.requires_grad=True
    opt = optim.Adam([delta], lr=1e-3)
    init_loss = 1e8
    
    
    for j in range(n_epochs):
        for obs, gt in zip(test_data.observations, test_data.ground_truth):
            obs = torch.from_numpy(np.asarray(obs)[None, None])
            gt = torch.from_numpy(np.asarray(gt)[None, None])
            obs = obs.cuda()
            #print(obs.shape)
            gt = gt.cuda()
            clean_recon = forward_pass(reconstructor, args.method, fbp_op_mod, obs).detach()
            criterion = criterion.cuda()
            #attack
            pred = forward_pass(reconstructor, args.method, fbp_op_mod, obs+delta)
            loss = -criterion(pred, clean_recon)
            opt.zero_grad()
            loss.backward()
            opt.step()
            delta.data.clamp_(-epsilon, epsilon)
        
        
    torch.save(delta.detach().cpu(),f'{experiment_dir}/universal_adv.pt')
    

    
        
if __name__ == '__main__':
    args = parse_args()
    main(args)
