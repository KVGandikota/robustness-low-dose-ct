import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0"#,1,2,3"
import sys
sys.path.append('/data/vsa_vaishnavi/inverse_prob_adversarial/dip-ct-benchmark/')
import torch
from dival import DataPairs
from dliplib.reconstructors import get_reconstructor
from dliplib.utils.helper import load_standard_dataset
from odl import Operator
from odl.contrib.torch import OperatorModule
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging
import importlib
import kornia
try:
    from skimage.metrics import structural_similarity
except ImportError:
    # fallback for scikit-image <= 0.15.0
    from skimage.measure import compare_ssim as structural_similarity
    
import localized as ds
from network import BasicResnet
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing-untargeted attacks')
    parser.add_argument('--dataset', type=str, default='lodopab', help='data set for testing lodopab, lodopab_200')
    parser.add_argument('--eps', type=float, default=0.01, help='adversarial noise strength. option 0.01, 0.025, 0.05')
    parser.add_argument('--size_part', type=float, default=1.0, help ='fraction of data samples used in training')
    parser.add_argument('--log_dir', type=str, default=None, help='Experiment root')
    parser.add_argument('--method', type=str, default='fbpunet', help='method to test. Options are fbp, fbpunet, learnedpd, learnedgd, iradonmap')
    parser.add_argument('--niter', type=int, default=None, help='number of iterations for adversarial attack')
    parser.add_argument('--count', type=int, default=100, help='number of samples for testing')
    parser.add_argument('--restarts', type=int, default=1, help='number ofrandom restarts')
    parser.add_argument('--datapath', type=str, default='/data/lodopab-new2', help='path to dataset with nodule positions')

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

#Mask to localize patch in image and sinogram
def get_mask(ray_trafo,loc, nosz=16, shape_val=362):
    
    mask = torch.zeros([loc.shape[0],1,shape_val,shape_val]).to(loc.device)
    
    for i in range(loc.shape[0]):
        mask[i,:,int(loc[i,0]-nosz):int(loc[i,0]+nosz),int(loc[i,1]-nosz):int(loc[i,1]+nosz)] = 1
        
    mask_sino = ray_trafo(mask) > 0    
    return mask,mask_sino

#soft mask
def get_softmask(ray_trafo,loc, nosz=16, shape_val=362):
    mask = torch.zeros([loc.shape[0],1,shape_val,shape_val]).to(loc.device)
    for i in range(loc.shape[0]):      
        mask[i,:,int(loc[i,0]-nosz):int(loc[i,0]+nosz),int(loc[i,1]-nosz):int(loc[i,1]+nosz)] = 1
    gauss = kornia.filters.GaussianBlur2d((5, 5), (3, 3)).to(loc.device)
    sino_blurmask =ray_trafo(gauss(mask))
    mask_sino = sino_blurmask/sino_blurmask.max()  
    return mask,mask_sino

#Function to perform localized attacks
def localized_attack_restarts(num_restarts, eps, niter, criterion, obs, pos, clean_pred, clean_recon, reconstructor, classifier, method, fbp_op_mod, ray_trafo):
    losses = []
    advrecons = []
    adv_noises = []
    _,mask_sino =  get_softmask(ray_trafo,pos.unsqueeze(0),nosz=16,  shape_val=clean_recon.shape[2])
    
    for i in range(num_restarts):            
        #initialize adversarial noise
        epsilon = (eps)*(torch.max(obs)-torch.min(obs))        
        with torch.no_grad():
            delta = (torch.randn_like(obs)).clamp_(-epsilon, epsilon)
            recon = forward_pass(reconstructor, method, fbp_op_mod, obs+delta)
            crop = ds.crop_center(recon, pos.unsqueeze(0), size=16)
            pred = F.softmax(classifier((crop-crop.mean())/crop.std()),dim=1)
            init_loss = -criterion(pred, clean_pred).detach().item()
            del_max = delta.detach()
        
        delta.requires_grad=True
        opt = optim.Adam([delta], lr=1e-3)
        
        if niter is None:
            niter = 50 #set maximum number of adversarial steps
            
        #generate adversarial noise, 
        #iterate till miscalssification or max niter
        success=False
        num_iter=0
        while not success and num_iter<niter:
            recon = forward_pass(reconstructor, method, fbp_op_mod, obs+delta)
            crop = ds.crop_center(recon, pos.unsqueeze(0), size=16)
            pred = F.softmax(classifier((crop-crop.mean())/crop.std()),dim=1)
            loss = -criterion(pred, clean_pred)
            opt.zero_grad()
            loss.backward()
            if pred.max(1)[1]!=clean_pred.max(1)[1]:
                del_max.data = delta.data
                success = True
                break
            if loss.item()<init_loss:
                init_loss=loss.item()
                del_max.data = delta.data
            opt.step()
            
            delta.data *= mask_sino
            delta.data.clamp_(-epsilon, epsilon)
           
            num_iter += 1           
            
        #Adversarial_output  
        with torch.no_grad():
            advpred = forward_pass(reconstructor, method, fbp_op_mod, obs+delta)
            loss = -criterion(pred,clean_pred)
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
        args.log_dir = f'localized_{args.dataset}_{args.method}_niter{args.niter}_eps{args.eps}'
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
    #new datset with nodule positions provide the correct datapath
    dataset = ds.lodopab_datapos(data_path = args.datapath)
    
    #forward radon transform
    ray_trafo = OperatorModule(load_standard_dataset(args.dataset).ray_trafo)

    count = args.count # number of samples
    ground_truth = dataset.data['gt'][:count]
    observations = dataset.data['obs'][:count]
    positions = dataset.data['pos'][:count]
    
    #Load classifier network
    classnet_path = 'network_100a_old.pt'
    classifier = BasicResnet()
    classifier.load_state_dict(torch.load(classnet_path))
    classifier = classifier.cuda()
    classifier.eval()
    
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
    advobs_psnr=[]
    interior_psnr=[]
    exterior_psnr=[]
    interior_psnr_clean=[]
    exterior_psnr_clean=[]
    attack_recon_clean_obs_psnr=[]
    attack_recon_adv_obs_psnr=[]
    attack_success=0
    idx=0
    for obs, gt, pos in zip(observations, ground_truth, positions):
        criterion = nn.BCELoss()#for attacking classifier        
        obs = obs.cuda()
        gt = gt.cuda()
        pos = pos.cuda()
        criterion = criterion.cuda()
        #clean recon
        clean_recon = forward_pass(reconstructor, args.method, fbp_op_mod, obs).detach()
        #Classifier prediction of clean reconstruction
        crop = ds.crop_center(clean_recon, pos.unsqueeze(0), size=16).detach()
        clean_pred = F.softmax(classifier((crop - crop.mean()) / crop.std()),dim=1).round().detach()
        '''Adversarial attack'''
        #Attack params
        eps = args.eps
        niter = args.niter
        num_restarts = args.restarts
        
        advrecon,delta = localized_attack_restarts(num_restarts, eps, niter, criterion, obs, pos, clean_pred, clean_recon, reconstructor, classifier, args.method, fbp_op_mod,ray_trafo)
        
        #print('adversarial attack done')
        
        torch.save(delta.detach().cpu(),f'{experiment_dir}/adv{idx}.pt')
        
        clean_recon = torch.clamp(clean_recon, 0,1)
        advrecon = torch.clamp(advrecon, 0,1)
        torch.save(clean_recon.detach().cpu(),f'{experiment_dir}/clean{idx}.pt')
        torch.save(advrecon.detach().cpu(),f'{experiment_dir}/eps_{args.eps}{idx}.pt')
              
        crop=ds.crop_center(advrecon, pos.unsqueeze(0), size=16)
        adv_pred = classifier((crop-crop.mean())/crop.std()).detach()
        #print(F.softmax(adv_pred,dim=1))
        if adv_pred.max(1)[1]!=clean_pred.max(1)[1]:
            #print('Evaluate:')
            #print('apred',adv_pred)
            #print(F.softmax(adv_pred,dim=1))
            #print('clean_pred',clean_pred)
            attack_success += 1
        
        mask,_ =  get_mask(ray_trafo,pos.unsqueeze(0),nosz=16,  shape_val=clean_recon.shape[2])#exterior zeroed
        maskn = 1-mask#nodulepatch zeroed
        
        with torch.no_grad():
            #calculate and save metrics
            clean_psnr.append(psnr(clean_recon, gt))
            clean_ssim.append(ssim(clean_recon, gt))
            
            attack_psnr.append(psnr(advrecon, gt))
            attack_ssim.append(ssim(advrecon, gt))
            
            clean_recon_clean_obs_psnr.append(psnr(obs,ray_trafo(clean_recon)))
            advobs_psnr.append(psnr(obs, obs+delta))
            
            
            #calculate interior exterior errors 
            err_int=(torch.sum((advrecon*mask-gt*mask)**2,dim=[2,3])/1024.).cpu().numpy()
            err_ext=(torch.sum((advrecon*maskn-gt*maskn)**2,dim=[2,3])/(362**2-1024)).cpu().numpy()
            err_int_clean=(torch.sum((clean_recon*mask-gt*mask)**2,dim=[2,3])/1024.).cpu().numpy()
            err_ext_clean=(torch.sum((clean_recon*maskn-gt*maskn)**2,dim=[2,3])/(362**2-1024)).cpu().numpy()
            
            #calculate interior exterior psnrs
            gt = gt.cpu().detach().numpy()
            data_range = (np.max(gt) - np.min(gt))
            
            #attack
            interior_psnr.append(20*np.log10(data_range) - 10*np.log10(err_int))
            exterior_psnr.append(20*np.log10(data_range) - 10*np.log10(err_ext))
            
            #clean
            interior_psnr_clean.append(20*np.log10(data_range) - 10*np.log10(err_int_clean))
            exterior_psnr_clean.append(20*np.log10(data_range) - 10*np.log10(err_ext_clean))
            attack_recon_clean_obs_psnr.append(psnr(obs,ray_trafo(advrecon)))
            attack_recon_adv_obs_psnr.append(psnr(obs+delta,ray_trafo(advrecon)))
            
            idx += 1        
        
    clean_psnr = np.array(clean_psnr, dtype=np.float32).mean()
    clean_ssim = np.array(clean_ssim, dtype=np.float32).mean()
    clean_recon_clean_obs_psnr = np.array(clean_recon_clean_obs_psnr, dtype=np.float32).mean()
    attack_psnr = np.array(attack_psnr, dtype=np.float32).mean()
    attack_ssim = np.array(attack_ssim, dtype=np.float32).mean()
    advobs_psnr = np.array(advobs_psnr, dtype=np.float32).mean()
    interior_psnr = np.array(interior_psnr, dtype=np.float32).mean()
    exterior_psnr = np.array(exterior_psnr, dtype=np.float32).mean()
    interior_psnr_clean = np.array(interior_psnr_clean, dtype=np.float32).mean()
    exterior_psnr_clean = np.array(exterior_psnr_clean, dtype=np.float32).mean()
    attack_recon_clean_obs_psnr = np.array(attack_recon_clean_obs_psnr, dtype=np.float32).mean()
    attack_recon_adv_obs_psnr = np.array(attack_recon_adv_obs_psnr, dtype=np.float32).mean()    

    log_string('End of testing... ')
    log_string(f'clean psnr: {clean_psnr} \n'
               f'clean_ssim: {clean_ssim}  \n'               
               f'interior_psnr_clean : {interior_psnr_clean } \n'
               f'exterior_psnr_clean: {exterior_psnr_clean}\n'
               f'clean_recon_clean_obs_psnr: {clean_recon_clean_obs_psnr} \n'
               f'attack psnr: {attack_psnr}  \n'
               f'attack_ssim: {attack_ssim}  \n'               
               f'interior_psnr_attack: {interior_psnr} \n'
               f'exterior_psnr_attack: {exterior_psnr} \n'
               f'attack_recon_clean_obs_psnr: {attack_recon_clean_obs_psnr}\n'
               f'attack_recon_adv_obs_psnr: {attack_recon_adv_obs_psnr}\n'
               f'advobs_psnr: {advobs_psnr} \n'
               f'attack_success: {attack_success}\n')    
    
        
if __name__ == '__main__':
    args = parse_args()
    main(args)
