import glob
import numpy as np
import torch
import h5py
import os
from torchvision import transforms

class lodopab_datapos(torch.utils.data.Dataset):
    def __init__(self, data_path = "/data/lodopab-new2"):
        ''' Dataset for targeted attack
        args:
            data_path: path to dataset
        '''
        all_files =glob.glob(f'{data_path}/*.hdf5')
        gt_files =[]
        obs_files=[]
        pos_files=[] 
        for file in all_files:
            if 'ground_truth' in file:
                gt_files.append(file)
                gt_files.sort()
            elif 'observation' in file:
                obs_files.append(file)
                obs_files.sort()
            elif 'pos' in file:
                pos_files.append(file)
                pos_files.sort()
        
        gt_tensor=torch.cat([torch.cat([torch.tensor(h5py.File(files, "r")[key][()], dtype=torch.float32) for key in list(h5py.File(files, "r").keys())],axis=0) for files in gt_files],axis=0)

        obs_tensor=torch.cat([torch.cat([torch.tensor(h5py.File(files, "r")[key][()], dtype=torch.float32) for key in list(h5py.File(files, "r").keys())],axis=0) for files in obs_files],axis=0)
        
        pos_tensor=torch.cat([torch.cat([torch.tensor(h5py.File(files, "r")[key][()], dtype=torch.float32) for key in list(h5py.File(files, "r").keys())],axis=0) for files in pos_files],axis=0)
        
        gt_tensor = torch.reshape(gt_tensor, (gt_tensor.shape[0], 1, 1, gt_tensor.shape[1], gt_tensor.shape[2]))
        obs_tensor = torch.reshape(obs_tensor, (obs_tensor.shape[0], 1, 1, obs_tensor.shape[1], obs_tensor.shape[2]))
        #pos_tensor = torch.reshape(pos_tensor, (pos_tensor.shape[0], 1,  pos_tensor.shape[1]))
        
        if len(gt_tensor)==len(obs_tensor)==len(pos_tensor):
            #pos_tensor = torch.reshape(pos_tensor, (pos_tensor.shape[0], 1,  pos_tensor.shape[1]))
            self.data = {'gt': gt_tensor,
                          'obs': obs_tensor,
                          'pos': pos_tensor}
            print("Loaded {} dataitems...".format(len(obs_tensor)))
        elif len(gt_tensor)==len(pos_tensor) and len(obs_tensor)<len(pos_tensor):
            
            #filter out files without nodules 
            val_indices=torch.sum(torch.abs(pos_tensor),dim=1)!=0.
            gt_valid = gt_tensor[val_indices]
            pos_valid = pos_tensor[val_indices]
            obs_valid = obs_tensor
            
            #filter out files where 32x32 nodule patch
            # doesnot lie fully with in 362x362 image
            val_indices = (torch.sum(torch.sign(pos_valid-345),dim=1)==-2)*(torch.sum(torch.sign(pos_valid-16),dim=1)==2).bool() 
            gt_valid = gt_valid[val_indices]
            pos_valid = pos_valid[val_indices]
            obs_valid = obs_valid[val_indices]
            print(pos_valid.shape)
            pos_valid = pos_valid.roll(1,-1)
            
            '''#filter out files where nodule position is negative 
            #this can happen due to center cropping in lodopab
            val_indices=torch.sum(torch.sign(pos_valid),dim=1)!=0.
            gt_valid = gt_valid[val_indices]
            pos_valid = pos_valid[val_indices]
            obs_valid = obs_tensor[val_indices]
            
            #
            val_indices=torch.sum(torch.sign(pos_valid-345),dim=1)==-2.
            gt_valid = gt_valid[val_indices]
            pos_valid = pos_valid[val_indices]
            obs_valid = obs_valid[val_indices]'''
            
            #filter out files where 32x32 nodule does not lie in 362x362
            
            self.data = {'gt': gt_valid,
                          'obs': obs_valid,
                          'pos': pos_valid}
            print("Loaded {} dataitems...".format(len(gt_valid)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ''' Returns: ground truth, low dose sinogram, nodule pos
        '''
        data_item = self.data[idx]
        gt = self.data['gt'][idx]
        obs = self.data['obs'][idx]
        pos = self.data['pos'][idx]
        return gt, obs, pos 

def crop_center(input, center, size):
    ''' Crops at center of the input. 
    args:
        input: input image (size: bsz x c x w x h)
        center: location to crop
        size: size of crop
    '''

    cropped = torch.zeros([center.shape[0],1,size*2,size*2]).to(input.device)
    for i in range(center.shape[0]):
        if torch.floor(center[i,1])==center[i,1]:
            center[i,1]+=1e-4
        if torch.floor(center[i,0])==center[i,0]:
            center[i,0]+=1e-4
        left_crop = -int(torch.floor(0 + center[i,1]-size))
        right_crop = -int(torch.floor(input.shape[2] - (center[i,1]+size)))-1
        upper_crop = -int(torch.floor(0 + center[i,0]-size))
        lower_crop = -int(torch.floor(input.shape[3] - (center[i,0]+size)))-1
        cropped[i]=torch.nn.functional.pad(input[i],(left_crop,right_crop,upper_crop,lower_crop))
    return cropped
