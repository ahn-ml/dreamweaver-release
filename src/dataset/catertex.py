from torch.utils.data import Dataset
import torch
from typing import Optional, Callable
import os

from torchvision import transforms
import argparse
import h5py
import numpy as np
from einops import rearrange
from torchvision.transforms import functional as F

class CaterTexDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        img_size: int,
        video_len: int,
        num_train_images: Optional[int] = None,
        transform: Optional[Callable] = None, 
        stochastic_sample: Optional[bool] = True,
    ):
        self.img_size = img_size
        self.video_len = video_len
        self.split = split
        self.stochastic_sample = stochastic_sample
        
        if split == 'train':
            dataset = h5py.File(f'{root}/train.h5','r')
            self.images = dataset['videos']
        elif split == 'test':
            dataset = h5py.File(f'{root}/test.h5','r')
            self.images = dataset['videos']
            self.labels = dataset['labels']
            self.masks = dataset['masks']
        
        if num_train_images:
            inds = np.sort((torch.randperm(len(self.images))).cuda()[:num_train_images].cpu().numpy())
            
            self.images = self.images[inds]
            if split == 'test':
                self.labels = self.labels[inds]
                self.masks = self.masks[inds]
            
        print(f'{split} dataset has # {len(self.images)}')
        
        normalize = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.transform = normalize if transform is None else transform
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):        
        # stochastic video sample
        if self.stochastic_sample:
            start_idx = torch.randperm(len(self.images[idx]) - self.video_len + 1)[0]
        else:
            start_idx = 0
        
        # transform PIL -> normalized tensor
        transformed_image=self.images[idx][start_idx:start_idx+self.video_len]
        transformed_image=F.resize(torch.tensor(rearrange(transformed_image, 'f h w c -> f c h w')/255.0), self.img_size)
        
        if self.split =="test":
            label = self.labels[idx][start_idx:start_idx+self.video_len]
            mask = F.resize(torch.tensor(self.masks[idx][start_idx:start_idx+self.video_len]), self.img_size)
        
        source = transformed_image[:self.video_len]
        if self.split == "train":
            return self.transform(source.float())
        else:
            return self.transform(source.float()), label, mask
        

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--video-len", type=int, default=5)
    parser.add_argument("--dataset-path", type=str, default="moving-sprites")
    args = parser.parse_args()
    
    train_dataset = CaterTexDataset(args.dataset_path, "train", args.image_size, args.video_len)