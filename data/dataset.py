import os
from tqdm import tqdm
import shutil
from typing import Tuple, List
from glob import glob
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from .augmentations import train_global_transform, train_private_transform, test_global_transform

class GrowthDataset(Dataset):
    def __init__(self, df, train_mode, private_transforms, global_transforms):
        self.df = df
        self.train_mode = train_mode
        self.private_transforms = private_transforms
        self.global_transforms = global_transforms
        
    def __getitem__(self, index):
        before_path = self.df.iloc[index]['before_file_path']
        before_image = cv2.imread(before_path)

        after_path = self.df.iloc[index]['after_file_path']
        after_image = cv2.imread(after_path)

        if self.private_transforms is not None:
            before_image = self.private_transforms(image=before_image)['image']
            after_image = self.private_transforms(image=after_image)['image']

        if self.global_transforms is not None:
            trans_image = self.global_transforms(image=before_image, image1=after_image)
            before_image = trans_image['image']
            after_image = trans_image['image1']
        
        if self.train_mode == 'test':
            return before_image, after_image
        else:
            time_delta = float(self.df.iloc[index]['time_delta'])
            return before_image, after_image, time_delta

    def __len__(self):
        return len(self.df)

def prepare_dataloader(df, mode, args):
    if mode == 'train':
        dataset = GrowthDataset(df, mode, train_private_transform, train_global_transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        
    elif mode == 'valid':
        dataset = GrowthDataset(df, 'train', None, test_global_transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    else:
        if args.tta:
            dataset = GrowthDataset(df, mode, None, train_global_transform)
        else:
            dataset = GrowthDataset(df, mode, None, test_global_transform)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)
        
    return loader