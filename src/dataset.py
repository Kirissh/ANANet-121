import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter
import os

class HEp2Dataset(Dataset):
    def __init__(self, csv_path, config, split='train'):
        self.df = pd.read_csv(csv_path)
        self.config = config
        self.split = split
        
        img_size = config['data']['image_size']
        mean = tuple(config['augmentation']['normalize_mean'])
        std = tuple(config['augmentation']['normalize_std'])
        
        if split == 'train' and config['augmentation']['use_augmentation']:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=config['augmentation']['rotation_limit'], p=0.8),
                A.RandomBrightnessContrast(
                    brightness_limit=config['augmentation']['brightness_limit'],
                    contrast_limit=config['augmentation']['contrast_limit'], p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.ElasticTransform(p=0.3),
                A.GridDistortion(p=0.3),
                A.Resize(img_size, img_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        mask_path = row['mask_path']
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask / 255.0).astype(np.float32)
        
        transformed = self.transform(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask'].unsqueeze(0)  # (1, H, W)
        
        # Binarize mask again just in case augmentation interpolated it
        transformed_mask = (transformed_mask > 0.5).float()

        return {
            'image': transformed_image,
            'mask': transformed_mask,
            'label': torch.tensor(row['label'], dtype=torch.long),
            'class_name': row['class_name'],
            'image_path': image_path
        }

def get_class_weights(csv_path):
    df = pd.read_csv(csv_path)
    counts = Counter(df['label'])
    total = len(df)
    num_classes = len(counts)
    
    weights = [total / (num_classes * counts[i]) for i in range(num_classes)]
    return torch.FloatTensor(weights)

def create_dataloaders(config):
    splits_dir = os.path.join(config['data']['root_dir'], 'splits')
    
    train_ds = HEp2Dataset(os.path.join(splits_dir, 'train.csv'), config, 'train')
    val_ds = HEp2Dataset(os.path.join(splits_dir, 'val.csv'), config, 'val')
    test_ds = HEp2Dataset(os.path.join(splits_dir, 'test.csv'), config, 'test')
    
    train_loader = DataLoader(
        train_ds, batch_size=config['training']['batch_size'], 
        shuffle=True, num_workers=config['training']['num_workers'],
        pin_memory=True, persistent_workers=config['training']['num_workers'] > 0
    )
    val_loader = DataLoader(
        val_ds, batch_size=config['training']['batch_size'], 
        shuffle=False, num_workers=config['training']['num_workers'],
        pin_memory=True, persistent_workers=config['training']['num_workers'] > 0
    )
    test_loader = DataLoader(
        test_ds, batch_size=config['training']['batch_size'], 
        shuffle=False, num_workers=config['training']['num_workers'],
        pin_memory=True, persistent_workers=config['training']['num_workers'] > 0
    )
    
    return train_loader, val_loader, test_loader
