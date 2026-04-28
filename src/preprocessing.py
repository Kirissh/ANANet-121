import cv2
import numpy as np
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_otsu_mask(image_bgr: np.ndarray) -> np.ndarray:
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)
    # Fill holes
    h, w = mask.shape
    mask_floodfill = mask.copy()
    mask_c = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(mask_floodfill, mask_c, (0,0), 1)
    mask_floodfill_inv = cv2.bitwise_not(mask_floodfill)
    mask = mask | mask_floodfill_inv
    return mask.astype(np.float32)

def generate_clahe_mask(image_bgr: np.ndarray) -> np.ndarray:
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img_gray)
    mask = cv2.adaptiveThreshold(enhanced, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype(np.float32)

def preprocess_dataset(raw_dir: str, output_dir: str, mask_type: str, image_size: int):
    image_out_dir = os.path.join(output_dir, 'images')
    mask_out_dir = os.path.join(output_dir, 'masks')
    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)
    
    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    
    for cls in classes:
        os.makedirs(os.path.join(image_out_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(mask_out_dir, cls), exist_ok=True)
        
        # supports common image extensions
        files = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.bmp'):
            files.extend(glob.glob(os.path.join(raw_dir, cls, ext)))
            
        for f in files:
            img = cv2.imread(f)
            if img is None:
                continue
                
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
            
            if mask_type == 'otsu':
                mask = generate_otsu_mask(img)
            else:
                mask = generate_clahe_mask(img)
                
            fname = os.path.basename(f)
            # save
            cv2.imwrite(os.path.join(image_out_dir, cls, fname), img)
            cv2.imwrite(os.path.join(mask_out_dir, cls, fname), (mask * 255).astype(np.uint8))

def create_splits(processed_dir: str, train: float, val: float, test: float, seed: int):
    image_dir = os.path.join(processed_dir, 'images')
    mask_dir = os.path.join(processed_dir, 'masks')
    splits_dir = os.path.join(processed_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    
    data = []
    classes = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    for cls in classes:
        files = os.listdir(os.path.join(image_dir, cls))
        for f in files:
            image_path = os.path.join(image_dir, cls, f)
            mask_path = os.path.join(mask_dir, cls, f)
            data.append({
                'image_path': image_path,
                'mask_path': mask_path,
                'label': class_to_idx[cls],
                'class_name': cls
            })
            
    df = pd.DataFrame(data)
    
    # Stratified splits
    if test > 0:
        train_val, test_df = train_test_split(df, test_size=test, stratify=df['label'], random_state=seed)
    else:
        train_val = df
        test_df = pd.DataFrame(columns=df.columns)

    val_ratio = val / (train + val)
    train_df, val_df = train_test_split(train_val, test_size=val_ratio, stratify=train_val['label'], random_state=seed)
    
    train_df.to_csv(os.path.join(splits_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(splits_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(splits_dir, 'test.csv'), index=False)
