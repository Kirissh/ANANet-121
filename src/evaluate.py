import os
import argparse
import yaml
import torch
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report
from src.dataset import create_dataloaders
from src.models.full_model import HEp2AnaNet
try:
    import torch_directml
    HAS_DIRECTML = True
except ImportError:
    HAS_DIRECTML = False

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(checkpoint_path, config, device):
    model = HEp2AnaNet(config).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model

@torch.no_grad()
def evaluate_full(model, test_loader, config, device):
    all_preds = []
    all_labels = []
    all_probs = []
    
    results_dir = config['evaluation']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    for batch in tqdm(test_loader, desc="Evaluating"):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        labels = batch['label'].to(device)
        
        logits = model(images, masks)['logits']
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        
    acc = accuracy_score(all_labels, all_preds)
    f1_m = f1_score(all_labels, all_preds, average='macro')
    f1_w = f1_score(all_labels, all_preds, average='weighted')
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except Exception:
        auc = 0.0
        
    metrics = {
        'accuracy': acc,
        'f1_macro': f1_m,
        'f1_weighted': f1_w,
        'mcc': mcc,
        'auc_roc': auc
    }
    print("Test Metrics:", metrics)
    
    # Confusion Matrix
    class_names = config['data']['class_names']
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
        
    return metrics

def run_evaluation(config_path, checkpoint_path, n_gradcam):
    config = load_config(config_path)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif HAS_DIRECTML:
        device = torch_directml.device()
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    _, _, test_loader = create_dataloaders(config)
    
    model = load_model(checkpoint_path, config, device)
    evaluate_full(model, test_loader, config, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--n_gradcam', type=int, default=10)
    args = parser.parse_args()
    
    run_evaluation(args.config, args.checkpoint, args.n_gradcam)
