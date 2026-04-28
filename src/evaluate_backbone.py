import torch
import torch.nn as nn
import yaml
import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from src.dataset import create_dataloaders
from src.pretrain_backbone import BackboneTrainer

def evaluate_backbone(config_path, checkpoint_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device setup
    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    # Load dataloaders (val_loader contains the 20% validation split)
    _, val_loader, _ = create_dataloaders(config)
    
    # Load model
    model = BackboneTrainer(config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    # The saved state in pretrain_backbone had 'backbone_state_dict'
    # But wait, BackboneTrainer has .backbone inside it.
    model.backbone.load_state_dict(state['backbone_state_dict'])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating Backbone"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            logits = model(images)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Metrics
    class_names = config['data']['class_names']
    report = classification_report(all_labels, all_preds, target_names=class_names)
    cm = confusion_matrix(all_labels, all_preds)
    
    results_dir = config['evaluation']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    # Save Report
    print("\nClassification Report:")
    print(report)
    with open(os.path.join(results_dir, 'backbone_classification_report.txt'), 'w') as f:
        f.write(report)
        
    # Save Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
    plt.title('Backbone Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'backbone_confusion_matrix.png'))
    plt.close()
    
    print(f"\nResults saved to {results_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    evaluate_backbone(args.config, args.checkpoint)
