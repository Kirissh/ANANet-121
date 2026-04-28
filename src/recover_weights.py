import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
from tqdm import tqdm
from src.dataset import create_dataloaders, get_class_weights
from src.pretrain_backbone import BackboneTrainer
from sklearn.metrics import f1_score

def recover_weights(config_path, backbone_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device setup
    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Recovering weights on device: {device}")

    train_loader, val_loader, _ = create_dataloaders(config)
    class_weights = get_class_weights(os.path.join(config['data']['root_dir'], 'splits', 'train.csv')).to(device)
    
    model = BackboneTrainer(config).to(device)
    
    # Load the 99% backbone
    state = torch.load(backbone_path, map_location=device)
    model.backbone.load_state_dict(state['backbone_state_dict'])
    print("Backbone loaded. Freezing backbone to train head...")
    
    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
        
    optimizer = optim.AdamW(model.head.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    best_f1 = 0.0
    for epoch in range(1, 11): # 10 quick epochs for the head
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Head Training")
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # Eval
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                logits = model(images)
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        print(f"Epoch {epoch} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            # Save FULL MODEL
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_f1': val_f1
            }, 'Verified_SOTA_HEp2_Model.pth')
            print(f"Saved new best full model with F1: {val_f1:.4f}")
            if val_f1 > 0.99:
                print("Target 0.99 reached. Stopping recovery.")
                break

if __name__ == '__main__':
    recover_weights('configs/config.yaml', 'Final_DenseNet_Backbone.pth')
