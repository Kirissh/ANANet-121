import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import os
from tqdm import tqdm
from src.dataset import create_dataloaders, get_class_weights
from src.models.densenet_backbone import DenseNetBackbone
from src.utils import set_seed, get_logger, save_checkpoint, EarlyStopping, MetricTracker
from sklearn.metrics import f1_score

class BackboneTrainer(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config['model']
        self.backbone = DenseNetBackbone(
            variant=model_cfg['densenet_variant'],
            pretrained=model_cfg['densenet_pretrained'],
            out_channels=model_cfg['densenet_out_channels']
        )
        # Simple Global Average Pooling + Head for pre-training
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(model_cfg['densenet_out_channels'], config['data']['num_classes'])

    def forward(self, x):
        features = self.backbone(x)
        pooled = self.pool(features).flatten(1)
        return self.head(pooled)

def train_backbone(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['data']['seed'])
    checkpoint_dir = 'checkpoints/pretrain'
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = get_logger('pretrain', os.path.join(checkpoint_dir, 'pretrain.log'))
    
    # Device setup
    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Pre-training on device: {device}")

    train_loader, val_loader, _ = create_dataloaders(config)
    class_weights = get_class_weights(os.path.join(config['data']['root_dir'], 'splits', 'train.csv')).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    model = BackboneTrainer(config).to(device)
    num_epochs = config['training']['epochs']
    optimizer = optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']), weight_decay=float(config['training']['weight_decay']))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))
    
    best_f1 = 0.0
    for epoch in range(1, num_epochs + 1): # respect config epochs
        model.train()
        metrics = MetricTracker()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Pretrain")
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            acc = (logits.argmax(1) == labels).float().mean().item()
            metrics.update({'loss': loss.item(), 'acc': acc}, n=images.size(0))
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{acc:.4f}"})
            
        # Validation
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
        logger.info(f"Epoch {epoch} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({'backbone_state_dict': model.backbone.state_dict()}, os.path.join(checkpoint_dir, 'backbone_best.pth'))
            logger.info("Saved best backbone weights")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    train_backbone(args.config)
