import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import os
from tqdm import tqdm
from src.dataset import create_dataloaders, get_class_weights
from src.models.full_model import HEp2AnaNet, get_parameter_groups
from src.utils import set_seed, get_logger, save_checkpoint, EarlyStopping, MetricTracker
from sklearn.metrics import f1_score
try:
    import torch_directml
    HAS_DIRECTML = True
except ImportError:
    HAS_DIRECTML = False

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def build_scheduler(optimizer, config, steps_per_epoch):
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['epochs'] * steps_per_epoch, eta_min=float(config['training']['min_lr'])
    )
    return scheduler

def train_one_epoch(model, loader, optimizer, scaler, scheduler, device, config, epoch, criterion_cls):
    model.train()
    metrics = MetricTracker()
    use_amp = config['training']['use_amp']
    clip_norm = float(config['training']['gradient_clip_norm'])
    mask_lambda = float(config['model']['gam_loss_weight'])
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} Train")
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        if device.type == 'cuda' and use_amp:
            with torch.amp.autocast('cuda', enabled=True):
                outputs = model(images, masks, labels)
                logits = outputs['logits']
                loss_cls = criterion_cls(logits, labels)
                loss_mask = outputs.get('mask_loss', 0.0)
                loss = loss_cls + mask_lambda * loss_mask
        else:
            outputs = model(images, masks, labels)
            logits = outputs['logits']
            loss_cls = criterion_cls(logits, labels)
            loss_mask = outputs.get('mask_loss', 0.0)
            loss = loss_cls + mask_lambda * loss_mask
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        
        metrics.update({
            'loss': loss.item(),
            'cls_loss': loss_cls.item(),
            'mask_loss': loss_mask.item() if isinstance(loss_mask, torch.Tensor) else loss_mask,
            'accuracy': acc
        }, n=images.size(0))
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{acc:.4f}"})
        
    return metrics.compute()

@torch.no_grad()
def validate(model, loader, criterion_cls, device, config):
    model.eval()
    metrics = MetricTracker()
    use_amp = config['training']['use_amp']
    mask_lambda = float(config['model']['gam_loss_weight'])
    
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc="Validation")
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        labels = batch['label'].to(device)
        
        if device.type == 'cuda' and use_amp:
            with torch.amp.autocast('cuda', enabled=True):
                outputs = model(images, masks, labels)
                logits = outputs['logits']
                loss_cls = criterion_cls(logits, labels)
                loss_mask = outputs.get('mask_loss', 0.0)
                loss = loss_cls + mask_lambda * loss_mask
        else:
            outputs = model(images, masks, labels)
            logits = outputs['logits']
            loss_cls = criterion_cls(logits, labels)
            loss_mask = outputs.get('mask_loss', 0.0)
            loss = loss_cls + mask_lambda * loss_mask
            
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        
        metrics.update({
            'loss': loss.item(),
            'cls_loss': loss_cls.item(),
            'mask_loss': loss_mask.item() if isinstance(loss_mask, torch.Tensor) else loss_mask,
            'accuracy': acc
        }, n=images.size(0))
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    comp_metrics = metrics.compute()
    comp_metrics['f1_macro'] = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    comp_metrics['f1_weighted'] = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return comp_metrics

def main(config_path):
    config = load_config(config_path)
    set_seed(config['data']['seed'])
    
    logger = get_logger('train', os.path.join(config['training']['checkpoint_dir'], 'train.log'))
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif HAS_DIRECTML:
        device = torch_directml.device()
    else:
        device = torch.device('cpu')
        
    logger.info(f"Using device: {device}")
    
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    class_weights = get_class_weights(os.path.join(config['data']['root_dir'], 'splits', 'train.csv')).to(device)
    criterion_cls = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=float(config['training']['label_smoothing']))
    
    model = HEp2AnaNet(config).to(device)
    
    # Load Pre-trained Backbone if specified
    if args.pretrained_backbone:
        ptr_checkpoint = torch.load(args.pretrained_backbone, map_location=device)
        model.backbone.load_state_dict(ptr_checkpoint['backbone_state_dict'])
        logger.info(f"Successfully loaded pre-trained backbone from {args.pretrained_backbone}")
        
    model.backbone.freeze_backbone()
    
    param_groups = get_parameter_groups(model, config)
    optimizer = optim.AdamW(param_groups, weight_decay=float(config['training']['weight_decay']))
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch)
    
    scaler_enabled = config['training']['use_amp'] and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=scaler_enabled)
    
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'],
        mode='max'
    )
    
    start_epoch = 1
    epochs = config['training']['epochs']
    freeze_epochs = config['model']['densenet_freeze_epochs']
    
    for epoch in range(start_epoch, epochs + 1):
        if epoch == freeze_epochs + 1:
            logger.info("Unfreezing DenseNet block4")
            model.backbone.unfreeze_backbone('denseblock4')
        elif epoch == freeze_epochs + 10:
            logger.info("Unfreezing DenseNet transition3 + denseblock3")
            model.backbone.unfreeze_backbone('transition3')
            model.backbone.unfreeze_backbone('denseblock3')
        elif epoch == freeze_epochs + 20:
            logger.info("Unfreezing DenseNet transition2 + denseblock2")
            model.backbone.unfreeze_backbone('transition2')
            model.backbone.unfreeze_backbone('denseblock2')
        elif epoch == freeze_epochs + 30:
            logger.info("Unfreezing entire backbone")
            model.backbone.unfreeze_backbone()
            
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, scheduler, device, config, epoch, criterion_cls)
        val_metrics = validate(model, val_loader, criterion_cls, device, config)
        
        logger.info(f"Epoch {epoch}")
        logger.info(f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"Train Acc:  {train_metrics['accuracy']:.4f} | Val Acc:  {val_metrics['accuracy']:.4f}")
        logger.info(f"Val F1_M:   {val_metrics['f1_macro']:.4f} | Val F1_W: {val_metrics['f1_weighted']:.4f}")
        
        is_best = False
        if early_stopping(val_metrics['f1_macro']):
            logger.info(f"Early stopping at epoch {epoch}")
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_f1_macro': val_metrics['f1_macro']
            }, is_best=False, checkpoint_dir=config['training']['checkpoint_dir'], epoch=epoch)
            break
            
        if val_metrics['f1_macro'] == early_stopping.best_metric:
            is_best = True
            
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_f1_macro': val_metrics['f1_macro']
        }, is_best=is_best, checkpoint_dir=config['training']['checkpoint_dir'], epoch=epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--pretrained_backbone', type=str, default=None, help='Path to pre-trained backbone weights')
    args = parser.parse_args()
    main(args.config)
