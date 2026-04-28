import torch
import torch.nn as nn
from src.models.densenet_backbone import DenseNetBackbone
from src.models.vit_encoder import ViTEncoder
from src.models.guided_attention import MaskBackgroundLoss

class HEp2AnaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_cfg = config['model']
        
        self.backbone = DenseNetBackbone(
            variant=model_cfg['densenet_variant'],
            pretrained=model_cfg['densenet_pretrained'],
            out_channels=model_cfg['densenet_out_channels']
        )
        
        self.vit = ViTEncoder(
            in_channels=model_cfg['densenet_out_channels'],
            spatial_size=7, # assuming 224x224 input yields 7x7 for DenseNet-121
            dim=model_cfg['vit_dim'],
            depth=model_cfg['vit_depth'],
            heads=model_cfg['vit_heads'],
            mlp_ratio=model_cfg['vit_mlp_ratio'],
            dropout=model_cfg['vit_dropout'],
            attn_dropout=model_cfg['vit_attn_dropout'],
            gam_alpha=model_cfg['gam_alpha'],
            gam_epsilon=float(model_cfg['gam_epsilon'])
        )
        
        vit_dim = model_cfg['vit_dim']
        head_dropout = model_cfg['head_dropout']
        num_classes = config['data']['num_classes']
        
        self.head = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Dropout(head_dropout),
            nn.Linear(vit_dim, vit_dim // 2),
            nn.GELU(),
            nn.Dropout(head_dropout / 2),
            nn.Linear(vit_dim // 2, num_classes)
        )
        
        self.mask_loss_fn = MaskBackgroundLoss()

    def forward(self, image, mask, labels=None):
        feature_map = self.backbone(image)
        cls_out, all_attn_weights, mask_weights = self.vit(feature_map, mask)
        logits = self.head(cls_out)
        
        output = {
            'logits': logits,
            'attn_weights': all_attn_weights,
            'mask_weights': mask_weights
        }
        
        if labels is not None:
            # We will handle primary CE loss in the training loop/losses.py
            # Here we just compute the mask loss
            mask_loss = self.mask_loss_fn(all_attn_weights[-1], mask_weights)
            output['mask_loss'] = mask_loss
            
        return output

def get_parameter_groups(model, config):
    base_lr = float(config['training']['learning_rate'])
    backbone_lr = base_lr * float(config['training']['backbone_lr_multiplier'])
    
    backbone_params = []
    vit_head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            vit_head_params.append(param)
            
    return [
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': vit_head_params, 'lr': base_lr}
    ]
