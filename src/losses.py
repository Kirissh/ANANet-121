import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[labels]
            loss = alpha_t * loss
            
        return loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None, label_smoothing=0.0, mask_loss_weight=0.2, use_focal=False):
        super().__init__()
        self.mask_loss_weight = mask_loss_weight
        if use_focal:
            self.cls_criterion = FocalLoss(alpha=class_weights)
        else:
            self.cls_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    def forward(self, logits, labels, attn_weights, mask_tokens):
        # The cross entropy
        l_cls = self.cls_criterion(logits, labels)
        
        # Mask Background Loss
        # Extract mean attention across heads from the last ViT layer
        # attn_weights is usually a list of attns per layer. We take the last one.
        if isinstance(attn_weights, list):
            attn = attn_weights[-1]
        else:
            attn = attn_weights
            
        # attn: (B, heads, N+1, N+1)
        # mask_tokens: (B, N, 1)
        mean_attn = attn.mean(dim=1) # (B, N+1, N+1)
        cls_attn = mean_attn[:, 0, 1:] # (B, N)
        bg_mask = 1.0 - mask_tokens.squeeze(-1) # 1 where background
        
        l_mask = (cls_attn * bg_mask).mean()
        
        loss = l_cls + self.mask_loss_weight * l_mask
        
        return {
            'total': loss,
            'classification': l_cls,
            'mask_attention': l_mask
        }
