import torch
import torch.nn as nn
import torch.nn.functional as F

class GuidedMultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_dropout=0.0, alpha=10.0, epsilon=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = dim // num_heads
        self.alpha = alpha
        self.epsilon = epsilon

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_dropout)

    def forward(self, x, mask_weights):
        B, N_plus_1, D = x.shape
        N = mask_weights.shape[1]

        q = self.q(x).reshape(B, N_plus_1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N_plus_1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N_plus_1, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)

        # Prepend a ones column for CLS token: (B, N+1, 1)
        cls_mask = torch.ones((B, 1, 1), device=mask_weights.device)
        full_mask = torch.cat([cls_mask, mask_weights], dim=1) # (B, N+1, 1)

        mask_bias = self.alpha * torch.log(full_mask + self.epsilon)
        mask_bias = mask_bias.unsqueeze(1).transpose(2, 3) # (B, 1, 1, N+1)
        
        scores = scores + mask_bias
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        output = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, N_plus_1, D)
        output = self.out(output)
        return output, attn

class GuidedTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, attn_dropout=0.0, alpha=10.0, epsilon=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GuidedMultiHeadAttention(dim, num_heads, attn_dropout, alpha, epsilon)
        self.norm2 = nn.LayerNorm(dim)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask_weights):
        attn_out, attn_weights = self.attn(self.norm1(x), mask_weights)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights

class MaskBackgroundLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attn_weights, mask_weights):
        # attn_weights: (B, heads, N+1, N+1)
        # mask_weights: (B, N, 1) weights where 0 = background
        
        # average over heads:
        attn = attn_weights.mean(dim=1) # (B, N+1, N+1)
        
        # we want to penalize attention paid to background tokens (mask_weights == 0)
        # background_mask: 1 if background, 0 if cell
        # soften the background check
        bg_mask = 1.0 - mask_weights.squeeze(-1) # (B, N)
        
        # CLS attention to other tokens:
        cls_attn = attn[:, 0, 1:] # (B, N)
        
        loss = (cls_attn * bg_mask).mean()
        return loss
