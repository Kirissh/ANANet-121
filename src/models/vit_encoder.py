import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.guided_attention import GuidedTransformerBlock

class ViTEncoder(nn.Module):
    def __init__(self, in_channels, spatial_size, dim, depth, heads, mlp_ratio, dropout, attn_dropout, gam_alpha=10.0, gam_epsilon=1e-6):
        super().__init__()
        self.spatial_size = spatial_size
        self.patch_proj = nn.Conv2d(in_channels, dim, kernel_size=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        
        num_patches = spatial_size ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self._init_pos_embed()
        
        self.blocks = nn.ModuleList([
            GuidedTransformerBlock(dim, heads, mlp_ratio, dropout, attn_dropout, gam_alpha, gam_epsilon)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def _init_pos_embed(self):
        import math
        pe = torch.zeros(self.spatial_size, self.spatial_size, self.pos_embed.size(-1))
        dim = self.pos_embed.size(-1) // 2
        
        # 2D Positional Encoding
        for row in range(self.spatial_size):
            for col in range(self.spatial_size):
                for i in range(dim):
                    pe[row, col, 2*i] = math.sin(row / (10000 ** (2 * i / dim)))
                    if 2*i+1 < pe.size(-1):
                        pe[row, col, 2*i+1] = math.cos(row / (10000 ** (2 * i / dim)))
                        
        pe = pe.view(-1, self.pos_embed.size(-1))
        with torch.no_grad():
            self.pos_embed[:, 1:, :] = pe.unsqueeze(0)

    def forward(self, feature_map, mask):
        B, C, H, W = feature_map.shape
        x = self.patch_proj(feature_map) # (B, dim, H, W)
        x = x.flatten(2).transpose(1, 2) # (B, H*W, dim)
        
        # Resize mask to token grid size (B, 1, H, W)
        mask_resized = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)
        mask_weights = mask_resized.flatten(2).transpose(1, 2) # (B, H*W, 1)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, 1+H*W, dim)
        
        x = x + self.pos_embed

        all_attn_weights = []
        for blk in self.blocks:
            x, attn = blk(x, mask_weights)
            all_attn_weights.append(attn)

        x = self.norm(x)
        cls_output = x[:, 0]
        
        return cls_output, all_attn_weights, mask_weights
