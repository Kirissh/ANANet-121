import torch
import torch.nn as nn
import timm

class DenseNetBackbone(nn.Module):
    def __init__(self, variant='densenet121', pretrained=True, out_channels=1024):
        super().__init__()
        # By setting num_classes=0 and global_pool='', timm Models return spatial feature maps
        self.encoder = timm.create_model(variant, pretrained=pretrained, num_classes=0, global_pool='')
        self.out_channels = out_channels

    def forward(self, x):
        return self.encoder(x)

    def freeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, unfreeze_from_layer=None):
        if unfreeze_from_layer is None:
            for param in self.parameters():
                param.requires_grad = True
        else:
            unfreeze = False
            for name, param in self.named_parameters():
                if unfreeze_from_layer in name:
                    unfreeze = True
                if unfreeze:
                    param.requires_grad = True
