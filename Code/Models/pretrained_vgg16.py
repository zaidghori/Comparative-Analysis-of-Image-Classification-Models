import torch
import torch.nn as nn
from torchvision import models


class PretrainedVGG16(nn.Module):
    """VGG16 using pretrained weights from torchvision."""
    
    def __init__(self, num_classes=8, dropout=0.5, freeze_features=True):
        """
        Initialize pretrained VGG16 model.
        
        Args:
            num_classes: Number of output classes
            dropout: Dropout probability for classifier
            freeze_features: Whether to freeze feature extraction layers
        """
        super(PretrainedVGG16, self).__init__()
        
        # Load pretrained VGG16
        vgg16 = models.vgg16(pretrained=True)
        
        # Copy feature layers
        self.features = vgg16.features
        
        # Freeze feature layers if specified
        if freeze_features:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # Replace classifier with custom head
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def unfreeze_features(self):
        """Unfreeze all feature extraction layers."""
        for param in self.features.parameters():
            param.requires_grad = True