import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional layers
        self.feature_extractor = nn.Sequential(
            # TODO
        )

        self.avg_pooling = nn.AdaptiveAvgPool2d((7, 7))

        # Linear layers
        self.classifier = nn.Sequential(
            # TODO
        )
        
    def forward(self, x):
        # TODO
        raise NotImplementedError