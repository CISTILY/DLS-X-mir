import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ResNet50(nn.Module):
    def __init__(self, pretrained=True, embedding_dim=None):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=pretrained)
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])

        in_features = self.resnet50[7][2].bn3.num_features
        self.fc = nn.Linear(in_features, embedding_dim) if embedding_dim else None

    def forward(self, x):
        x = self.resnet50(x)
        x = torch.flatten(x, 1)
        if self.fc:
            x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x


class DenseNet121(nn.Module):
    def __init__(self, pretrained=True, embedding_dim=None):
        super(DenseNet121, self).__init__()
        self.densenet121 = models.densenet121(pretrained=pretrained)
        self.densenet121 = nn.Sequential(*list(self.densenet121.children())[:-1])

        self.densenet121[0].add_module('relu', nn.ReLU(inplace=True))
        self.densenet121.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))

        in_features = self.densenet121[0].norm5.num_features
        self.fc = nn.Linear(in_features, embedding_dim) if embedding_dim else None

    def forward(self, x):
        x = self.densenet121(x)
        x = torch.flatten(x, 1)
        if self.fc:
            x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x

class ViT(nn.Module):
    def __init__(self, pretrained=True, embedding_dim=None, model_name="vit_b_16"):
        super(ViT, self).__init__()
        # load pretrained ViT
        self.vit = getattr(models, model_name)(weights="IMAGENET1K_V1" if pretrained else None)

        # remove classifier head
        in_features = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity()

        # optional projection
        self.fc = nn.Linear(in_features, embedding_dim) if embedding_dim else None

    def forward(self, x):
        x = self.vit(x)  # vit already flattens output [B, D]
        if self.fc:
            x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x
