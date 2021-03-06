import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50


class Net(nn.Module):
    def __init__(self, n_classes=20, in_channels=3, pretrained=True):
        super(Net, self).__init__()

        self.n_classes = n_classes
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.resnet50 = resnet50.resnet50(pretrained=pretrained, in_channels=in_channels, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.classifier = nn.Conv2d(2048, self.n_classes, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    def forward(self, x):

        x = self.stage1(x)
        if self.pretrained:
            x = self.stage2(x).detach()
        else:  # Train from scratch for L8Biome
            x = self.stage2(x)

        x = self.stage3(x)
        x = self.stage4(x)

        x = torchutils.gap2d(x, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, self.n_classes)

        return x

    def train(self, mode=True):
        if self.pretrained and self.in_channels == 3:  # Don't retrain stem if pretrained and RGB inputs
            for p in self.resnet50.conv1.parameters():
                p.requires_grad = False
            for p in self.resnet50.bn1.parameters():
                p.requires_grad = False

    def trainable_parameters(self):
        return list(self.backbone.parameters()), list(self.newly_added.parameters())


class CAM(Net):

    def __init__(self, **kwargs):
        super(CAM, self).__init__(**kwargs)

    def forward(self, x):

        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)

        x = F.conv2d(x, self.classifier.weight)
        x = F.relu(x)

        x = x[0] + x[1].flip(-1)

        return x
