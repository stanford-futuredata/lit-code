'''
Resnet Implementation Heavily Inspired by the torchvision resnet implementation
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn import init

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    """An implementation of a basic residual block
       Args:
           inplanes (int): input channels
           planes (int): output channels
           stride (int): filter stride (default is 1)
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='cifar10'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes,planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.out_size = planes

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'cifar10':
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            else:
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, option='imagenet'):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.out_size = planes * 4

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, dataset="cifar10", num_classes= 10, in_planes = None):
        super(ResNet, self).__init__()
        self.dataset = dataset
        if in_planes:
            self.in_planes = in_planes
        elif "cifar" in dataset:
            self.in_planes = 16
        else:
            self.in_planes = 64

        if dataset == "cifar10":
            num_classes = 10
        elif dataset == "imagenet":
            num_classes = 1000

        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        ip = self.in_planes
        self.layer1 = self._make_layer(block, ip, layers[0], stride=1)
        self.layer2 = self._make_layer(block, ip * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, ip * 4, layers[2], stride=2)

        if ("cifar" in dataset) or ("svhn" in dataset):
            self.linear = nn.Linear(ip * 4 * block.expansion, num_classes)
            self.conv1 = nn.Conv2d(3, ip, kernel_size=3, stride=1, padding=1, bias=False)
            self.layer4 = None
        else:
            self.linear = nn.Linear(ip * 8 * block.expansion, num_classes)
            self.conv1 = nn.Conv2d(3, ip, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer4 = self._make_layer(block, ip * 8, layers[3], stride=2)
            self.avgpool = nn.AvgPool2d(7, stride=1)


        #Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i in range(len(strides)):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride, self.dataset))
            if i == 0: self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)


    def forward(self, x, get_features = False):
        if get_features: features = OrderedDict()

        out = self.relu(self.bn1(self.conv1(x)))
        if self.layer4: out = self.maxpool(out)
        if get_features:
            features[0] = out.detach()

        out = self.layer1(out)
        if get_features:
            features[1] = out.detach()

        out = self.layer2(out)
        if get_features:
            features[2] = out.detach()

        out = self.layer3(out)
        if get_features:
            features[3] = out.detach()

        if self.layer4:
            out = self.layer4(out)
            if get_features:
                features[4] = out.detach()
            out = self.avgpool(out)
        else:
            out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        # Fully connected layer to get to the class
        out = self.linear(out)
        if get_features:
            return features, out.detach()
        return out

class CifarModel():
    @staticmethod
    def resnet20(**kwargs):
        return ResNet(BasicBlock, [3, 3, 3], **kwargs)
    @staticmethod
    def resnet32(**kwargs):
        return ResNet(BasicBlock, [5, 5, 5], **kwargs)
    @staticmethod
    def resnet44(**kwargs):
        return ResNet(BasicBlock, [7, 7, 7], **kwargs)
    @staticmethod
    def resnet56(**kwargs):
        return ResNet(BasicBlock, [9, 9, 9], **kwargs)
    @staticmethod
    def resnet110(**kwargs):
        return ResNet(BasicBlock, [18, 18, 18], **kwargs)
    @staticmethod
    def resnet1202(**kwargs):
        return ResNet(BasicBlock, [200, 200, 200], **kwargs)

resnet_models = {
    "cifar": {
        "resnet20": CifarModel.resnet20,
        "resnet32": CifarModel.resnet32,
        "resnet44": CifarModel.resnet44,
        "resnet56": CifarModel.resnet56,
        "resnet110": CifarModel.resnet110,
        "resnet1202": CifarModel.resnet1202
    },
}


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))
