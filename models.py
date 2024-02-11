import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing


# Linear model.
class LinearModel(nn.Module):
    """Linear model class."""
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = self.fc(x)
        return x


# CNN model. Note we only use this for CIFAR such that the architecture is
# specific. If with MNIST type of data, it may require some minor changes.
# Hardcode some values here for the architecture.
class ConvolutionalNeuralNet(nn.Module):
    """Convolutional neural network model."""
    def __init__(self):
        super(ConvolutionalNeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Multilayer perceptron.
# Hardcode some values here for the architecture.
class MultilayerPerceptron(nn.Module):
    """Multilayer perceptron model."""
    def __init__(self):
        super(MultilayerPerceptron, self).__init__()
        self.linear1 = nn.Linear(784, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return F.log_softmax(x, dim=1)


# Build custom module for logistic regression.
# Hardcode some values here for the model.
class LogisticRegression(torch.nn.Module):
    """Logistic regression model."""
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(784, 10)
    # make predictions
    def forward(self, x):
        x = torch.sigmoid(self.linear(x))
        return x

# DenseNet Model.
# Hardcode some values here for the architecture.
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(growth_rate, growth_rate, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = torch.cat((x, out), 1)
        return out

class DenseNet(nn.Module):
    def __init__(self, growth_rate, num_layers, num_classes):
        super(DenseNet, self).__init__()
        self.initial_conv = nn.Conv2d(3, 2 * growth_rate, kernel_size=3, padding=1)  # Input has 3 channels
        self.features = nn.Sequential()
        
        # Adding DenseBlocks
        in_channels = 2 * growth_rate
        for i in range(num_layers):
            self.features.add_module(f'DenseBlock_{i+1}', DenseBlock(in_channels, growth_rate))
            in_channels += growth_rate
        
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)
        
    def forward(self, x):
        out = self.initial_conv(x)
        out = self.features(out)
        out = self.global_avg_pooling(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()
        self.in_channels = 16

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # Create 5 layers ResNet20
        self.layer1 = self._make_layer(16, 5, stride=1)
        self.layer2 = self._make_layer(32, 5, stride=2)
        self.layer3 = self._make_layer(64, 5, stride=2)
        self.layer4 = self._make_layer(128, 2, stride=2) 
        self.layer5 = self._make_layer(256, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)  # First block stride is 2, others are 1
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# Model types
ModelType = typing.Union[
    LinearModel,
    ConvolutionalNeuralNet,
    MultilayerPerceptron,
    LogisticRegression,
    DenseNet,
    ResNet20
]