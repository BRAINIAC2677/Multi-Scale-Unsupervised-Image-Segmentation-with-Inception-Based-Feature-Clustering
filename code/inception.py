import torch
from torch import nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))


class Inception_block(nn.Module):
    def __init__(
        self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
    ):
        super(Inception_block, self).__init__()
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=1),
            conv_block(red_3x3, out_3x3, kernel_size=(3, 3), padding=1),
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=1),
            conv_block(red_5x5, out_5x5, kernel_size=5, padding=2),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_1x1pool, kernel_size=1),
        )

    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1
        )


class InceptionNet(nn.Module):
    def __init__(self,input_dim, intermediate_channels = 15, nChannel = 100, nConv = 2):
        super(InceptionNet, self).__init__()
        self.nConv = nConv
        self.conv1 = nn.Conv2d(input_dim, intermediate_channels, kernel_size=1, stride=1, padding=0 )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)

        # inception_block(in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool)
        self.inception3a = Inception_block(intermediate_channels, 160, 96, 64, 16, 16, 16)
        self.bn_i_1 = nn.BatchNorm2d(256)

        self.inception3b = nn.ModuleList()
        self.bn_i_2 = nn.ModuleList()

        if nConv >= 1:
            self.inception3b.append(Inception_block(256, 96, 32, 16, 16, 8, 8))
            self.bn_i_2.append(nn.BatchNorm2d(128))

            for i in range(nConv-1):
                self.inception3b.append(Inception_block(128, 96, 32, 16, 16, 8, 8))
                self.bn_i_2.append(nn.BatchNorm2d(128))

        r = nChannel

        print('last layer size:', r)
        if nConv>=1:
            self.conv3 = nn.Conv2d(128, r, kernel_size=1, stride=1, padding=0 )
        else:
            self.conv3 = nn.Conv2d(256, r, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(r)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        
        x = self.inception3a(x)
        x = F.relu( x )
        x = self.bn_i_1(x)

        for i in range(self.nConv):
            x = self.inception3b[i](x)
            x = F.relu( x )
            x = self.bn_i_2[i](x)

        x = self.conv3(x)
        x = self.bn3(x)
        return x
