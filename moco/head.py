import pdb
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1) 


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4 

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class WiderBottleneck(nn.Module):
    expansion = 8

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(WiderBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 8, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 8)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DilateBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(DilateBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               bias=False, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class Head(nn.Module):
    def __init__(self, arch_type, inplanes):
        super(Head, self).__init__()
        self.inplanes = inplanes
        head = []
        if arch_type == 'add_stage4':
            layer = self._make_layer(Bottleneck, 512, 3, stride=2) # layer4 in res50
            head.append(layer)
            head.append(nn.AvgPool2d(7, stride=1))
            head.append(Flatten())
            head.append(nn.Linear(2048, 2048))
            head.append(nn.ReLU())
            head.append(nn.Linear(2048, 128))


        if arch_type == 'add_stage3_lessconv':
            layer = self._make_layer(Bottleneck, 256, 2, stride=2) # layer3 in res50 with less conv
            head.append(layer)
            head.append(nn.AdaptiveAvgPool2d((1,1)))
            head.append(Flatten())
            head.append(nn.Linear(1024, 2048))
            head.append(nn.ReLU())
            head.append(nn.Linear(2048, 128))


        if arch_type == 'add_stage3_1conv':
            layer = self._make_layer(Bottleneck, 256, 1, stride=2) # layer3 in res50 with 1 conv
            head.append(layer)
            head.append(nn.AdaptiveAvgPool2d((1,1)))
            head.append(Flatten())
            head.append(nn.Linear(1024, 2048))
            head.append(nn.ReLU())
            head.append(nn.Linear(2048, 128))




        if arch_type == 'add_stage4_lessconv':
            layer = self._make_layer(Bottleneck, 512, 2, stride=2) # layer4 in res50 with less conv
            head.append(layer)
            head.append(nn.AdaptiveAvgPool2d((1,1)))
            head.append(Flatten())
            head.append(nn.Linear(2048, 2048))
            head.append(nn.ReLU())
            head.append(nn.Linear(2048, 128))



        if arch_type == 'add_stage4_1conv':
            layer = self._make_layer(Bottleneck, 512, 1, stride=2) # layer4 in res50 with less conv
            head.append(layer)
            head.append(nn.AdaptiveAvgPool2d((1,1)))
            head.append(Flatten())
            head.append(nn.Linear(2048, 2048))
            head.append(nn.ReLU())
            head.append(nn.Linear(2048, 128))




        if arch_type == 'add_stage4_wider':
            layer = self._make_layer(WiderBottleneck, 512, 3, stride=2) # wider layer4 in res50, expansion = 8
            head.append(layer)
            head.append(nn.AdaptiveAvgPool2d((1,1)))
            head.append(Flatten())
            head.append(nn.Linear(4096, 4096))
            head.append(nn.ReLU())
            head.append(nn.Linear(4096, 128))



        if arch_type == 'add_stage4_dilate':
            layer = self._make_layer(DilateBottleneck, 512, 3, stride=1)
            head.append(layer)
            head.append(nn.AdaptiveAvgPool2d((1,1)))
            head.append(Flatten())
            head.append(nn.Linear(2048, 2048))
            head.append(nn.ReLU())
            head.append(nn.Linear(2048, 128))

        if arch_type == 'add_stage4_1conv_dilate':
            layer = self._make_layer(DilateBottleneck, 512, 1, stride=1)
            head.append(layer)
            head.append(nn.AdaptiveAvgPool2d((1,1)))
            head.append(Flatten())
            head.append(nn.Linear(2048, 2048))
            head.append(nn.ReLU())
            head.append(nn.Linear(2048, 128))


        if arch_type == 'add_stage4_2conv_dilate':
            layer = self._make_layer(DilateBottleneck, 512, 2, stride=1)
            head.append(layer)
            head.append(nn.AdaptiveAvgPool2d((1,1)))
            head.append(Flatten())
            head.append(nn.Linear(2048, 2048))
            head.append(nn.ReLU())
            head.append(nn.Linear(2048, 128))





        self.head = nn.ModuleList(head)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)



    def forward(self, x):
        for m in self.head:
            x = m(x)
        return x

if __name__ == '__main__':
    head = Head('add_stage4_lessconv', 1024)
    print(head)
    x = torch.randn(8,1024,7,7)
    out = head(x)
    pdb.set_trace()
