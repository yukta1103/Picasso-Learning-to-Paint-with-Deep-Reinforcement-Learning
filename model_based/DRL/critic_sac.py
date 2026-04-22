# SAC twin Q-network critics
# weightNorm + TReLU instead of GroupNorm + ReLU

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm


class TReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0)

    def forward(self, x):
        return F.relu(x - self.alpha) + self.alpha


def conv3x3(in_planes, out_planes, stride=1):
    return weightNorm(nn.Conv2d(in_planes, out_planes, kernel_size=3,
                                stride=stride, padding=1, bias=True))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                weightNorm(nn.Conv2d(in_planes, self.expansion * planes,
                                    kernel_size=1, stride=stride, bias=True)),
            )
        self.relu_1 = TReLU()
        self.relu_2 = TReLU()

    def forward(self, x):
        out = self.relu_1(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        return self.relu_2(out)


class QNetwork(nn.Module):
    """
    Single Q(s,a) network.
    Matches ResNet_wobn from critic.py exactly.
    Input: (B, 12, 128, 128) — [canvas0(3), canvas1(3), gt(3), stepnum(1), coord(2)]
    Output: (B, 1)
    """

    def __init__(self, num_inputs=12, depth=18):
        super().__init__()
        from DRL.actor import cfg
        block, num_blocks = cfg(depth)
        self.in_planes = 64

        self.conv1  = conv3x3(num_inputs, 64, stride=2)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc     = nn.Linear(512 * block.expansion, 1)
        self.relu_1 = TReLU()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, canvas0, canvas1, gt, stepnum, coord):
        x = torch.cat([canvas0, canvas1, gt, stepnum, coord], dim=1)
        x = self.relu_1(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        return self.fc(x)  # (B, 1)


class TwinCritic(nn.Module):
    def __init__(self, num_inputs=12, depth=18):
        super().__init__()
        self.q1 = QNetwork(num_inputs, depth)
        self.q2 = QNetwork(num_inputs, depth)

    def forward(self, canvas0, canvas1, gt, stepnum, coord):
        return (self.q1(canvas0, canvas1, gt, stepnum, coord),
                self.q2(canvas0, canvas1, gt, stepnum, coord))

    def q1_only(self, canvas0, canvas1, gt, stepnum, coord):
        return self.q1(canvas0, canvas1, gt, stepnum, coord)