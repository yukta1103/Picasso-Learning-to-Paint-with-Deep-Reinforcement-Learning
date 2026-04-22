# SAC actor — Gaussian policy with tanh squashing

import torch
import torch.nn as nn
import torch.nn.functional as F
from DRL.actor import BasicBlock, cfg

LOG_STD_MAX =  2
LOG_STD_MIN = -5
EPS         =  1e-6


class ActorSAC(nn.Module):
    """
    Stochastic actor for SAC.

    Backbone : ResNet-18 with GroupNorm 
    Heads    : mean + log_std (both state-dependent for SAC)
    Output   : tanh-squashed action in [0,1]^65, corrected log_prob
    """

    def __init__(self, num_inputs=9, depth=18, action_dim=65):
        super().__init__()
        self.action_dim = action_dim

        block, num_blocks = cfg(depth)
        self.in_planes = 64

        self.conv1  = nn.Conv2d(num_inputs, 64, 3, stride=2, padding=1, bias=False)
        self.gn1    = nn.GroupNorm(_gn_groups(64), 64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        feat_dim = 512 * block.expansion

        self.mean_head    = nn.Linear(feat_dim, action_dim)
        self.log_std_head = nn.Linear(feat_dim, action_dim)

        self._init_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(_GNBasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
        nn.init.orthogonal_(self.mean_head.weight,    gain=0.01)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias,    0)
        nn.init.constant_(self.log_std_head.bias, 0)

    def _backbone(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        return x.view(x.size(0), -1)

    def forward(self, x):
        """
        Returns (action, log_prob) with tanh squashing + rescale to [0,1].
        Used during training (reparameterization trick via rsample).
        """
        feat    = self._backbone(x)
        mean    = self.mean_head(feat)
        log_std = self.log_std_head(feat).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std     = log_std.exp()

        # reparameterization sample
        dist    = torch.distributions.Normal(mean, std)
        x_t     = dist.rsample()                        

        # tanh squash to (-1, 1) then shift/scale to (0, 1)
        y_t     = torch.tanh(x_t)
        action  = (y_t + 1.0) / 2.0                     

        # log prob with tanh correction
        # log π(a|s) = log N(x_t) - sum log(1 - tanh^2(x_t)) - 65*log(2)
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1.0 - y_t.pow(2) + EPS)
        log_prob  = log_prob.sum(-1)                    

        return action, log_prob

    @torch.no_grad()
    def act(self, x):
        """Deterministic action for evaluation (use mean, no sampling)."""
        feat   = self._backbone(x)
        mean   = self.mean_head(feat)
        action = (torch.tanh(mean) + 1.0) / 2.0
        return action


class _GNBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3,
                               stride=stride, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(_gn_groups(planes), planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(_gn_groups(planes), planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.GroupNorm(_gn_groups(planes), planes),
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


def _gn_groups(channels):
    for g in range(min(32, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1