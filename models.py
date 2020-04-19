import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as M
from senet import *

class SFT_torch(nn.Module):
    def __init__(self, sigma=0.1, *args, **kwargs):
        super(SFT_torch, self).__init__(*args, **kwargs)
        self.sigma = sigma

    def forward(self, emb_org):
        emb_org_norm = torch.norm(emb_org, 2, 1, True).clamp(min=1e-12)
        emb_org_norm = torch.div(emb_org, emb_org_norm)
        W = torch.mm(emb_org_norm, emb_org_norm.t())
        W = torch.div(W, self.sigma)
        T = F.softmax(W, 1)
        emb_sft = torch.mm(T, emb_org)
        return emb_sft

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])

class MaxPool(nn.Module):
    def forward(self, x):
        avgpool_x = F.avg_pool2d(x, x.shape[2:])
        maxpool_x = F.avg_pool2d(x, x.shape[2:])

        x = torch.cat([avgpool_x, maxpool_x], dim=1)
        return x

# class res34(nn.Module):
#     def __init__(self, num_classes=50, dropout=True):
#         super().__init__()
#         self.net = M.resnet34(pretrained=True)
#         self.base = nn.Sequential(self.net.conv1,
#                                   self.net.bn1,
#                                   self.net.relu,
#                                   self.net.maxpool,
#                                   self.net.layer1,
#                                   self.net.layer2,
#                                   self.net.layer3,
#                                   self.net.layer4)
#         self.apool = nn.AdaptiveAvgPool2d(1)
#         self.sft = SFT_torch()
#         self.fc = nn.Sequential(nn.Dropout(),
#                                 nn.Linear(512, 50))

#     def forward(self, x):
#         x = self.base(x)
#         x = self.apool(x).view(x.size(0), -1)
#         x = self.sft(x)
#         # x = torch.cat([max_x, avg_x], dim=1)
#         logit = self.fc(x)

#         return logit

class res34(nn.Module):
    def __init__(self, num_classes=5, dropout=True):
        super().__init__()
        self.net = M.resnet34(pretrained=True)
        self.net.avgpool = AvgPool()
        if dropout:
            self.net.fc = nn.Sequential(
                # nn.BatchNorm1d(512),
                #nn.BatchNorm1d(512).bias.requires_grad_(False).apply(weights_init_kaiming), 
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes),
            )
        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        return self.net(x)

# class res34_pool(nn.Module):
#     def __init__(self, num_classes=50, dropout=True):
#         super().__init__()
#         self.net = M.resnet34(pretrained=True)

#         self.base = nn.Sequential(
#                 self.net.conv1,
#                 self.net.bn1,
#                 self.net.relu,
#                 self.net.maxpool,
#                 self.net.layer1,
#                 self.net.layer2,
#                 self.net.layer3,
#                 self.net.layer4
#             )
#         self.bottom_up_att = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0,
#                                bias=False)
#         self.top_down_att = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0,
#                                bias=False)

#         self.gap = nn.AdaptiveAvgPool2d(1)

#     def forward(self, x):
#         bs = x.shape[0]
#         x = self.base(x)
#         x_1 = self.bottom_up_att(x)
#         x_2 = self.top_down_att(x)
#         x = x_1 * x_2
#         x = self.gap(x).view(-1, 50)
#         return x


class res50(nn.Module):
    def __init__(self, num_classes=5, dropout=True):
        super().__init__()
        self.net = M.resnet50(pretrained=True)
        self.net.avgpool = AvgPool()
        if dropout:
            self.net.fc = nn.Sequential(
                #nn.BatchNorm1d(2048),
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes),
            )
        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        return self.net(x)

class se50(nn.Module):
    def __init__(self, num_classes=5, dropout=True):
        super().__init__()
        # self.net = create_net(net_cls, pretrained=pretrained)
        self.net = se_resnext50_32x4d()
        # self.net = M.resnet50(pretrained=True)
        self.net.avg_pool = AvgPool()
        if dropout:
            self.net.last_linear = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.last_linear.in_features, num_classes),
            )
        else:
            self.net.last_linear = nn.Linear(self.net.last_linear.in_features, num_classes)

    def fresh_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        return self.net(x)

class se101(nn.Module):
    def __init__(self, num_classes=5, dropout=True):
        super().__init__()
        # self.net = create_net(net_cls, pretrained=pretrained)
        self.net = se_resnext101_32x4d()
        # self.net = M.resnet50(pretrained=True)
        self.net.avg_pool = AvgPool()
        if dropout:
            self.net.last_linear = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.last_linear.in_features, num_classes),
            )
        else:
            self.net.last_linear = nn.Linear(self.net.last_linear.in_features, num_classes)

    def fresh_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        return self.net(x)

class se154(nn.Module):
    def __init__(self, num_classes=5, dropout=True):
        super().__init__()
        # self.net = create_net(net_cls, pretrained=pretrained)
        self.net = senet154()
        # self.net = M.resnet50(pretrained=True)
        self.net.avg_pool = AvgPool()
        self.net.dropout = nn.Dropout()
        if dropout:
            self.net.last_linear = nn.Sequential(
                nn.Linear(self.net.last_linear.in_features, num_classes),
            )
        else:
            self.net.last_linear = nn.Linear(self.net.last_linear.in_features, num_classes)

    def fresh_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        return self.net(x)

# class se154(nn.Module):
#     def __init__(self, num_classes=50, dropout=True):
#         super().__init__()
#         self.net = senet154()
#         self.base = nn.Sequential(
#                 self.net.layer0,
#                 self.net.layer1,
#                 self.net.layer2,
#                 self.net.layer3,
#                 self.net.layer4,
#             )
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#                 nn.Dropout(),
#                 nn.Linear(self.net.last_linear.in_features, num_classes),
#             )
#     def forward(self, x):
#         x = self.base(x)
#         x = self.gap(x).view(x.shape[0], -1)
#         x = self.fc(x)
#         return x
