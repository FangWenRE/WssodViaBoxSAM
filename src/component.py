import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ASPP_simple(nn.Module):
    """docstring for ASPP_simple, simple means no ReLU """

    def __init__(self, inplanes, planes, rates=[1, 6, 12, 18]):
        super(ASPP_simple, self).__init__()

        self.aspp0 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1,
                                             stride=1, padding=0, dilation=1, bias=False),
                                   nn.BatchNorm2d(planes))
        self.aspp1 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                             stride=1, padding=rates[1], dilation=rates[1], bias=False),
                                   nn.BatchNorm2d(planes))
        self.aspp2 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                             stride=1, padding=rates[2], dilation=rates[2], bias=False),
                                   nn.BatchNorm2d(planes))
        self.aspp3 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                             stride=1, padding=rates[3], dilation=rates[3], bias=False),
                                   nn.BatchNorm2d(planes))

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False))

        self.reduce = nn.Sequential(
            nn.Conv2d(planes * 5, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.global_avg_pool(x)
        x4 = F.interpolate(x4, x3.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x0, x1, x2, x3, x4), dim=1)
        x = self.reduce(x)
        return x


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.__init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.relu(self.bn(x))
        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, planes, rates):
        super(ASPP, self).__init__()

        self.aspp1 = ASPP_module(inplanes, planes, rate=rates[0])
        self.aspp2 = ASPP_module(inplanes, planes, rate=rates[1])
        self.aspp3 = ASPP_module(inplanes, planes, rate=rates[2])
        self.aspp4 = ASPP_module(inplanes, planes, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, planes, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(planes),
                                             nn.ReLU()
                                             )

        self.conv1 = nn.Conv2d(planes * 5, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

    # Channel Attention


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


# Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class AttentionDecoder(nn.Module):
    def __init__(self, in_planes):
        super(AttentionDecoder, self).__init__()
        self.sp_attention = SpatialAttention()
        self.ch_attention = ChannelAttention(in_planes)
        self.conv_1 = nn.Conv2d(in_planes, 256, 1, stride=1, padding=0)
        self.conv_2 = nn.Sequential(nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(256), nn.ReLU())

    def forward(self, x, input=None):
        x_sp = torch.mul(x, self.sp_attention(x))
        x_ch = torch.mul(x_sp, self.ch_attention(x_sp))
        if input != None and x_ch.size()[1] != input.size()[1]:
            x_ch = self.conv_1(x_ch)
            out = self.conv_2(input + x_ch)
            return out
        else:
            return x_ch


class AttentionNoConv(nn.Module):
    def __init__(self, in_planes):
        super(AttentionNoConv, self).__init__()
        self.sp_attention = SpatialAttention()
        self.ch_attention = ChannelAttention(in_planes)

    def forward(self, x, input):
        x_sp = torch.mul(x, self.sp_attention(x))
        x_ch = torch.mul(x_sp, self.ch_attention(x_sp))
        out = input + x_ch
        return out
