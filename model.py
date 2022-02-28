import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Net(nn.Module):
    def __init__(self, angRes):
        super(Net, self).__init__()
        self.num_cascade = 2
        mindisp = -4
        maxdisp = 4
        self.angRes = angRes
        self.maxdisp = maxdisp
        self.init_feature = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(16),
            ResB(16), ResB(16), ResB(16), ResB(16),
            ResB(16), ResB(16), ResB(16), ResB(16),
            nn.Conv3d(16, 16, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(16, 8, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.lastconv = nn.Conv3d(8, 8, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False)
        self.build_cost = BuildCost(8, 512, angRes, mindisp, maxdisp)
        self.aggregate = Aggregate(512, 160, mindisp, maxdisp)

    def forward(self, x, dispGT=None):
        lf = rearrange(x, 'b c (a1 h) (a2 w) -> b c a1 a2 h w', a1=self.angRes, a2=self.angRes)
        x = rearrange(x, 'b c (a1 h) (a2 w) -> b c (a1 a2) h w', a1=self.angRes, a2=self.angRes)
        b, c, _, h, w = x.shape
        feat = self.init_feature(x)
        feat = self.lastconv(feat)
        if dispGT is not None:
            mask = Generate_mask(lf, dispGT)
            cost = self.build_cost(feat, mask)
            disp = self.aggregate(cost)
        else:
            mask = torch.ones(1, self.angRes ** 2, h, w).to(x.device)
            cost = self.build_cost(feat, mask)
            disp = self.aggregate(cost)
            mask = Generate_mask(lf, disp)
            cost = self.build_cost(feat, mask)
            disp = self.aggregate(cost)

        return disp


class BuildCost(nn.Module):
    def __init__(self, channel_in, channel_out, angRes, mindisp, maxdisp):
        super(BuildCost, self).__init__()
        self.oacc = ModulateConv2d(channel_in, channel_out, kernel_size=angRes, stride=1, bias=False)
        self.angRes = angRes
        self.mindisp = mindisp
        self.maxdisp = maxdisp
        self.channel_att = channel_out
        self.channel_in = channel_in

    def forward(self, x, mask):
        b, c, aa, h, w = x.shape
        x = rearrange(x, 'b c (a1 a2) h w -> (b a1 a2) c h w', a1=self.angRes, a2=self.angRes)
        bdr = (self.angRes // 2) * self.maxdisp
        pad = nn.ZeroPad2d((bdr, bdr, bdr, bdr))
        x_pad = pad(x)
        x_pad = rearrange(x_pad, '(b a1 a2) c h w -> b c (a1 h) (a2 w)', a1=self.angRes, a2=self.angRes)
        h_pad, w_pad = h + 2 * bdr, w + 2 * bdr
        mask_avg = torch.mean(mask, dim=1)
        cost = []
        for d in range(self.mindisp, self.maxdisp + 1):
            dila = [h_pad - d, w_pad - d]
            self.oacc.dilation = dila
            crop = (self.angRes // 2) * (d - self.mindisp)
            if d == self.mindisp:
                feat = x_pad
            else:
                feat = x_pad[:, :, crop: -crop, crop: -crop]
            current_cost = self.oacc(feat, mask)
            cost.append(current_cost / mask_avg.unsqueeze(1).repeat(1, current_cost.shape[1], 1, 1))
        cost = torch.stack(cost, dim=2)

        return cost


class Aggregate(nn.Module):
    def __init__(self, inC, channel, mindisp, maxdisp):
        super(Aggregate, self).__init__()
        self.sq = nn.Sequential(
            nn.Conv3d(inC, channel, 1, 1, 0, bias=False), nn.BatchNorm3d(channel), nn.LeakyReLU(0.1, inplace=True))
        self.Conv1 = nn.Sequential(
            nn.Conv3d(channel, channel, 3, 1, 1, bias=False), nn.BatchNorm3d(channel), nn.LeakyReLU(0.1, inplace=True))
        self.Conv2 = nn.Sequential(
            nn.Conv3d(channel, channel, 3, 1, 1, bias=False), nn.BatchNorm3d(channel), nn.LeakyReLU(0.1, inplace=True))
        self.Resb1 = ResB3D(channel)
        self.Resb2 = ResB3D(channel)
        self.Conv3 = nn.Sequential(
            nn.Conv3d(channel, channel, 3, 1, 1, bias=False), nn.BatchNorm3d(channel), nn.LeakyReLU(0.1, inplace=True))
        self.Conv4 = nn.Conv3d(channel, 1, 3, 1, 1, bias=False)
        self.softmax = nn.Softmax(1)
        self.mindisp = mindisp
        self.maxdisp = maxdisp

    def forward(self, psv):
        buffer = self.sq(psv)
        buffer = self.Conv1(buffer)
        buffer = self.Conv2(buffer)
        buffer = self.Resb1(buffer)
        buffer = self.Resb2(buffer)
        buffer = self.Conv3(buffer)
        score = self.Conv4(buffer)
        attmap = self.softmax(score.squeeze(1))
        temp = torch.zeros(attmap.shape).to(attmap.device)
        for d in range(self.maxdisp - self.mindisp + 1):
            temp[:, d, :, :] = attmap[:, d, :, :] * (self.mindisp + d)
        disp = torch.sum(temp, dim=1, keepdim=True)
        return disp


class ResB(nn.Module):
    def __init__(self, feaC):
        super(ResB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(feaC, feaC, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.BatchNorm3d(feaC),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(feaC, feaC, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.BatchNorm3d(feaC))

    def forward(self, x):
        out = self.conv(x)
        return x + out


class ResB3D(nn.Module):
    def __init__(self, channels):
        super(ResB3D, self).__init__()
        self.body = nn.Sequential(
            nn.Conv3d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(channels))
        self.calayer = CALayer(channels, 9)

    def forward(self, x):
        buffer = self.body(x)
        return self.calayer(buffer) + x

class CALayer(nn.Module):
    def __init__(self, channel, num_views):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((num_views, 1, 1))
        self.conv_du = nn.Sequential(
                nn.Conv3d(channel, channel // 16, 1, 1, 0, bias=True),
                nn.BatchNorm3d(channel // 16),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv3d(channel // 16, channel, 1, 1, 0, bias=True),
                nn.BatchNorm3d(channel),
                nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


def Generate_mask(lf, disp):
    b, c, angRes, _, h, w = lf.shape
    x_base = torch.linspace(0, 1, w).repeat(1, h, 1).to(lf.device)
    y_base = torch.linspace(0, 1, h).repeat(1, w, 1).transpose(1, 2).to(lf.device)
    center = (angRes - 1) // 2
    img_ref = lf[:, :, center, center, :, :]
    img_res = []
    for u in range(angRes):
        for v in range(angRes):
            img = lf[:, :, u, v, :, :]
            if (u == center) & (v == center):
                img_warped = img
            else:
                du, dv = u - center, v - center
                img_warped = warp(img, -disp, du, dv, x_base, y_base)
            img_res.append(abs((img_warped - img_ref)))
    mask = torch.cat(img_res, dim=1)
    out = (1 - mask) ** 2
    return out


def warp(img, disp, du, dv, x_base, y_base):

    b, _, h, w = img.size()
    x_shifts = dv * disp[:, 0, :, :] / w
    y_shifts = du * disp[:, 0, :, :] / h
    flow_field = torch.stack((x_base + x_shifts, y_base + y_shifts), dim=3)
    img_warped = F.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros')
    return img_warped


class ModulateConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, dilation=1, bias=False):
        super(ModulateConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.flatten = nn.Unfold(kernel_size=1, stride=1, dilation=1, padding=0)
        self.fuse = nn.Conv2d(channel_in * kernel_size * kernel_size, channel_out,
                              kernel_size=1, stride=1, padding=0, bias=bias, groups=channel_in)

    def forward(self, x, mask):
        mask_flatten = self.flatten(mask)
        Unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation)
        x_unfold = Unfold(x)
        x_unfold_modulated = x_unfold * mask_flatten.repeat(1, x.shape[1], 1)
        Fold = nn.Fold(output_size=(mask.shape[2], mask.shape[3]), kernel_size=1, stride=1)
        x_modulated = Fold(x_unfold_modulated)
        out = self.fuse(x_modulated)
        return out


if __name__ == "__main__":
    angRes = 9
    net = Net(angRes).cuda()
    from thop import profile
    input = torch.randn(1, 1, 32 * angRes, 32 * angRes).cuda()
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (params / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))