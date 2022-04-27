"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import copy
import math

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from core.wing import FAN


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, batch_size, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self.batch_size = batch_size
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        # self.conv1 = nn.Conv1d(dim_in, dim_in, 3, 1, 1)
        # self.conv2 = nn.Conv1d(dim_in, dim_out, 3, 1, 1)
        # print('build res conv1 2 :',dim_in, dim_out)
        self.conv1 = nn.Linear(dim_in, dim_in)
        self.conv2 = nn.Linear(dim_in, dim_out)
        # if self.normalize:
        #     # self.norm1 = nn.InstanceNorm1d(dim_in, affine=True)
        #     # self.norm2 = nn.InstanceNorm1d(dim_in, affine=True)
        #     self.norm1 = nn.InstanceNorm1d(self.batch_size, affine=True)
        #     self.norm2 = nn.InstanceNorm1d(self.batch_size, affine=True)
        if self.learned_sc:
            # self.conv1x1 = nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False)
            self.conv1x1 = nn.Linear(dim_in, dim_out, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        # if self.downsample:
        #     x = F.avg_pool1d(x, 2)
        return x

    def _residual(self, x, is_training=True):
        # if self.normalize and is_training:
        #     x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        # if self.downsample:
        #     x = F.avg_pool1d(x, 2)
        # if self.normalize and is_training:
        #     x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, is_training=True):
        # print('self._shortcut(x).shape', self._shortcut(x).shape)
        # print('self._residual(x).shape', self._residual(x).shape)
        x = self._shortcut(x) + self._residual(x, is_training)
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        # self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        # h = h.view(h.size(0), h.size(1), 1, 1)
        h = h.view(h.size(0), h.size(1))
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        # x = self.norm(x)
        # print('gamma.shape :', gamma.shape)
        # print('beta.shape :', beta.shape)
        # print('x.shape :', x.shape)
        return (1 + gamma) * x + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=256, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=256):
        # self.conv1 = nn.Conv1d(dim_in, dim_out, 3, 1, 1)
        # self.conv2 = nn.Conv1d(dim_out, dim_out, 3, 1, 1)
        self.conv1 = nn.Linear(dim_in, dim_out)
        self.conv2 = nn.Linear(dim_out, dim_out)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            # self.conv1x1 = nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False)
            self.conv1x1 = nn.Linear(dim_in, dim_out, bias=False)

    def _shortcut(self, x):
        # if self.upsample:
        #     x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        # if self.upsample:
        #     x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


# class HighPass(nn.Module):
#     def __init__(self, w_hpf, device):
#         super(HighPass, self).__init__()
#         self.register_buffer('filter',
#                              torch.tensor([[-1, -1, -1],
#                                            [-1, 8., -1],
#                                            [-1, -1, -1]]) / w_hpf)

#     def forward(self, x):
#         filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
#         return F.conv1d(x, filter, padding=1, groups=x.size(1))


class Generator(nn.Module):
    def __init__(self, batch_size, xvec_size=512, style_dim=256, max_conv_dim=512, w_hpf=1):
        super().__init__()
        # dim_in = 2**14 // xvec_size
        dim_in = xvec_size
        self.xvec_size = xvec_size
        # self.from_rgb = nn.Conv1d(1, dim_in, 3, 1, 1)
        # self.from_rgb = nn.Linear(dim_in, dim_in)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        # self.to_rgb = nn.Sequential(
        #     nn.InstanceNorm1d(batch_size, affine=True),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(dim_in, xvec_size))
        self.to_rgb = nn.Sequential(
        nn.LeakyReLU(0.2),
        nn.Linear(dim_in, xvec_size))           
        # self.to_rgb = nn.Sequential(
        #     nn.InstanceNorm1d(dim_in, affine=True),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(dim_in, 512))

        # down/up-sampling blocks
        repeat_num = int(np.log2(xvec_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            # dim_out = min(dim_in*2, max_conv_dim)
            dim_out = dim_in
            self.encode.append(
                ResBlk(dim_in, dim_out, batch_size=batch_size,normalize=True, downsample=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim,
                               w_hpf=w_hpf, upsample=True))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, batch_size=batch_size, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))

        # if w_hpf > 0:
        #     device = torch.device(
        #         'cuda' if torch.cuda.is_available() else 'cpu')
        #     self.hpf = HighPass(w_hpf, device)

    def forward(self, x, s, masks=None, is_training=True):
        # x = self.from_rgb(x)
        # print("Generating")
        cache = {}
        for block in self.encode:
            # print('in encoder')
            # print('x.shape', x.shape)
            # print('s.shape', s.shape)
            # if (masks is not None) and (x.size(2) in [32, 64, 128]):
            if (masks is not None) and (x.size(2) in [512]):
                cache[x.size(2)] = x
            x = block(x, is_training)
        for block in self.decode:
            # print('in decoder')
            x = block(x, s)
            # if (masks is not None) and (x.size(2) in [32, 64, 128]):
            if (masks is not None) and (x.size(2) in [512]):
                mask = masks[0] if x.size(2) in [32] else masks[1]
                mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
                x = x + self.hpf(mask * cache[x.size(2)])
        return self.to_rgb(x)
        # return x


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=256, num_domains=2):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, style_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        # print(out)
        # print(out.shape)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        # print(s.shape)
        return s



class StyleEncoder(nn.Module):
    def __init__(self, xvec_size=512, style_dim=256, num_domains=2, max_conv_dim=512):
        super().__init__()
        # dim_in = 2**14 // xvec_size
        dim_in = xvec_size
        blocks = []
        # blocks += [nn.Conv1d(1, dim_in, 3, 1, 1)]
        # blocks += [nn.Linear(xvec_size, dim_in)]

        repeat_num = int(np.log2(xvec_size)) - 2
        for _ in range(repeat_num):
            # dim_out = min(dim_in*2, max_conv_dim)
            dim_out = max(int(dim_in/2), style_dim)
            blocks += [ResBlk(dim_in, dim_out, batch_size=None, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        # blocks += [nn.Conv1d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.Linear(dim_out, dim_out)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s
        # return torch.index_select(out, dim=1, index=y)


class Discriminator(nn.Module):
    def __init__(self, xvec_size=512, num_domains=2, max_conv_dim=512):
        super().__init__()
        # dim_in = 2**14 // xvec_size
        dim_in = xvec_size
        blocks = []
        # blocks += [nn.Conv1d(1, dim_in, 3, 1, 1)]
        # blocks += [nn.Linear(xvec_size, dim_in)]

        repeat_num = int(np.log2(xvec_size)) - 2
        for _ in range(repeat_num):
            # dim_out = min(dim_in*4, max_conv_dim)
            # dim_out = min(dim_in*2, max_conv_dim)
            dim_out = max(int(dim_in/2), num_domains)
            blocks += [ResBlk(dim_in, dim_out, batch_size=None, downsample=True)]
            dim_in = dim_out

        # blocks += [nn.LeakyReLU(0.2)]
        # blocks += [nn.Conv1d(dim_out, dim_out, 4, 1, 0)]
        # blocks += [nn.Linear(dim_out, dim_out)]
        blocks += [nn.LeakyReLU(0.2)]
        # blocks += [nn.Conv1d(dim_out, num_domains, 1, 1, 0)]
        blocks += [nn.Linear(dim_out, num_domains)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        # print('x.shape :', x.shape)
        out = self.main(x)        
        # print(out.shape)
        out = torch.squeeze(out)
        # out = out.view(out.size(0), -1)  # (batch, num_domains)
        # print(out.shape)
        # print("---------------------------------------------")
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx,y]
        # print(out.shape)
        # print("---------------------------------------------")
        return out


def build_model(args):
    generator = nn.DataParallel(Generator(args.batch_size, args.xvec_size, args.style_dim, w_hpf=0))
    mapping_network = nn.DataParallel(MappingNetwork(args.latent_dim, args.style_dim, args.num_domains))
    style_encoder = nn.DataParallel(StyleEncoder(args.xvec_size, args.style_dim, args.num_domains))
    discriminator = nn.DataParallel(Discriminator(args.xvec_size, args.num_domains))
    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)

    nets = Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator)
    nets_ema = Munch(generator=generator_ema,
                     mapping_network=mapping_network_ema,
                     style_encoder=style_encoder_ema)

    # if args.w_hpf > 0:
    #     fan = nn.DataParallel(FAN(fname_pretrained=args.wing_path).eval())
    #     fan.get_heatmap = fan.module.get_heatmap
    #     nets.fan = fan
    #     nets_ema.fan = fan

    return nets, nets_ema
