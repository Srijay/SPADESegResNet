"""

The SPADESegResNet model

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc, norm):
        super().__init__()


        if norm == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif norm == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif norm == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        ks = 3
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # define normalization layers
        self.norm_0 = SPADE(fin, 3, norm='instance')
        self.norm_1 = SPADE(fmiddle, 3, norm='instance')
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, 3, norm='instance')

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
      super(ResnetBlock, self).__init__()
      self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
      conv_block = []
      p = 0
      if padding_type == 'reflect':
        conv_block += [nn.ReflectionPad2d(1)]
      elif padding_type == 'replicate':
        conv_block += [nn.ReplicationPad2d(1)]
      elif padding_type == 'zero':
        p = 1
      else:
        raise NotImplementedError('padding [%s] is not implemented' % padding_type)

      conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                     norm_layer(dim),
                     activation]

      if use_dropout:
        conv_block += [nn.Dropout(0.5)]

      p = 0
      if padding_type == 'reflect':
        conv_block += [nn.ReflectionPad2d(1)]
      elif padding_type == 'replicate':
        conv_block += [nn.ReplicationPad2d(1)]
      elif padding_type == 'zero':
        p = 1
      else:
        raise NotImplementedError('padding [%s] is not implemented' % padding_type)
      conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                     norm_layer(dim)]

      return nn.Sequential(*conv_block)

    def forward(self, x):
      out = x + self.conv_block(x)
      return out

class SPADEResNet(torch.nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=5, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(SPADEResNet, self).__init__()
        activation = nn.ReLU(True)

        downsampler = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            downsampler += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]
        self.downsampler = nn.Sequential(*downsampler)

        ### resnet blocks
        mult = 2 ** n_downsampling
        self.resnetblocks1 = SPADEResnetBlock(ngf * mult, ngf * mult)
        self.resnetblocks2 = SPADEResnetBlock(ngf * mult, ngf * mult)
        self.resnetblocks3 = SPADEResnetBlock(ngf * mult, ngf * mult)
        self.resnetblocks4 = SPADEResnetBlock(ngf * mult, ngf * mult)
        self.resnetblocks5 = SPADEResnetBlock(ngf * mult, ngf * mult)

        ### upsample
        upsampler = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            upsampler += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]

        upsampler += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        # upsampler += [nn.Conv2d(2*ngf, ngf, kernel_size=3, padding=1),
        #           norm_layer(ngf),
        #           nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1),
        #           nn.Tanh()] #For larger to smaller region network

        self.upsampler = nn.Sequential(*upsampler)

    def forward(self, input):
        downsampled = self.downsampler(input)
        resnet1 = self.resnetblocks1(downsampled, input)
        resnet2 = self.resnetblocks1(resnet1, input)
        resnet3 = self.resnetblocks1(resnet2, input)
        resnet4 = self.resnetblocks1(resnet3, input)
        resnet5 = self.resnetblocks1(resnet4, input)
        upsampled = self.upsampler(resnet5)
        return upsampled

