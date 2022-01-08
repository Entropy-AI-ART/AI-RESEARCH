"""
-------------------------------------------------
Copied and Modified from: https://github.com/NVlabs/stylegan2-ada-pytorch
-------------------------------------------------
"""

import numpy as np
# from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.bias_act import bias_act, activation_funcs
from training.modulatedconv import modulated_conv2d
from utils import upfirdn2d

class PixelNormLayer(nn.Module):
    def __init__(self, dim= 1, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.dim = dim

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim= self.dim, keepdim=True) + self.epsilon)

class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        act      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = act
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act(x, b, act=self.activation)
        return x

# class EqualizedLinear(nn.Module):
#     """Linear layer with equalized learning rate and custom learning rate multiplier."""

#     def __init__(self, input_size, output_size, gain=1, use_wscale=True, lr_multiplier=1, bias=True\
#     , act='linear', alpha=None, bias_init= 0):
#         super().__init__()
#         self.act = act
#         self.alpha = alpha
#         he_std = gain * input_size ** (-0.5)  # He init
#         # Equalized learning rate and custom learning rate multiplier.
#         if use_wscale:
#             init_std = 1.0 / lr_multiplier
#             self.w_mul = he_std * lr_multiplier
#         else:
#             init_std = he_std / lr_multiplier
#             self.w_mul = lr_multiplier
#         self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) * init_std)
        
#         if bias:
#             self.bias = torch.nn.Parameter(torch.full([output_size], np.float32(bias_init))) if bias else None
#             self.b_mul = lr_multiplier
#         else:
#             self.bias = None

#     def forward(self, x):
#         bias = self.bias
#         out = F.linear(x, self.weight * self.w_mul)
#         return bias_act(out, b=bias, dim=1, act=self.act, alpha=self.alpha, gain=self.b_mul, clamp=None)

class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = activation_funcs[activation]['def_gain']

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        # print('convw', w.max(), w.min())
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1) # slightly faster
        # print('conv', x.max(), x.min())
        x = upfirdn2d.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)
        # print('conv', x.max(), x.min())
        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        # print('conv', x.max(), x.min())
        return x

# class ModulatedConv2dLayer(nn.Module):
#     def __init__(self, 
#                 in_channels, 
#                 out_channels, 
#                 style_dim, 
#                 kernel_size, 
#                 up=False, 
#                 down=False, 
#                 demodulate=True, 
#                 resample_kernel=None, 
#                 gain=1, 
#                 use_wscale=True, 
#                 lrmul=1, 
#                 fused_modconv=True):
                
#         super(ModulatedConv2dLayer, self).__init()
#         assert not (up and down)
#         assert kernel_size >= 1 and kernel_size % 2 == 1
#         self.kernel_size = kernel_size
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.up = up
#         self.down = down
#         self.gain = gain
#         self.use_wscale = use_wscale
#         self.fused_modconv= fused_modconv
#         self.resample_kernel = resample_kernel
#         self.demodulate = demodulate

#         self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_kernel))

#         # Get weight.
#         self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]))
        
#     def forward(self, x, styles, noise= None):
#         batch_size, x_in_channels, xh, xw = x.size()
#         _, s_in_channels = styles.size()
        
#         assert x_in_channels == self.in_channels
#         assert s_in_channels == self.in_channels

#         # Calculate per-sample weights and demodulation coefficients.
#         w = None
#         dcoefs = None
#         if self.demodulate or self.fused_modconv:
#             w = self.weight.unsqueeze(0) # [NOIkk]
#             w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
#         if self.demodulate:
#             dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
#         if self.demodulate and self.fused_modconv:
#             w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

#         if not self.fused_modconv:
#             x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
#             x = upfirdn2d.conv2d_resample.conv2d_resample(x=x, w=self.weight.to(x.dtype), f=self.resample_filter
#             , up= self.up, down= self.down, padding= self.padding, flip_weight= self.flip_weight)
#             if self.demodulate and noise is not None:
#                 x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1) + noise.to(x.dtype)
#             elif self.demodulate:
#                 x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
#             elif noise is not None:
#                 x = x.add_(noise.to(x.dtype))
#             return x

#         #Execute as one fused op
#         x = x.reshape(1, -1, *x.size()[2:])
#         w = self.weight.reshape(-1, self.in_channels, self.kernel_size, self.kernel_size)
#         x = upfirdn2d.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, \
#         up=self.up, down=self.down, padding=self.padding, groups=batch_size, flip_weight=self.flip_weight)
#         x = x.reshape(batch_size, -1, *x.shape[2:])
#         if noise is not None:
#             x = x.add_(noise)
#         return x

class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = True,         # Enable noise input?
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last   = False,        # Use channels_last format for the weights?
    ):
        super().__init__()
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = activation_funcs[activation]['def_gain']

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        # memory_format = torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size])) #.to(memory_format=memory_format))
        
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        assert self.weight.size(1) == x.size(1)
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength
        # print('synw', w.max(), w.min())

        flip_weight = (self.up == 1) # slightly faster
        # print('syn', x.max(), x.min())
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)
        # print('syn', x.max(), x.min())
        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        # print('syn', x.max(), x.min())
        return x


class ToRGBLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        # memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size])) #.to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x

class MinibatchStdLayer(nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        # with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
        G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x
