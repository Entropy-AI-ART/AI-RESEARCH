"""
-------------------------------------------------
Copied and Modified from: https://github.com/NVlabs/stylegan2-ada-pytorch
-------------------------------------------------
"""

import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


activation_funcs = {
    'linear':   {'func': lambda x, **_:         x, 'def_gain' : 1, 'def_alpha': 0},
    'relu':     {'func': lambda x, **_:         torch.nn.functional.relu(x), 'def_gain' : np.sqrt(2), 'def_alpha': 0},
    'lrelu':    {'func': lambda x, alpha, **_:  torch.nn.functional.leaky_relu(x, alpha), 'def_gain' : np.sqrt(2), 'def_alpha': 0.2},
    'tanh':     {'func': lambda x, **_:         torch.tanh(x), 'def_gain' : 1, 'def_alpha': 0},
    'sigmoid':  {'func': lambda x, **_:         torch.sigmoid(x), 'def_gain' : 1, 'def_alpha': 0},
    'elu':      {'func': lambda x, **_:         torch.nn.functional.elu(x), 'def_gain' : 1, 'def_alpha': 0},
    'selu':     {'func': lambda x, **_:         torch.nn.functional.selu(x), 'def_gain' : 1, 'def_alpha': 0},
    'softplus': {'func': lambda x, **_:         torch.nn.functional.softplus(x), 'def_gain' : 1, 'def_alpha': 0},
    'swish':    {'func': lambda x, **_:         torch.sigmoid(x) * x, 'def_gain' : np.sqrt(2), 'def_alpha': 0}
}

def bias_act(x, b=None, dim=1, act='linear', alpha=None, gain=None, clamp=None):
    r"""Fused bias and activation function.

    Adds bias `b` to activation tensor `x`, evaluates activation function `act`,
    and scales the result by `gain`. Each of the steps is optional. In most cases,
    the fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports first and second order gradients,
    but not third order gradients.

    Args:
        x:      Input activation tensor. Can be of any shape.
        b:      Bias vector, or `None` to disable. Must be a 1D tensor of the same type
                as `x`. The shape must be known, and it must match the dimension of `x`
                corresponding to `dim`.
        dim:    The dimension in `x` corresponding to the elements of `b`.
                The value of `dim` is ignored if `b` is not specified.
        act:    Name of the activation function to evaluate, or `"linear"` to disable.
                Can be e.g. `"relu"`, `"lrelu"`, `"tanh"`, `"sigmoid"`, `"swish"`, etc.
                See `activation_funcs` for a full list. `None` is not allowed.
        alpha:  Shape parameter for the activation function, or `None` to use the default.
        gain:   Scaling factor for the output tensor, or `None` to use default.
                See `activation_funcs` for the default scaling of each activation function.
                If unsure, consider specifying 1.
        clamp:  Clamp the output values to `[-clamp, +clamp]`, or `None` to disable
                the clamping (default).

    Returns:
        Tensor of the same shape and datatype as `x`.
    """
    assert isinstance(x, torch.Tensor)
    assert clamp is None or clamp >= 0
    spec = activation_funcs[act]
    alpha = float(alpha if alpha is not None else spec['def_alpha'])
    gain = float(gain if gain is not None else spec['def_gain'])
    clamp = float(clamp if clamp is not None else -1)

    # Add bias.
    if b is not None:
        assert isinstance(b, torch.Tensor) and b.ndim == 1
        assert 0 <= dim < x.ndim
        assert b.shape[0] == x.shape[dim]
        x = x + b.reshape([-1 if i == dim else 1 for i in range(x.ndim)])

    # Evaluate activation function.
    alpha = float(alpha)
    x = spec['func'](x, alpha=alpha)

    # Scale by gain.
    gain = float(gain)
    if gain != 1:
        x = x * gain

    # Clamp.
    if clamp >= 0:
        x = x.clamp(-clamp, clamp) # pylint: disable=invalid-unary-operand-type
    return x
