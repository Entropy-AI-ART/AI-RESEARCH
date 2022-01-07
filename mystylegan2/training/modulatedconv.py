"""
-------------------------------------------------
Copied and Modified from: https://github.com/NVlabs/stylegan2-ada-pytorch
-------------------------------------------------
"""

from utils.upfirdn2d import conv2d_resample

def modulated_conv2d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size, x_in_channels, _, _ = x.size()
    out_channels, in_channels, kh, kw = weight.size()
    _, s_in_channels = styles.size()
    
    assert x_in_channels == in_channels # [NIHW]
    assert s_in_channels == in_channels # [NI]

    # # Pre-normalize inputs to avoid FP16 overflow.
    # if x.dtype == torch.float16 and demodulate:
    #     weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
    #     styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        # print('mcov', x.max(), x.min())
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        # print('mcov', x.max(), x.min())
        x = conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        # print('mcov', x.max(), x.min())
        if demodulate and noise is not None:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1) + noise.to(x.dtype)
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        # print('mcov', x.max(), x.min())
        return x

    # # Execute as one fused op using grouped convolution.
    # with misc.suppress_tracer_warnings(): # this value will be treated as a constant
    #     batch_size = int(batch_size)
    # misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    # print('mcov', x.max(), x.min())
    x = conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    # print('mcov', x.max(), x.min())
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    # print('mcov', x.max(), x.min())
    
    return x