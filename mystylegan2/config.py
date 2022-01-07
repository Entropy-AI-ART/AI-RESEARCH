from yacs.config import CfgNode as CN

cfg = CN()

cfg.training = CN()
# trainning
cfg.training.output_dir = "cryptopunk"
cfg.training.device = 'cuda'
cfg.training.device_id = '0'
cfg.training.num_works = 2
cfg.training.num_samples = 36
cfg.training.feedback_factor = 10
cfg.training.checkpoint_factor = 10
cfg.training.epochs = 100
cfg.training.batch_sizes = 128

cfg.stylegan2 = CN()
# Stylegan
cfg.stylegan2.d_repeats = 1
cfg.stylegan2.use_ema = True
cfg.stylegan2.ema_decay = 0.999
cfg.stylegan2.style_mixing_prob = 0.7
cfg.stylegan2.G_reg_interval = 4
cfg.stylegan2.D_reg_interval = 16
# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.dataname = "cryptopunk" # cryptopunk
cfg.dataset.datapath = "/content/drive/My Drive/cryptopunk/data/onlyhuman.csv" # cryptopunk
cfg.dataset.image_root = "/content/drive/My Drive/cryptopunk" # cryptopunk
# cfg.dataset.folder = True
cfg.dataset.resolution = 128
cfg.dataset.channels = 3

cfg.model = CN()
# ---------------------------------------------------------------------------- #
# Options for Generator
# ---------------------------------------------------------------------------- #
cfg.model.G = CN()
cfg.model.G.z_dim = 512
cfg.model.G.w_dim = 512
cfg.model.G.c_dim = 0
cfg.model.G.img_resolution = cfg.dataset.resolution
cfg.model.G.img_channels = cfg.dataset.channels

# Mapping
cfg.model.G.mapping_kwargs = CN()
cfg.model.G.mapping_kwargs.num_layers = 5
# cfg.model.G.mapping_kwargs.num_ws = None
cfg.model.G.mapping_kwargs.embed_features  = None     # Label embedding dimensionality, None = same as w_dim.
cfg.model.G.mapping_kwargs.layer_features  = None     # Number of intermediate features in the mapping layers, None = same as w_dim.
cfg.model.G.mapping_kwargs.activation      = 'lrelu'  # Activation function: 'relu', 'lrelu', etc.
cfg.model.G.mapping_kwargs.lr_multiplier   = 0.01     # Learning rate multiplier for the mapping layers.
cfg.model.G.mapping_kwargs.w_avg_beta      = 0.995    # Decay for tracking the moving average of W during training, None = do not track.
# cfg.model.G.mapping.truncation_psi = 0.7
# cfg.model.G.mapping.truncation_cutoff = 8

# Synthesis
cfg.model.G.synthesis_kwargs = CN()
cfg.model.G.synthesis_kwargs.channel_base    = 32768 // 16    # Overall multiplier for the number of channels.
cfg.model.G.synthesis_kwargs.channel_max     = 512      # Maximum number of channels in any layer.
# cfg.model.G.synthesis.num_fp16_res    = 0,        # Use FP16 for the N highest resolutions.

cfg.model.G.synthesis_kwargs.architecture = 'resnet'
# ---------------------------------------------------------------------------- #
# Options for Discriminator
# ---------------------------------------------------------------------------- #
cfg.model.D = CN()
cfg.model.D.c_dim = cfg.model.G.c_dim                          # Conditioning label (C) dimensionality.
cfg.model.D.img_resolution = cfg.model.G.img_resolution                 # Input resolution.
cfg.model.D.img_channels = cfg.model.G.img_channels                   # Number of input color channels.
cfg.model.D.architecture        = 'resnet' # Architecture: 'orig', 'skip', 'resnet'.
cfg.model.D.channel_base        = 32768 // 16    # Overall multiplier for the number of channels.
cfg.model.D.channel_max         = 512      # Maximum number of channels in any layer.
cfg.model.D.num_fp16_res        = 0        # Use FP16 for the N highest resolutions.
cfg.model.D.conv_clamp          = None     # Clamp the output of convolution layers to +-X, None = disable clamping.
cfg.model.D.cmap_dim            = None     # Dimensionality of mapped conditioning label, None = default.

cfg.model.D.block_kwargs = CN()
cfg.model.D.block_kwargs.resample_filter     = [1,3,3,1]    # Low-pass filter to apply when resampling activations.
cfg.model.D.block_kwargs.fp16_channels_last  = False      # Use channels-last memory format with FP16?
cfg.model.D.block_kwargs.freeze_layers       = 0            # Freeze-D: Number of layers to freeze.

cfg.model.D.mapping_kwargs = CN()
# cfg.model.D.mapping     = {},       # Arguments for MappingNetwork.

cfg.model.D.epilogue_kwargs = CN()
# cfg.model.D.epilogue     = {},       # Arguments for DiscriminatorEpilogue.

# ---------------------------------------------------------------------------- #
# Options for Generator Optimizer
# ---------------------------------------------------------------------------- #
cfg.G_opt = CN()
cfg.G_opt.lr= 0.002
cfg.G_opt.betas = [0, 0.99]
cfg.G_opt.eps = 1e-8

# ---------------------------------------------------------------------------- #
# Options for Discriminator Optimizer
# ---------------------------------------------------------------------------- #
cfg.D_opt = CN()
cfg.D_opt.lr= 0.002
cfg.D_opt.betas = [0, 0.99]
cfg.D_opt.eps = 1e-8

#Loss stylegan2
cfg.loss = CN()
cfg.loss.r1_gamma = 10
cfg.loss.pl_decay = 0.1
cfg.loss.pl_weight = 1.0
cfg.loss.pl_batch_shrink = int(cfg.training.batch_sizes / 2)
cfg.loss.device = cfg.training.device