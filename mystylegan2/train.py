import argparse
import os
import shutil

import torch
from torch.backends import cudnn

from data import make_dataset
from training.training import StyleGAN
from utils import (copy_files_and_create_dirs,
                   list_dir_recursively_with_ignore, make_logger, load, visualize_tensor)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="StyleGAN pytorch implementation.")
    parser.add_argument('--config', default='')
    
    parser.add_argument("--generator_file", action="store", type=str, default=None,
                        help="pretrained Generator file (compatible with my code)")
    parser.add_argument("--gen_shadow_file", action="store", type=str, default=None,
                        help="pretrained gen_shadow file")
    parser.add_argument("--discriminator_file", action="store", type=str, default=None,
                        help="pretrained Discriminator file (compatible with my code)")
    parser.add_argument("--gen_optim_file", action="store", type=str, default=None,
                        help="saved state of generator optimizer")
    parser.add_argument("--dis_optim_file", action="store", type=str, default=None,
                        help="saved_state of discriminator optimizer")
    parser.add_argument("--checkpoint_factor", action="store", type=int, default=1,
                        help="Save checkpoint period")
    args = parser.parse_args()

    from config import cfg as opt
    if not args.config == '':
        opt.merge_from_file(args.config)

    # make output dir
    output_dir = opt.training.output_dir
    if not os.path.exists(output_dir):
        # raise KeyError("Existing path: ", output_dir)
        os.makedirs(output_dir)

        # copy codes and config file
        files = list_dir_recursively_with_ignore('.', ignores=['diagrams', 'configs'])
        files = [(f[0], os.path.join(output_dir, "src", f[1])) for f in files]
        copy_files_and_create_dirs(files)
        if not args.config == '':
            shutil.copy2(args.config, output_dir)

    # logger
    logger = make_logger("project", opt.training.output_dir, 'log')

    # device
    if opt.training.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.training.device_id
        num_gpus = len(opt.training.device_id.split(','))
        logger.info("Using {} GPUs.".format(num_gpus))
        logger.info("Training on {}.\n".format(torch.cuda.get_device_name(0)))
        cudnn.benchmark = True
    device = torch.device(opt.training.device)

    # create the dataset for training
    dataset = make_dataset(opt.dataset)
    opt.model.G.c_dim = len(dataset.list_attributes)
    # print(dataset.list_attributes)
    # exit(0)

    # init the network
    style_gan = StyleGAN(G_kwargs= opt.model.G, 
                         D_kwargs= opt.model.D, 
                         G_opt_kwargs= opt.G_opt, 
                         D_opt_kwargs= opt.D_opt, 
                         loss_kwargs= opt.loss, 
                         G_reg_interval= opt.stylegan2.G_reg_interval, 
                         D_reg_interval= opt.stylegan2.D_reg_interval,
                         d_repeats=opt.stylegan2.d_repeats,
                         use_ema=opt.stylegan2.use_ema,
                         ema_decay=opt.stylegan2.ema_decay,
                         device=device)
    opt.freeze()

    # Resume training from checkpoints
    if args.generator_file is not None:
        logger.info("Loading generator from: %s", args.generator_file)
        # style_gan.gen.load_state_dict(torch.load(args.generator_file))
        # Load fewer layers of pre-trained models if possible
        load(style_gan.gen, args.generator_file)
    else:
        logger.info("Training from scratch...")

    if args.discriminator_file is not None:
        logger.info("Loading discriminator from: %s", args.discriminator_file)
        style_gan.dis.load_state_dict(torch.load(args.discriminator_file))

    if args.gen_shadow_file is not None and opt.use_ema:
        logger.info("Loading shadow generator from: %s", args.gen_shadow_file)
        # style_gan.gen_shadow.load_state_dict(torch.load(args.gen_shadow_file))
        # Load fewer layers of pre-trained models if possible
        load(style_gan.gen_shadow, args.gen_shadow_file)

    if args.gen_optim_file is not None:
        logger.info("Loading generator optimizer from: %s", args.gen_optim_file)
        style_gan.gen_optim.load_state_dict(torch.load(args.gen_optim_file))

    if args.dis_optim_file is not None:
        logger.info("Loading discriminator optimizer from: %s", args.dis_optim_file)
        style_gan.dis_optim.load_state_dict(torch.load(args.dis_optim_file))

    # train the network
    style_gan.train(dataset=dataset,
                  num_workers=opt.training.num_works,
                  epochs=opt.training.epochs,
                  batch_sizes=opt.training.batch_sizes,
                  logger=logger,
                  output=output_dir,
                  num_samples=opt.training.num_samples,
                  feedback_factor=opt.training.feedback_factor,
                  checkpoint_factor=opt.training.checkpoint_factor)