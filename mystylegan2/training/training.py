"""
-------------------------------------------------
Copied and Modified from: https://github.com/NVlabs/stylegan2-ada-pytorch
https://github.com/huangzh13/StyleGAN.pytorch/blob/master/models/Losses.py
-------------------------------------------------
"""

import torch
import copy
import numpy as np
import torch.nn as nn
import time
import timeit
import datetime
import os
import matplotlib.pyplot as plt
import PIL
from training import update_average
from training.networks import Discriminator, MappingNetwork, SynthesisNetwork, Generator
from training.loss import Stylegan2Loss
from data.utils import get_batchRandomAttribute
from data import get_data_loader


class StyleGAN:
    def __init__(self,
                G_kwargs                = {},       # Options for generator network.
                D_kwargs                = {},       # Options for discriminator network.
                G_opt_kwargs            = {},       # Options for generator optimizer.
                D_opt_kwargs            = {},       # Options for discriminator optimizer.
                # augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
                loss_kwargs             = {},       # Options for loss function.
                G_reg_interval          = 4,        # How often to perform regularization for G? None = disable lazy regularization.
                D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
                d_repeats=1,
                use_ema=False, 
                ema_decay=0.999, 
                device=torch.device("cpu")):
        """
        Wrapper around the Generator and the Discriminator.
        :param structure: 'fixed' = no progressive growing, 'linear' = human-readable
        :param resolution: Input resolution. Overridden based on dataset.
        :param num_channels: Number of input color channels. Overridden based on dataset.
        :param latent_size: Latent size of the manifold used by the GAN
        :param g_args: Options for generator network.
        :param d_args: Options for discriminator network.
        :param g_opt_args: Options for generator optimizer.
        :param d_opt_args: Options for discriminator optimizer.
        :param loss: the loss function to be used
                     Can either be a string =>
                          ["wgan-gp", "wgan", "lsgan", "lsgan-with-sigmoid",
                          "hinge", "standard-gan" or "relativistic-hinge"]
                     Or an instance of GANLoss
        :param drift: drift penalty for the
                      (Used only if loss is wgan or wgan-gp)
        :param d_repeats: How many times the discriminator is trained per G iteration.
        :param use_ema: boolean for whether to use exponential moving averages
        :param ema_decay: value of mu for ema
        :param device: device to run the GAN on (GPU / CPU)
        """
        self.device = device
        self.d_repeats = d_repeats
        self.G_reg_interval = G_reg_interval
        self.D_reg_interval = D_reg_interval

        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # Create the Generator and the Discriminator
        # mapping_kwargs= G_kwargs.mapping, 
        #                     synthesis_kwargs= G_kwargs.synthesis
        self.gen = Generator(**G_kwargs).to(self.device)

        # block_kwargs= D_kwargs.block,
        #                         mapping_kwargs= D_kwargs.mapping,
        #                         epilogue_kwargs= D_kwargs.epilogue, 
        self.dis = Discriminator(**D_kwargs).to(self.device)

        # if code is to be run on GPU, we can use DataParallel:
        # TODO

        # Lazy regularization.
        if self.D_reg_interval == 0:
            self.D_reg_gain = 1
        if self.D_reg_interval != 1:
            self.D_reg_gain = self.D_reg_interval
            mb_ratio = self.D_reg_interval / (self.D_reg_interval + 1)
            D_opt_kwargs.lr = D_opt_kwargs.lr * mb_ratio
            D_opt_kwargs.betas = [beta ** mb_ratio for beta in D_opt_kwargs.betas]

        if self.G_reg_interval == 0:
            self.G_reg_gain = 1
        elif self.G_reg_interval != 1:
            self.G_reg_gain = self.G_reg_interval
            mb_ratio = self.G_reg_interval / (self.G_reg_interval + 1)
            G_opt_kwargs.lr = G_opt_kwargs.lr * mb_ratio
            G_opt_kwargs.betas = [beta ** mb_ratio for beta in G_opt_kwargs.betas]

        # define the optimizers for the discriminator and generator
        self.__setup_gen_optim(**G_opt_kwargs)
        self.__setup_dis_optim(**D_opt_kwargs)

        # define the loss function used for training the GAN
        self.loss = Stylegan2Loss(dis= self.dis, **loss_kwargs)

        # Use of ema
        if self.use_ema:
            # create a shadow copy of the generator
            self.gen_shadow = copy.deepcopy(self.gen)
            # updater function:
            self.ema_updater = update_average
            # initialize the gen_shadow weights equal to the weights of gen
            self.ema_updater(self.gen_shadow, self.gen, beta=0)

    def __setup_gen_optim(self, lr, betas, eps):
        self.gen_optim = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(betas[0], betas[1]), eps=eps)

    def __setup_dis_optim(self, lr, betas, eps):
        self.dis_optim = torch.optim.Adam(self.dis.parameters(), lr=lr, betas=(betas[0], betas[1]), eps=eps)

    def optimize_discriminator(self, noise, real_batch, labels=None, gain= 1, reg= True):
        """
        performs one step of weight update on discriminator using the batch of data
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
        :param depth: current depth of optimization
        :param alpha: current alpha for fade-in
        :return: current loss (Wasserstein loss)
        """
        
        real_samples = real_batch

        self.dis_optim.zero_grad()
        
        # generate a batch of samples
        fake_samples = self.gen(noise, labels).detach()

        loss, penalty = self.loss.dis_loss(
                real_samples, fake_samples, labels= labels, gain= gain, reg= reg)

        # optimize discriminator
        # loss.backward()
        # if penalty is not None:
        #     penalty.backward()
        self.dis_optim.step()

        return loss, penalty

    def optimize_generator(self, noise, real_batch, labels=None, gain= 1, reg= True):
        """
        performs one step of weight update on generator for the given batch_size
        :param noise: input random noise required for generating samples
        :param real_batch: batch of real samples
        :param depth: depth of the network at which optimization is done
        :param alpha: value of alpha for fade-in effect
        :return: current loss (Wasserstein estimate)
        """

        # real_samples = self.__progressive_down_sampling(real_batch, depth, alpha)
        real_samples = real_batch

        # generate fake samples:
        fake_samples = self.gen(noise, labels, return_ws= False)

        self.gen_optim.zero_grad()
        if reg:
            shirnk_noise = noise[:self.loss.pl_batch_shrink]
            shirnk_labels = labels[:self.loss.pl_batch_shrink]
            shirnk_fake_samples, shirnk_gen_ws = self.gen(shirnk_noise, shirnk_labels, return_ws= True)
            loss, penalty = self.loss.gen_loss(
                    real_samples, fake_samples, shirnk_fake_samples, shirnk_gen_ws, 
                    labels= labels, gain= gain, reg= reg)
        else:
            loss, penalty = self.loss.gen_loss(
                    real_samples, fake_samples, None, None, 
                    labels= labels, gain= gain, reg= reg)

        
        # # optimize the generator
        # loss.backward()
        # if penalty is not None:
        #     penalty.backward()

        # Gradient Clipping
        # nn.utils.clip_grad_norm_(self.gen.parameters(), max_norm=10.)
        self.gen_optim.step()

        # if use_ema is true, apply ema to the generator parameters
        if self.use_ema:
            self.ema_updater(self.gen_shadow, self.gen, self.ema_decay)

        # return the loss value
        return loss, penalty

    @staticmethod
    def create_grid(samples, scale_factor, img_file):
        """
        utility function to create a grid of GAN samples
        :param samples: generated samples for storing
        :param scale_factor: factor for upscaling the image
        :param img_file: name of file to write
        :return: None (saves a file)
        """
        from torch.nn.functional import interpolate
        from torchvision.utils import save_image

        # upsample the image
        if scale_factor > 1:
            samples = interpolate(samples, scale_factor=scale_factor)

        # save the images:
        save_image(samples, img_file, nrow=int(np.sqrt(len(samples))),
                   normalize=True, scale_each=True, pad_value=128, padding=1)

    def train(self, 
                dataset, 
                num_workers, 
                epochs, 
                batch_sizes,  
                logger, 
                output,
                num_samples=36,
                feedback_factor=100, 
                checkpoint_factor=1):
        """
        Utility method for training the GAN. Note that you don't have to necessarily use this
        you can use the optimize_generator and optimize_discriminator for your own training routine.
        :param dataset: object of the dataset used for training.
                        Note that this is not the data loader (we create data loader in this method
                        since the batch_sizes for resolutions can be different)
        :param num_workers: number of workers for reading the data. def=3
        :param epochs: list of number of epochs to train the network for every resolution
        :param batch_sizes: list of batch_sizes for every resolution
        :param fade_in_percentage: list of percentages of epochs per resolution used for fading in the new layer
                                   not used for first resolution, but dummy value still needed.
        :param logger:
        :param output: Output dir for samples,models,and log.
        :param num_samples: number of samples generated in sample_sheet. def=36
        :param start_depth: start training from this depth. def=0
        :param feedback_factor: number of logs per epoch. def=100
        :param checkpoint_factor:
        :return: None (Writes multiple files to disk)
        """

        # assert self.depth <= len(epochs), "epochs not compatible with depth"
        # assert self.depth <= len(batch_sizes), "batch_sizes not compatible with depth"
        # assert self.depth <= len(fade_in_percentage), "fade_in_percentage not compatible with depth"

        # turn the generator and discriminator into train mode
        self.gen.train()
        self.dis.train()
        if self.use_ema:
            self.gen_shadow.train()

        # create a global time counter
        global_time = time.time()

        # create fixed_input for debugging
        fixed_input = torch.randn(num_samples, self.gen.z_dim).to(self.device)
        
        fixed_labels = None
        if self.gen.c_dim:
            # fixed_labels = torch.linspace(
            #     0, self.n_classes - 1, num_samples).to(torch.int64).to(self.device)
            fixed_labels = torch.tensor(get_batchRandomAttribute(dataset.list_attributes, num_samples), dtype= torch.float32).to(self.device)
        # config depend on structure
        logger.info("Starting the training process ... \n")
        # if self.structure == 'fixed':
        #     start_depth = 0
        step = 1  # counter for number of iterations
        # logger.info("Max depth: %d", self.depth)
        # for current_depth in range(start_depth, self.depth):
        #     current_res = np.power(2, current_depth + 2)
        #     logger.info("Currently working on depth: %d", current_depth)
        #     logger.info("Current resolution: %d x %d" % (current_res, current_res))

        # ticker = 1

        # Choose training parameters and configure training ops.
        
        # TODO

        data = get_data_loader(dataset, batch_sizes, num_workers)

        for epoch in range(1, epochs + 1):
            start = timeit.default_timer()  # record time at the start of epoch

            logger.info("Epoch: [%d]" % epoch)
            # total_batches = len(iter(data))
            total_batches = len(data)

            # fade_point = int((fade_in_percentage[current_depth] / 100)
            #                  * epochs[current_depth] * total_batches)

            for i, batch in enumerate(data, 1):
                # calculate the alpha for fading in the layers
                # alpha = ticker / fade_point if ticker <= fade_point else 1

                # extract current batch of data for training
                if self.gen.c_dim:
                    images, labels = batch
                    # print(images)
                    # print(labels)
                    labels = labels.to(self.device)
                else:
                    images, _ = batch
                    labels = None
                # print(images.max(), images.min())
                images = images * 2 - 1
                
                images = images.to(self.device)

                gan_input = torch.randn(images.shape[0], self.gen.z_dim).to(self.device)

                # optimize the discriminator:
                if self.D_reg_interval:
                    if step % self.D_reg_interval == 0:
                        dis_loss, Dr1 = self.optimize_discriminator(gan_input, images, labels, gain= self.D_reg_interval, reg= True)
                    else:
                        dis_loss, Dr1 = self.optimize_discriminator(gan_input, images, labels, gain= 1, reg= False)
                else:
                    dis_loss, Dr1 = self.optimize_discriminator(gan_input, images, labels, gain= 1, reg= False)

                # optimize the generator:
                # print(self.G_reg_interval)
                if self.G_reg_interval:
                    if step % self.G_reg_interval == 0:
                        gen_loss, pl = self.optimize_generator(gan_input, images, labels, gain= self.G_reg_interval, reg= True)
                    else:
                        gen_loss, pl = self.optimize_generator(gan_input, images, labels, gain= 1, reg= False)
                else:
                    gen_loss, pl = self.optimize_generator(gan_input, images, labels, gain= 1, reg= False)  

                # provide a loss feedback
                if i % int(total_batches / feedback_factor + 1) == 0 or i == 1:
                    elapsed = time.time() - global_time
                    elapsed = str(datetime.timedelta(seconds=elapsed)).split('.')[0]
                    
                    logger.info(
                        "Elapsed: [%s] Step: %d  Batch: %d  D_Loss: %f  R1: %f  G_Loss: %f  PL: %f"
                        % (elapsed, step, i, dis_loss, Dr1, gen_loss, pl))

                    # create a grid of samples and save it
                    os.makedirs(os.path.join(output, 'samples'), exist_ok=True)
                    gen_img_file = os.path.join(output, 'samples', "gen"
                                                + "_" + str(epoch) + "_" + str(i) + ".jpg")

                    with torch.no_grad():                            
                        # self.create_grid(
                        #     samples=self.gen(fixed_input, current_depth, alpha, labels_in=fixed_labels).detach() if not self.use_ema
                        #     else self.gen_shadow(fixed_input, current_depth, alpha, labels_in=fixed_labels).detach(),
                        #     scale_factor=int(
                        #         np.power(2, self.depth - current_depth - 1)) if self.structure == 'linear' else 1,
                        #     img_file=gen_img_file,
                        # )
                        samples=self.gen(fixed_input, fixed_labels).detach() if not self.use_ema \
                            else self.gen_shadow(fixed_input, fixed_labels).detach()
                        plt.figure(figsize= (15, 15))
                        num_sample = samples.size(0)
                        edge = int(np.sqrt(num_sample))
                        logger.info('range (%f, %f)'%(samples[0].min(), samples[0].max()))
                        # exit(0)
                        lo, hi = -1, 1
                        # img = (img - lo) * (255 / (hi - lo))
                        # img = np.rint(img).clip(0, 255).astype(np.uint8)
                        # exit(0)
                        numpy_images = samples.add(-lo).mul(255 / (hi - lo)).round().clamp(0, 255).permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
                        for i, img in enumerate(numpy_images):
                            plt.subplot(edge, edge, i + 1)
                            plt.imshow(PIL.Image.fromarray(img, 'RGB'))
                            plt.axis('off')
                        plt.savefig(gen_img_file)
                        plt.close()
                        

                # increment the alpha ticker and the step
                # ticker += 1
                step += 1

            elapsed = timeit.default_timer() - start
            elapsed = str(datetime.timedelta(seconds=elapsed)).split('.')[0]
            logger.info("Time taken for epoch: %s\n" % elapsed)

            if epoch % checkpoint_factor == 0:
                save_dir = os.path.join(output, 'models')
                os.makedirs(save_dir, exist_ok=True)
                gen_save_file = os.path.join(save_dir, "GAN_GEN" + "_" + str(epoch) + ".pth")
                dis_save_file = os.path.join(save_dir, "GAN_DIS" + "_" + str(epoch) + ".pth")
                gen_optim_save_file = os.path.join(
                    save_dir, "GAN_GEN_OPTIM" + "_" + str(epoch) + ".pth")
                dis_optim_save_file = os.path.join(
                    save_dir, "GAN_DIS_OPTIM" + "_" + str(epoch) + ".pth")

                torch.save(self.gen.state_dict(), gen_save_file)
                logger.info("Saving the model to: %s\n" % gen_save_file)
                torch.save(self.dis.state_dict(), dis_save_file)
                torch.save(self.gen_optim.state_dict(), gen_optim_save_file)
                torch.save(self.dis_optim.state_dict(), dis_optim_save_file)

                # also save the shadow generator if use_ema is True
                if self.use_ema:
                    gen_shadow_save_file = os.path.join(
                        save_dir, "GAN_GEN_SHADOW" + "_" + str(epoch) + ".pth")
                    torch.save(self.gen_shadow.state_dict(), gen_shadow_save_file)
                    logger.info("Saving the model to: %s\n" % gen_shadow_save_file)

        logger.info('Training completed.\n')
        