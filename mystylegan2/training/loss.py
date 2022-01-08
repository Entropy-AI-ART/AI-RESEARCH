"""
-------------------------------------------------
   Description:  Module implementing various loss functions
                 Copy from: https://github.com/akanimax/pro_gan_pytorch
                 https://github.com/NVlabs/stylegan2-ada-pytorch
                 https://github.com/huangzh13/StyleGAN.pytorch/blob/master/models/Losses.py
                 """

import numpy as np
import torch
import torch.nn as nn
# from torch.nn import BCEWithLogitsLoss

# =============================================================
# Interface for the losses
# =============================================================

class GANLoss:
    """ Base class for all losses
        @args:
        dis: Discriminator used for calculating the loss
             Note this must be a part of the GAN framework
    """

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps):
        """
        calculate the discriminator loss using the following data
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps):
        """
        calculate the generator loss
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("gen_loss method has not been implemented")

class Stylegan2Loss(GANLoss):
    def __init__(self, dis, r1_gamma, pl_decay, pl_weight, pl_batch_shrink, device='cpu'):
        super().__init__(dis)
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)

    def path_lengthPenalty(self, shrink_gen_img, shrink_gen_ws, gain):
        # batch_size = gen_z.shape[0] // self.pl_batch_shrink
        # gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)

        pl_noise = torch.randn_like(shrink_gen_img) / np.sqrt(shrink_gen_img.size(2) * shrink_gen_img.size(3))
        # with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
        pl_grads = torch.autograd.grad(outputs=[(shrink_gen_img * pl_noise).sum()]\
        , inputs=[shrink_gen_ws], create_graph=True, only_inputs=True)[0]

        pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
        pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
        self.pl_mean.copy_(pl_mean.detach())
        pl_penalty = (pl_lengths - pl_mean).square()
        
        loss_Gpl = pl_penalty * self.pl_weight
        
        # with torch.autograd.profiler.record_function('Gpl_backward'):
        return (shrink_gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain)

    def R1Penalty(self, real_img, labels= None, reg= True):

        # # TODO: use_loss_scaling, for fp16
        # apply_loss_scaling = lambda x: x * torch.exp(x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))
        # undo_loss_scaling = lambda x: x * torch.exp(-x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))

        real_img = torch.autograd.Variable(real_img, requires_grad=True)
        real_logits = self.dis(real_img, labels)
        if not reg:
            return real_logits
        # real_logit = apply_loss_scaling(torch.sum(real_logit))
        # real_grads = torch.autograd.grad(outputs=real_logit, inputs=real_img,
        #                                  grad_outputs=torch.ones(real_logit.size()).to(real_img.device),
        #                                  create_graph=True, retain_graph=True)[0].view(real_img.size(0), -1)
        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img], 
        create_graph=True, only_inputs=True)[0]
        r1_penalty = r1_grads.square().sum([1,2,3])
        loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
        # real_grads = undo_loss_scaling(real_grads)
        # r1_penalty = torch.sum(torch.mul(real_grads, real_grads))
        return real_logits, loss_Dr1

    def dis_loss(self, real_samps, fake_samps, labels= None, gain= 1, reg= True):
        # Obtain predictions
        r_preds = self.dis(real_samps, labels)
        f_preds = self.dis(fake_samps, labels)
        # exit(0)

        lossDgen = torch.mean(nn.Softplus()(f_preds))
        lossDgen.backward()

        lossDreal = nn.Softplus()(-r_preds)
        loss = lossDgen.item() + torch.mean(lossDreal).item()

        if self.r1_gamma != 0.0 and reg:
            real_logit, r1_penalty = self.R1Penalty(real_samps.detach(), labels, reg= reg)
            (real_logit * 0 + lossDreal + r1_penalty).mean().mul(gain).backward()
          
        else:
            real_logit = self.R1Penalty(real_samps.detach(), labels, reg= reg)
            (real_logit * 0 + lossDreal).mean().mul(gain).backward()
          
        return loss, r1_penalty.mean().item() if reg else 0

    def gen_loss(self, _, fake_samps, shirnk_image_gen, gen_ws_skirnk, gain, labels= None, reg= True):
        f_preds = self.dis(fake_samps, labels)
        # shirnk_image_gen = fake_samps[:self.pl_batch_shrink]
        # gen_ws_skirnk = gen_ws[:self.pl_batch_shrink]
        loss_Gmain = torch.mean(nn.Softplus()(-f_preds))
        loss_Gmain.backward()
        # print(self.pl_weight, reg)
        if self.pl_weight != 0.0 and reg:
            gl_penalty = self.path_lengthPenalty(shirnk_image_gen, gen_ws_skirnk, gain)
            gl_penalty.backward()

            return loss_Gmain.item(), gl_penalty.item()

        return loss_Gmain.item(), 0