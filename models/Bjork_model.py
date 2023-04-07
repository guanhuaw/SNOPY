import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util.metrics import PSNR
# import pytorch_msssim
import time
import sys
import os
import numpy as np
from util.hfen import hfen
from .losses import pns
from mirtorch.alg import CG, FISTA, POGM, power_iter
from mirtorch.linear import LinearMap, FFTCn, NuSense, Sense, FFTCn, Identity, Diff2dgram, Diff3dgram, Gmri, \
    Wavelet2D, \
    NuSenseGram
from mirtorch.prox import Prox, L1Regularizer
from models.mirtorch_pkg import NuSense_om, Gram_inv, NuSenseGram_om, Gram_inv_diff
import torchkbnufft as tkbn
from .sgld import SGLD


# The reconstruction here follows the MoDL DOI: 10.1109/TMI.2018.2865356
class BJORKModel(BaseModel):
    def name(self):
        return 'BJORKModel'

    # Initialize the model
    def initialize(self, opt):
        BaseModel.initialize(self, opt)  # ATTENTION HERE: NEED TO ALTER THE DEFAULT PLAN
        self.netSampling = networks.define_G(opt, opt.which_model_netD_I, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_I = networks.define_G(opt, opt.which_model_netG_I, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.model_names = ['Sampling', 'G_I']
            self.loss_names = ['G_I_L1', 'G_I_L2', 'PSNR', 'grad', 'slew', 'pns']
        else:
            self.model_names = ['Sampling', 'G_I']
            self.loss_names = ['PSNR']

        self.visual_names = ['Ireal', 'Ifake', 'Iunder', 'ktraj', 'Idcf', 'grad', 'slew']
        # self.visual_names = []

        self.num_shots = opt.num_shots
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionMSE = torch.nn.MSELoss()
        if self.isTrain:
            self.optimizers = []
            if self.opt.sgld:
                self.optimizer_S = SGLD(list(
                    self.netSampling.parameters()), lr=self.opt.ReconVSTraj * opt.lr_traj, num_burn_in_steps = 0)
            else:
                self.optimizer_S = torch.optim.Adam(list(
                    self.netSampling.parameters()), lr=self.opt.ReconVSTraj * opt.lr_traj, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_S)
            self.optimizer_G = torch.optim.Adam(
                list(self.netG_I.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
        self.update_alt_cnt = 0

        if self.opt.contrast_condition is not None:
            self.contrast_idx = np.load(self.opt.contrast_condition)['idx']
            self.contrast_value = np.load(self.opt.contrast_condition)['value']
            self.gradient_idx = np.load(self.opt.contrast_condition)['gidx']
            self.gradient_value = np.load(self.opt.contrast_condition)['gvalue']


    def set_input(self, input):

        self.Ireal = input['I'].to(self.device)
        self.smap = input['smap'].to(self.device)
        self.image_paths = input['path']
        self.batchsize = self.smap.shape[0]
        if self.isTrain and not self.opt.no_flip:
            if np.random.rand() > 0.5:
                self.Ireal = torch.flip(self.Ireal, [-3])
                self.smap = torch.flip(self.smap, [-3])
            if np.random.rand() > 0.5:
                self.Ireal = torch.flip(self.Ireal, [-2])
                self.smap = torch.flip(self.smap, [-2])
            if np.random.rand() > 0.5:
                self.Ireal = torch.flip(self.Ireal, [-1])
                self.smap = torch.flip(self.smap, [-1])


    def backward_G(self):

        self.loss_G_I_L1 = self.criterionL1(self.Ifake, self.Ireal) * self.opt.loss_content_I_l1
        self.loss_G_I_L2 = self.criterionMSE(self.Ifake, self.Ireal) * self.opt.loss_content_I_l2
        self.loss_G_CON_I = self.loss_G_I_L1 + self.loss_G_I_L2
        softgrad = torch.nn.Softshrink(self.opt.gradmax * 0.995)
        softslew = torch.nn.Softshrink(self.opt.slewmax * 0.995)
        softpi = torch.nn.Softshrink(torch.pi)
        pt = torch.norm(self.pt, dim=0)
        softpns = torch.nn.Softshrink(self.opt.pth)
        self.loss_pns = torch.sum(softpns(pt))*self.opt.loss_pns
        if self.opt.iso_constraint:
            self.loss_grad = torch.sum(torch.pow(softgrad(torch.norm(self.grad, dim=0)), 2))*self.opt.loss_grad
            self.loss_slew = torch.sum(torch.pow(softslew(torch.norm(self.slew, dim=0)), 2))*self.opt.loss_slew
        else:
            self.loss_grad = torch.sum(torch.pow(softgrad(torch.abs(self.grad)), 2))*self.opt.loss_grad
            self.loss_slew = torch.sum(torch.pow(softslew(torch.abs(self.slew)), 2))*self.opt.loss_slew
        if self.opt.contrast_condition is not None:
            # Reshape the traj
            ktraj_tmp = self.ktraj.reshape(self.ktraj.shape[0], self.ktraj.shape[1], self.opt.num_shots, self.opt.nfe)
            self.loss_contrast = (torch.norm(ktraj_tmp[:,self.contrast_idx[0],:,self.contrast_idx[1]]-torch.tensor(self.contrast_value).to(ktraj_tmp))+ \
                                 torch.norm(self.grad[self.gradient_idx[0],:,self.gradient_idx[1]]-torch.tensor(self.gradient_value).to(ktraj_tmp)))*self.opt.loss_contrast
        else:
            self.loss_contrast = 0
        self.loss_pi = self.opt.loss_pi * torch.sum(softpi(torch.abs(self.ktraj)))
        self.loss_G = self.loss_G_CON_I + self.loss_grad + self.loss_slew + self.loss_pns + self.loss_contrast + self.loss_pi
        self.loss_G.backward()

    def forward(self):
        self.ktraj, self.grad, self.slew = self.netSampling(1)
        self.pt = pns(self.slew)
        self.A = NuSense_om(self.smap, self.ktraj, numpoints=self.opt.numpoints, grid_size=self.opt.grid_size, norm='ortho')
        self.Ireal = self.Ireal.unsqueeze(1)
        self.kunder = self.A * (self.Ireal)

        self.Iunder = self.A.H * self.kunder
        self.dcf = tkbn.calc_density_compensation_function(self.ktraj.detach(), im_size=self.Ireal.shape[-3:])
        self.Idcf = self.A.H * (self.kunder * self.dcf).detach()
        AIdcf = self.A * self.Idcf
        self.Idcf = self.Idcf * torch.sum(torch.conj(AIdcf) * self.kunder) / (torch.norm(AIdcf) ** 2)
        self.Ifake = self.Idcf
        P = Gram_inv(self.smap, self.ktraj, self.opt.CGlambda, self.opt.CGtol, max_iter=10, norm='ortho',
                      numpoints=self.opt.numpoints, grid_size=self.opt.grid_size, alert=False, x0=self.Ifake)
        for i in range(self.opt.num_blocks):
            self.Ifake = torch.view_as_complex(
                self.netG_I(torch.view_as_real(self.Ifake.squeeze(1)).permute(0, 4, 1, 2, 3).contiguous())
                    .permute(0, 2, 3, 4, 1).contiguous()).unsqueeze(1)
            P.x0 = self.Ifake
            self.Ifake = P * (self.Iunder + self.opt.CGlambda * self.Ifake)
        self.Ireal = torch.view_as_real(self.Ireal.squeeze(1)).permute(0, 4, 1, 2, 3)
        self.Ifake = torch.view_as_real(self.Ifake.squeeze(1)).permute(0, 4, 1, 2, 3)
        self.Idcf = torch.view_as_real(self.Idcf.squeeze(1)).permute(0, 4, 1, 2, 3)
        self.Iunder = torch.view_as_real(self.Iunder.squeeze(1)).permute(0, 4, 1, 2, 3)
        self.loss_PSNR = PSNR(self.Ireal, self.Ifake)

    def optimize_parameters(self):
        start = time.time()
        if self.update_alt_cnt < self.opt.ReconVSTraj:
            self.forward()
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
            self.update_alt_cnt = self.update_alt_cnt + 1
        else:
            self.forward()
            self.optimizer_G.zero_grad()
            self.optimizer_S.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
            self.optimizer_S.step()
            self.update_alt_cnt = 1

        print('Epoch time', time.time() - start)

