import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import time
from util.util import bspline2_1ndsynth, \
    bilinear_interpolate_torch_gridsample, \
    center_crop
from packaging import version
import torchkbnufft

if version.parse(torchkbnufft.__version__) <= version.parse("1.0.0"):
    from torchkbnufft import AdjMriSenseNufft, MriSenseNufft, KbNufft, AdjKbNufft, ToepSenseNufft
import math
import scipy
import scipy.linalg
from .didn3d import DIDN3D, create_standard_module
from einops import rearrange, reduce, repeat
import scipy.io as sio

# from util.cg_block import cg_block
###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
        print('data parallel initialization')
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(opt, which_model_netG, init_type='normal', init_gain=0.02, gpu_ids=[], device=None):
    netG = None
    norm_layer = get_norm_layer(norm_type=opt.norm)
    if which_model_netG == 'didn':
        netG = DIDN3D(opt)
    elif which_model_netG == "samplingBspline":
        netG = SamplingLayerBspline3D(opt.num_shots, opt.nfe, decim=opt.decim_rate, dt=opt.dt,
                                      res=[opt.resx, opt.resy, opt.resz], init_traj=opt.mask_path, ext=opt.padding,
                                      device=device)
    elif which_model_netG == "SOS":
        netG = SOS(opt.num_shots, opt.nfe, opt.nkz, decim=opt.decim_rate, dt=opt.dt,
               res=opt.resx, init_angle=opt.mask_path, device='cuda:0')

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, init_gain, gpu_ids)



##############################################################################
# Classes
##############################################################################
class SamplingLayerBspline2D(nn.Module):
    """
        The unit system are as follow:
            k: cycle / cm
            g: Gauss / cm
            s: Gauss / cm / s
            dt: s
            gamma: Hz / Gauss
            fov/res: cm
        Input:
            init_traj: [ndim, nacq]
            num_shots: number of shots
            nechos: num of points per acquisition
            decim: decimation rate
        Return:
            traj: [ndim, nshot, nechos]
            grad: [nshot, num_kernels]
            slew: [nshot, num_kernels]
    """

    def __init__(self, num_shots, nfe, decim=4, gamma=4.2576e+03, dt=4e-6, res=0.1, init_traj=None, ndims=2, gpu_ids=[],
                 ext=40):
        super(SamplingLayerBspline2D, self).__init__()
        self.nfe = nfe
        self.num_kernels = nfe // decim
        self.ndims = ndims
        self.num_shots = num_shots
        self.coeff = torch.ones(self.ndims, num_shots, self.num_kernels).to(gpu_ids[0])
        self.kmax = 1 / res
        self.gamma = gamma
        self.dt = dt
        self.decim = decim

        if decim > 1:
            # Paddind length
            self.ext = ext
            # build B:
            self.B, self.dB1, self.dB2 = bspline2_1ndsynth(torch.ones(self.num_kernels), nfe, dt, gamma, self.ext)
            self.B = self.B.to(gpu_ids[0])
            self.dB1 = self.dB1.to(gpu_ids[0])
            self.dB2 = self.dB2.to(gpu_ids[0])
            B = self.B.permute(1, 0).cpu().numpy()

        traj_ref = np.load(init_traj)
        if len(traj_ref.shape) == 2:
            traj_ref = np.reshape(traj_ref, (self.ndims, self.num_shots, self.nfe))
        traj_ref = traj_ref / np.pi * self.kmax / 2
        if decim == 1:
            self.coeff = torch.tensor(traj_ref).to(dtype=self.coeff.dtype, device=gpu_ids[0])
        else:
            for ii in range(self.ndims):
                for jj in range(self.num_shots):
                    traj_ref_i = np.zeros(nfe + self.ext)
                    traj_ref_i[0:self.ext // 2] = traj_ref[ii, jj, 0]
                    traj_ref_i[nfe + self.ext // 2:nfe + self.ext] = traj_ref[ii, jj, nfe - 1]
                    traj_ref_i[self.ext // 2:nfe + self.ext // 2] = traj_ref[ii, jj, :]
                    self.coeff[ii, jj, :] = torch.tensor(np.linalg.lstsq(B, traj_ref_i)[0]).to(
                        dtype=self.coeff.dtype, device=self.coeff.device)
        self.coeff = torch.nn.Parameter(self.coeff)

    def forward(self, _):
        if self.decim == 1:
            self.traj = self.coeff * 1
            self.gradient = (self.traj[:, :, :-1] - self.traj[:, :, 1:]) / self.gamma / self.dt
            self.slew = (self.gradient[:, :, :-1] - self.gradient[:, :, 1:]) / self.dt
        else:
            self.traj = torch.matmul(self.coeff, self.B)[:, :, self.ext // 2:self.nfe + self.ext // 2]
            self.gradient = torch.matmul(self.coeff, self.dB1)
            self.slew = torch.matmul(self.coeff, self.dB2)
        self.traj = torch.reshape(self.traj, (self.ndims, self.num_shots * self.nfe)).unsqueeze(0)
        self.traj = self.traj / self.kmax * 2 * np.pi
        return self.traj, self.gradient, self.slew


class SamplingLayerBspline3D(nn.Module):
    """
        The unit system are as follow:
            k: cycle / cm
            g: Gauss / cm
            s: Gauss / cm / s
            dt: s
            gamma: Hz / Gauss
            fov/res: cm
        Input:
            init_traj: [ndim, nacq]
            num_shots: number of shots
            nechos: num of points per acquisition
            decim: decimation rate
        Return:
            traj: [ndim, nshot, nechos]
            grad: [ndim, nshot, num_kernels]
            slew: [ndim, nshot, num_kernels]
    """

    def __init__(self, num_shots, nfe, decim=4, gamma=4.2576e+03, dt=4e-6, res=[0.1, 0.1, 0.1], init_traj=None,
                 ext=40, device='cuda:0'):
        super(SamplingLayerBspline3D, self).__init__()
        self.nfe = nfe
        self.num_kernels = nfe // decim
        self.ndims = 3
        self.num_shots = num_shots
        self.coeff = torch.ones(self.ndims, num_shots, self.num_kernels)
        self.kmax = 1 / np.array(res)
        self.gamma = gamma
        self.dt = dt
        self.decim = decim
        self.device = device

        if decim > 1:
            # Paddind length
            self.ext = ext
            # build B:
            B, dB1, dB2 = bspline2_1ndsynth(torch.ones(self.num_kernels), nfe, dt, gamma, self.ext)
            self.register_buffer('B', B)
            self.register_buffer('dB1', dB1)
            self.register_buffer('dB2', dB2)
            B = self.B.permute(1, 0).cpu().numpy()
        # The shape of traj_ref should be [ndim, nshot, npoints]
        # Load the traj from pre-computed numpy file
        traj_ref = np.load(init_traj)
        if len(traj_ref.shape) == 2:
            traj_ref = np.reshape(traj_ref, (self.ndims, self.num_shots, self.nfe))
        for idim in range(self.ndims):
            traj_ref[idim, :, :] = traj_ref[idim, :, :] / np.pi * self.kmax[idim] / 2
        if decim == 1:
            self.coeff = torch.tensor(traj_ref).to(dtype=self.coeff.dtype, device=self.device)
        else:
            self.coeff = torch.zeros(self.ndims, num_shots, B.shape[1], device=self.device)
            for ii in range(self.ndims):
                for jj in range(self.num_shots):
                    traj_ref_i = np.zeros(nfe + self.ext)
                    traj_ref_i[0:self.ext // 2] = traj_ref[ii, jj, 0]
                    traj_ref_i[nfe + self.ext // 2:nfe + self.ext] = traj_ref[ii, jj, nfe - 1]
                    traj_ref_i[self.ext // 2:nfe + self.ext // 2] = traj_ref[ii, jj, :]
                    self.coeff[ii, jj, :] = torch.from_numpy(np.linalg.lstsq(B, traj_ref_i, rcond=None)[0]).to(
                        dtype=self.coeff.dtype, device=self.device)
        self.coeff = torch.nn.Parameter(self.coeff)

    def forward(self, _, lr=None):
        # Extract the locations for maximum gradient and slew rate
        if self.decim == 1:
            traj = self.coeff * 1
            gradient = (traj[:, :, :-1] - traj[:, :, 1:]) / self.gamma / self.dt
            slew = (gradient[:, :, :-1] - gradient[:, :, 1:]) / self.dt
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            traj = torch.matmul(self.coeff, self.B)[:, :, self.ext // 2:self.nfe + self.ext // 2]
            gradient = (traj[:, :, :-1] - traj[:, :, 1:]) / self.gamma / self.dt
            slew = (gradient[:, :, :-1] - gradient[:, :, 1:]) / self.dt

        for idim in range(self.ndims):
            traj[idim, :, :] = traj[idim, :, :] / self.kmax[idim] * 2 * np.pi
        traj = torch.reshape(traj, (self.ndims, self.num_shots * self.nfe)).unsqueeze(0)
        return traj, gradient, slew



class SOS(nn.Module):
    """
        Optimize the parameters of stack-of-star, isotropic(!). Experiment 3.2.2 in the SNOPY paper.
        The unit system are as follow:
            k: cycle / cm
            g: Gauss / cm
            s: Gauss / cm / s
            dt: s
            gamma: Hz / Gauss
            fov/res: cm
        Input:
            num_shots: number of shots in each kz
            num_slices: number of kz
        Return:
            traj: [ndim, nshot*nfe]
            grad: [ndim, nshot, num_kernels]
            slew: [ndim, nshot, num_kernels]

        Follow the rotation matrix definition in: https://en.wikipedia.org/wiki/Rotation_matrix
        R = R_z(alpha)R_y(beta)R_z(gamma)
    """

    def __init__(self, num_shots, nfe, nkz, decim=4, gamma=4.2576e+03, dt=4e-6, res=0.1, init_angle=None,
                 device='cuda:0'):
        super(SOS, self).__init__()
        self.nfe = nfe
        self.ndims = 3
        self.num_shots = num_shots
        self.kmax = 1 / res
        self.gamma = gamma
        self.dt = dt
        self.device = device
        self.nkz = nkz
        if init_angle is not None:
            init_angle = torch.tensor(np.load(init_angle))
        else:
            torch.manual_seed(0)
            init_angle = np.pi * torch.rand(self.nkz, self.num_shots)
        self.angle = torch.nn.Parameter(init_angle)

    def forward(self, _, lr=None):
        # Extract the locations for maximum gradient and slew rate
        # angle[0] is alpha
        self.spoke = (torch.linspace(-np.pi, np.pi, self.nfe) / np.pi * self.kmax / 2).to(dtype=self.angle.dtype,
                                                                                          device=self.angle.device)
        traj = torch.zeros([self.ndims, self.nkz, self.num_shots, self.nfe]).to(dtype=self.angle.dtype,
                                                                                device=self.angle.device)
        traj[0] = torch.cos(self.angle.clone().unsqueeze(-1)) * self.spoke
        traj[1] = torch.sin(self.angle.clone().unsqueeze(-1)) * self.spoke
        traj[2] = repeat(torch.linspace(-np.pi, np.pi, self.nkz) / np.pi * self.kmax / 2, 'h -> h w t',
                         w=self.num_shots, t=self.nfe).to(dtype=self.angle.dtype, device=self.angle.device)
        traj = traj.reshape(self.ndims, self.nkz * self.num_shots, self.nfe)
        gradient = (traj[:, :, :-1] - traj[:, :, 1:]) / self.gamma / self.dt
        slew = (gradient[:, :, :-1] - gradient[:, :, 1:]) / self.dt
        traj = traj / self.kmax * 2 * np.pi
        traj = torch.reshape(traj, (self.ndims, self.nkz * self.num_shots * self.nfe)).unsqueeze(0)
        return traj, gradient, slew