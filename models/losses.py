import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.autograd as autograd
import numpy as np
import torchvision.models as models
import util.util as util
from util.image_pool import ImagePool


###############################################################################
# Functions
###############################################################################


class PerceptualLoss():

    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def initialize(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss

def pns(slew, dt = 4e-6, chronaxie=359e-6, rheobase=26.5, alpha=0.370):
    '''
    The convolutional PNS constraint based on: 
    Schulte, R. F., & Noeske, R. (2015). Peripheral nerve stimulation‐optimal gradient waveform design. Magnetic resonance in medicine, 74(2), 518-522.
    :param slew: we follow the unit (Gauss/cm/s) in networks.py, and transform it to T/m/s
    :param dt: 4e-6s:
    :param chronaxie: s
    :param rheobase: T/s
    :param alpha: m
    :return:
    Scanner  Gradient coil   chronaxie rheobase alpha  gmax  smax
    MR750w   XRMW            360e-6    20.0     0.324  33    120
    MR750    XRM             334e-6    23.4     0.333  50    200
    HDx      TRM WHOLE       370e-6    23.7     0.344  23    77
    HDx      TRM ZOOM        354e-6    29.1     0.309  40    150
    UHP      HRMB            359d-6    26.5     0.370  100   200
    Premier  HRMW            642.4d-6  17.9     0.310  70    200
    '''
    slew = slew/100
    [ndim, nshot, nfe] = slew.shape
    slew_min = rheobase / alpha
    decay = 1/(torch.pow(chronaxie+(torch.range(0,nfe-1)-0.5)*dt,2).unsqueeze(0).unsqueeze(0)).to(slew)
    decay_padded = decay
    slew_padded = slew
    pt_1 = 100*dt*chronaxie/slew_min*torch.fft.ifft(torch.fft.ifftshift(torch.fft.fftshift(torch.fft.fft(decay_padded, dim=2))*torch.fft.fftshift(torch.fft.fft(slew_padded, dim=2))))
    return pt_1
