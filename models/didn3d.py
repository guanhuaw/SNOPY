"""
Copyright (c) 2019 Imperial College London.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from iunets import iUNet
from iunets.layers import AdditiveCoupling


class PixelShuffle3d(nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''

    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height,
                                             in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)


class _Residual_Block(nn.Module):
    def __init__(self, num_chans=64):
        super(_Residual_Block, self).__init__()
        bias = True
        # res1
        self.conv1 = nn.Conv3d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv3d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu4 = nn.PReLU()
        # res1
        # concat1

        self.conv5 = nn.Conv3d(num_chans, num_chans * 2, kernel_size=3, stride=2, padding=1, bias=bias)
        self.relu6 = nn.PReLU()

        # res2
        self.conv7 = nn.Conv3d(num_chans * 2, num_chans * 2, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu8 = nn.PReLU()
        # res2
        # concat2

        self.conv9 = nn.Conv3d(num_chans * 2, num_chans * 4, kernel_size=3, stride=2, padding=1, bias=bias)
        self.relu10 = nn.PReLU()

        # res3
        self.conv11 = nn.Conv3d(num_chans * 4, num_chans * 4, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu12 = nn.PReLU()
        # res3

        self.conv13 = nn.Conv3d(num_chans * 4, num_chans * 16, kernel_size=1, stride=1, padding=0, bias=bias)
        self.up14 = PixelShuffle3d(2)

        # concat2
        self.conv15 = nn.Conv3d(num_chans * 4, num_chans * 2, kernel_size=1, stride=1, padding=0, bias=bias)
        # res4
        self.conv16 = nn.Conv3d(num_chans * 2, num_chans * 2, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu17 = nn.PReLU()
        # res4

        self.conv18 = nn.Conv3d(num_chans * 2, num_chans * 8, kernel_size=1, stride=1, padding=0, bias=bias)
        self.up19 = PixelShuffle3d(2)

        # concat1
        self.conv20 = nn.Conv3d(num_chans * 2, num_chans, kernel_size=1, stride=1, padding=0, bias=bias)
        # res5
        self.conv21 = nn.Conv3d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu22 = nn.PReLU()
        self.conv23 = nn.Conv3d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu24 = nn.PReLU()
        # res5

        self.conv25 = nn.Conv3d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        res1 = x
        out = self.relu4(self.conv3(self.relu2(self.conv1(x))))
        out = torch.add(res1, out)
        cat1 = out

        out = self.relu6(self.conv5(out))
        res2 = out
        out = self.relu8(self.conv7(out))
        out = torch.add(res2, out)
        cat2 = out

        out = self.relu10(self.conv9(out))
        res3 = out

        out = self.relu12(self.conv11(out))
        out = torch.add(res3, out)

        out = self.up14(self.conv13(out))

        out = torch.cat([out, cat2], 1)
        out = self.conv15(out)
        res4 = out
        out = self.relu17(self.conv16(out))
        out = torch.add(res4, out)

        out = self.up19(self.conv18(out))

        out = torch.cat([out, cat1], 1)
        out = self.conv20(out)
        res5 = out
        out = self.relu24(self.conv23(self.relu22(self.conv21(out))))
        out = torch.add(res5, out)

        out = self.conv25(out)
        out = torch.add(out, res1)

        return out


def my_module_fn(in_channels, **kwargs):
    channel_split_pos = in_channels // 2

    conv_layer = nn.Conv3d(in_channels=channel_split_pos,
                           out_channels=in_channels - channel_split_pos,
                           kernel_size=3,
                           padding=1)
    nonlinearity = nn.PReLU()
    F = nn.Sequential(conv_layer, nonlinearity)

    return AdditiveCoupling(F, channel_split_pos)


class StandardBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_in_channels,
                 num_out_channels,
                 depth=2,
                 zero_init=True,
                 normalization="instance",
                 **kwargs):
        super(StandardBlock, self).__init__()

        conv_op = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dim - 1]

        self.seq = nn.ModuleList()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels

        for i in range(depth):

            current_in_channels = max(num_in_channels, num_out_channels)
            current_out_channels = max(num_in_channels, num_out_channels)

            if i == 0:
                current_in_channels = num_in_channels
            if i == depth - 1:
                current_out_channels = num_out_channels

            self.seq.append(
                conv_op(
                    current_in_channels,
                    current_out_channels,
                    3,
                    padding=1,
                    bias=False))
            # torch.nn.init.kaiming_uniform_(self.seq[-1].weight,
            #                                a=0.01,
            #                                mode='fan_out',
            #                                nonlinearity='leaky_relu')
            torch.nn.init.normal_(self.seq[-1].weight,
                                  0.0,
                                  0.000002)
            if normalization == "instance":
                norm_op = [nn.InstanceNorm1d,
                           nn.InstanceNorm2d,
                           nn.InstanceNorm3d][dim - 1]
                self.seq.append(norm_op(current_out_channels, affine=True))

            elif normalization == "group":
                self.seq.append(
                    nn.GroupNorm(
                        np.min(1, current_out_channels // 8),
                        current_out_channels,
                        affine=True)
                )

            elif normalization == "batch":
                norm_op = [nn.BatchNorm1d,
                           nn.BatchNorm2d,
                           nn.BatchNorm3d][dim - 1]
                self.seq.append(norm_op(current_out_channels, eps=1e-3))

            else:
                print("No normalization specified.")

            self.seq.append(nn.LeakyReLU(inplace=True))

        # Initialize the block as the zero transform, such that the coupling
        # becomes the coupling becomes an identity transform (up to permutation
        # of channels)
        if zero_init:
            torch.nn.init.zeros_(self.seq[-2].weight)
            torch.nn.init.zeros_(self.seq[-2].bias)

        self.F = nn.Sequential(*self.seq)

    def forward(self, x):
        x = self.F(x)
        return x


def get_num_channels(input_shape_or_channels):
    """
    Small helper function which outputs the number of
    channels regardless of whether the input shape or
    the number of channels were passed.
    """
    if hasattr(input_shape_or_channels, '__iter__'):
        return input_shape_or_channels[0]
    else:
        return input_shape_or_channels


def create_standard_module(in_channels, **kwargs):
    dim = kwargs.pop('dim', 2)
    depth = kwargs.pop('depth', 2)
    num_channels = get_num_channels(in_channels)
    num_F_in_channels = num_channels // 2
    num_F_out_channels = num_channels - num_F_in_channels

    module_index = kwargs.pop('module_index', 0)
    # For odd number of channels, this switches the roles of input and output
    # channels at every other layer, e.g. 1->2, then 2->1.
    if np.mod(module_index, 2) == 0:
        (num_F_in_channels, num_F_out_channels) = (
            num_F_out_channels, num_F_in_channels
        )
    return AdditiveCoupling(
        F=StandardBlock(
            dim,
            num_F_in_channels,
            num_F_out_channels,
            depth=depth,
            **kwargs),
        channel_split_pos=num_F_out_channels
    )


class DIDN3D(nn.Module):
    def __init__(self, opt, n_res_blocks=3):
        super().__init__()
        self.opt = opt
        self.conv1 = nn.Conv3d(opt.input_nc, opt.ngf, kernel_size=3, padding=1)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv3d(opt.ngf, opt.ngf, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.PReLU()
        self.n_res_blocks = n_res_blocks
        recursive = []
        for i in range(self.n_res_blocks):
            recursive.append(_Residual_Block(num_chans=opt.ngf))

        self.recursive = torch.nn.ModuleList(recursive)
        self.conv_mid = nn.Conv3d(opt.ngf * self.n_res_blocks, opt.ngf, kernel_size=1, stride=1, padding=0)
        self.relu3 = nn.PReLU()
        self.conv_mid2 = nn.Conv3d(opt.ngf, opt.ngf, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.PReLU()
        self.subpixel = PixelShuffle3d(2)
        self.conv_output = nn.Conv3d(opt.ngf // 8, opt.output_nc, kernel_size=3, stride=1, padding=1)
        self.pad_data = True

    def calculate_downsampling_padding3d(self, tensor, num_pool_layers):
        # calculate pad size
        factor = 2 ** num_pool_layers
        imshape = np.array(tensor.shape[-3:])
        paddings = np.ceil(imshape / factor) * factor - imshape
        paddings = paddings.astype(np.int) // 2
        p3d = (paddings[2], paddings[2], paddings[1], paddings[1], paddings[0], paddings[0])
        return p3d

    def pad3d(self, tensor, p3d):
        if np.any(p3d):
            # order of padding is reversed. that's messed up.
            tensor = F.pad(tensor, p3d)
        return tensor

    def unpad3d(self, tensor, shape):
        if tensor.shape == shape:
            return tensor
        else:
            return self.center_crop(tensor, shape)

    def center_crop(self, data, shape):
        """
        Apply a center crop to the input real image or batch of real images.

        Args:
            data (torch.Tensor): The input tensor to be center cropped. It should have at
                least 2 dimensions and the cropping is applied along the last two dimensions.
            shape (int, int): The output shape. The shape should be smaller than the
                corresponding dimensions of data.

        Returns:
            torch.Tensor: The center cropped image
        """
        assert 0 < shape[0] <= data.shape[-3]
        assert 0 < shape[1] <= data.shape[-2]
        assert 0 < shape[2] <= data.shape[-1]
        w_from = (data.shape[-3] - shape[0]) // 2
        h_from = (data.shape[-2] - shape[1]) // 2
        t_from = (data.shape[-1] - shape[2]) // 2
        w_to = w_from + shape[0]
        h_to = h_from + shape[1]
        t_to = t_from + shape[2]
        return data[..., w_from:w_to, h_from:h_to, t_from:t_to]

    def forward(self, x):
        if self.pad_data:
            orig_shape3d = x.shape[-3:]
            p3d = self.calculate_downsampling_padding3d(x, 3)
            x = self.pad3d(x, p3d)
        residual = x
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))

        recons = []
        for i in range(self.n_res_blocks):
            out = self.recursive[i](out) + out
            recons.append(out)

        out = torch.cat(recons, 1)

        out = self.relu3(self.conv_mid(out))
        residual2 = out
        out = self.relu4(self.conv_mid2(out))
        out = torch.add(out, residual2)

        out = self.subpixel(out)
        out = self.conv_output(out)
        if not self.opt.no_global_residual:
            out = torch.add(out, residual)

        if self.pad_data:
            out = self.unpad3d(out, orig_shape3d)

        return out
