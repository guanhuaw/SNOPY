import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import scipy.io as sp
import numpy as np
import torch
from util.util import generate_mask_alpha, generate_mask_beta
import scipy.ndimage
from util.util import fft2, ifft2, cplx_to_tensor, complex_conj, complex_matmul, absolute, roll
import h5py
from mirtorch.linear import LinearMap, FFTCn, NuSense, Sense, FFTCn, Identity, Diff2dgram, Gmri, Wavelet2D, Diff3dgram
from skimage.transform import resize

'''
    Loader of Calgary brain dataset
'''


class calgarydataset(BaseDataset):
    def initialize(self, opt):
        self.cluster_indices = [[] for x in range(14 * 25)]
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase)
        self.mask_type = opt.mask_type
        self.A_paths = make_dataset(self.dir_A, opt.datalabel)
        self.A_paths = sorted(self.A_paths)
        self.A_size = len(self.A_paths)

    def __getitem__(self, index):
        if self.opt.model == 'PNS':
            A_path = self.A_paths[index % self.A_size]
            return {'path': A_path}
        else:
            A_path = self.A_paths[index % self.A_size]
            try:
                A_temp = h5py.File(A_path, 'r')
            except:
                print(A_path)
            I = torch.tensor(A_temp['I'][()], dtype=torch.cfloat)
            s = torch.tensor(A_temp['S'][()], dtype=torch.cfloat)
            A_temp.close()
            I = I / torch.max(torch.abs(I))
            ncoil, nx, ny, nz = s.shape
            if self.opt.isTrain:
                if nx == self.opt.nx:
                    shiftx = 0
                else:
                    shiftx = np.random.randint(nx - self.opt.nx)
                if ny == self.opt.ny:
                    shifty = 0
                else:
                    shifty = np.random.randint(ny - self.opt.ny)
                if nz == self.opt.nz:
                    shiftz = 0
                else:
                    shiftz = np.random.randint(nz - self.opt.nz)
            else:
                shiftx = nx//2-self.opt.nx//2
                shifty = ny // 2 - self.opt.ny // 2
                shiftz = nz // 2 - self.opt.nz // 2
            I = I[shiftx:shiftx + self.opt.nx, shifty:shifty + self.opt.ny, shiftz:shiftz + self.opt.nz]
            s = s[:, shiftx:shiftx + self.opt.nx, shifty:shifty + self.opt.ny, shiftz:shiftz + self.opt.nz]
            if self.opt.axial:
                I = I.permute(1, 2, 0)
                s = s.permute(0, 2, 3, 1)
            return {'I': I, 'path': A_path, 'smap': s}

    def __len__(self):
        return self.A_size

    def name(self):
        return 'skmteaDataset'
