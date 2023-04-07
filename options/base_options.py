import argparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--dataroot', required=True,
                            help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--input_nc', type=int, default=2, help='# of input image channels of NN')
        parser.add_argument('--output_nc', type=int, default=2, help='# of output image channels of NN')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--which_model_netD_I', type=str, default='samplingBspline',
                            help='selects model for sampling trajectory')
        parser.add_argument('--which_model_netG_I', type=str, default='resnet_9blocks',
                            help='selects model to use for netG')
        parser.add_argument('--n_layers_D_I', type=int, default=2, help='only used if which_model_netD == n_layers')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--name', type=str, default='experiment_name',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--dataset_mode', type=str, default='unaligned',
                            help='chooses how datasets are loaded. [unaligned | aligned | single]')
        parser.add_argument('--model', type=str, default='cycle_gan',
                            help='chooses which model to use. cycle_gan, pix2pix, test')
        parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--norm', type=str, default='instance',
                            help='instance normalization or batch normalization')
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--display_winsize', type=int, default=1024, help='display window size')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost",
                            help='visdom server of the web display')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop',
                            help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        parser.add_argument('--no_flip', action='store_true',
                            help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--init_type', type=str, default='normal',
                            help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str,
                            help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{which_model_netG}_size{loadSize}')
        parser.add_argument('--loss_content_I_l1', type=float, default=0, help='content loss, l1')
        parser.add_argument('--loss_content_I_l2', type=float, default=0, help='content loss, l2')
        parser.add_argument('--loss_pns', type=float, default=1, help='The ratio of SSIM loss')
        parser.add_argument('--loss_slew', type=float, default=1, help='The ratio of SSIM loss')
        parser.add_argument('--loss_grad', type=float, default=1, help='The ratio of gradient strength loss')
        parser.add_argument('--loss_contrast', type=float, default=1000000, help='The ratio of image contrast penalty')
        parser.add_argument('--loss_pi', type=float, default=0, help='The ratio of maximum trajectory penalty')
        parser.add_argument('--mask_path', type=str, default=None, help='type of kspace mask')
        parser.add_argument('--angle_path', type=str, default=None, help='type of rotation angle')
        parser.add_argument('--mask_type', type=str, default='random_alpha',
                            help='type of kspace mask: uniform random_alpha mat')
        parser.add_argument('--alpha', type=float, default=1.0, help='the ratio of original kspace')
        parser.add_argument('--beta', type=float, default=1, help='the ratio of density layer')
        parser.add_argument('--noise_level', type=float, default=0,
                            help='the magnitude of additive noise to the k-space')
        parser.add_argument('--num_blocks', type=int, default=10, help='number of blocks in unrolled networks')
        parser.add_argument('--CGtol', type=float, default=0.000001, help='convergence tolerance in CG')
        parser.add_argument('--CGlambda', type=float, default=0.001, help='lambda in CG')
        parser.add_argument('--datalabel', type=str, default='',
                            help='certain label of file name to be included in the dataset')
        parser.add_argument('--nx', type=int, default=48, help='nx')
        parser.add_argument('--ny', type=int, default=48, help='ny')
        parser.add_argument('--nz', type=int, default=48, help='ny')
        parser.add_argument('--datamode', type=str, default='magic', help='which dataset')
        parser.add_argument('--num_shots', type=int, default=5, help='how many shots')
        parser.add_argument('--nfe', type=int, default=320, help='how many sampling points per echo')
        parser.add_argument('--num_plane', type=int, default=64, help='planes for multi-axis spiral')
        parser.add_argument('--decim_rate', type=int, default=4, help='how many folds to accelerate')
        parser.add_argument('--dt', type=float, default=0.000004, help='dwell time')
        parser.add_argument('--resx', type=float, default=0, help='resolution of x direction in cm')
        parser.add_argument('--resy', type=float, default=0, help='resolution of y direction in cm')
        parser.add_argument('--resz', type=float, default=0, help='resolution of z direction in cm')
        parser.add_argument('--ReconVSTraj', type=int, default=10, help='how many updates of recon against sampling')
        parser.add_argument('--no_global_residual', action='store_true', help='add residual connection for unet/didn')
        parser.add_argument('--padding', type=int, default=40, help='how many padding for the b-spline kernels')
        parser.add_argument('--gradmax', type=float, default=5, help='maximum gradient')
        parser.add_argument('--slewmax', type=float, default=20000, help='maximum slewrate')
        parser.add_argument('--pth', type=float, default=100, help='pns threshold')
        parser.add_argument('--use_rough', action='store_true', help='use QPWLS instead of CG-SENSE.')
        parser.add_argument('--grid_size', type=float, default=2, help='grid size for nufft')
        parser.add_argument('--numpoints', type=int, default=6, help='number of points in nufft')
        parser.add_argument('--iso_constraint', action='store_true',
                            help='use the isotropic constraint on the slew rate')
        parser.add_argument('--sgld', action='store_true', help='use sgld as the optimizer')
        parser.add_argument('--contrast_condition', type=str, default=None, help="constraint on image contrast")
        parser.add_argument('--axial', action='store_true', help='filp the calgary dataset')

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with the new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):

        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
