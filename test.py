import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import ntpath
import numpy as np
import torch

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test

    img_dir = os.path.join(web_dir, 'images')
    log_name = os.path.join(img_dir, 'loss_log.txt')
    PSNR = []
    SSIM = []
    HFEN = []
    print(len(dataset))
    for i, data in enumerate(dataset):
        torch.cuda.reset_peak_memory_stats()
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        losses = model.get_current_losses()
        img_path = model.get_image_paths()
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, max(opt.nx,opt.ny,opt.nz))

        short_path = ntpath.basename(img_path[0])
        name = os.path.splitext(short_path)[0]
        message = name
        for k, v in losses.items():
            if k == 'PSNR':
                PSNR = np.append(PSNR,v)
            message += '%s: %.5f ' % (k, v)
        print(message)
        with open(log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    webpage.save()
    with open(log_name, "a") as log_file:
        log_file.write('averagepsnr%s\n' % (np.sum(PSNR) / i))
    psnr_name = os.path.join(img_dir, 'PSNR.npy')
    np.save(psnr_name, PSNR)
    print('psnr%f'%(np.sum(PSNR) / i))

