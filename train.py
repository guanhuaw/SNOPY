import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
import torch
import numpy as np
import os

if __name__ == '__main__':
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    opt.phase = 'val'
    batchSize = opt.batchSize
    opt.batchSize = 1  # test code only supports batchSize = 1
    data_loader_evaluate = CreateDataLoader(opt)
    dataset_evaluate = data_loader_evaluate.load_data()
    dataset_size_evaluate = len(data_loader_evaluate)
    opt.phase = 'train'
    opt.batchSize = batchSize
    print('#evaluating images = %d' % dataset_size_evaluate)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()
            # print('max',torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()


            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))

                model.save_networks('latest')

                if hasattr(model, 'ktraj'):
                    model.save_traj(epoch, total_steps)


            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        if epoch % opt.val_epoch_freq == 0:
            PSNR = []
            with torch.no_grad():
                for i, data in enumerate(dataset):
                    iter_start_time = time.time()
                    model.set_input(data)
                    model.test()
                    losses = model.get_current_losses()
                    for k, v in losses.items():
                        if k == 'PSNR':
                            PSNR = np.append(PSNR, v)
            print('Validate the model at the end of epoch %d, iters %d psnr%f' % (epoch, total_steps, np.sum(PSNR) / (i + 1)))


        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

