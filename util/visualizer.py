import numpy as np
import os
import ntpath
import time
from . import util
from . import html
from . import metrics
from PIL import Image
import h5py

# save image to the disk
def save_images(webpage, visuals, image_path, w):
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = im_data.data.cpu().numpy()
        image_name = '%s_%s.h5' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        hf = h5py.File(save_path, 'w')
        hf.create_dataset(label, data=im_data.data.cpu().numpy())
        hf.close()
        image_numpy = []
        if 'kfake' in label or 'kunder' in label or 'kreal' in label or 'ksimu' in label or 'k_simu_1' in label or 'ex_pattern' in label:
            # print(label)
            image_numpy = util.tensor2imk3D(im_data, w, w)
        elif 'traj' in label:
            image_traj = im_data[0, :, :].transpose(0, 1).data.cpu().float().numpy()
            image_traj = image_traj[0::10, :]
        elif 'slew' in label:
            image_slew = im_data[:, 0, :].transpose(0, 1).data.cpu().float().numpy()
        elif 'grad' in label:
            image_grad = im_data[:, 0, :].transpose(0, 1).data.cpu().float().numpy()
        elif 'pt1' in label:
            image_pt1 = im_data[:, 0, :].transpose(0, 1).data.cpu().float().numpy()
        elif 'pt' in label:
            image_pt = im_data[:, 0, :].transpose(0, 1).data.cpu().float().numpy()

        else:
            image_numpy = util.tensor2im3D(im_data, w, w)
        if len(image_numpy)>0:
            im = np.concatenate((image_numpy[0],image_numpy[1],image_numpy[2]), axis = 1)

            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)

            util.save_image(im, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)

    webpage.add_images(ims, txts, links, width=w)
    webpage.add_images(ims, txts, links, width=w)


class Visualizer():
    def __init__(self, opt):
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.nx = opt.display_nx
        self.ny = opt.display_ny
        self.name = opt.name
        self.opt = opt
        self.saved = False
        if self.display_id > 0:
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, raise_exceptions=True)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    def throw_visdom_connection_error(self):
        print('\n\nCould not connect to Visdom server (https://github.com/facebookresearch/visdom) for displaying training progress.\nYou can suppress connection to Visdom using the option --display_id -1. To install visdom, run \n$ pip install visdom\n, and start the server by \n$ python -m visdom.server.\n\n')
        exit(1)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, save_result):
        if self.display_id > 0:  # show images in the browser
            ncols = self.ncols
            # if ncols > 0:
            # ncols = min(ncols, len(visuals))
            ncols = 3
            h = self.nx
            w = self.ny
            table_css = """<style>
                    table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                    table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                    </style>""" % (w, h)
            title = self.name
            label_html = ''
            label_html_row = ''
            images = []

            idx = 0
            for label, image in visuals.items():
                image_numpy = []
                if 'kfake' in label or 'kunder' in label or 'kreal' in label or 'ksimu' in label or 'k_simu_1' in label or 'ex_pattern' in label:
                    # print(label)
                    image_numpy = util.tensor2imk3D(image,h,w)
                    label_html += '<td>%s</td>' % label
                elif 'traj' in label:
                    image_traj = image[0, :, :].transpose(0,1).data.cpu().float().numpy()
                    image_traj = image_traj[0::10, :]
                elif 'slew' in label:
                    image_slew = image[:, 0, :].transpose(0,1).data.cpu().float().numpy()
                elif 'grad' in label:
                    image_grad = image[:, 0, :].transpose(0,1).data.cpu().float().numpy()
                elif 'pt1' in label:
                    image_pt1 = image[:, 0, :].transpose(0,1).data.cpu().float().numpy()
                elif 'pt' in label:
                    image_pt = image[:, 0, :].transpose(0,1).data.cpu().float().numpy()

                else:
                    image_numpy = util.tensor2im3D(image,h,w)
                    label_html += '<td>%s</td>' % label
                # label_html_row += '<td>%s</td>' % label
                if len(image_numpy)>0:
                    for i in range(len(image_numpy)):
                        images.append(image_numpy[i].transpose([2, 0, 1]))
                # idx += 1
                # if idx % ncols == 0:

                # label_html_row = ''
            # white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
            # while idx % ncols != 0:
            #     images.append(white_image)
            #     label_html_row += '<td></td>'
            #     idx += 1
            # if label_html_row != '':
            #     label_html += '<tr>%s</tr>' % label_html_row
            # pane col = image row
            try:
                self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                padding=2, opts=dict(title=title + ' images'))
                label_html = '<table>%s</table>' % label_html
                self.vis.text(table_css + label_html, win=self.display_id + 2,
                              opts=dict(title=title + ' labels'))
                try:
                    image_traj
                except NameError:
                    image_traj = None
                try:
                    image_slew
                except NameError:
                    image_slew = None
                try:
                    image_grad
                except NameError:
                    image_grad = None
                try:
                    image_pt1
                except NameError:
                    image_pt1 = None
                try:
                    image_pt
                except NameError:
                    image_pt = None
                if image_traj is not None:
                    scatteropts= dict(markersize=1, markerborderwidth=0)
                    self.vis.scatter(image_traj, win=self.display_id + 3, opts=scatteropts)
                if image_slew is not None:
                    self.vis.line(image_slew, win=self.display_id + 4)
                if image_grad is not None:
                    self.vis.line(image_grad, win=self.display_id + 5)
                if image_pt is not None:
                    self.vis.line(image_pt, win=self.display_id + 6)
                if image_pt1 is not None:
                    self.vis.line(image_pt1, win=self.display_id + 7)

            except ConnectionError:
                self.throw_visdom_connection_error()


    # losses: dictionary of error labels and values
    def plot_current_losses(self, epoch, counter_ratio, opt, losses):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except ConnectionError:
            self.throw_visdom_connection_error()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, i, losses, t, t_data):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
