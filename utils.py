# -*- coding: utf-8 -*-
"""Implements some utils

TODO:
"""
import os
import random

from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class Visualizer:
    def __init__(self, save_dir, show_step=10, image_size=30):
        self.save_dir = save_dir

        self.transform = transforms.Compose([transforms.ToPILImage()])

        self.show_step = show_step
        self.step = 0

        self.lr_image_ph = None
        self.hr_image_ph = None
        self.fake_hr_image_ph = None

    def show(self, inputsG, inputsD_real, inputsD_fake):

        self.step += 1
        if self.step == self.show_step:
            self.step = 0

            i = random.randint(0, inputsG.size(0) -1)

            lr_image = self.transform(inputsG[i])
            hr_image = self.transform(inputsD_real[i])
            fake_hr_image = self.transform(inputsD_fake[i])

            if self.lr_image_ph is None:
                self.lr_image_ph = self.lr_plot.imshow(lr_image)
                self.hr_image_ph = self.hr_plot.imshow(hr_image)
                self.fake_hr_image_ph = self.fake_plot.imshow(fake_hr_image)
            else:
                self.lr_image_ph.set_data(lr_image)
                self.hr_image_ph.set_data(hr_image)
                self.fake_hr_image_ph.set_data(fake_hr_image)

            self.figure.canvas.draw()

    def save(self, inputsG, inputsD_real, inputsD_fake, epoch, batch):

        self.step += 1
        if self.step == self.show_step:
            self.step = 0

            i = random.randint(0, inputsG.size(0) -1)

            lr_image = self.transform(inputsG[i])
            hr_image = self.transform(inputsD_real[i])
            fake_hr_image = self.transform(inputsD_fake[i])

            fig, axarr = plt.subplots(1, 3, figsize=(12,10))
            # fig.suptitle(str(caption).decode('utf8'), fontsize=9)
            axarr[0].imshow(lr_image, interpolation='none')
            axarr[0].set_title('lr image')
            axarr[0].axis('off')

            axarr[1].imshow(hr_image, interpolation='none')
            axarr[1].set_title('hr image')
            axarr[1].axis('off')

            axarr[2].imshow(fake_hr_image, interpolation='none')
            axarr[2].set_title('fake hr image')
            axarr[2].axis('off')

            plt.savefig(os.path.join(self.save_dir, '%d_%d.png') % (epoch, batch), bbox_inches='tight')
            print ("Saved samples to %s " % self.save_dir)
