import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.utils as vutils

from model.pix2pix_model import netG, netD_64, netD_128, netD_256


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def load_network(config):
    generator = netG(input_dim=3, num_filter=64, output_dim=3)
    # generator.apply(weights_init)
    print(generator)

    discriminators = []
    if config['branch'] > 0:
        discriminators.append(netD_64(input_dim=6, num_filter=64, output_dim=1))
    if config['branch'] > 1:
        discriminators.append(netD_128(input_dim=6, num_filter=64, output_dim=1))
    if config['branch'] > 2:
        discriminators.append(netD_256(input_dim=6, num_filter=64, output_dim=1))

    # for i in range(len(discriminators)):
    #    discriminators[i].apply(weights_init)

    if config['cuda']:
        generator.cuda()
        for i in range(len(discriminators)):
            discriminators[i].cuda()

    return generator, discriminators, len(discriminators)

def define_optimizer(netG, netsD):
    optimizer_D = []
    for i in range(len(netsD)):
        opt = optim.Adam(netsD[i].parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_D.append(opt)

    optimizer_G = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    return optimizer_G, optimizer_D

class GAN_Trainer(nn.Module):
    def __init__(self, config, dataloader):
        super(GAN_Trainer, self).__init__()
        self.config = config
        self.dataloader = dataloader

        self.G, self.D, self.num_D = load_network(config)
        self.optimizerG, self.optimizerD = define_optimizer(self.G, self.D)
        self.criterion_gan = nn.BCELoss()
        self.criterion_mse = nn.L1Loss()
        # make downsampled images
        self.downsample = nn.AvgPool2d(2, stride=2)

        if config['cuda']:
            self.criterion_mse.cuda()
            self.criterion_gan.cuda()

    def update_D(self, idx, iterations):

        netD, optimizerD = self.D[idx], self.optimizerD[idx]

        netD.zero_grad()
        # print(self.real_imgs[idx].size())
        # real_ab = torch.cat((self.real_imgs_A[idx], self.real_imgs_B[idx]), 1)
        real_logits = netD(self.real_imgs_B[idx], self.real_imgs_A[idx])
        # fake_ab = torch.cat((self.real_imgs_A[idx], self.fake_imgs[idx]), 1)
        fake_logits = netD(self.fake_imgs[idx].detach(), self.real_imgs_A[idx])
        real_labels = Variable(self.real_labels.data.resize_(real_logits.size()).fill_(1))
        fake_labels = Variable(self.fake_labels.data.resize_(fake_logits.size()).fill_(0))

        errD_real = self.criterion_gan(real_logits, real_labels)
        errD_fake = self.criterion_gan(fake_logits, fake_labels)

        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()
        if iterations % 100 == 0:
            pass
        return errD

    def update_G(self, iterations):
        self.G.zero_grad()
        errG_total = 0.
        for i in range(self.num_D):
            netD = self.D[i]
            # fake_ab = torch.cat((self.real_imgs_A[i], self.fake_imgs[i]), 1)
            outputs = netD(self.fake_imgs[i], self.real_imgs_A[i])
            real_labels_ = Variable(torch.FloatTensor(outputs.size()).fill_(1))
            real_labels_v = real_labels_.cuda()
            errG = self.criterion_gan(outputs, real_labels_v)
            err_mse = self.criterion_mse(self.fake_imgs[i], self.real_imgs_B[i])
            errG_total = errG_total + errG + 50. * err_mse
        errG_total.backward()
        self.optimizerG.step()
        if iterations % 100 == 0:
            pass
        return errG_total

    def train(self):
        for i in range(self.num_D):
            print(self.D[i])
        batchsize = 1

        self.real_labels = \
            Variable(torch.FloatTensor(batchsize).fill_(1))
        self.fake_labels = \
            Variable(torch.FloatTensor(batchsize).fill_(0))
        if self.config['cuda']:
            self.real_labels = self.real_labels.cuda()
            self.fake_labels = self.fake_labels.cuda()

        for epoch in range(self.config['niter']):

            for i, (input, target) in enumerate(self.dataloader):

                input_A = Variable(input.cuda())
                input_B = Variable(target.cuda())
                # make downsampled input
                self.real_imgs_A = [input_A]
                for _ in range(self.num_D - 1):
                    real_img = self.downsample(self.real_imgs_A[-1])
                    self.real_imgs_A.append(real_img)
                self.real_imgs_A.reverse()

                self.real_imgs_B = [input_B]
                for _ in range(self.num_D-1):
                    real_img = self.downsample(self.real_imgs_B[-1])
                    self.real_imgs_B.append(real_img)
                self.real_imgs_B.reverse()
                # (1) generate fake images
                self.fake_imgs = self.G(input_A)

                # (2) update D networks
                errD_total = 0.
                for idx in range(self.num_D):
                    errd = self.update_D(idx, i)
                    errD_total += errd

                # (3) update G networks
                errG_total = self.update_G(i)

                if i % 10 == 0:
                    print('[%d]|[%d] loss_D: %.4f loss_G: %.4f' %(epoch, self.config['niter'], errD_total, errG_total))
                    
                    vutils.save_image(input_A.data,
                              '%s/real_samples_A_epoch_%03d.png' % (self.config['outf'], epoch),
                              normalize=True)
                    vutils.save_image(input_B.data,
                              '%s/real_samples_B_epoch_%03d.png' % (self.config['outf'], epoch),
                              normalize=True)
                    vutils.save_image(self.fake_imgs[0].data,
                              '%s/fake_samples_0_epoch_%03d.png' % (self.config['outf'], epoch),
                              normalize=True)
                    vutils.save_image(self.fake_imgs[1].data,
                              '%s/fake_samples_1_epoch_%03d.png' % (self.config['outf'], epoch),
                              normalize=True)
                    vutils.save_image(self.fake_imgs[2].data,
                              '%s/fake_samples_2_epoch_%03d.png' % (self.config['outf'], epoch),
                              normalize=True)
            # save the generator
            if epoch % 40 == 0:
                torch.save(self.G.state_dict(), '%s/generator_epoch_%d.pkl' % (self.config['outf'], epoch))
                # torch.save(D.state_dict(), model_dir + 'discriminator_param.pkl')

    def test(self):
        """
        :return:
        """
        self.G.eval()

        for i, (input, target) in enumerate(self.dataloader):

            input_A = Variable(input.cuda())
            input_B = Variable(target.cuda())
            # make downsampled input
            self.real_imgs_A = [input_A]
            for _ in range(self.num_D - 1):
                real_img = self.downsample(self.real_imgs_A[-1])
                self.real_imgs_A.append(real_img)
            self.real_imgs_A.reverse()

            self.real_imgs_B = [input_B]
            for _ in range(self.num_D - 1):
                real_img = self.downsample(self.real_imgs_B[-1])
                self.real_imgs_B.append(real_img)
            self.real_imgs_B.reverse()
            # (1) generate fake images
            self.fake_imgs = self.G(input_A)

            vutils.save_image(input_A.data,
                              '%s/real_samples_A_%03d.png' % (self.config['outf_test'], i),
                              normalize=True, padding=0)
            vutils.save_image(input_B.data,
                              '%s/real_samples_B_%03d.png' % (self.config['outf_test'], i),
                              normalize=True, padding=0)
            vutils.save_image(self.fake_imgs[0].data,
                              '%s/fake_samples_0_%3d.png' % (self.config['outf_test'], i),
                              normalize=True, padding=0)
            vutils.save_image(self.fake_imgs[1].data,
                              '%s/fake_samples_1_%03d.png' % (self.config['outf_test'], i),
                              normalize=True, padding=0)
            vutils.save_image(self.fake_imgs[2].data,
                              '%s/fake_samples_2_%03d.png' % (self.config['outf_test'], i),
                              normalize=True, padding=0)



