import numpy as np
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.utils as vutils

from model.networks import ResnetGenerator, patchD, Discriminator_r
from roi_align.roi_align import RoIAlign
from src.proposal import Proposal
from utils.tools import Local, weights_init


def to_varabile(arr, requires_grad=False, is_cuda=True):
    tensor = torch.from_numpy(arr)
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var


class trainer_gan(nn.Module):
    def __init__(self, config, data_loader):
        super(trainer_gan, self).__init__()
        self.config = config
        self.input_nc = config['input_nc']
        self.batchSize = config['batchSize']
        self.real_A = torch.FloatTensor(self.batchSize, self.input_nc, config['fineSize'], config['fineSize'])
        self.real_B = torch.FloatTensor(self.batchSize, self.input_nc, config['fineSize'], config['fineSize'])
        self.label = torch.FloatTensor(self.batchSize)
        self.label_r = torch.FloatTensor(self.batchSize)
        self.real_label = 1.
        self.fake_label = 0.
        # using which netG
        if config['netG'] == 'unet':
            # self.netG = unet
            pass
        elif config['netG'] == 'resnet':
            self.netG = ResnetGenerator(config['input_nc'], config['output_nc'], config['ngf'], use_dropout=True, n_blocks=9)
        elif config['netG'] == 'dilated':
            # self.netG = dilatedGenerator
            pass
        # using which netD
        if config['netD'] == 'patchd':
            self.netD = patchD(config['input_nc'], config['output_nc'], config['ndf'])
        if config['reviser'] == 'patchd':
            self.netD_r = patchD(config['input_nc'], config['output_nc'], config['ndf'])
        elif config['reviser'] == 'imaged':
            self.netD_r = Discriminator_r(config['input_nc'], config['output_nc'], config['ndf'])

        # weight init
        self.netG.apply(weights_init('gaussian'))
        self.netD.apply(weights_init('gaussian'))
        self.netD_r.apply(weights_init('gaussian'))

        # define the optimizer
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))
        self.optimizer_D_r = torch.optim.Adam(self.netD_r.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))
        # define the criterion
        self.criterion_gan = nn.BCELoss()
        self.criterion_l1 = nn.L1Loss()
        self.criterion_mse = nn.MSELoss()

        if config['cuda']:
            self.real_A = Variable(self.real_A.cuda())
            self.real_B = Variable(self.real_B.cuda())
            self.label = Variable(self.label.cuda())
            self.label_r = Variable(self.label_r.cuda())
        # roi_align like proposal
        self.proposal = Proposal(config)

        # loss information
        self.loss_names = ['loss_d', 'loss_g', 'errL1', 'loss_dr', 'loss_gr', 'errL1_r']
        self.data_loader = data_loader

        print(self.netG, self.netD, self.netD_r)

    def train(self, epoch):
        for i, image in enumerate(self.data_loader):
            if self.config['AtoB'] == 'AtoB':
                imgA = image[0]
                imgB = image[1]
            else:
                imgA = image[1]
                imgB = image[0]

            # train netD with real data
            # self.real_A.data.resize_(imgA.size()).copy_(imgA)
            # self.real_B.data.resize_(imgB.size()).copy_(imgB)
            real_A, real_B = Variable(imgA.cuda()), Variable(imgB.cuda())

            if self.config['gan_type'] == 'CGAN':
                fake_B, fake_B_, fake_Br, real_Br = self.update_cgan(real_A, real_B)

            # print the logging information
            if i % 10 == 0:
                losses = self.get_current_losses()
                message = '([%d/%d][%d/%d])' % (epoch, self.config['nepoch'], i, len(self.data_loader))
                for k, v in losses.items():
                    message += '%s: %.3f ' % (k, v)
                print(message)

            # visualization
            if i % 20 == 0:
                vutils.save_image(real_A.data, '%s/realA_epoch_%03d.png' % (self.config['outf'], epoch),
                                  normalize=True)
                vutils.save_image(real_B.data, '%s/realB_epoch_%03d.png' % (self.config['outf'], epoch),
                                  normalize=True)
                vutils.save_image(fake_B.data, '%s/fakeB_epoch_%03d.png' % (self.config['outf'], epoch),
                                  normalize=True)
                vutils.save_image(fake_B_.data, '%s/fakeB_5step_epoch_%03d.png' % (self.config['outf'], epoch),
                                  normalize=True)
                vutils.save_image(fake_Br.data, '%s/fakeB_local_epoch_%03d.png' % (self.config['outf'], epoch),
                                  normalize=True)
                vutils.save_image(real_Br.data, '%s/realB_local_epoch_%03d.png' % (self.config['outf'], epoch),
                                  normalize=True)

    def update_cgan(self, real_A, real_B):

        real_AB = torch.cat((real_A, real_B), 1)
        # train netD
        self.netD.zero_grad()
        output = self.netD(real_AB)
        self.label.data.resize_(output.size())
        self.label.data.fill_(self.real_label)
        self.errd_real = self.criterion_gan(output, self.label)
        self.errd_real.backward()

        fake_B = self.netG(real_A)
        self.label.data.fill_(self.fake_label)
        fake_AB = torch.cat((real_A, fake_B), 1)
        output = self.netD(fake_AB.detach())
        self.errd_fake = self.criterion_gan(output, self.label)
        self.errd_fake.backward()
        self.loss_d = (self.errd_fake + self.errd_real) / 2
        self.optimizer_D.step()

        # train netG
        self.netG.zero_grad()
        self.label.data.fill_(self.real_label)
        output = self.netD(fake_AB)
        self.errGAN = self.criterion_gan(output, self.label)
        self.errL1 = self.criterion_l1(fake_B, real_B)
        self.loss_g = self.errGAN + 100. * self.errL1
        self.loss_g.backward()
        self.optimizer_G.step()

        # training with reviser
        for n_step in range(self.config['n_step']):
            fake_B_ = self.netG(real_A)
            fake_AB = torch.cat((real_A, fake_B_), 1)
            output = self.netD(fake_AB.detach())
            # proposal
            fake_ABm, real_Br, fake_Br = self.proposal(real_AB, fake_AB, output, real_B, fake_B_)
            # train with real
            self.netD_r.zero_grad()
            output_r = self.netD_r(real_AB)
            self.label_r.data.resize_(output_r.size())
            labelv_r = Variable(self.label_r.data.fill_(self.real_label))
            errD_real_r = self.criterion_gan(output_r, labelv_r)
            errD_real_r.backward()

            # train with fake
            labelv_r = Variable(self.label_r.data.fill_(self.fake_label))
            output_r = self.netD_r(fake_ABm.detach())
            errD_fake_r = self.criterion_gan(output_r, labelv_r)
            errD_fake_r.backward()

            self.loss_dr = (errD_real_r + errD_fake_r) / 2
            self.optimizer_D_r.step()

            # train Generator with reviser
            self.netG.zero_grad()
            labelv_r = Variable(labelv_r.data.fill_(self.real_label))
            output_r = self.netD_r(fake_ABm)
            self.errGAN_r = self.criterion_gan(output_r, labelv_r)
            self.errL1_r = self.criterion_l1(fake_Br, real_Br)
            self.loss_gr = self.errGAN_r + 10 * self.errL1_r
            self.loss_gr.backward()
            self.optimizer_G.step()

        return fake_B, fake_B_, fake_Br, real_Br

    def update_wgan_gp(self):
        pass

    # return traning losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = getattr(self, name)
        return errors_ret

    def save(self, epoch):
        torch.save(self.netG.state_dict(), '%s/netG_epoch_%03d.pth' % (self.config['outf'], epoch))
        torch.save(self.netD.state_dict(), '%s/netD_epoch_%03d.pth' % (self.config['outf'], epoch))
        torch.save(self.netD_r.state_dict(), '%s/netD_r_epoch_%03d.pth' % (self.config['outf'], epoch))

    def test(self):
        self.netG.eval()
        save_path = os.path.join(self.config['outf'], 'test')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for i, image in enumerate(self.data_loader):
            if self.config['AtoB'] == 'AtoB':
                imgA = image[0]
                imgB = image[1]
            else:
                imgA = image[1]
                imgB = image[0]

            real_A, real_B = Variable(imgA.cuda()), Variable(imgB.cuda())
            fake_B = self.netG(real_A)

            vutils.save_image(real_A.data, '%s/realA_%03d.png' % (save_path, i),
                              normalize=True)
            vutils.save_image(real_B.data, '%s/realB_%03d.png' % (save_path, i),
                              normalize=True)
            vutils.save_image(fake_B.data, '%s/fakeB_%03d.png' % (save_path, i),
                              normalize=True)

class trainer_stackgan():
    def __init__(self):
        super(trainer_stackgan, self).__init__()









