import numpy as np
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.utils as vutils

from model.networks import ResnetGenerator, patchD, MultiscaleDiscriminator, LocalEnhancer, VGGLoss
from roi_align.roi_align import RoIAlign
from src.proposal import Proposal
from utils.tools import weights_init, tensor2label


def to_varabile(arr, requires_grad=False, is_cuda=True):
    tensor = torch.from_numpy(arr)
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var


class trainer_gan(nn.Module):
    def __init__(self, config, data_loader, resume_epoch):
        super(trainer_gan, self).__init__()
        self.config = config
        self.input_nc = config['input_nc']
        self.batchSize = config['batchSize']
        self.real_A = torch.FloatTensor(self.batchSize, self.input_nc, config['fineSizeH'], config['fineSizeW'])
        self.real_B = torch.FloatTensor(self.batchSize, self.input_nc, config['fineSizeH'], config['fineSizeW'])
        self.label = torch.FloatTensor(self.batchSize)
        self.label_r = torch.FloatTensor(self.batchSize)
        self.real_label = 1.
        self.fake_label = 0.
        self.old_lr = config['lr']
        self.resume_epoch = resume_epoch
        # using which netG
        if config['netG'] == 'resnet':
            self.netG = ResnetGenerator(config['input_nc'], config['output_nc'], config['ngf'], use_dropout=False, n_blocks=9)
        elif config['netG'] == 'dilated':
            # self.netG = dilatedGenerator
            pass
        elif config['netG'] == 'local':
            self.netG = LocalEnhancer(config['input_nc'], config['output_nc'])
        # using which netD
        if config['netD'] == 'patchd':
            self.netD = patchD(config['input_nc'], config['output_nc'], config['ndf'])
        elif config['netD'] == 'multiscale':
            self.netD = MultiscaleDiscriminator(config['input_nc'], config['output_nc'], config['ndf'], use_sigmoid=False)
        if config['reviser'] == 'patchd':
            self.netD_r = patchD(config['input_nc'], config['output_nc'], config['ndf'])

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
        self.criterion_vgg = VGGLoss()

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

    #Encode label images
    def encode_input(self, label_map):
        size = label_map.size()
        oneHot_size = (size[0], 35, size[2], size[3])
        input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
        input_label = Variable(input_label, volatile=False)
        return input_label

    def update_learning_rate(self, epoch):
        lrd = self.config['lr'] / 100
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def train(self, epoch):
        for i, image in enumerate(self.data_loader):
            if self.config['AtoB'] == 'AtoB':
                imgA = self.encode_input(image[0])
                imgB = image[1]
            real_A, real_B = Variable(imgA), Variable(imgB.cuda())
            if self.config['gan_type'] == 'CGAN':
                fake_B, fake_B_, fake_Br, real_Br = self.update_cgan(real_A, real_B)

            # print the logging information
            if i % 100 == 0:
                losses = self.get_current_losses()
                message = '([%d/%d][%d/%d])' % (epoch, self.config['nepoch'], i, len(self.data_loader))
                for k, v in losses.items():
                    message += '%s: %.3f ' % (k, v)
                print(message)

            # visualization
            if i % 20 == 0:
                vutils.save_image(tensor2label(image[0][0], 35), '%s/realA_epoch_%03d.png' % (self.config['outf'], epoch),
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
        # save the generator
        if epoch % 10 == 0:
            torch.save(self.netG.state_dict(), '%s/generator_epoch_%d.pkl' % (self.config['outf'], epoch))
        if epoch % 199 == 0:
            torch.save(self.netG.state_dict(), '%s/generator_epoch_%d.pkl' % (self.config['outf'], epoch))

    def dis_MS(self, netD, real_AB, fake_AB):
        output_r = netD(real_AB)
        output_f = netD(fake_AB.detach())
        loss_D = 0.
        for i, (out_r, out_f) in enumerate(zip(output_r, output_f)):
            real_label = torch.FloatTensor(out_r.size()).fill_(self.real_label).cuda()
            fake_label = torch.FloatTensor(out_f.size()).fill_(self.fake_label).cuda()
            loss_D_real = self.criterion_mse(out_r, real_label)
            loss_D_fake = self.criterion_mse(out_f, fake_label)
            loss_D += (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def gen_MS(self, fake_AB):
        output = self.netD(fake_AB)
        loss_g = 0.
        for i, out in enumerate(output):
            label = torch.FloatTensor(out.size()).fill_(self.real_label).cuda()
            loss_g += self.criterion_mse(out, label)
        return loss_g

    def update_cgan(self, real_A, real_B):

        real_AB = torch.cat((real_A, real_B), 1)
        # train netD
        self.netD.zero_grad()
        fake_B = self.netG(real_A)
        fake_AB = torch.cat((real_A, fake_B), 1)
        self.loss_d = self.dis_MS(self.netD, real_AB, fake_AB)
        self.optimizer_D.step()

        # train netG
        self.netG.zero_grad()
        self.errGAN = self.gen_MS(fake_AB)
        self.errL1 = self.criterion_l1(fake_B, real_B)
        self.errVGG = self.criterion_vgg(fake_B, real_B)
        self.loss_g = self.errGAN + 10. * self.errL1 + self.errVGG * 10.
        self.loss_g.backward()
        self.optimizer_G.step()

        # training with reviser
        for n_step in range(self.config['n_step']):
            fake_B_ = self.netG(real_A)
            fake_AB = torch.cat((real_A, fake_B_), 1)
            output, _, _ = self.netD(fake_AB.detach())
            # proposal
            fake_ABm, real_Br, fake_Br = self.proposal(real_AB, fake_AB, output, real_B, fake_B_)
            # train with real
            self.netD_r.zero_grad()
            output_r = self.netD_r(real_AB)
            self.label_r.data.resize_(output_r.size())
            labelv_r = Variable(self.label_r.data.fill_(self.real_label))
            errD_real_r = self.criterion_mse(output_r, labelv_r)
            errD_real_r.backward()

            # train with fake
            labelv_r = Variable(self.label_r.data.fill_(self.fake_label))
            output_r = self.netD_r(fake_ABm.detach())
            errD_fake_r = self.criterion_mse(output_r, labelv_r)
            errD_fake_r.backward()

            self.loss_dr = (errD_real_r + errD_fake_r) / 2
            self.optimizer_D_r.step()

            # train Generator with reviser
            self.netG.zero_grad()
            labelv_r = Variable(labelv_r.data.fill_(self.real_label))
            output_r = self.netD_r(fake_ABm)
            self.errGAN_r = self.criterion_mse(output_r, labelv_r)
            self.errL1_r = self.criterion_l1(fake_Br, real_Br)
            self.loss_gr = self.errGAN_r + 10 * self.errL1_r
            self.loss_gr.backward()
            self.optimizer_G.step()

        return fake_B, fake_B_, fake_Br, real_Br

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

    def resume(self):
        self.netG.load_state_dict(torch.load('%s/netG_epoch_%03d.pth' % (self.config['outf'], self.resume_epoch)))
        self.netD.load_state_dict(torch.load('%s/netD_epoch_%03d.pth' % (self.config['outf'], self.resume_epoch)))
        self.netD_r.load_state_dict(torch.load('%s/netD_r_epoch_%03d.pth' % (self.config['outf'], self.resume_epoch)))

    def test(self):
        self.netG.eval()
        save_path = os.path.join(self.config['outf'], 'test')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for i, image in enumerate(self.data_loader):
            if self.config['AtoB'] == 'AtoB':
                imgA = self.encode_input(image[0])
                imgB = image[1]
            else:
                imgA = image[1]
                imgB = self.encode_input(image[0])

            real_A, real_B = Variable(imgA.cuda()), Variable(imgB.cuda())
            fake_B = self.netG(real_A)

            vutils.save_image(tensor2label(image[0][0], 35), '%s/realA_%03d.png' % (save_path, i),
                              normalize=True)
            vutils.save_image(real_B.data, '%s/realB_%03d.png' % (save_path, i),
                              normalize=True)
            vutils.save_image(fake_B.data, '%s/fakeB_%03d.png' % (save_path, i),
                              normalize=True)

class trainer_stackgan():
    def __init__(self):
        super(trainer_stackgan, self).__init__()