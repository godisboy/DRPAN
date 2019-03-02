import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from roi_align.roi_align import RoIAlign


def to_varabile(arr, requires_grad=False, is_cuda=True):
    tensor = torch.from_numpy(arr)
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var


class Proposal(nn.Module):
    def __init__(self):
        super(Proposal, self).__init__()
        self.width = 1
        self.height = 1
        self.region_width = 70
        self.region_height = 70
        self.stride = 1
        # using 5 layers PatchGAN
        self.receptive_field = 70.
        self.roialign = RoIAlign(self.region_height, self.region_width, transform_fpcoor=True)
        # use mask operation or not

    def _localize(self, score_map, input):
        """
        width range: (feature_width - w_width) / stride + 1
        :param score_map:
        :param input:
        :return:
        """
        batch_size = score_map.size(0)
        ax_tmp_fake = np.ones((batch_size, 3))
        ax_tmp_real = np.zeros((batch_size, 3))
        pro_height = (score_map.size(2) - self.height) / self.stride + 1
        pro_width = (score_map.size(3) - self.width) / self.stride + 1

        for n in range(batch_size):
            for i in range(pro_width):
                for j in range(pro_height):
                    _x, _y = i * self.stride, j * self.stride
                    region_score = score_map[n, :, _x:_x + self.stride, _y:_y + self.stride].mean()
                    if ax_tmp_real[n][2] < region_score.cpu().data.numpy():
                        ax_tmp_real[n] = _x, _y, region_score.cpu().data.numpy()
                    if ax_tmp_fake[n][2] > region_score.cpu().data.numpy():
                        ax_tmp_fake[n] = _x, _y, region_score.cpu().data.numpy()

        _img_stride = (input.size(2) - self.receptive_field) // score_map.size(2)
        ax_transform_fake = ax_tmp_fake[:, :2] * _img_stride + self.receptive_field
        ax_transform_real = ax_tmp_real[:, :2] * _img_stride + self.receptive_field
        return ax_transform_fake, ax_transform_real

    def forward_A(self, real_B, fake_B, real_A, score_map):
        ax_fake, ax_real = self._localize(score_map, real_B)
        fake_Br, real_Ar, fake_Bf, real_Af= [], [], [], []
        for i in range(real_B.size(0)):
            x, y = ax_fake[i, :]
            # Takes all the image
            boxes = np.asarray([[y, x, y + self.region_height, x + self.region_width]], dtype=np.float32)
            box_index_data = np.asarray([0], dtype=np.int32)
            boxes = to_varabile(boxes, requires_grad=False, is_cuda=True)
            box_index = to_varabile(box_index_data, requires_grad=False, is_cuda=True)
            fake_Bf.append(self.roialign(fake_B[i].view(-1, 3, real_B.size(2), real_B.size(3)), boxes, box_index))
            real_Af.append(self.roialign(real_A[i].view(-1, 3, real_A.size(2), real_A.size(3)), boxes, box_index))

        fake_Bf, real_Af = torch.cat(fake_Bf, dim=0), torch.cat(real_Af, dim=0)
        fake_ABf = torch.cat((real_Af, fake_Bf), 1)

        for i in range(real_B.size(0)):
            x, y = ax_real[i, :]
            # Takes all the image
            boxes = np.asarray([[y, x, y + self.region_height, x + self.region_width]], dtype=np.float32)
            box_index_data = np.asarray([0], dtype=np.int32)
            boxes = to_varabile(boxes, requires_grad=False, is_cuda=True)
            box_index = to_varabile(box_index_data, requires_grad=False, is_cuda=True)
            fake_Br.append(self.roialign(fake_B[i].view(-1, 3, real_B.size(2), real_B.size(3)), boxes, box_index))
            real_Ar.append(self.roialign(real_A[i].view(-1, 3, real_A.size(2), real_A.size(3)), boxes, box_index))

        fake_Br, real_Ar = torch.cat(fake_Br, dim=0), torch.cat(real_Ar, dim=0)
        real_ABr = torch.cat((real_Ar, fake_Br), 1)

        return fake_Br, real_Ar, fake_Bf, real_Af, fake_ABf, real_ABr

    def forward_B(self, real_A, fake_A, real_B, score_map):
        ax_fake, ax_real = self._localize(score_map, real_A)
        fake_Ar, real_Br, fake_Af, real_Bf = [], [], [], []
        for i in range(real_A.size(0)):
            x, y = ax_fake[i, :]
            # Takes all the image
            boxes = np.asarray([[y, x, y + self.region_height, x + self.region_width]], dtype=np.float32)
            box_index_data = np.asarray([0], dtype=np.int32)
            boxes = to_varabile(boxes, requires_grad=False, is_cuda=True)
            box_index = to_varabile(box_index_data, requires_grad=False, is_cuda=True)
            fake_Af.append(self.roialign(fake_A[i].view(-1, 3, real_A.size(2), real_A.size(3)), boxes, box_index))
            real_Bf.append(self.roialign(real_B[i].view(-1, 3, real_B.size(2), real_B.size(3)), boxes, box_index))

        fake_Af, real_Bf = torch.cat(fake_Af, dim=0), torch.cat(real_Bf, dim=0)
        fake_BAf = torch.cat((real_Bf, fake_Af), 1)

        for i in range(real_A.size(0)):
            x, y = ax_real[i, :]
            # Takes all the image
            boxes = np.asarray([[y, x, y + self.region_height, x + self.region_width]], dtype=np.float32)
            box_index_data = np.asarray([0], dtype=np.int32)
            boxes = to_varabile(boxes, requires_grad=False, is_cuda=True)
            box_index = to_varabile(box_index_data, requires_grad=False, is_cuda=True)
            fake_Ar.append(self.roialign(fake_A[i].view(-1, 3, real_A.size(2), real_A.size(3)), boxes, box_index))
            real_Br.append(self.roialign(real_B[i].view(-1, 3, real_B.size(2), real_B.size(3)), boxes, box_index))

        fake_Ar, real_Br = torch.cat(fake_Ar, dim=0), torch.cat(real_Br, dim=0)
        real_BAr = torch.cat((real_Br, fake_Ar), 1)

        return fake_Ar, real_Br, fake_Af, real_Bf, fake_BAf, real_BAr





    # def _mask_operation_R_B(self, real_A, fake_A, real_B, rec_B, ax):
    #     # _ax = np.expand_dims(ax, axis=1)
    #     # _ax = np.repeat(_ax, real_AB.size(1), axis=1)
    #     mask = Variable(torch.zeros(real_A.size(0), 3, real_A.size(2), real_A.size(3)).cuda())
    #     for i in range(real_A.size(0)):
    #         x, y = ax[i, :].astype(int)
    #         mask[i, :, x:x + int(self.receptive_field), y:y + int(self.receptive_field)] = 1.
    #     fake_mA = fake_A * mask + real_A * (1 - mask)
    #     real_Bl = real_B * mask
    #     rec_Bl = rec_B * mask
    #     return fake_mA, real_Bl, rec_Bl
    #
    # def _mask_operation_R_A(self, real_B, fake_B, real_A, rec_A, ax_fake, ax_real):
    #     # _ax = np.expand_dims(ax, axis=1)
    #     # _ax = np.repeat(_ax, real_AB.size(1), axis=1)
    #     mask_fake = Variable(torch.zeros(real_B.size(0), 3, real_B.size(2), real_B.size(3)).cuda())
    #     mask_real = Variable(torch.zeros(real_B.size(0), 3, real_B.size(2), real_B.size(3)).cuda())
    #     for i in range(real_B.size(0)):
    #         x, y = ax_fake[i, :].astype(int)
    #         mask_fake[i, :, x:x + int(self.receptive_field), y:y + int(self.receptive_field)] = 1.
    #     for i in range(real_B.size(0)):
    #         x, y = ax_real[i, :].astype(int)
    #         mask_real[i, :, x:x + int(self.receptive_field), y:y + int(self.receptive_field)] = 1.
    #     real_Al = real_A * mask
    #     rec_Al = rec_A * mask
    #     return real_Al, rec_Al
