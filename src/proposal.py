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
    def __init__(self, config):
        super(Proposal, self).__init__()
        self.width = config['window_width']
        self.height = config['window_height']
        self.region_width = config['region_width']
        self.region_height = config['region_height']
        self.stride = 1
        # using 5 layers PatchGAN
        self.receptive_field = 70.
        self.roialign = RoIAlign(self.region_height, self.region_width, transform_fpcoor=True)
        # use mask operation or not
        self.mask_opt = config['mask_operation']

    def _localize(self, score_map, input):
        """
        width range: (feature_width - w_width) / stride + 1
        :param score_map:
        :param input:
        :return:
        """
        batch_size = score_map.size(0)
        ax_tmp = np.zeros((batch_size, 3))
        pro_height = (score_map.size(2) - self.height) / self.stride + 1
        pro_width = (score_map.size(3) - self.width) / self.stride + 1

        for n in range(batch_size):
            for i in range(pro_width):
                for j in range(pro_height):
                    _x, _y = i * self.stride, j * self.stride
                    region_score = score_map[n, :, _x:_x + self.stride, _y:_y + self.stride].mean()
                    if ax_tmp[n][2] < region_score.cpu().data.numpy():
                        ax_tmp[n] = _x, _y, region_score.cpu().data.numpy()

        _img_stride = (input.size(2) - self.receptive_field) // score_map.size(2)
        ax_transform = ax_tmp[:, :2] * _img_stride + self.receptive_field
        return ax_transform

    def _mask_operation(self, real_AB, fake_AB, ax):
        # _ax = np.expand_dims(ax, axis=1)
        # _ax = np.repeat(_ax, real_AB.size(1), axis=1)
        mask = Variable(torch.zeros(real_AB.size(0), 6, real_AB.size(2), real_AB.size(3)).cuda())
        for i in range(real_AB.size(0)):
            x, y = ax[i, :].astype(int)
            mask[i, :, x:x + self.stride, y:y + self.stride] = 1.
        fake_ABm = fake_AB * mask + real_AB * (1 - mask)
        return fake_ABm

    def forward(self, real_AB, fake_AB, score_map, real_B, fake_B):
        ax = self._localize(score_map, real_AB)
        fake_Br, real_Br, real_ABr, fake_ABr = [], [], [], []
        for i in range(real_AB.size(0)):
            x, y = ax[i, :]
            # Takes all the image
            boxes = np.asarray([[y, x, y + self.region_height, x + self.region_width]], dtype=np.float32)
            box_index_data = np.asarray([0], dtype=np.int32)
            boxes = to_varabile(boxes, requires_grad=False, is_cuda=True)
            box_index = to_varabile(box_index_data, requires_grad=False, is_cuda=True)
            fake_Br.append(self.roialign(fake_B[i].view(-1, 3, real_B.size(2), real_B.size(3)), boxes, box_index))
            real_Br.append(self.roialign(real_B[i].view(-1, 3, real_B.size(2), real_B.size(3)), boxes, box_index))
            if not self.mask_opt:
                real_ABr.append(self.roialign(real_AB[i].view(-1, 6, real_B.size(2), real_B.size(3)), boxes, box_index))
                fake_ABr.append(self.roialign(fake_AB[i].view(-1, 6, real_B.size(2), real_B.size(3)), boxes, box_index))

        fake_Br, real_Br = torch.cat(fake_Br, dim=0), torch.cat(real_Br, dim=0)
        if self.mask_opt:
            return self._mask_operation(real_AB, fake_AB, ax), real_Br, fake_Br
        else:
            fake_ABr, real_ABr = torch.cat(fake_ABr, dim=0), torch.cat(real_ABr, dim=0)
            return fake_ABr, real_ABr, real_Br, fake_Br