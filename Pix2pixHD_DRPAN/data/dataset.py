# Custom dataset
from PIL import Image
import torch
import torch.utils.data as data
from torch.autograd import Variable
import os
import random

class Aligned_Dataset(data.Dataset):
    def __init__(self, image_dir, subfolder='train', direction='AtoB', transform=None, resize_scale=None,
                 crop_size=None, fliplr=False, Stack=True):
        """
        :param image_dir:
        :param subfolder:
        :param direction:
        :param transform:
        :param resize_scale:
        :param crop_size:
        :param fliplr:
        :param Stack: If stack, resize the images with different size to form the "image pairs": [A, B, A_re, B_re]
        """
        super(Aligned_Dataset, self).__init__()
        self.input_path = os.path.join(image_dir, subfolder)
        self.image_filenames = [x for x in sorted(os.listdir(self.input_path))]
        self.direction = direction
        self.transform = transform
        self.resize_scale = resize_scale
        self.crop_size = crop_size
        self.fliplr = fliplr
        self.stack = Stack

    ###########################Encode label images from pix2pixHD######################################
    # def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):
    # @staticmethod
    # def encode_input(self, label_map):
    #     # if self.opt.label_nc == 0:
    #     #     input_label = label_map.data.cuda()
    #     # else:
    #     # create one-hot vector for label map
    #     size = label_map.size()
    #     print(size)
    #     # oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
    #     oneHot_size = (size[0], 35, size[2], size[3])
    #     # input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    #     input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    #     input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
    #     # if self.opt.data_type == 16:
    #     #     input_label = input_label.half()
    #
    #     # get edges from instance map
    #     # if not self.opt.no_instance:
    #     #     inst_map = inst_map.data.cuda()
    #     #     edge_map = self.get_edges(inst_map)
    #     #     input_label = torch.cat((input_label, edge_map), dim=1)
    #     input_label = Variable(input_label, volatile=False)
    #
    #     # real images for training
    #     # if real_image is not None:
    #     #     real_image = Variable(real_image.data.cuda())
    #
    #     # # instance map for feature encoding
    #     # if self.use_features:
    #     #     # get precomputed feature maps
    #     #     if self.opt.load_features:
    #     #         feat_map = Variable(feat_map.data.cuda())
    #     #     if self.opt.label_feat:
    #     #         inst_map = label_map.cuda()
    #
    #     # return input_label, inst_map, real_image, feat_map
    #     return input_label
    #     #self.encode = encode_input()

    def __getitem__(self, index):
        # Load Image
        img_fn = os.path.join(self.input_path, self.image_filenames[index])
        img = Image.open(img_fn)
        if self.direction == 'AtoB':
            input = img.crop((0, 0, img.width // 2, img.height))
            target = img.crop((img.width // 2, 0, img.width, img.height))
        elif self.direction == 'BtoA':
            input = img.crop((img.width // 2, 0, img.width, img.height))
            target = img.crop((0, 0, img.width // 2, img.height))
        # preprocessing
        if self.resize_scale:
            input = input.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)
            target = target.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)

        if self.crop_size:
            x = random.randint(0, self.resize_scale - self.crop_size + 1)
            y = random.randint(0, self.resize_scale - self.crop_size + 1)
            input = input.crop((x, y, x + self.crop_size, y + self.crop_size))
            target = target.crop((x, y, x + self.crop_size, y + self.crop_size))
        if self.fliplr:
            if random.random() < 0.5:
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform is not None:
            input = self.transform(input)*255.0
            target = self.transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)