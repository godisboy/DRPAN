import random
import sys
from optparse import OptionParser

import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn

from src.trainer import trainer_gan
from utils.tools import get_config
from data.dataset import Aligned_Dataset

parser = OptionParser()
parser.add_option('--config', type=str, help='net configuration')
parser.add_option('--cuda', action='store_true', help='enables_cuda')
parser.add_option('--gpu_ids', default=0, type=int, help='enables cuda')
parser.add_option('--manualSeed', type=int, help='manual seed')


def main(argv):
    (opt, args) = parser.parse_args(argv)
    print(opt)
    config = get_config(opt.config)

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print('Random Seed: ', opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
        torch.cuda.set_device(opt.gpu_ids)
    cudnn.benchmark = True

    # loading data set
    transform = transforms.Compose([transforms.Resize((config['fineSize'], config['fineSize'])),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    dataset = Aligned_Dataset(config['dataPath'], direction='AtoB', transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=True, num_workers=int(4))
    # setup model
    trainer = trainer_gan(config, train_loader)
    if opt.cuda:
        trainer.cuda()
    # training
    for epoch in range(config['nepoch']):
        trainer.train(epoch)


if __name__ == '__main__':
    main(sys.argv)







































