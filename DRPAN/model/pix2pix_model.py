import torch.nn as nn
import torch


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, activation=True, batch_norm=True):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
        self.activation = activation
        self.lrelu = torch.nn.LeakyReLU(0.2, True)
        self.batch_norm = batch_norm
        self.bn = torch.nn.BatchNorm2d(output_size)

    def forward(self, x):
        if self.activation:
            out = self.conv(self.lrelu(x))
        else:
            out = self.conv(x)

        if self.batch_norm:
            return self.bn(out)
        else:
            return out


class ConvTBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, batch_norm=True, dropout=False):
        super(ConvTBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding)
        self.bn = torch.nn.BatchNorm2d(output_size)
        self.drop = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU(True)
        self.batch_norm = batch_norm
        self.dropout = dropout

    def forward(self, x):
        if self.batch_norm:
            out = self.bn(self.deconv(self.relu(x)))
        else:
            out = self.deconv(self.relu(x))

        if self.dropout:
            return self.drop(out)
        else:
            return out

class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(),
            nn.Conv2d(channel_num, channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel_num)
        )

    def forward(self, input):
        res = input
        out = self.block(input)
        out += res
        return out

class Get_Image_netG(nn.Module):
    def __init__(self, ngf):
        super(Get_Image_netG, self).__init__()
        self.gf_dim = ngf
        self.img_G = nn.Sequential(
            nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.img_G(input)

class netG(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(netG, self).__init__()

        # Encoder
        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 8)
        self.conv5 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv6 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv7 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv8 = ConvBlock(num_filter * 8, num_filter * 8, batch_norm=False)
        # Decoder
        self.deconv1 = ConvTBlock(num_filter * 8, num_filter * 8, dropout=True)
        self.deconv2 = ConvTBlock(num_filter * 8 * 2, num_filter * 8, dropout=True)
        self.deconv3 = ConvTBlock(num_filter * 8 * 2, num_filter * 8, dropout=True)
        self.deconv4 = ConvTBlock(num_filter * 8 * 2, num_filter * 8)
        self.deconv5 = ConvTBlock(num_filter * 8 * 2, num_filter * 4)
        self.deconv6 = ConvTBlock(num_filter * 4 * 2, num_filter * 2)
        self.deconv7 = ConvTBlock(num_filter * 2 * 2, num_filter)
        self.deconv8 = ConvTBlock(num_filter * 2, output_dim, batch_norm=False)

        # Generator ResBlock
        self.num_res = 2
        self.Next_G2 = self.make_layer(256)
        self.Next_G3 = self.make_layer(128)

        self.Img_G1 = Get_Image_netG(128)
        self.Img_G2 = Get_Image_netG(64)

    def make_layer(self, channel_num):
        layers = []
        for i in range(self.num_res):
            layers += [ResBlock(channel_num)]
        return nn.Sequential(*layers)

    def forward(self, x):
        fake_imgs = []
        # Encoder
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)
        enc6 = self.conv6(enc5)
        enc7 = self.conv7(enc6)
        enc8 = self.conv8(enc7)
        # Decoder with skip-connections
        dec1 = self.deconv1(enc8)
        # (512) * 2 * 2
        dec1 = torch.cat([dec1, enc7], 1)
        dec2 = self.deconv2(dec1)
        # (512) * 4 * 4
        dec2 = torch.cat([dec2, enc6], 1)
        dec3 = self.deconv3(dec2)
        # (512) * 8 * 8
        dec3 = torch.cat([dec3, enc5], 1)
        dec4 = self.deconv4(dec3)
        # (512) * 16 * 16
        dec4 = torch.cat([dec4, enc4], 1)
        dec5 = self.deconv5(dec4)
        # (256) * 32 * 32
        dec5 = torch.cat([dec5, enc3], 1)
        dec6 = self.deconv6(dec5)
        fake_img1 = self.Img_G1(dec6)
        fake_imgs.append(fake_img1)
        # (128) * 64 * 64
        dec6 = torch.cat([dec6, enc2], 1)
        dec6 = self.Next_G2(dec6)
        dec7 = self.deconv7(dec6)
        fake_img2 = self.Img_G2(dec7)
        fake_imgs.append(fake_img2)
        # (64) * 128 * 128
        dec7 = torch.cat([dec7, enc1], 1)
        dec7 = self.Next_G3(dec7)
        dec8 = self.deconv8(dec7)
        # 3 * 256 * 256
        fake_img3 = torch.nn.Tanh()(dec8)
        fake_imgs.append(fake_img3)
        return fake_imgs


# define multiscale discriminator

class netD_256(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(netD_256, self).__init__()

        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 8)
        # add a layer for 512
        self.conv5 = ConvBlock(num_filter * 8, num_filter * 8, stride=1)
        self.conv6 = ConvBlock(num_filter * 8, output_dim, activation=False, stride=1, batch_norm=False)

    def forward(self, x, label):
        x = torch.cat([x, label], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        out = torch.nn.Sigmoid()(x)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal(m.conv.weight, mean, std)


class netD_128(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(netD_128, self).__init__()

        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)
        # add a layer for 256
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 8, stride=1)
        self.conv5 = ConvBlock(num_filter * 8, output_dim, activation=False, stride=1, batch_norm=False)

    def forward(self, x, label):
        x = torch.cat([x, label], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        out = torch.nn.Sigmoid()(x)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal(m.conv.weight, mean, std)


class netD_64(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(netD_64, self).__init__()

        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4, stride=2)
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 4, stride=2)
        self.conv5 = ConvBlock(num_filter * 4, num_filter * 8, stride=2)
        # add a layer for 128
        self.conv6 = ConvBlock(num_filter * 8, output_dim, activation=False, stride=1, padding=0, batch_norm=False)

    def forward(self, x, label):
        x = torch.cat([x, label], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        out = torch.nn.Sigmoid()(x)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal(m.conv.weight, mean, std)






