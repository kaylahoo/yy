import torch
import torch.nn as nn

from src.UNet import UNet
from src.VQGAN import VQGAN
from src.partialconv2d import PartialConv2d  # 加
from src.partialconv2d import PConvBNActiv  # 加
from src.depconv2d import DepConvBNActiv
import torch.nn.functional as F  # 加


# from timm.models.layers import DropPath


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)  # 初始化正态分布N ( 0 , std=1 )
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)  # 初始化为 正态分布~ N ( 0 , std )

                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # kaiming针对relu函数提出的初始化方法
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)  # 初始化为常数

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class InpaintGenerator(BaseNetwork):
    # class Generator(nn.Module):
    def __init__(self, codebook_size=256):
        super(InpaintGenerator, self).__init__()

        self.vqgan = VQGAN(codebook_size)
        self.unet = UNet()

    def forward(self, x, mask):
        # 使用 VQ-GAN 模型生成特征码
        vq_code = self.vqgan(x)

        # 将掩膜信息送入 U-Net 模型进行处理，得到修复后的图像
        inpainted_img = self.unet(x * (1 - mask) + vq_code * mask)

        return inpainted_img


# def __init__(self, residual_blocks=8, init_weights=True):
#     super(InpaintGenerator, self).__init__() #在子类中调用父类方法
#
#     self.encoder = nn.Sequential(
#         nn.ReflectionPad2d(3),#镜像填充
#         nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
#         nn.InstanceNorm2d(64, track_running_stats=False), #一个channel内做归一化，算H*W的均值
#         nn.ReLU(True),
#
#         nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
#         nn.InstanceNorm2d(128, track_running_stats=False),
#         nn.ReLU(True),
#
#         nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
#         nn.InstanceNorm2d(256, track_running_stats=False),
#         nn.ReLU(True)
#     )
#
#     blocks = []
#     for _ in range(residual_blocks):
#         block = ResnetBlock(256, 2)
#         blocks.append(block)
#
#     self.middle = nn.Sequential(*blocks)
#
#     self.decoder = nn.Sequential(
#         nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
#         nn.InstanceNorm2d(128, track_running_stats=False),
#         nn.ReLU(True),
#
#         nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
#         nn.InstanceNorm2d(64, track_running_stats=False),
#         nn.ReLU(True),
#
#         nn.ReflectionPad2d(3),
#         nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
#     )
#
#     if init_weights:
#         self.init_weights()
# def forward(self, x):
#     x = self.encoder(x)
#     x = self.middle(x)
#     x = self.decoder(x)
#     x = (torch.tanh(x) + 1) / 2
#
#     return x

# csy
# def __init__(self, residual_blocks=8, init_weights=True):
#     super(InpaintGenerator, self).__init__()
#
#     n = 32
#     # rough
#     self.enc_r1 = nn.Sequential(
#         nn.ReflectionPad2d(3),
#         nn.Conv2d(in_channels=7, out_channels=n, kernel_size=7, stride=1, padding=0),
#         nn.InstanceNorm2d(n, track_running_stats=False),
#         nn.ReLU(True)
#     )  # 256
#
#     self.Pconv_r2 = PartialConv2d(in_channels=n, out_channels=2*n, kernel_size=7, stride=2, padding=3,
#                                   return_mask=True)
#     self.enc_r2 = nn.Sequential(
#         nn.InstanceNorm2d(2*n, track_running_stats=False),
#         nn.ReLU(True)
#     )  # 128
#
#     self.Pconv_r3 = PartialConv2d(in_channels=2*n, out_channels=4*n, kernel_size=3, stride=2, padding=1,
#                                   return_mask=True)
#     self.enc_r3 = nn.Sequential(
#         nn.InstanceNorm2d(4*n, track_running_stats=False),
#         nn.ReLU(True)
#     )  # 64
#
#     self.enc_r4 = nn.Sequential(
#         nn.Conv2d(in_channels=4*n, out_channels=8*n, kernel_size=3, stride=2, padding=1),
#         nn.InstanceNorm2d(8*n, track_running_stats=False),
#         nn.ReLU(True)
#     )  # 32
#
#
#     # fine
#     self.enc_f1 = nn.Sequential(
#         nn.ReflectionPad2d(2),
#         nn.Conv2d(in_channels=7, out_channels=n, kernel_size=5, stride=1, padding=0),
#         nn.InstanceNorm2d(n, track_running_stats=False),
#         nn.ReLU(True)
#     )  # 256
#
#     self.Pconv_f2 = PartialConv2d(in_channels=n, out_channels=2*n, kernel_size=5, stride=2, padding=2, return_mask=True)
#     self.enc_f2 = nn.Sequential(
#         nn.InstanceNorm2d(2*n, track_running_stats=False),
#         nn.ReLU(True)
#     )  # 128
#
#     self.Pconv_f3 = PartialConv2d(in_channels=2*n, out_channels=4*n, kernel_size=3, stride=2, padding=1, return_mask=True)
#     self.enc_f3 = nn.Sequential(
#         nn.InstanceNorm2d(4*n, track_running_stats=False),
#         nn.ReLU(True)
#     )  # 64
#
#     self.enc_f4 = nn.Sequential(
#         nn.Conv2d(in_channels=4*n, out_channels=8*n, kernel_size=3, stride=2, padding=1),
#         nn.InstanceNorm2d(8*n, track_running_stats=False),
#         nn.ReLU(True)
#     )  # 32
#
#     # bottleneck
#     blocks = []
#     for i in range(residual_blocks-1):
#         block = ResnetBlock(dim=8*n, dilation=2, use_attention_norm=False)
#         blocks.append(block)
#     self.middle = nn.Sequential(*blocks)
#     self.chan_att_norm = ResnetBlock(dim=8*n, dilation=2, use_attention_norm=True)
#
#     # decoder
#     self.dec_1 = nn.Sequential(
#         nn.ConvTranspose2d(in_channels=8*n, out_channels=4*n, kernel_size=4, stride=2, padding=1),
#         nn.InstanceNorm2d(4*n, track_running_stats=False),
#         nn.ReLU(True)
#     )  # 64
#
#     self.dec_2 = nn.Sequential(
#         nn.ConvTranspose2d(in_channels=4*n, out_channels=2*n, kernel_size=4, stride=2, padding=1),
#         nn.InstanceNorm2d(2*n, track_running_stats=False),
#         nn.ReLU(True)
#     )  # 128
#
#     self.dec_3 = nn.Sequential(
#         nn.ConvTranspose2d(in_channels=2*n+3, out_channels=n, kernel_size=4, stride=2, padding=1),
#         nn.InstanceNorm2d(n, track_running_stats=False),
#         nn.ReLU(True)
#     )  # 256
#
#     self.dec_4 = nn.Sequential(
#         nn.Conv2d(in_channels=n, out_channels=3, kernel_size=1, padding=0)
#     )  # 256
#
#
#
#     if init_weights:
#         self.init_weights()
#
# def forward(self, x, mask, stage):
#
#     if stage is 0:
#         structure = x[:, -3:, ...]
#         structure = F.interpolate(structure, scale_factor=0.5, mode='bilinear')
#
#         x = self.enc_r1(x)
#         x, m = self.Pconv_r2(x, mask)
#         x = self.enc_r2(x)
#         f1 = x.detach()
#         x, _ = self.Pconv_r3(x, m)
#         x = self.enc_r3(x)
#         x = self.enc_r4(x)
#         x = self.middle(x)
#         x, l_c1 = self.chan_att_norm(x)
#         x = self.dec_1(x)
#         x = self.dec_2(x)
#         x, l_p1 = self.pos_attention_norm1(x, f1, m)
#         att_structure = self.pos_attention_norm1(structure, structure, m, reuse=True)
#         att_structure = att_structure.detach()
#         x = torch.cat((x, att_structure), dim=1)
#         x = self.dec_3(x)
#         x = self.dec_4(x)
#         x_rough = (torch.tanh(x) + 1) / 2
#         orth_loss = (l_c1 + l_p1) / 2
#         return x_rough, orth_loss
#
#     if stage is 1:
#         residual = x[:, -3:, ...]
#         residual = F.interpolate(residual, scale_factor=0.5, mode='bilinear')
#
#         x = self.enc_f1(x)
#         x, m = self.Pconv_f2(x, mask)
#         x = self.enc_f2(x)
#         f2 = x.detach()
#         x, _ = self.Pconv_f3(x, m)
#         x = self.enc_f3(x)
#         x = self.enc_f4(x)
#         x = self.middle(x)
#         x, l_c2 = self.chan_att_norm(x)
#         x = self.dec_1(x)
#         x = self.dec_2(x)
#         x, l_p2 = self.pos_attention_norm2(x, f2, m)
#         att_residual = self.pos_attention_norm2(residual, residual, m, reuse=True)
#         att_residual = att_residual.detach()
#         x = torch.cat((x, att_residual), dim=1)
#         x = self.dec_3(x)
#         x = self.dec_4(x)
#         x_fine = (torch.tanh(x) + 1) / 2
#         orth_loss = (l_c2 + l_p2) / 2
#         return x_fine, orth_loss


class EdgeGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, use_spectral_norm=True, init_weights=True):
        super(EdgeGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
                          use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
                          use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
                          use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
                          use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0)
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        inplace = True
        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=inplace),  # nn.LeakyReLU给非负值赋予一个非零斜率
        )  # spectral_norm利用pytorch自带的频谱归一化函数，给设定好的网络进行频谱归一化，主要用于生成对抗网络的鉴别器

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=inplace),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=inplace),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=inplace),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            #  谱归一化，为了约束GAN的鉴别器映射函数
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
