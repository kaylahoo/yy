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
    def __init__(self,in_channels=4,init_weights=True):
        super(InpaintGenerator, self).__init__()

        #定义编码器
        self.encoder_conv1 = PConvBNActiv(in_channels, out_channels=64, bn = False, sample='down-7')
        self.encoder_conv2 = PConvBNActiv(in_channels=64, out_channels=128,sample='down-5'  )
        self.encoder_conv3 = PConvBNActiv(in_channels=128, out_channels=256,sample='down-5' )
        self.encoder_conv4 = PConvBNActiv(in_channels=256, out_channels=512, sample='down-3')

        # 定义解码器（对称的 UNet 结构）
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv4 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

        if init_weights:
            self.init_weights()

    def forward(self, images_masks,masks):
        # 编码器部分
        x = images_masks
        print(x.shape)
        y = torch.cat((masks, masks, masks,masks), dim=1)
        print(y.shape)
        x,y= self.encoder_conv1(x,y)
        x_downsample_1 = F.max_pool2d(x, 2, stride=2)
        y_downsample_1 = F.max_pool2d(y, 2, stride=2)

        x_downsample_1,y_downsample_1 = self.encoder_conv2(x_downsample_1,y_downsample_1)
        #y_downsample_1 = F.relu(self.encoder_conv2(y_downsample_1,y_downsample_1))

        x_downsample_2 = F.max_pool2d(x_downsample_1, 2, stride=2)
        y_downsample_2 = F.max_pool2d(y_downsample_1, 2, stride=2)

        x_downsample_2,y_downsample_2= self.encoder_conv3(x_downsample_2,y_downsample_2)
        #y_downsample_2 = F.relu(self.encoder_conv3(y_downsample_2,y_downsample_2))

        x_downsample_3 = F.max_pool2d(x_downsample_2, 2, stride=2)
        y_downsample_3 = F.max_pool2d(y_downsample_2, 2, stride=2)

        x_downsample_3,y_downsample_3 = self.encoder_conv4(x_downsample_3, y_downsample_3)




        # 解码器部分
        x_upsample_1 = torch.cat([self.upconv1(x_downsample_3), x_downsample_2], dim=1)
        #y_upsample_1 = torch.cat([self.upconv1(y_downsample_2), y_downsample_1], dim=1)

        x_upsample_2 = torch.cat([self.upconv2(x_upsample_1), x_downsample_1], dim=1)
        #y_upsample_2 = torch.cat([self.upconv2(y_upsample_1), y_downsample_1], dim=1)

        x_upsample_3 = torch.cat([self.upconv3(x_upsample_2), images_masks], dim=1)
        #y_upsample_3 = torch.cat([self.upconv3(y_upsample_2), masks], dim=1)

        x6 = self.upconv4(x_upsample_3)
        #y6 = self.upconv4(y_upsample_3)

        return torch.tanh(x6)




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
class VectorQuantization(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantization, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_layer.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, inputs):
        """
        :param inputs: shape [batch_size, channels, height, width]
        :return:
            quantized: nearest embedding after input projected to embedding space
            encoding_indices: index of nearest embeddings for each element in inputs
            perplexity: exp(-1 * sum_all(p_{i} * log(p_{i})))
        """
        # Flatten input tensor to be B x (C*H*W)
        input_flattened = inputs.view(-1, self.embedding_dim)  # flatten different to 'vector_quantize'

        # Calculate distance between input and embeddings, get nearest embedding index
        distances = (
            (input_flattened ** 2).sum(dim=1, keepdim=True)
            + (self.embedding_layer.weight ** 2).sum(dim=1)
            - 2 * torch.matmul(input_flattened, self.embedding_layer.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(dim=1)
        quantized = self.embedding_layer(encoding_indices).view(inputs.shape)

        # Calculate commitment loss
        e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2)
        q_latent_loss = torch.mean((quantized - inputs.detach()) ** 2)
        commitment_loss = self.commitment_cost * (q_latent_loss + e_latent_loss)


        quantized = inputs + (quantized - inputs).detach()  # смещаем квантованные значения
        return quantized, encoding_indices, commitment_loss


# VQGAN Encoder with VectorQuantization
class VQEncoder(nn.Module):
    def __init__(self, embedding_dim=64, num_embeddings=256, commitment_cost=0.25):
        super(VQEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, embedding_dim, kernel_size=3, stride=2, padding=1),
        )
        self.vector_quantization = VectorQuantization(num_embeddings=num_embeddings, embedding_dim=embedding_dim,
                                                      commitment_cost=commitment_cost)

    def forward(self, x):
        h = self.encoder(x)
        vq_x, _, _, commitment_loss = self.vector_quantization(h)
        return vq_x, commitment_loss


# VQGAN Decoder with VectorQuantization
class VQDecoder(nn.Module):
    def __init__(self, embedding_dim=64, num_embeddings=256):
        super(VQDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=3, padding=1),
            nn.Tanh(),
        )
        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, z_e):
        x = self.decoder(z_e) + 0.5#在 VQGAN 模型中，由于像素值的范围一般是从 0 到 1，因此将解码后的输出进行 + 0.5 的操作可将其范围变为 0.5 到 1.5，这样可以更好地匹配输入像素值范围，使得生成图像更自然。
        return x


class EdgeGenerator(BaseNetwork):
    def __init__(self, embedding_dim=64, num_embeddings=256, commitment_cost=0.25):
        super(EdgeGenerator, self).__init__()
        self.encoder = VQEncoder(embedding_dim=embedding_dim, num_embeddings=num_embeddings,
                                 commitment_cost=commitment_cost)
        self.decoder = VQDecoder(embedding_dim=embedding_dim, num_embeddings=num_embeddings)

    def forward(self, x):
        # Encoder output
        quantized_x, commitment_loss = self.encoder(x)

        # Decoder output
        z_e = self.encoder.vector_quantization.embedding_layer.weight.transpose(0, 1).cuda()  # [z_dim, K]
        z_q = self.encoder.vector_quantization(quantized_x)[1].squeeze()

        return self.decoder(z_e[z_q]), commitment_loss


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
