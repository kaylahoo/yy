import torch
import torch.nn as nn
import torch.nn.functional as F# 定义 U-Net 模型，这里我们简化了 U-Net 的结构

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder_conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.encoder_conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_conv1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)

        self.output_conv = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x):
        # 编码器部分
        x = F.relu(self.encoder_conv1(x))
        x_downsample_1 = F.max_pool2d(x, 2, stride=2)

        x_downsample_1 = F.relu(self.encoder_conv2(x_downsample_1))
        x_downsample_2 = F.max_pool2d(x_downsample_1, 2, stride=2)

        x_downsample_2 = F.relu(self.encoder_conv3(x_downsample_2))

        # 解码器部分
        x_upsample_1 = torch.cat([self.upconv1(x_downsample_2), x_downsample_1], dim=1)
        x_upsample_1 = F.relu(self.decoder_conv1(x_upsample_1))

        x_upsample_2 = torch.cat([self.upconv2(x_upsample_1), x], dim=1)

        return torch.tanh(self.output_conv(x_upsample_2))