import torch
import torch.nn as nn
import torch.nn.functional as F


class VQGAN(nn.Module):
    def __init__(self, codebook_size=256):
        super(VQGAN, self).__init__()
        self.codebook_size = codebook_size
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv8 = nn.Conv2d(512, self.codebook_size, kernel_size=1)
        self.embedding = nn.Embedding(self.codebook_size, 512)

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        x = F.relu(self.conv5(x), inplace=True)
        x = F.relu(self.conv6(x), inplace=True)
        x = F.relu(self.conv7(x), inplace=True)
        # Quantize
        # 将张量调整为形状为(批次大小, 通道数(codebook_size), 高度, 宽度)的形式
        x_reshaped = self.conv8(x).view(x.size(0), self.codebook_size, -1)
        print(x_reshaped.shape)
        # 使用自适应最大池化将每个通道的张量映射到单个值上，并返回可以用来还原的索引
        quantized, indices = F.adaptive_max_pool2d(x_reshaped, (1, 1))
        print(quantized.shape)
        # 计算 permute() 函数中需要的参数，然后使用它对张量进行重新排序
        permute_order = (0, 1, 2, 3)   # 总共四个维度，没有需要删除的维度
        quantized = quantized.permute(*permute_order)
        emb = self.embedding.weight.unsqueeze(0)
        dist = torch.norm(emb - quantized.unsqueeze(1), dim=2)
        _, indices = torch.min(dist, dim=1)
        return self.embedding(indices).permute(0, 2, 1)


