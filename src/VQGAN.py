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
        quantized, indices = F.adaptive_max_pool2d(self.conv8(x).view(x.size(0), self.codebook_size, -1), (1, 1))


        # 打印张量和元组的形状，以确保所有的维度都被覆盖了且顺序正确
        print("Shape of original tensor: ", quantized.shape)

        quantized = quantized.permute(0, 2, 1)
        emb = self.embedding.weight.unsqueeze(0)
        dist = torch.norm(emb - quantized.unsqueeze(1), dim=2)
        _, indices = torch.min(dist, dim=1)
        return self.embedding(indices).permute(0, 2, 1)


