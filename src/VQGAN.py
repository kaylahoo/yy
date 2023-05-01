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
        #x_reshaped = self.conv8(x).view(x.size(0), self.codebook_size, -1)
        #print(x_reshaped.shape)

        # 第一步：对 x_reshaped 进行自适应最大池化并提取 quantized 和 indices
        #quantized_fn = lambda x: F.adaptive_max_pool2d(x, (1, 1)).squeeze(dim=-1).squeeze(dim=-1)
        #quantized = quantized_fn(x_reshaped)
        #print(quantized.shape)
        #indices = quantized.argmin(dim=-1, keepdim=True)
        #print(indices.shape)

        # 第二步：修改 indices 的形状
        #num_features = x_reshaped.size(-1)

        #indices = indices.unsqueeze(-1).expand(x.size(0), self.codebook_size, num_features)
        #print(indices.shape)

        #permute_order = (0, 2, 1)  # 已经获得 **num_features x codebook_sizex1** 的张量，故将池化后的两个维度删除
        #quantized = quantized.permute(*permute_order)

        # 第三步：计算欧氏距离和最近邻索引，并返回该值
        #distances = torch.norm(quantized.unsqueeze(dim=1) - self.embedding.weight.unsqueeze(0), dim=-1)
        #indices = distances.argmin(dim=2)

        # 对嵌入向量应用排列并返回张量
        #return self.embedding(indices).permute(0, 2, 1)

        # Quantize

        x_reshaped = self.conv8(x).view(x.size(0), self.codebook_size, -1)
        print(x_reshaped.shape)

        # 第一步：对 x_reshaped 进行自适应最大池化并提取 quantized 和 indices
        quantized_fn = lambda x: F.adaptive_max_pool2d(x, (1, 1)).squeeze(dim=-1).squeeze(dim=-1)
        quantized = quantized_fn(x_reshaped)
        indices = quantized.argmin(dim=-1, keepdim=True)

        # 第二步：将 indices、quantized 张量经过维度转换，使得张量的维度为 (batch_size, codebook_size, num_features, 1)

        quantized_expanded = quantized.unsqueeze(dim=2)

        # 第三步：计算欧氏距离和最近邻索引，并返回该值
        distances = torch.norm(quantized_expanded - self.embedding.weight.unsqueeze(dim=0).unsqueeze(dim=2), dim=1)
        indices = distances.argmin(dim=1)

        # 对嵌入向量应用排列并返回张量，这里将最后一个维度转置成 num_features x codebook_size x 1
        return self.embedding(indices).permute(0, 2, 1).unsqueeze(dim=-1)

