import torch
import torch.nn as nn
import torch.nn.functional as F




class VQGAN(nn.Module):
    def __init__(self, codebook_size):
        super(VQGAN, self).__init__()
        self.codebook_size = codebook_size
        # 加载预训练模型
        self.vqgan = torch.hub.load('openai/dall-e', '12-b4412b')
        self.quantize = nn.Embedding(self.codebook_size, 256)

    def forward(self, x):
        x = self.vqgan(x)['decoder_out']
        x = self.quantize(x.argmax(dim=1))
        return x