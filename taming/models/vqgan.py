import torch
import pytorch_lightning as pl

from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,#输入的配置参数
                 n_embed,#向量化所用参数的数量
                 embed_dim,#投影到向量空间中的嵌入维度
                 ckpt_path=None,#预训练模型路径，设置为None表示为从头开始
                 ignore_keys=[],#忽落预训练过程中的权重信息
                 image_key="image",#输入图像的键名
                 colorize_nlabels=None,#色彩标签数量
                 monitor=None,#应用于记录日志和保存检查点等的Monitor实例
                 remap=None,#设置被替代的索引映射表
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key#用来存储输入数据的名称
        self.encoder = Encoder(**ddconfig)#
        self.decoder = Decoder(**ddconfig)#
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)#
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)#从给定的于训练模型路径中加载权重等参数
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")#使用CPU加载
        if "state_dict" in sd.keys():
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        print("Strict load")
        self.load_state_dict(sd, strict=True)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)#矢量化后的代码，输入的loss以及附加信息
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        dec = self.decoder(quant)#解码器输出的解码图片
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)#通过给定的代码向量重构未采样数据
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff#返回重构好的图像以及额外的损失信息
