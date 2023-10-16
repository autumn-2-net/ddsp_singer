from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ddsp.vocoder import CombSub, Sins
# from ddspx.vocoder import CombSub
from model.attention.attention_lay import attn_lay
from model.diffusion.ddpm import GaussianDiffusion
from model.fastspeech.fastspeech import fastspeech2


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: Optional[float] = None,drop_path: float=0.0,drop_out: float=0.0

    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv



        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()
        self.dropout=nn.Dropout(drop_out) if drop_out > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, ) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)


        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        x=self.dropout(x)

        x = residual + self.drop_path (x)
        return x


class Pencoder(nn.Module):
    def __init__(self,indim,outdim,dim,lays, heads=4, dim_head=64,hideen_dim=None,):
        super().__init__()
        self.attn_lay=nn.ModuleList([attn_lay(dim=dim, heads=heads, dim_head=dim_head,hideen_dim=hideen_dim,kernel_size=1) for _ in range(lays)])
        self.inl=nn.Linear(indim,dim) if indim!=dim else nn.Identity()
        self.outl = nn.Linear(dim,outdim) if outdim!=dim else nn.Identity()
        self.norm=nn.LayerNorm(dim)

    def forward(self,x,mask=None):
        x=self.inl(x)
        # x=x+taskemb[:, None, :]
        for i in self.attn_lay:
            x=i(x,mask)
        x=self.norm(x)
        x = self.outl(x)
        return x


class fs2_decode(nn.Module):
    def __init__(self,encoder_hidden,out_dims,n_chans,kernel_size,dropout_rate,n_layers,parame=None):
        super().__init__()
        self.inconv=nn.Conv1d(encoder_hidden, n_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.conv = nn.ModuleList([ConvNeXtBlock(dim=n_chans,intermediate_dim=int(n_chans*2.5),layer_scale_init_value=1e-6,drop_out=dropout_rate)  for _ in range(n_layers)])
        self.outconv=nn.Conv1d(n_chans, out_dims, kernel_size, stride=1, padding=(kernel_size - 1) // 2)




    def forward(self, x,):
        x=x.transpose(1, 2)
        x=self.inconv(x)
        for i in self.conv:
            x=i(x)
        x=self.outconv(x).transpose(1, 2)

        return x
        pass

class ddsp_singer(nn.Module):
    def __init__(self,config,vocab_size):
        super().__init__()
        # self.svcin=nn.Linear(768,config['fs2_hidden_size'])




        self.pitch_embed = nn.Linear(1, config['fs2_hidden_size'])  ###############
        # self.Pencoder=Pencoder(indim=config['fs2_hidden_size'],outdim=config['condition_dim'],dim=config['mixenvc_hidden_size'],lays=config['mixenvc_lays'], heads=config['mixenvc_heads'], dim_head=config['mixenvc_dim_head'],hideen_dim=config.get('mixenvc_latent_dim'),)
        self.Pencoder = Pencoder(indim=config['fs2_hidden_size'], outdim=256,
                                 dim=config['mixenvc_hidden_size'], lays=config['mixenvc_lays'],
                                 heads=config['mixenvc_heads'], dim_head=config['mixenvc_dim_head'],
                                 hideen_dim=config.get('mixenvc_latent_dim'), )
        self.wnd = fs2_decode(encoder_hidden=256, out_dims=128, n_chans=384, kernel_size=7, dropout_rate=0.05,
                              n_layers=5, )

        self.fs2=fastspeech2(vocab_size=vocab_size,config=config)

        # self.ddsp=CombSub(
        #     sampling_rate=args.data.sampling_rate,
        #     block_size=args.data.block_size,
        #     win_length=args.data.n_fft,
        #     n_mag_harmonic=args.model.n_mag_harmonic,
        #     n_mag_noise=args.model.n_mag_noise,
        #     n_mels=args.data.n_mels)
        self.ddsp=CombSub(
            sampling_rate=44100,
            block_size=512,
            win_length=2048,
            n_mag_harmonic=512,
            n_mag_noise=256,
            n_mels=128)
        # self.ddsp=Sins(
        #     sampling_rate=44100,
        #     block_size=512,
        #     win_length=2048,
        #     n_harmonics=128,
        #     n_mag_noise=256,
        #     n_mels=128)
        # self.ddsp=CombSub(
        #     sampling_rate=44100,
        #     block_size=512,
        #     n_mag_allpass=256,
        #     n_mag_harmonic=512,
        #     n_mag_noise=256,
        #     n_mels=128)


    def forward(self,f0,txt_tokens=None, mel2ph=None, key_shift=None, speed=None, infer=True, **kwargs):

        condition=self.fs2(txt_tokens=txt_tokens,mel2ph=mel2ph,  key_shift=key_shift, speed=speed,
           **kwargs)
        f0_mel = (1 + f0 / 700).log()
        pitch_embed = self.pitch_embed(f0_mel[:, :, None])
        condition=condition+pitch_embed
        condition=self.wnd (self.Pencoder(condition))

        # condition=target
        signal, _, (s_h, s_n) = self.ddsp(condition, torch.unsqueeze(f0,dim=-1), infer=infer)
        return condition,signal,s_h, s_n



    # def forwardsvs(self,txt_tokens,mel2ph,key_shift,speed,kwargs):
    #     return self.fs2(txt_tokens=txt_tokens,mel2ph=mel2ph,  key_shift=key_shift, speed=speed,
    #        **kwargs)

