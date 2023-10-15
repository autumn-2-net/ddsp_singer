import torch
import torch.nn as nn
import torch.nn.functional as F

from ddsp.vocoder import CombSub
from model.attention.attention_lay import attn_lay
from model.diffusion.ddpm import GaussianDiffusion
from model.fastspeech.fastspeech import fastspeech2


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


class ddsps(nn.Module):
    def __init__(self,config,vocab_size):
        super().__init__()
        # self.svcin=nn.Linear(768,config['fs2_hidden_size'])




        self.pitch_embed = nn.Linear(1, config['fs2_hidden_size'])  ###############
        # self.Pencoder=Pencoder(indim=config['fs2_hidden_size'],outdim=config['condition_dim'],dim=config['mixenvc_hidden_size'],lays=config['mixenvc_lays'], heads=config['mixenvc_heads'], dim_head=config['mixenvc_dim_head'],hideen_dim=config.get('mixenvc_latent_dim'),)
        self.Pencoder = Pencoder(indim=config['fs2_hidden_size'], outdim=128,
                                 dim=config['mixenvc_hidden_size'], lays=config['mixenvc_lays'],
                                 heads=config['mixenvc_heads'], dim_head=config['mixenvc_dim_head'],
                                 hideen_dim=config.get('mixenvc_latent_dim'), )

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


    def forward(self,f0,txt_tokens=None, mel2ph=None, key_shift=None, speed=None, infer=True, **kwargs):

        condition=self.fs2(txt_tokens=txt_tokens,mel2ph=mel2ph,  key_shift=key_shift, speed=speed,
           **kwargs)
        f0_mel = (1 + f0 / 700).log()
        pitch_embed = self.pitch_embed(f0_mel[:, :, None])
        condition=condition+pitch_embed
        condition=self.Pencoder(condition)

        signal, _, (s_h, s_n) = self.ddsp(condition, torch.unsqueeze(f0,dim=-1), infer=infer)
        return condition,signal,s_h, s_n



    # def forwardsvs(self,txt_tokens,mel2ph,key_shift,speed,kwargs):
    #     return self.fs2(txt_tokens=txt_tokens,mel2ph=mel2ph,  key_shift=key_shift, speed=speed,
    #        **kwargs)

