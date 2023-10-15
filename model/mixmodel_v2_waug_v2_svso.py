from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention.attention_lay import attn_lay
from model.diffusion.ddpm import GaussianDiffusion
from model.fastspeech.fast_speech import FastSpeech2Encoder
from model.fastspeech.fastspeech import fastspeech2
from modules.fastspeech.acoustic_encoder import FastSpeech2Acoustic


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


class fs2_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,y, x):
        x=(x - (-5)) / (0 - (-5)) * 2 - 1
        return nn.L1Loss()(y,x)


class fs2_decode(nn.Module):
    def __init__(self,encoder_hidden,out_dims,n_chans,kernel_size,dropout_rate,n_layers,parame=None):
        super().__init__()
        self.inconv=nn.Conv1d(encoder_hidden, n_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.conv = nn.ModuleList([ConvNeXtBlock(dim=n_chans,intermediate_dim=int(n_chans*2.5),layer_scale_init_value=1e-6,drop_out=dropout_rate)  for _ in range(n_layers)])
        self.outconv=nn.Conv1d(n_chans, out_dims, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.loss=fs2_loss()


    def build_loss(self):

        return fs2_loss()

    def forward(self, x,infer,*args,**kwargs):
        x=x.transpose(1, 2)
        x=self.inconv(x)
        for i in self.conv:
            x=i(x)
        x=self.outconv(x).transpose(1, 2)
        if infer:
            x=(x + 1) / 2 * (0 - (-5)) + (-5)
        return x
        pass
    def tforward(self, x,gt=None):
        x=x.transpose(1, 2)
        x=self.inconv(x)
        for i in self.conv:
            x=i(x)
        x=self.outconv(x).transpose(1, 2)
        # if infer:
        #     x=(x + 1) / 2 * (0 - (-5)) + (-5)
        # return self.loss(x,gt)
        if gt is not None:
            gt=gt
        return x, gt
        pass

class mixecn(nn.Module):
    def __init__(self,indim,outdim,dim,lays, heads=4, dim_head=64,hideen_dim=None,):
        super().__init__()
        self.attn_lay=nn.ModuleList([attn_lay(dim=dim, heads=heads, dim_head=dim_head,hideen_dim=hideen_dim,kernel_size=1) for _ in range(lays)])
        self.inl=nn.Linear(indim,dim) if indim!=dim else nn.Identity()
        self.outl = nn.Linear(dim,outdim) if outdim!=dim else nn.Identity()

    def forward(self,x,taskemb=None,mask=None):
        x=self.inl(x)
        # x=x+taskemb[:, None, :]
        for i in self.attn_lay:
            x=i(x,mask)
        x = self.outl(x)
        return x


class ssvm(nn.Module):
    def __init__(self,config,vocab_size):
        super().__init__()
        self.svcin=nn.Linear(768,config['fs2_hidden_size'])


        self.taskemb=nn.Embedding(2,config['fs2_hidden_size'])

        self.pitch_embed_svc = nn.Linear(1, config['fs2_hidden_size'])  ###############
        self.pitch_embed_svs = nn.Linear(1, config['fs2_hidden_size'])
        self.mixencoder=mixecn(indim=config['fs2_hidden_size'],outdim=config['fs2_hidden_size'],dim=config['mixenvc_hidden_size'],lays=config['mixenvc_lays'], heads=config['mixenvc_heads'], dim_head=config['mixenvc_dim_head'],hideen_dim=config.get('mixenvc_latent_dim'),)
        self.mixencoder2 = mixecn(indim=config['fs2_hidden_size'], outdim=config['fs2_hidden_size'],
                                 dim=config['mixenvc_hidden_size'], lays=config['mixenvc_lays'],
                                 heads=config['mixenvc_heads'], dim_head=config['mixenvc_dim_head'],
                                 hideen_dim=config.get('mixenvc_latent_dim'), )

        self.fs2=fastspeech2(vocab_size=vocab_size,config=config)
        # self.fs2 = FastSpeech2Acoustic(vocab_size=vocab_size, config=config)
        # self.fs2 = FastSpeech2Encoder(input_size=vocab_size, use_embedding_to_input=True)
        # input_size = 1024,
        # max_seq_len = 4096,
        # num_layers = 4,
        # hidden_size = 256,
        # ffn_kernel_size = 9,
        # dropout = 0.1,
        # num_heads = 2,
        # ffn_padding = "SAME",
        # ffn_act = "gelu",
        # padding_idx = 0,
        # use_embedding_to_input = False

        # self.diffusion = GaussianDiffusion(config=config,
        #     out_dims=config['audio_num_mel_bins'],
        #     num_feats=1,
        #     timesteps=config['timesteps'],
        #     k_step=config['K_step'],
        #     denoiser_type=config['diff_decoder_type'],
        #     denoiser_args=config['decoder_arg'],
        #     spec_min=config['spec_min'],
        #     spec_max=config['spec_max']
        # )
        self.wnd=fs2_decode(encoder_hidden=256,out_dims=128,n_chans=384,kernel_size=7,dropout_rate=0.,n_layers=5,)

    def forward(self, promot,f0,tasktype,svsmask=None,svcmask=None,txt_tokens=None, mel2ph=None, cvec_feature=None, key_shift=None, speed=None,
     gt_mel=None, infer=True, **kwargs):

        svsu=0 in tasktype
        svcu = 1 in tasktype
        # torch.nonzero(torch.tensor([0, 0, 0, 0, 1, 1]) == 0).size()[0]
        if svcu and svsu:
            # mask=torch.cat([svsmask,svcmask],dim=0)
            mask=svsmask
            svst=torch.nonzero(tasktype == 0).size()[0]
            f0_svs=f0[:svst]
            # f0_svc =f0[svst:]
            svsc = self.forwardsvs(txt_tokens=txt_tokens, mel2ph=mel2ph, key_shift=key_shift, speed=speed, kwargs=kwargs,f0=f0_svs)
            # svcc=self.forwardsvc(cvec_feature,f0=f0_svc)
            gt_mel=gt_mel[:svst]
            promot=promot[:svst]
            # condition = torch.cat([svsc,svcc])#+self.taskemb(tasktype)[:, None, :]
            condition=svsc
        elif svsu:
            mask=svsmask
            svsc=self.forwardsvs(txt_tokens=txt_tokens,mel2ph=mel2ph,key_shift=key_shift,speed=speed,kwargs=kwargs,f0=f0)

            condition = svsc#+self.taskemb(tasktype)[:, None, :]

        elif svcu:
            mask=svcmask
            svcc=self.forwardsvc(cvec_feature,f0=f0)


            condition=svcc#+self.taskemb(tasktype)[:, None, :]



        if infer:


            # mel_pred = self.diffusion(condition, src_spec=None,promot=promot, infer=True)
            mel_pred=self.wnd.tforward(condition)[0]
            if mask is not None:
                mel_pred *= mask.float()[:, :, None]
            return mel_pred
        else:

            # x_recon, noise = self.diffusion(condition, gt_spec=gt_mel,promot=promot, infer=False)
            x_recon, noise =self.wnd.tforward(condition,gt_mel)
            mask=None
            return x_recon, noise,mask


    def forwardsvs(self,txt_tokens,mel2ph,key_shift,speed,kwargs,f0):
        f0_mel = (1 + f0 / 700).log()
        pitch_embed = self.pitch_embed_svs (f0_mel[:, :, None])

        # return self.mixencoder2(self.fs2(txt_tokens=txt_tokens,mel2ph=mel2ph,  key_shift=key_shift, speed=speed,
        #    **kwargs)+pitch_embed)

        return self.fs2(txt_tokens=txt_tokens, mel2ph=mel2ph, key_shift=key_shift, speed=speed,
                                     **kwargs) + pitch_embed
    def forwardsvc(self,feature,f0):
        # torch
        f0_mel = (1 + f0 / 700).log()
        pitch_embed = self.pitch_embed_svc (f0_mel[:, :, None])
        return self.mixencoder(self.svcin(feature.transpose(1,2))+pitch_embed)
