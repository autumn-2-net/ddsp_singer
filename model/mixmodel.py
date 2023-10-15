import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention.attention_lay import attn_lay
from model.diffusion.ddpm import GaussianDiffusion
from model.fastspeech.fastspeech import fastspeech2


class mixecn(nn.Module):
    def __init__(self,indim,outdim,dim,lays, heads=4, dim_head=64,hideen_dim=None,):
        super().__init__()
        self.attn_lay=nn.ModuleList([attn_lay(dim=dim, heads=heads, dim_head=dim_head,hideen_dim=hideen_dim,kernel_size=1) for _ in range(lays)])
        self.inl=nn.Linear(indim,dim) if indim!=dim else nn.Identity()
        self.outl = nn.Linear(dim,outdim) if outdim!=dim else nn.Identity()

    def forward(self,x,taskemb,mask=None):
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


        self.taskemb=nn.Embedding(2,config['mixenvc_hidden_size'])

        self.pitch_embed = nn.Linear(1, config['fs2_hidden_size'])  ###############
        self.mixencoder=mixecn(indim=config['fs2_hidden_size'],outdim=config['condition_dim'],dim=config['mixenvc_hidden_size'],lays=config['mixenvc_lays'], heads=config['mixenvc_heads'], dim_head=config['mixenvc_dim_head'],hideen_dim=config.get('mixenvc_latent_dim'),)


        self.fs2=fastspeech2(vocab_size=vocab_size,config=config)

        self.diffusion = GaussianDiffusion(config=config,
            out_dims=config['audio_num_mel_bins'],
            num_feats=1,
            timesteps=config['timesteps'],
            k_step=config['K_step'],
            denoiser_type=config['diff_decoder_type'],
            denoiser_args=config['decoder_arg'],
            spec_min=config['spec_min'],
            spec_max=config['spec_max']
        )

    def forward(self, promot,f0,tasktype,svsmask=None,svcmask=None,txt_tokens=None, mel2ph=None, cvec_feature=None, key_shift=None, speed=None,
     gt_mel=None, infer=True, **kwargs):

        svsu=0 in tasktype
        svcu = 1 in tasktype
        if svcu and svsu:
            mask=torch.cat([svsmask,svcmask],dim=0)
            svsc = self.forwardsvs(txt_tokens=txt_tokens, mel2ph=mel2ph, key_shift=key_shift, speed=speed, kwargs=kwargs)
            svcc=self.forwardsvc(cvec_feature)
            f0_mel = (1 + f0 / 700).log()
            pitch_embed = self.pitch_embed(f0_mel[:, :, None])
            condition = self.mixencoder(torch.cat([svsc,svcc],dim=0)+pitch_embed, taskemb=self.taskemb(tasktype), mask=mask)
        elif svsu:
            mask=svsmask
            svsc=self.forwardsvs(txt_tokens=txt_tokens,mel2ph=mel2ph,key_shift=key_shift,speed=speed,kwargs=kwargs)
            f0_mel = (1 + f0 / 700).log()
            pitch_embed = self.pitch_embed(f0_mel[:, :, None])
            condition = self.mixencoder(svsc+pitch_embed, taskemb=self.taskemb(tasktype), mask=svsmask)

        elif svcu:
            mask=svcmask
            svcc=self.forwardsvc(cvec_feature)
            f0_mel = (1 + f0 / 700).log()
            pitch_embed = self.pitch_embed(f0_mel[:, :, None])

            condition=self.mixencoder(svcc+pitch_embed,taskemb=self.taskemb(tasktype),mask=svcmask)

        # f0_mel = (1 + f0 / 700).log()
        # pitch_embed = self.pitch_embed(f0_mel[:, :, None])
        # condition=condition+pitch_embed

        if infer:


            mel_pred = self.diffusion(condition, src_spec=None,promot=promot, infer=True)
            if mask is not None:
                mel_pred *= mask.float()[:, :, None]
            return mel_pred
        else:

            x_recon, noise = self.diffusion(condition, gt_spec=gt_mel,promot=promot, infer=False)
            return x_recon, noise,mask


    def forwardsvs(self,txt_tokens,mel2ph,key_shift,speed,kwargs):
        return self.fs2(txt_tokens=txt_tokens,mel2ph=mel2ph,  key_shift=key_shift, speed=speed,
           **kwargs)
    def forwardsvc(self,feature):
        # torch
        return self.svcin(feature.transpose(1,2))
