import pathlib
import random

import numpy as np

from model.mixmodel_v2 import ssvm
from train_utils.Tdataset import SVS_Dataset, SVC_Dataset
from train_utils.ssvv_BatchSampler import MIX_Dataset
from utils_model.base_model import BaseTask
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from vocoders.nsf_hifigan.nsf_hifigan import NsfHifiGAN


def collates(minibatch):
    maxl=0
    maxpm=0
    max_ph=0
    svsd=[]
    svcd=[]
    for i in minibatch:
        if i['type']=='svs':
            if i['datal']>maxl:
                maxl=i['datal']
            if len(i['ph_idx'])>max_ph:
                max_ph=len(i['ph_idx'])
            if len(i['promot'])>maxpm:
                maxpm=len(i['promot'])
            ssv=i
            ssv['tasktype']=0
            del ssv['type']
            svsd.append(ssv)
        else:
            ssv = i
            ssv['tasktype'] = 1
            del ssv['type']

            svcd.append(ssv)

    svsd1 = []
    # {'type': 'svs', 'f0': f0.astype(np.float32), 'gtmel': mel, 'mel2ph': mel2ph, 'ph_idx': ph_idx,
    #  'datal': len(mel)}
    for i in svsd:
        pads = maxl - i['datal']
        datx = {}

        for j in i:
            if j not in ['tasktype', 'ph_idx', 'datal', 'promot']:

                if j in ['gtmel']:
                    d = torch.tensor(i[j])
                    d = F.pad(d.transpose( 0, 1), ( 0, pads), "constant", 0).transpose( 0, 1)
                    datx[j] = d
                else:
                    d = torch.tensor(i[j])
                    d = F.pad(d, ( 0, pads), "constant", 0)
                    datx[j] = d
            if j == 'ph_idx':
                d = torch.tensor(i[j])
                phpads = max_ph - len(d)
                d = F.pad(d, (0, phpads), "constant", 0)
                datx[j] = d
            if j == 'promot':
                d = torch.tensor(i[j])
                phpads = maxpm- len(d)
                d = F.pad(d.transpose( 0, 1), (0, phpads), "constant", 0).transpose( 0, 1)
                datx[j] = d
            if j == 'tasktype':
                datx[j] = torch.tensor(i[j])
        datx['mask'] = torch.tensor([True for _ in range(i['datal'])] + [False for _ in range(pads)])
        svsd1.append(datx)

    svcd1 = []
    for i in svcd:
        pads = maxl - i['datal']
        datx = {}

        if i['datal'] <= maxl:


            for j in i:
                if j not in ['tasktype', 'ph_idx', 'datal', 'promot','feature']:
                    # d = torch.tensor(i[j])
                    # d = F.pad(d, (0,0,0, pads), "constant", 0)
                    # datx[j] = d
                    if j in ['gtmel']:
                        d = torch.tensor(i[j])
                        d = F.pad(d.transpose( 0, 1), (0, pads), "constant", 0).transpose( 0, 1)
                        datx[j] = d
                    else:
                        d = torch.tensor(i[j])
                        d = F.pad(d, (0, pads), "constant", 0)
                        datx[j] = d
                if j=='feature':
                    d = torch.tensor(i[j])
                    d = F.pad(d, (0, pads), "constant", 0)
                    datx[j] = d

                if j == 'promot':
                    d = torch.tensor(i[j])
                    if len(d) > maxpm:
                        startc = random.randint(0, len(d) - 1 - maxpm)
                        endc = startc + maxpm
                        d = torch.tensor(i[j])[startc:endc]
                        # d= F.pad(d, (0, pads), "constant", 0)
                        datx[j] = d



                    else:
                        phpads = maxpm - len(d)
                        d = F.pad(d.transpose( 0, 1), (0, phpads), "constant", 0).transpose( 0, 1)
                        datx[j] = d

                if j == 'tasktype':
                    datx[j] = torch.tensor(i[j])
        else:
            pads = 0
            start = random.randint(0, i['datal'] - 1 - maxl)
            end = start + maxl
            for j in i:
                if j not in ['tasktype', 'ph_idx', 'datal', 'promot','feature']:
                    d = torch.tensor(i[j])[start:end]
                    # d= F.pad(d, (0, pads), "constant", 0)
                    datx[j] = d

                if j=='feature':
                    d = torch.tensor(i[j])[:,start:end]
                    # d= F.pad(d, (0, pads), "constant", 0)
                    datx[j] = d
                if j == 'promot':
                    d = torch.tensor(i[j])
                    if len(d) > maxpm:
                        startc = random.randint(0, len(d) - 1 - maxpm)
                        endc = startc + maxpm
                        d = torch.tensor(i[j])[startc:endc]
                        # d= F.pad(d, (0, pads), "constant", 0)
                        datx[j] = d



                    else:
                        phpads = maxpm - len(d)
                        d = F.pad(d.transpose( 0, 1), (0, phpads), "constant", 0).transpose( 0, 1)
                        datx[j] = d

                if j == 'tasktype':
                    datx[j] = torch.tensor(i[j])
        if i['datal'] <= maxl:
            datx['mask'] = torch.tensor([True for _ in range(i['datal'])] + [False for _ in range(pads)])
        else:
            datx['mask'] = torch.tensor([True for _ in range(maxl)] + [False for _ in range(pads)])
        svcd1.append(datx)


    tmpsvsd={}
    for i in svsd1:
        for j in i:
            ttt=tmpsvsd.get(j)
            if ttt is None:
                tmpsvsd[j]=[]
            if j!='datal':
                tmpsvsd[j].append(i[j])

    tmpsvcd={}
    for i in svcd1:
        for j in i:
            ttt=tmpsvcd.get(j)
            if ttt is None:
                tmpsvcd[j]=[]
            if j!='datal':
                tmpsvcd[j].append(i[j])
    tmpsvsd1={}
    for i in tmpsvsd:
        if i=='mask':
            tmpsvsd1['svsmask']=torch.stack(tmpsvsd[i],dim=0)
        else:
            tmpsvsd1[i] = torch.stack(tmpsvsd[i], dim=0)
    tmpsvcd1={}
    for i in tmpsvcd:
        if i=='mask':
            tmpsvcd1['svcmask']=torch.stack(tmpsvcd[i],dim=0)
        if i=='feature':
            tmpsvcd1['cvec_feature']=torch.stack(tmpsvcd[i],dim=0)
        else:
            tmpsvcd1[i] = torch.stack(tmpsvcd[i], dim=0)

    finx={}
    for i in set(list(tmpsvsd1.keys())+list(tmpsvcd1.keys())):
        svsf=tmpsvsd1.get(i)
        svcf = tmpsvcd1.get(i)

        if svcf is not None and svsf is not None:
            finx[i]=torch.cat([svsf,svcf],dim=0)
        elif svcf is not None:
            finx[i]=svcf
        elif svsf is not None:
            finx[i] = svsf

    return finx



def collates_val(minibatch):
    xxx={}
    for i in minibatch:
        for j in i:
            if j=='datal' or j=='type':
                continue
            if j=='feature' and i['type']=='svc':
                xxx['cvec_feature'] = torch.tensor(i[j]).unsqueeze(0)
                continue

            xxx[j]=torch.tensor(i[j]).unsqueeze(0)
    return xxx














class DiffusionNoiseLoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type
        if self.loss_type == 'l1':
            self.loss = nn.L1Loss(reduction='none')
        elif self.loss_type == 'l2':
            self.loss = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError()

    @staticmethod
    def _mask_nonpadding(x_recon, noise, nonpadding=None):
        if nonpadding is not None:
            # nonpadding = nonpadding.transpose(1, 2).unsqueeze(1)

            return x_recon.masked_fill(~nonpadding.unsqueeze(1).unsqueeze(1), 0), noise.masked_fill(~nonpadding.unsqueeze(1).unsqueeze(1), 0)
        else:
            return x_recon, noise

    def _forward(self, x_recon, noise):
        return self.loss(x_recon, noise)

    def forward(self, x_recon: torch.Tensor, noise: torch.Tensor, nonpadding: torch.Tensor = None) -> torch.Tensor:
        """
        :param x_recon: [B, 1, M, T]
        :param noise: [B, 1, M, T]
        :param nonpadding: [B, T, M]
        """
        x_recon, noise = self._mask_nonpadding(x_recon, noise, nonpadding)
        return self._forward(x_recon, noise).mean()

def spec_to_figure(spec, vmin=None, vmax=None):
    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    fig = plt.figure(figsize=(12, 9))
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
    plt.tight_layout()
    return fig
def load_dict_list(paths):
    phl=[]
    with open(paths, "r",encoding="utf-8") as f:
        ff=f.read().strip().split('\n')
    for i in ff:
        phx=i.strip().split('\t')[1]
        phx2=phx.strip().split(' ')
        for i in phx2:
            if i!='':
                phl.append(i.strip())
    return phl


class vcc:
    def __init__(self):
        self.vc=NsfHifiGAN(r'D:\propj\Disa\checkpoints\nsf_hifigan/model',use_natural_log=False).cuda()


    @torch.no_grad()
    def spec2wav(self, mel, f0, key_shift=0):


        y = self.vc.spec2wav(mel=torch.tensor(mel).transpose(0,1).cuda(), f0=torch.tensor(f0).cuda(), key_shift=key_shift)

        return y
class ssvc(BaseTask):
    def __init__(self,config):
        super().__init__(config)
        vbc=[]
        for i in config['dict_path']:
            vbc=vbc+load_dict_list(i)
        vbc=list(set(vbc))
        self.model=ssvm(config=config,vocab_size=len(vbc)+1+2)
        self.logged_gt_wav = set()


        self.train_dataset=MIX_Dataset(svs_data_set=SVS_Dataset(paths =str(pathlib.Path(config['data_index_path'])/'train_svs'),config=config,vocab_list=vbc),svc_data_set=SVC_Dataset(paths =str(pathlib.Path(config['data_index_path'])/'train_svc'),config=config))
        self.valid_dataset=MIX_Dataset(svs_data_set=SVS_Dataset(paths =str(pathlib.Path(config['data_index_path'])/'val_svs') ,config=config,vocab_list=vbc),svc_data_set=SVC_Dataset(paths =str(pathlib.Path(config['data_index_path'])/'val_svc'),config=config))
        self.train_dataset_collates=collates
        self.valid_dataset_collates = collates_val

        self.use_vocoder=True
        self.vocoder=vcc()

        self.required_variances = []

    def build_losses_and_metrics(self):

        self.mel_loss = DiffusionNoiseLoss(loss_type=self.config['diff_loss_type'])

    def run_model(self, sample, infer=False):
        #
        # testdl=self.train_dataloader()

        txt_tokens = sample.get('ph_idx')  # [B, T_ph]
        target = sample['gtmel']  # [B, T_s, M]
        mel2ph = sample.get('mel2ph' ) # [B, T_s]
        f0 = sample['f0']
        promot = sample['promot']
        variances = {
            v_name: sample[v_name]
            for v_name in self.required_variances
        }
        key_shift = sample.get('key_shift')
        speed = sample.get('speed')
        tasktype=sample['tasktype']
        svsmask = sample.get('svsmask')
        svcmask = sample.get('svcmask')
        cvec_feature = sample.get('cvec_feature')


        # if self.config['use_spk_id']:
        #     spk_embed_id = sample['spk_ids']
        # else:
        # spk_embed_id = None
        # output = self.model(
        #     txt_tokens, mel2ph=mel2ph, f0=f0, promot=promot, **variances,
        #     key_shift=key_shift, speed=speed, spk_embed_id=spk_embed_id,
        #     gt_mel=target, infer=infer
        # )

        output = self.model(
            promot=promot, f0=f0, tasktype=tasktype, svsmask=svsmask, svcmask=svcmask, txt_tokens=txt_tokens, mel2ph=mel2ph,
            cvec_feature=cvec_feature, key_shift=key_shift, speed=speed,
            gt_mel=target, infer=infer, **variances
        )

        if infer:
            return output
        else:
            losses = {}

            # if output.aux_out is not None:
            #     aux_out = output.aux_out
            #     aux_mel_loss = self.lambda_aux_mel_loss * self.aux_mel_loss(aux_out, target)
            #     losses['aux_mel_loss'] = aux_mel_loss


            x_recon, x_noise,mask = output
            mel_loss = self.mel_loss(x_recon, x_noise, nonpadding=mask)
            losses['mel_loss'] = mel_loss

            return losses

    # def on_train_start(self):
    #     if self.use_vocoder and self.vocoder.get_device() != self.device:
    #         self.vocoder.to_device(self.device)
    #
    # def _on_validation_start(self):
    #     if self.use_vocoder and self.vocoder.get_device() != self.device:
    #         self.vocoder.to_device(self.device)

    def _validation_step(self, sample, batch_idx):
        losses = self.run_model(sample, infer=False)

        if batch_idx < self.config['num_valid_plots'] \
                and (self.trainer.distributed_sampler_kwargs or {}).get('rank', 0) == 0:
            mel_out = self.run_model(sample, infer=True)

            if self.use_vocoder:
                self.plot_wav(
                    batch_idx,typess=('svc' if 1 in  sample['tasktype'] else 'svs'), gt_mel=sample['gtmel'],
                    aux_mel=None, diff_mel=mel_out,
                    f0=sample['f0']
                )
            # if mel_out.aux_out is not None:
            #     self.plot_mel(batch_idx, sample['mel'], mel_out.aux_out, name=f'auxmel_{batch_idx}')
            # if mel_out.diff_out is not None:
            self.plot_mel(batch_idx, sample['gtmel'], mel_out, name=f'diffmel_{batch_idx}')

        # return losses, sample['size']
        return losses, 1

    def plot_wav(self, batch_idx, gt_mel, typess,aux_mel=None, diff_mel=None, f0=None):
        gt_mel = gt_mel[0].cpu().numpy()
        if aux_mel is not None:
            aux_mel = aux_mel[0].cpu().numpy()
        if diff_mel is not None:
            diff_mel = diff_mel[0].cpu().numpy()
        f0 = f0[0].cpu().numpy()
        if batch_idx not in self.logged_gt_wav:
            gt_wav = self.vocoder.spec2wav(gt_mel, f0=f0)
            self.logger.experiment.add_audio(f'gt_{batch_idx}_{typess}', gt_wav, sample_rate=self.config['audio_sample_rate'],
                                             global_step=self.global_step)
            self.logged_gt_wav.add(batch_idx)
        if aux_mel is not None:
            aux_wav = self.vocoder.spec2wav(aux_mel, f0=f0)
            self.logger.experiment.add_audio(f'aux_{batch_idx}_{typess}', aux_wav, sample_rate=self.config['audio_sample_rate'],
                                             global_step=self.global_step)
        if diff_mel is not None:
            diff_wav = self.vocoder.spec2wav(diff_mel, f0=f0)
            self.logger.experiment.add_audio(f'diff_{batch_idx}_{typess}', diff_wav, sample_rate=self.config['audio_sample_rate'],
                                             global_step=self.global_step)

    def plot_mel(self, batch_idx, spec, spec_out, name=None):
        name = f'mel_{batch_idx}' if name is None else name
        vmin = self.config['mel_vmin']
        vmax = self.config['mel_vmax']
        spec_cat = torch.cat([(spec_out - spec).abs() + vmin, spec, spec_out], -1)
        self.logger.experiment.add_figure(name, spec_to_figure(spec_cat[0], vmin, vmax), self.global_step)