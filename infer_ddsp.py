import json
import pathlib
from collections import OrderedDict

import numpy as np
import torch
import torchaudio
from lightning.pytorch.loggers import TensorBoardLogger
from tqdm import tqdm

from PL_callbacks.save_checkpoint import ModelCheckpoints
from utils.VE_u import get_mel2ph_torch
from utils.config_loader import get_config
from utils.data_orgmelE import wav2spec
from utils.datapre_ph import LengthRegulator
from utils_model.ddsp_sinder_task import ddsp_ss

# from utils_model.ssvc import ssvc

import lightning as pl

#
config=get_config('configs/a_v2.yaml')
# config=get_config('configs/a1.yaml')
config.update({'infer':True})
timestep = config['hop_size'] / config['audio_sample_rate']
# models_ssvc=ssvc(config=config)

vbc = []
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
for i in config['dict_path']:
    vbc = vbc + load_dict_list(i)
vbc = list(set(vbc))
vocab_list = sorted(vbc+['AP', 'SP'])
vocab_map = {}
# self.keyaugpb=0.5
# self.keyaugpb = key_aug
for idx, i in enumerate(vocab_list):
    vocab_map[i] = idx + 1
def loadckpt(path,configs):
    models_ssvc = ddsp_ss(config=configs)
    # models_ssvc.load_from_checkpoint(path)
    models_ssvc.load_state_dict(torch.load(path)['state_dict'],strict=False) ################################
    return models_ssvc.cuda()

def resample_align_curve(points: np.ndarray, original_timestep: float, target_timestep: float, align_length: int):
    t_max = (len(points) - 1) * original_timestep
    curve_interp = np.interp(
        np.arange(0, t_max, target_timestep),
        original_timestep * np.arange(len(points)),
        points
    ).astype(points.dtype)
    delta_l = align_length - len(curve_interp)
    if delta_l < 0:
        curve_interp = curve_interp[:align_length]
    elif delta_l > 0:
        curve_interp = np.concatenate((curve_interp, np.full(delta_l, fill_value=curve_interp[-1])), axis=0)
    return curve_interp

def build_ds(dcf):
    bt={}

    ph_l = dcf['ph_seq'].strip().split(' ')
    ph_idx = [vocab_map[i] for i in ph_l]
    txt_tokens = torch.LongTensor(ph_idx).to('cuda')  # => [B, T_txt]
    # batch['tokens'] = txt_tokens
    # bt['ph_idx']=txt_tokens
    bt['ph_idx'] = txt_tokens.unsqueeze( dim=0)
    lr=LengthRegulator()
    ph_dur = torch.from_numpy(np.array(dcf['ph_dur'].split(), np.float32)).to('cuda')
    ph_acc = torch.round(torch.cumsum(ph_dur, dim=0) / timestep + 0.5).long()
    durations = torch.diff(ph_acc, dim=0, prepend=torch.LongTensor([0]).to('cuda'))[None]  # => [B=1, T_txt]
    mel2ph = lr(durations, txt_tokens == 0)  # => [B=1, T]
    bt['mel2ph']=mel2ph
    # bt['mel2ph'] = torch.unsqueeze(mel2ph, dim=0)
    length = mel2ph.size(1)
    bt['f0'] = torch.from_numpy(resample_align_curve(
        np.array(dcf['f0_seq'].split(), np.float32),
        original_timestep=float(dcf['f0_timestep']),
        target_timestep=timestep,
        align_length=length
    )).to('cuda')[None]
    bt['tasktype']=torch.tensor([[0]]).to('cuda')
    # bt['key_shift'] = torch.tensor([[0.]]).to('cuda')
    bt['key_shift'] = torch.tensor([[-4.]]).to('cuda')

    return bt









def loadds(dsp,):
    # wva, melss = wav2spec(pm, device='cpu', config=config)
    # melss = torch.from_numpy(melss).cuda()
    # if maxppm is not None:
    #     melss = melss[:maxppm, :]

    # melss:torch.tensor()
    # melss = torch.unsqueeze(melss, dim=0)

    # bt['promot'] = melss.cuda()

    with open(dsp,'r',encoding='utf8') as f:
        jsx=json.load(f)
    ccl=[]
    for i in jsx:
        a=build_ds(i)
        # a['promot'] = melss.cuda()
        ccl.append(a)
    return ccl



if __name__ == '__main__':
    config=get_config('configs/a_v2.yaml')

    config.update({'infer':True})

    # models_ssvc=ssvc(config=config)
    models_ssvc=loadckpt(r'D:\propj\ddsp_singer\ckpt\ddsp_s8\model_ckpt_steps_63999.ckpt',configs=config)
    ccx=loadds('穿越.ds',)
    melss=[]
    sss=[]
    f0e=[]
    for i in tqdm(ccx):
        mel =models_ssvc.run_model(i,infer=True,inx=True)
        melss.append(mel)
    #     f0e.append(i['f0'])
    # for i ,f0 in zip(melss,f0e):
    #     sss.append(models_ssvc.vocoder.spec2wav(i.squeeze( dim=0), f0=f0))
    # for i in sss:
    wavss=torch.cat(melss,dim=-1)
    torchaudio.save('cpxmn.wav', wavss.detach().cpu(), sample_rate=44100)

    pass