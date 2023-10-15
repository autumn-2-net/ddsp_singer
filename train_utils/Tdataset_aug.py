import csv
import random
import re

import numpy as np
import torch

from utils.VE_u import get_pitch_parselmouth, get_mel2ph_torch
from utils.data_orgmelE import wav2spec
from utils.datapre_ph import LengthRegulator

from torch.multiprocessing import  current_process

is_main_process = not bool(re.match(r'((.*Process)|(SyncManager)|(.*PoolWorker))-\d+', current_process().name))

class SVS_Dataset:
    def __init__(self,paths ,config,vocab_list,key_aug=None):
        with open(paths, 'r') as csvfile:
            reader = list(csv.DictReader(csvfile))

        if not is_main_process:
            torch.set_num_threads(1)
        self.didx=reader
        self.dalen=len(self.didx)
        self.config=config
        self.lr=LengthRegulator()
        self.timestep = config['hop_size'] / config['audio_sample_rate']
        self.vocab_list = sorted(vocab_list+['AP', 'SP'])
        self.vocab_map={}
        # self.keyaugpb=0.5
        self.keyaugpb = key_aug
        for idx,i in enumerate(self.vocab_list):
            self.vocab_map[i]=idx+1






    def __getitem__(self, i):
        # return self.xs[i[0]], self.ys[i[0]]
        data=self.didx[i]
        data:dict
        datapath=data['path_wav']
        wav,mel=wav2spec(datapath,config=self.config,device='cpu',keyshift=0, speed=1
                 )
        key_shift=0.0
        if self.keyaugpb is not None:
            if random.random()<self.keyaugpb:
                key_shift=random.uniform(-5,5)
                _, melaug = wav2spec(datapath, config=self.config, device='cpu', keyshift=key_shift, speed=1
                                    )
            else:
                melaug=mel
        else:
            melaug=mel



        f0,_=get_pitch_parselmouth(wav_data=wav, length=len(mel), hparams=self.config, speed=1, interp_uv=self.config['interp_uv'])
        f0 *= 2 ** (key_shift / 12)
        length=len(mel)
        # ''.strip().split(' ')
        # sst=[float(i) for i in data['ph_dur'].strip().split(' ')]
        mel2ph=get_mel2ph_torch(
            self.lr, torch.tensor([float(i) for i in data['ph_dur'].strip().split(' ')]), length, self.timestep, device='cpu'
        ).cpu().numpy()
        ph_l=data['ph_seq'].strip().split(' ')
        ph_idx=[self.vocab_map[i] for i in ph_l]


        pml=len(mel)//3
        start = random.randint(0, len(mel) - 1 - pml)
        end = start + pml
        promot=mel[start:end]


        return{'type':'svs','f0':f0.astype(np.float32),'gtmel':melaug,'mel2ph':mel2ph,'ph_idx':ph_idx,'datal':len(mel),'promot':promot,'tasktype':0,'key_shift':key_shift}

    def __len__(self):
        return self.dalen

class SVC_Dataset:
    def __init__(self,paths ,config ):
        with open(paths, 'r') as csvfile:
            reader = list(csv.DictReader(csvfile))

        if not is_main_process:
            torch.set_num_threads(1)
        self.didx=reader
        self.dalen=len(self.didx)
        self.config=config


    def __getitem__(self, i):
        # return self.xs[i[0]], self.ys[i[0]]
        data=self.didx[i]
        data:dict
        datapath=data['path_wav']
        feature=np.load(data['path_feature'])
        wav,mel=wav2spec(datapath,config=self.config,device='cpu',keyshift=0, speed=1
                 )
        f0,_=get_pitch_parselmouth(wav_data=wav, length=len(mel), hparams=self.config, speed=1, interp_uv=self.config['interp_uv'])


        pml=len(mel)//3
        start = random.randint(0, len(mel) - 1 - pml)
        end = start + pml
        promot=mel[start:end]

        return {'type':'svc','f0':f0.astype(np.float32),'gtmel':mel,'datal':len(mel),'promot':promot,'feature':feature,'tasktype':1}

    def __len__(self):
        return self.dalen

if __name__=='__main__':
    pass