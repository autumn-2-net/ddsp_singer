import os
import numpy as np
import librosa
import torch
import pyworld as pw
import parselmouth
import argparse
import shutil



def get_f0(path_srcfile,sampling_rate,f0_extractor,f0_min,f0_max,hop_length,melL=None):
    # extract f0 using parselmouth
    x, _ = librosa.load(path_srcfile, sr=sampling_rate)

    if melL is not None:
        mel = melL
    else:
        mel = int(len(x)//512+1)
    # print(mel)
    if f0_extractor == 'parselmouth':
        f0 = parselmouth.Sound(x, sampling_rate).to_pitch_ac(
            time_step=hop_length / sampling_rate,
            voicing_threshold=0.6,
            pitch_floor=f0_min,
            pitch_ceiling=f0_max).selected_array['frequency']
        pad_size = (int(len(x) // hop_length) - len(f0) + 1) // 2
        f0 = np.pad(f0, [[pad_size, mel - len(f0) - pad_size]], mode='constant')

    # extract f0 using dio
    elif f0_extractor == 'dio':
        _f0, t = pw.dio(
            x.astype('double'),
            sampling_rate,
            f0_floor=f0_min,
            f0_ceil=f0_max,
            channels_in_octave=2,
            frame_period=(1000 * hop_length / sampling_rate))
        f0 = pw.stonemask(x.astype('double'), _f0, t, sampling_rate)
        f0 = f0.astype('float')[:mel]

    # extract f0 using harvest
    elif f0_extractor == 'harvest':
        f0, _ = pw.harvest(
            x.astype('double'),
            sampling_rate,
            f0_floor=f0_min,
            f0_ceil=f0_max,
            frame_period=(1000 * hop_length / sampling_rate))
        f0 = f0.astype('float')[:mel]

    else:
        raise ValueError(f" [x] Unknown f0 extractor: {f0_extractor}")

    uv = f0 == 0
    if len(f0[~uv]) > 0:
        # interpolate the unvoiced f0
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        uv = uv.astype('float')
        uv = np.min(np.array([uv[:-2], uv[1:-1], uv[2:]]), axis=0)
        uv = np.pad(uv, (1, 1))
        # save npy
        return f0,uv
        # os.makedirs(path_meldir, exist_ok=True)
        # np.save(path_melfile, mel)
        # os.makedirs(path_f0dir, exist_ok=True)
        # np.save(path_f0file, f0)
        # os.makedirs(path_uvdir, exist_ok=True)
        # np.save(path_uvfile, uv)
    else:
        print('\n[Error] F0 extraction failed: ' + path_srcfile)
        return None,None
        # os.makedirs(path_skipdir, exist_ok=True)
        # shutil.move(path_srcfile, path_skipdir)
        # print('This file has been moved to ' + os.path.join(path_skipdir, file))


if __name__=='__main__':
    import glob
    import torchaudio
    from tqdm import tqdm
    # from concurrent.futures import ProcessPoolExecutor
    # import random

    # import re
    # from torch.multiprocessing import Manager, Process, current_process, get_context
    #
    # is_main_process = not bool(re.match(r'((.*Process)|(SyncManager)|(.*PoolWorker))-\d+', current_process().name))


    lll = glob.glob(r'D:\propj\Disa\data\opencpop\raw\wavs/**.wav')
    # torch.set_num_threads(1)

    for i in tqdm(lll):
        f0,uv=get_f0(i,44100,'harvest',f0_min=40,f0_max=1600,hop_length=512,)
        # audio, sr = torchaudio.load(i)
        # audio = torch.clamp(audio[0], -1.0, 1.0)



# print('Preprocess the audio clips in :', path_srcdir)

# aaa=get_f0('./2099003695.wav',44100,'parselmouth',20,1000,512)
# print(list(aaa[0]))
# aaa

#data tip f0 \t time \t ph \t diao
#数据格式 一行 f0 制表符 音速时长 \t 音速 \t 音调