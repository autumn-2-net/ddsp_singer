import torch
from torchaudio.transforms import MelSpectrogram
import torchaudio
def get_mel_from_audio(
        # audio: torch.Tensor,
        sample_rate=44100,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        f_min=40,
        f_max=16000,
        n_mels=128,
        center=True,
        power=1.0,
        pad_mode="reflect",
        norm="slaney",
        mel_scale="slaney",
):
    # assert audio.ndim == 2, "Audio tensor must be 2D (1, n_samples)"
    # assert audio.shape[0] == 1, "Audio tensor must be mono"

    transform = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        center=center,
        power=power,
        pad_mode=pad_mode,
        norm=norm,
        mel_scale=mel_scale,
    )  # .to(audio.device)
    return transform


pass
# if __name__=='__main__':
#     import glob
#     from tqdm import tqdm
#     import random
#
#
#
#     glic={'audio_sample_rate': 44100, 'audio_num_mel_bins': 128,'hop_size': 512,'fft_size': 2048 ,'win_size': 2048,'fmin': 40 ,    'fmax': 16000}
#
#     lll=glob.glob(r'D:\propj\Disa\data\opencpop\raw\wavs/**.wav')
#     mel_spec_transform = get_mel_from_audio()
#
#     for i in tqdm(lll):
#         audio, sr = torchaudio.load(i)
#         audio = torch.clamp(audio[0], -1.0, 1.0)
#
#
#         with torch.no_grad():
#             spectrogram = mel_spec_transform(audio)
#             # spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20  #ds 是log10
#             spectrogram = torch.log(torch.clamp(spectrogram, min=1e-5))
#

if __name__=='__main__':
    import glob
    from tqdm import tqdm
    import random

    torch.set_num_threads(1)



    glic={'audio_sample_rate': 44100, 'audio_num_mel_bins': 128,'hop_size': 512,'fft_size': 2048 ,'win_size': 2048,'fmin': 40 ,    'fmax': 16000}

    lll=glob.glob(r'D:\propj\Disa\data\opencpop\raw\wavs/**.wav')
    mel_spec_transform = get_mel_from_audio()

    for i in tqdm(lll):
        audio, sr = torchaudio.load(i)
        audio = torch.clamp(audio[0], -1.0, 1.0)


        with torch.no_grad():
            spectrogram = mel_spec_transform(audio)
            # spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20  #ds 是log10
            spectrogram = torch.log(torch.clamp(spectrogram, min=1e-5))