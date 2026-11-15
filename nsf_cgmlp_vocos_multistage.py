"""Multi-stage cgMLP vocoder with filtered NSF in each stage's residual stream.

All stages keep the input frame count and hop unchanged; only channel widths
increase. The output contains input_frames * input_hop samples. All stages
analyze the same sample-rate excitation, preserving phase across filters.
The final head concatenates the last filtered excitation's
real/imaginary features with the backbone features before its linear projection.
There is no direct NSF skip into the final spectrum.
"""

import math

import torch
from torch import nn

from nsf_cgmlp_vocos import CgMLPBlock, CombtoothSource, VocosSameSTFT, VocosSameISTFT


class FilteredNSFStage(nn.Module):
    def __init__(self, dim, n_fft, hop_length, num_blocks, kernel_size,
                 expansion, dropout):
        super().__init__()
        self.n_bins = n_fft // 2 + 1
        self.source_stft = VocosSameSTFT(n_fft, hop_length)
        self.filter_norm = nn.LayerNorm(dim)
        self.filter_head = nn.Linear(dim, 2 * self.n_bins)
        nn.init.zeros_(self.filter_head.weight)
        nn.init.zeros_(self.filter_head.bias)
        self.excitation_proj = nn.Linear(2 * self.n_bins, dim)
        self.excitation_norm = nn.LayerNorm(dim)
        self.injection_scale = nn.Parameter(torch.full((dim,), 0.1))
        self.blocks = nn.Sequential(*[
            CgMLPBlock(dim, kernel_size, expansion, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x, source, return_aux=False):
        batch, frames, _ = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            source_spec = self.source_stft(source)
            if source_spec.shape[-1] != frames:
                raise ValueError('NSF spectrum and conditioning must have equal frame counts')
            control = self.filter_head(self.filter_norm(x.float())).transpose(1, 2)
            log_gain, phase_pi = control.chunk(2, dim=1)
            source_filter = torch.polar(log_gain.clamp(-12., 8.).exp(), math.pi * phase_pi)
            excitation = source_spec * source_filter
            excitation_ri = torch.view_as_real(
                excitation.transpose(1, 2).contiguous()
            ).reshape(batch, frames, 2 * self.n_bins)
            injection = self.excitation_norm(self.excitation_proj(excitation_ri))
        # Inject at the entrance to the stage, before its residual cgMLP stack.
        x = x + injection.to(x.dtype) * self.injection_scale.to(x.dtype)
        x = self.blocks(x)
        aux = None
        if return_aux:
            aux = {'source_spec': source_spec, 'filter': source_filter,
                   'excitation': excitation}
        return x, excitation_ri, aux


class MultiStageNSFCGMLPVocos(nn.Module):
    def __init__(self, n_mels=256, sample_rate=44100, hop_length=512,
                 dims=(768, 1024, 2048),
                 n_ffts=(2048, 1024, 1024), num_blocks=(2, 2, 2),
                 kernel_size=31, expansion=2.0, dropout=0.0):
        super().__init__()
        if not dims or any(not isinstance(d, int) or d <= 0 for d in dims):
            raise ValueError('dims must contain positive integers')
        if len(n_ffts) != len(dims) or len(num_blocks) != len(dims):
            raise ValueError('n_ffts and num_blocks must have one entry per stage')
        if not isinstance(hop_length, int) or hop_length <= 0:
            raise ValueError('hop_length must be a positive integer')
        if any(not isinstance(n, int) or n < 1 for n in num_blocks):
            raise ValueError('num_blocks must contain positive integers')
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.input_proj = nn.Linear(n_mels, dims[0])
        self.channel_projections = nn.ModuleList([
            nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in zip(dims[:-1], dims[1:])
        ])
        self.source = CombtoothSource(sample_rate, hop_length)
        self.stages = nn.ModuleList([
            FilteredNSFStage(dim, n_fft, hop_length, blocks, kernel_size, expansion, dropout)
            for dim, n_fft, blocks in zip(dims, n_ffts, num_blocks)
        ])
        self.final_norm = nn.LayerNorm(dims[-1])
        head_bins = 2 * (n_ffts[-1] // 2 + 1)
        self.spectral_head = nn.Linear(dims[-1] + head_bins, head_bins)
        self.istft = VocosSameISTFT(n_ffts[-1], hop_length)

    def forward(self, mel, f0, return_aux=False):
        if mel.ndim != 3:
            raise ValueError('conditioning must be [B,T,C] or [B,C,T]')
        if mel.shape[-1] != self.n_mels:
            if mel.shape[1] != self.n_mels:
                raise ValueError(f'Expected {self.n_mels} conditioning channels')
            mel = mel.transpose(1, 2)
        if f0.ndim == 3 and f0.shape[-1] == 1:
            f0 = f0[..., 0]
        if f0.ndim != 2 or f0.shape != mel.shape[:2] or mel.shape[1] == 0:
            raise ValueError('F0 must match the nonempty input frame grid [B,T]')
        with torch.autocast(device_type=mel.device.type, enabled=False):
            source = self.source(f0.float())
        x = self.input_proj(mel)
        stage_aux = []
        for index, stage in enumerate(self.stages):
            if index:
                x = self.channel_projections[index - 1](x)
            x, excitation_ri, aux = stage(x, source, return_aux=return_aux)
            if return_aux:
                stage_aux.append(aux)
        x = self.final_norm(x)
        with torch.autocast(device_type=mel.device.type, enabled=False):
            # Reuse the final stage's filtered source without detaching it or
            # adding it directly to the output spectrum. Keep real/imag in FP32.
            head_input = torch.cat([x.float(), excitation_ri], dim=-1)
            log_mag, phase = self.spectral_head(head_input).transpose(1, 2).chunk(2, dim=1)
            spec = torch.polar(log_mag.clamp(-12., 8.).exp(), phase)
            wave = self.istft(spec)
        if return_aux:
            return wave, {'source': source, 'stages': stage_aux, 'spec': spec}
        return wave
