"""Front cgMLP modulates NSF once; channel-expanding stages reuse that excitation.

All stages preserve the input frame count and hop. One FFT size controls source
analysis, filtering and final synthesis. Filtered NSF enters every residual
stage and is concatenated with the final features before the output linear.
"""

import math

import torch
from torch import nn

from nsf_cgmlp_vocos import CgMLPBlock, CombtoothSource, VocosSameSTFT, VocosSameISTFT


class NSFResidualStage(nn.Module):
    def __init__(self, dim, excitation_dim, num_blocks, kernel_size,
                 expansion, dropout):
        super().__init__()
        self.excitation_proj = nn.Linear(excitation_dim, dim)
        self.excitation_norm = nn.LayerNorm(dim)
        self.injection_scale = nn.Parameter(torch.full((dim,), 0.1))
        self.blocks = nn.Sequential(*[
            CgMLPBlock(dim, kernel_size, expansion, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x, excitation_ri):
        with torch.autocast(device_type=x.device.type, enabled=False):
            injection = self.excitation_norm(self.excitation_proj(excitation_ri))
        # Inject at the entrance to the stage, before its residual cgMLP stack.
        x = x + injection.to(x.dtype) * self.injection_scale.to(x.dtype)
        return self.blocks(x)


class MultiStageNSFCGMLPVocos(nn.Module):
    def __init__(self, n_mels=256, sample_rate=44100, hop_length=512,
                 n_fft=1024, dims=(768, 1024, 2048), num_front=2,
                 num_blocks=(2, 2, 2),
                 kernel_size=31, expansion=2.0, dropout=0.0):
        super().__init__()
        if not dims or any(not isinstance(d, int) or d <= 0 for d in dims):
            raise ValueError('dims must contain positive integers')
        if len(num_blocks) != len(dims):
            raise ValueError('num_blocks must have one entry per stage')
        if not isinstance(num_front, int) or num_front < 1:
            raise ValueError('num_front must be a positive integer')
        if not isinstance(hop_length, int) or hop_length <= 0:
            raise ValueError('hop_length must be a positive integer')
        if any(not isinstance(n, int) or n < 1 for n in num_blocks):
            raise ValueError('num_blocks must contain positive integers')
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_bins = n_fft // 2 + 1
        self.input_proj = nn.Linear(n_mels, dims[0])
        self.front = nn.Sequential(*[
            CgMLPBlock(dims[0], kernel_size, expansion, dropout)
            for _ in range(num_front)
        ])
        self.front_norm = nn.LayerNorm(dims[0])
        self.source = CombtoothSource(sample_rate, hop_length)
        self.source_stft = VocosSameSTFT(n_fft, hop_length)
        self.filter_head = nn.Linear(dims[0], 2 * self.n_bins)
        nn.init.zeros_(self.filter_head.weight)
        nn.init.zeros_(self.filter_head.bias)
        self.channel_projections = nn.ModuleList([
            nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in zip(dims[:-1], dims[1:])
        ])
        self.stages = nn.ModuleList([
            NSFResidualStage(dim, 2 * self.n_bins, blocks, kernel_size, expansion, dropout)
            for dim, blocks in zip(dims, num_blocks)
        ])
        self.final_norm = nn.LayerNorm(dims[-1])
        head_bins = 2 * self.n_bins
        self.spectral_head = nn.Linear(dims[-1] + head_bins, head_bins)
        self.istft = VocosSameISTFT(n_fft, hop_length)

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
        x = self.front_norm(self.front(self.input_proj(mel)))
        with torch.autocast(device_type=mel.device.type, enabled=False):
            source = self.source(f0.float())
            source_spec = self.source_stft(source)
            control = self.filter_head(x.float()).transpose(1, 2)
            log_gain, phase_pi = control.chunk(2, dim=1)
            source_filter = torch.polar(log_gain.clamp(-12., 8.).exp(), math.pi * phase_pi)
            excitation = source_spec * source_filter
            excitation_ri = torch.view_as_real(
                excitation.transpose(1, 2).contiguous()
            ).reshape(mel.shape[0], mel.shape[1], 2 * self.n_bins)
        stage_features = []
        for index, stage in enumerate(self.stages):
            if index:
                x = self.channel_projections[index - 1](x)
            x = stage(x, excitation_ri)
            if return_aux:
                stage_features.append(x)
        x = self.final_norm(x)
        with torch.autocast(device_type=mel.device.type, enabled=False):
            # Reuse the front-modulated source without detaching it or adding
            # it directly to the output spectrum. Keep real/imag in FP32.
            head_input = torch.cat([x.float(), excitation_ri], dim=-1)
            log_mag, phase = self.spectral_head(head_input).transpose(1, 2).chunk(2, dim=1)
            spec = torch.polar(log_mag.clamp(-12., 8.).exp(), phase)
            wave = self.istft(spec)
        if return_aux:
            return wave, {'source': source, 'source_spec': source_spec,
                          'filter': source_filter, 'excitation': excitation,
                          'stage_features': stage_features, 'spec': spec}
        return wave
