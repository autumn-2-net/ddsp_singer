"""Standalone NSF-Combtooth + cgMLP + Vocos-style iSTFT vocoder.

Inputs
------
mel: (B, T, n_mels) or (B, n_mels, T)   -- log-Mel conditioning
f0:  (B, T) or (B, T, 1)                 -- Hz, aligned to mel hop

Output
------
wave: (B, T * hop_length)                -- float32

The singer adapter also uses learned FastSpeech2 frame features as conditioning;
in that case n_mels is set to the frontend hidden size (no mel target required).

Only dependencies: PyTorch. No SVS frontend, speaker encoder, discriminator,
Sequence Packing, or auxiliary UV gate. For unvoiced intervals, pass a
continuous/interpolated F0 and let the learned filter/vocoder control energy.

The 'same' STFT/ISTFT alignment follows Vocos:
  - STFT: pad (n_fft-hop)//2 on both sides; center=False -> T frames
  - ISTFT: Hann-weighted inverse FFT + overlap-add/window-squared norm;
           crop the same padding -> T * hop waveform samples.

References:
  Vocos: https://github.com/gemelo-ai/vocos (vocos/spectral_ops.py)
  cgMLP: https://github.com/espnet/espnet (espnet2/asr/layers/cgmlp.py)
  Combtooth: https://github.com/yxlllc/DDSP-SVC (ddsp/vocoder.py, 6.3)

Example
-------
    model = NSFCGMLPVocos(n_mels=128, sample_rate=44100, n_fft=2048,
                          hop_length=512, dim=384, num_front=4, num_back=4,
                          kernel_size=31).cuda()
    mel = torch.randn(2, 80, 128, device="cuda")
    f0 = torch.full((2, 80), 220., device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        wave = model(mel, f0)           # (2, 40960), float32

The final log-magnitude and phase heads always run in float32. The model can
be trained end-to-end; gradient also propagates from the waveform to F0.
"""

import math
from typing import Union

import torch
from torch import nn
from torch.nn import functional as F


class CgMLPBlock(nn.Module):
    """Residual convolutional gated MLP; depthwise temporal kernel defaults to 31.

    The two halves of GELU-expanded features are gated by a depthwise-convolved
    version of the second half, similar to Branchformer's cgMLP (without attention).
    """

    def __init__(self, dim: int, kernel_size: int = 31,
                 expansion: float = 4.0, dropout: float = 0.0):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("kernel_size must be a positive odd integer")
        inner = max(2, int(dim * expansion))
        inner += inner % 2
        half = inner // 2

        self.pre_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, inner)
        self.gate_norm = nn.LayerNorm(half)
        self.dwconv = nn.Conv1d(half, half, kernel_size,
                                padding=kernel_size // 2, groups=half)
        self.fc2 = nn.Linear(half, dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = nn.Parameter(torch.full((dim,), 1e-2))

        # Start with a nearly identity-valued gating branch, stabilizing a
        # deeper residual stack. Learnable after the first optimization step.
        nn.init.normal_(self.dwconv.weight, 0.0, 1e-4)
        nn.init.ones_(self.dwconv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(self.pre_norm(x)))
        value, gate = h.chunk(2, dim=-1)
        gate = self.dwconv(self.gate_norm(gate).transpose(1, 2)).transpose(1, 2)
        h = self.fc2(self.dropout(value * gate))
        return x + self.dropout(h) * self.layer_scale.to(h.dtype)


class CombtoothSource(nn.Module):
    """yxlllc's wrapped-sinc periodic excitation, with frame-interpolated F0.

    Phase advances are scanned in float64 to avoid long-utterance drift; the
    output waveform and sample-wise sinc remain float32. This is a *periodic
    excitation approximation*, not a mathematically perfect anti-aliased source.
    """

    def __init__(self, sample_rate: int = 44100, hop_length: int = 512,
                 eps: float = 1e-5):
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.hop_length = int(hop_length)
        self.eps = float(eps)

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        if f0.ndim == 3 and f0.shape[-1] == 1:
            f0 = f0[..., 0]
        if f0.ndim != 2:
            raise ValueError("f0 must have shape [B,T] or [B,T,1]")

        # Critical phase arithmetic is not performed in the caller's AMP dtype.
        f0 = f0.float()
        B, T = f0.shape
        H = self.hop_length
        if T == 0:
            raise ValueError("f0 must contain at least one frame")

        s0 = f0 / self.sample_rate
        ds = F.pad(s0[:, 1:] - s0[:, :-1], (0, 1))
        n = torch.arange(H, device=f0.device, dtype=torch.float32)
        q = n + 1.0

        # Per-frame phase, in *cycles*. Integrates a linear F0 interpolation.
        local_phase = (s0[..., None] * q
                       + 0.5 * ds[..., None] * n * q / H)

        # Per-frame end advance, accumulated at float64 precision. Taking
        # modulo one *before* casting back to float32 prevents long-run loss
        # of fractional-cycle resolution.
        advance = (s0.double() * H
                   + 0.5 * ds.double() * (H - 1))
        phase_start = (torch.cumsum(advance, dim=1) - advance)
        phase_start = torch.remainder(phase_start, 1.0).float()
        phase = phase_start[..., None] + local_phase
        phase = phase - torch.round(phase)

        inst_step = s0[..., None] + ds[..., None] * (n / H)
        src = torch.sinc(phase / (inst_step + self.eps))
        return src.reshape(B, T * H)


class VocosSameSTFT(nn.Module):
    """Forward STFT matching Vocos's `padding='same'` frame placement.

    Vocos's mel feature extractor uses reflect padding for 'same'; we mirror
    that convention. For too-short audio to reflect-pad we use zeros instead.
    """

    def __init__(self, n_fft: int = 2048, hop_length: int = 512):
        super().__init__()
        if n_fft <= hop_length or (n_fft - hop_length) % 2:
            raise ValueError("Require n_fft > hop_length and even (n_fft-hop_length)")
        self.n_fft, self.hop_length = int(n_fft), int(hop_length)
        self.pad = (n_fft - hop_length) // 2
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        if wave.ndim != 2:
            raise ValueError("wave must be [B, samples]")
        if wave.shape[-1] % self.hop_length != 0:
            raise ValueError("wave length must be divisible by hop_length")
        with torch.autocast(device_type=wave.device.type, enabled=False):
            x = wave.float()
            pad_mode = "reflect" if x.shape[-1] > self.pad else "constant"
            x = F.pad(x, (self.pad, self.pad), mode=pad_mode)
            return torch.stft(x, n_fft=self.n_fft,
                              hop_length=self.hop_length,
                              win_length=self.n_fft,
                              window=self.window.float(),
                              center=False, return_complex=True)


class VocosSameISTFT(nn.Module):
    """Vocos-style 'same' inverse STFT using overlap-add and WOLA norm."""

    def __init__(self, n_fft: int = 2048, hop_length: int = 512):
        super().__init__()
        if n_fft <= hop_length or (n_fft - hop_length) % 2:
            raise ValueError("Require n_fft > hop_length and even (n_fft-hop_length)")
        self.n_fft, self.hop_length = int(n_fft), int(hop_length)
        self.pad = (n_fft - hop_length) // 2
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if spec.ndim != 3 or not torch.is_complex(spec):
            raise ValueError("spec must have shape [B,F,T] and complex dtype")
        B, Freq, T = spec.shape
        if Freq != self.n_fft // 2 + 1 or T < 1:
            raise ValueError("unexpected frequency or time dimensions")

        with torch.autocast(device_type=spec.device.type, enabled=False):
            spec = spec.to(torch.complex64)
            window = self.window.float()
            frames = torch.fft.irfft(spec, n=self.n_fft, dim=1)
            frames = frames * window[None, :, None]
            full_len = (T - 1) * self.hop_length + self.n_fft
            # An FFT frame is one im2col column; fold performs overlap-add.
            audio = F.fold(frames, output_size=(1, full_len),
                           kernel_size=(1, self.n_fft),
                           stride=(1, self.hop_length))[:, 0, 0, :]

            # Same normalizer as Vocos; one envelope works for every sample.
            win2 = window.square()[None, :, None].expand(1, self.n_fft, T)
            envelope = F.fold(win2, output_size=(1, full_len),
                              kernel_size=(1, self.n_fft),
                              stride=(1, self.hop_length))[0, 0, 0, :]
            L = T * self.hop_length
            audio = audio[:, self.pad:self.pad + L]
            envelope = envelope[self.pad:self.pad + L]
            return audio / envelope.clamp_min(1e-8)[None, :]


class NSFCGMLPVocos(nn.Module):
    """Simple standalone Mel+F0 NSF-Vocos, no packing/discriminator/UV gate.

    Pipeline:
       mel -> cgMLP(front) -> z -> FP32 complex filter H
       f0 -> Combtooth -> Vocos same-STFT S
       E = S*H -> real/imag projection -> fuse(E,z)
       -> cgMLP(back) -> FP32 logmag/phase head -> same-iSTFT -> wave
    """

    def __init__(self, n_mels: int = 128, sample_rate: int = 44100,
                 n_fft: int = 2048, hop_length: int = 512,
                 dim: int = 384, num_front: int = 4, num_back: int = 4,
                 kernel_size: int = 31, expansion: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.n_mels = int(n_mels)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_bins = n_fft // 2 + 1

        self.mel_proj = nn.Linear(n_mels, dim)
        self.front = nn.Sequential(*[
            CgMLPBlock(dim, kernel_size, expansion, dropout)
            for _ in range(num_front)
        ])
        self.front_norm = nn.LayerNorm(dim)

        self.source = CombtoothSource(sample_rate, hop_length)
        self.source_stft = VocosSameSTFT(n_fft, hop_length)

        # FP32 filter head: [log amplitude, phase / pi] per frequency bin.
        self.filter_head = nn.Linear(dim, 2 * self.n_bins)
        nn.init.zeros_(self.filter_head.weight)
        nn.init.zeros_(self.filter_head.bias)  # H=1+0j initially

        # E is a complex STFT; feed its real and imaginary parts to cgMLP.
        self.excitation_proj = nn.Linear(2 * self.n_bins, dim)
        self.excitation_norm = nn.LayerNorm(dim)
        self.fuse = nn.Linear(dim * 2, dim)

        self.back = nn.Sequential(*[
            CgMLPBlock(dim, kernel_size, expansion, dropout)
            for _ in range(num_back)
        ])
        self.back_norm = nn.LayerNorm(dim)

        # Vocos-like log-magnitude + phase STFT head in FP32.
        self.spectral_head = nn.Linear(dim, 2 * self.n_bins)
        self.istft = VocosSameISTFT(n_fft, hop_length)

    def forward(self, mel: torch.Tensor, f0: torch.Tensor,
                return_aux: bool = False) -> Union[torch.Tensor, tuple]:
        if mel.ndim != 3:
            raise ValueError("mel must be [B,T,M] or [B,M,T]")
        if mel.shape[-1] == self.n_mels:
            pass  # [B, T, M]
        elif mel.shape[1] == self.n_mels:
            mel = mel.transpose(1, 2)
        else:
            raise ValueError(f"Expected n_mels={self.n_mels}")

        if f0.ndim == 3 and f0.shape[-1] == 1:
            f0 = f0[..., 0]
        if f0.ndim != 2 or f0.shape != mel.shape[:2]:
            raise ValueError("f0 must have shape [B,T] (same T as mel)")
        B, T, _ = mel.shape
        if T == 0:
            raise ValueError("mel sequence is empty")

        # Backbone honors the caller's autocast policy (e.g. BF16).
        z = self.front_norm(self.front(self.mel_proj(mel)))

        # Numerically sensitive DSP, complex filter and its projection: FP32.
        with torch.autocast(device_type=mel.device.type, enabled=False):
            src = self.source(f0.float())
            source_spec = self.source_stft(src)          # [B,F,T] complex64
            ctrl = self.filter_head(z.float()).transpose(1, 2)
            log_gain, phase_pi = ctrl.chunk(2, dim=1)
            log_gain = log_gain.clamp(-12., 8.)
            src_filter = torch.polar(torch.exp(log_gain), math.pi * phase_pi)
            excitation_spec = source_spec * src_filter
            # [B,F,T] complex64 -> [B,T,2F] float32.
            excitation_ri = torch.view_as_real(
                excitation_spec.transpose(1, 2).contiguous()
            ).reshape(B, T, 2 * self.n_bins)
            excitation = self.excitation_norm(
                self.excitation_proj(excitation_ri)
            )

        z = self.fuse(torch.cat([z, excitation.to(z.dtype)], dim=-1))
        z = self.back_norm(self.back(z))

        with torch.autocast(device_type=mel.device.type, enabled=False):
            out = self.spectral_head(z.float()).transpose(1, 2)
            log_mag, phase = out.chunk(2, dim=1)
            # Stable log-polar complex head as in Vocos, with a looser clamp.
            spec = torch.polar(torch.exp(log_mag.clamp(-12., 8.)), phase)
            wave = self.istft(spec)                       # float32, [B,T*hop]

        if return_aux:
            return wave, {"s_h": source_spec, "filter": src_filter,
                          "excitation": excitation_spec, "spec": spec}
        return wave


if __name__ == "__main__":
    torch.manual_seed(1)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = NSFCGMLPVocos(dim=128, num_front=2, num_back=2).to(dev)
    model.train()
    mel = torch.randn(2, 9, 128, device=dev)
    f0 = (220.0 + 10 * torch.randn(2, 9, device=dev)).clamp_min(50.0)
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
        wave = model(mel, f0)
    loss = wave.square().mean()
    loss.backward()
    print("wave:", tuple(wave.shape), wave.dtype,
          "grad:", model.filter_head.weight.grad.abs().mean().item())
