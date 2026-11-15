"""Render a DiffSinger .ds file directly with a trained NSF Vocos singer."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio

from model.singer import build_singer
from utils.config_loader import get_config
from utils.datapre_ph import LengthRegulator


def load_vocabulary(config):
    phones = set()
    for path in config['dict_path']:
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                phones.update(line.strip().split('\t')[1].split())
    # Match SVS_Dataset's token ordering exactly.
    return {phone: i + 1 for i, phone in enumerate(sorted(list(phones) + ['AP', 'SP']))}


def build_sample(segment, vocab, config, device, transpose=0.0):
    tokens = torch.tensor([[vocab[p] for p in segment['ph_seq'].split()]],
                          dtype=torch.long, device=device)
    durations = torch.tensor([float(x) for x in segment['ph_dur'].split()], device=device)
    if durations.numel() != tokens.shape[1] or (durations < 0).any():
        raise ValueError('ph_dur must contain one nonnegative duration per phoneme')
    timestep = config['hop_size'] / config['audio_sample_rate']
    boundaries = torch.round(durations.cumsum(0) / timestep + 0.5).long()
    frame_durations = torch.diff(boundaries, prepend=boundaries.new_zeros(1))
    alignment = LengthRegulator()(frame_durations[None])
    frames = alignment.shape[1]
    if frames == 0:
        raise ValueError('Segment must contain at least one audio frame')
    f0 = np.asarray(segment['f0_seq'].split(), dtype=np.float32)
    f0_timestep = float(segment['f0_timestep'])
    if f0.size == 0 or not np.isfinite(f0).all() or (f0 < 0).any() or f0_timestep <= 0:
        raise ValueError('Expected nonnegative finite F0 and a positive f0_timestep')
    if config.get('interp_uv', True) and (f0 > 0).any():
        voiced = np.flatnonzero(f0 > 0)
        f0 = 2 ** np.interp(np.arange(f0.size), voiced, np.log2(f0[voiced]))
    f0 = np.interp(np.arange(frames) * timestep,
                   np.arange(f0.size) * f0_timestep, f0).astype(np.float32)
    sample = {
        'txt_tokens': tokens, 'mel2ph': alignment,
        'f0': torch.from_numpy(f0).to(device)[None] * (2 ** (transpose / 12)),
    }
    # F0 transposition controls pitch; key_shift is an independent timbre input.
    if config.get('use_key_shift_embed', False):
        sample['key_shift'] = torch.zeros(1, 1, device=device)
    if config.get('use_speed_embed', False):
        sample['speed'] = torch.ones(1, 1, device=device)
    for name in ('energy', 'breathiness'):
        if config.get(f'use_{name}_embed', False):
            raise ValueError(f'This .ds runner does not provide the required {name} curve')
    return sample


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--config', default='configs/nsf_cgmlp_vocos.yaml')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--transpose', type=float, default=0.0, help='Pitch shift in semitones')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    config = get_config(args.config)
    if config.get('backbone') != 'nsf_cgmlp_vocos':
        parser.error('Use the NSF Vocos configuration saved with the checkpoint')
    vocab = load_vocabulary(config)
    model = build_singer(config, vocab_size=max(vocab.values()) + 1).to(args.device).eval()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = checkpoint.get('state_dict', checkpoint)
    if any(k.startswith('model.') for k in state):
        state = {k[len('model.'):]: v for k, v in state.items() if k.startswith('model.')}
    model.load_state_dict(state, strict=True)
    with args.input.open(encoding='utf-8') as handle:
        segments = json.load(handle)
    if isinstance(segments, dict):
        segments = [segments]
    if not segments:
        raise ValueError('The .ds file has no segments')
    sample_rate = config['audio_sample_rate']
    rendered, cursor = [], 0
    for segment in segments:
        sample = build_sample(segment, vocab, config, args.device, args.transpose)
        wave = model(**sample, infer=True)[1][0].cpu()
        if not torch.isfinite(wave).all():
            raise ValueError('Non-finite waveform from checkpoint')
        start = round(float(segment['offset']) * sample_rate) if 'offset' in segment else cursor
        if start < 0:
            raise ValueError('Segment offset must be nonnegative')
        rendered.append((start, wave))
        cursor = max(cursor, start + wave.numel())
    output = torch.zeros(cursor)
    for start, wave in rendered:
        output[start:start + wave.numel()] += wave
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.output), output[None], sample_rate)
    print(f'Saved {args.output} ({cursor / sample_rate:.2f}s)')


if __name__ == '__main__':
    main()
