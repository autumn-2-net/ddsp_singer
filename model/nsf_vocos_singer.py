"""Text-conditioned NSF cgMLP Vocos singer, with F0 only in the DSP source."""

from torch import nn

from model.fastspeech.fastspeech import fastspeech2
from nsf_cgmlp_vocos import NSFCGMLPVocos


class NSFVocosSinger(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.fs2 = fastspeech2(vocab_size=vocab_size, config=config)
        # The vocoder's `n_mels` denotes input channels here: these are learned
        # frame features, not a supervised or externally generated mel spectrum.
        if config.get('backbone') == 'nsf_cgmlp_vocos_multistage':
            from nsf_cgmlp_vocos_multistage import MultiStageNSFCGMLPVocos
            self.vocos = MultiStageNSFCGMLPVocos(
                n_mels=config['fs2_hidden_size'],
                sample_rate=config['audio_sample_rate'],
                hop_length=config['hop_size'],
                **config.get('nsf_multistage_args', {}),
            )
        else:
            self.vocos = NSFCGMLPVocos(
                n_mels=config['fs2_hidden_size'],
                sample_rate=config['audio_sample_rate'],
                hop_length=config['hop_size'],
                n_fft=config['fft_size'],
                **config.get('nsf_vocos_args', {}),
            )

    def forward(self, f0, txt_tokens=None, mel2ph=None, key_shift=None,
                speed=None, infer=True, **kwargs):
        condition = self.fs2(
            txt_tokens=txt_tokens, mel2ph=mel2ph,
            key_shift=key_shift, speed=speed, **kwargs,
        )
        signal = self.vocos(condition, f0)
        # Preserve the task's output contract, without inventing harmonic/noise
        # components: NSF Vocos predicts one final complex spectrum.
        return condition, signal, None, None
