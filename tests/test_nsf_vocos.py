"""Small integration checks; no dataset, checkpoint or training job required."""
import unittest

import torch

from model.nsf_vocos_singer import NSFVocosSinger
from model.singer import build_singer
from model.ddsp_singer import ddsp_singer
from nsf_cgmlp_vocos import NSFCGMLPVocos


def small_config():
    return {
        'backbone': 'nsf_cgmlp_vocos',
        'fs2_hidden_size': 16, 'fs2_lays': 1, 'fs2_heads': 2,
        'fs2_dim_head': 8, 'fs2_kernel_size': 3,
        'mixenvc_hidden_size': 16, 'mixenvc_lays': 1,
        'mixenvc_heads': 2, 'mixenvc_dim_head': 8,
        'upss': True, 'useposte': True,
        'audio_sample_rate': 8000, 'hop_size': 8, 'fft_size': 32,
        'nsf_vocos_args': {
            'dim': 16, 'num_front': 1, 'num_back': 1,
            'kernel_size': 3, 'expansion': 2.0, 'dropout': 0.0,
        },
    }


class NSFVocosTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def make_inputs(self, device='cpu', frames=5):
        tokens = torch.tensor([[1, 2, 0], [2, 3, 0]], device=device)
        alignment = torch.ones(2, frames, dtype=torch.long, device=device)
        if frames > 1:
            alignment[:, frames // 2:] = 2
        f0 = torch.full((2, frames), 220.0, device=device)
        return tokens, alignment, f0

    def assert_finite_nonzero_grad(self, tensor):
        self.assertIsNotNone(tensor.grad)
        self.assertTrue(torch.isfinite(tensor.grad).all().item())
        self.assertGreater(tensor.grad.abs().sum().item(), 0.0)

    def check_backward(self, device='cpu', autocast=False):
        model = NSFVocosSinger(small_config(), vocab_size=6).to(device)
        tokens, alignment, f0 = self.make_inputs(device)
        f0.requires_grad_()
        with torch.autocast(device_type=device, dtype=torch.bfloat16,
                            enabled=autocast):
            condition, wave, harmonic, noise = model(
                f0=f0, txt_tokens=tokens, mel2ph=alignment, infer=False)
            loss = wave.square().mean()
        self.assertEqual(tuple(condition.shape), (2, 5, 16))
        self.assertEqual(tuple(wave.shape), (2, 40))
        self.assertEqual(wave.dtype, torch.float32)
        self.assertTrue(torch.isfinite(wave).all().item())
        self.assertIsNone(harmonic)
        self.assertIsNone(noise)
        loss.backward()
        self.assert_finite_nonzero_grad(model.fs2.txt_embed.weight)
        vocoders = [m for m in model.modules() if isinstance(m, NSFCGMLPVocos)]
        self.assertEqual(len(vocoders), 1)
        self.assert_finite_nonzero_grad(vocoders[0].filter_head.weight)
        self.assert_finite_nonzero_grad(vocoders[0].spectral_head.weight)
        self.assert_finite_nonzero_grad(f0)

    def test_forward_backward(self):
        self.check_backward()

    def test_f0_controls_wave_without_condition_embedding(self):
        model = NSFVocosSinger(small_config(), vocab_size=6).eval()
        self.assertFalse(any('pitch_embed' in name for name, _ in model.named_parameters()))
        self.assertFalse(hasattr(model, 'pitch_embed'))
        tokens, alignment, f0 = self.make_inputs()
        with torch.no_grad():
            first = model(f0=f0, txt_tokens=tokens, mel2ph=alignment)
            second = model(f0=f0 * 1.5, txt_tokens=tokens, mel2ph=alignment)
        torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
        self.assertGreater((first[1] - second[1]).abs().max().item(), 1e-7)

    def test_one_frame_zero_f0_and_padding(self):
        model = NSFVocosSinger(small_config(), vocab_size=6).eval()
        for frames in (1, 5):
            with self.subTest(frames=frames):
                tokens, alignment, f0 = self.make_inputs(frames=frames)
                f0.zero_()
                if frames > 1:
                    alignment[:, -2:] = 0
                with torch.no_grad():
                    condition, wave, _, _ = model(
                        f0=f0, txt_tokens=tokens, mel2ph=alignment)
                self.assertEqual(tuple(wave.shape), (2, frames * 8))
                self.assertTrue(torch.isfinite(condition).all().item())
                self.assertTrue(torch.isfinite(wave).all().item())

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                         'CUDA BF16 is unavailable')
    def test_cuda_bf16_backward(self):
        self.check_backward(device='cuda', autocast=True)

    def test_factory_new_and_legacy_default(self):
        config = small_config()
        self.assertIsInstance(build_singer(config, vocab_size=6), NSFVocosSinger)
        del config['backbone']
        self.assertIsInstance(build_singer(config, vocab_size=6), ddsp_singer)


if __name__ == '__main__':
    unittest.main()
