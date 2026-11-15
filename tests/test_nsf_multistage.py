"""Tiny multistage NSF checks; no data, checkpoints or training jobs."""
import unittest

import torch

from model.nsf_vocos_singer import NSFVocosSinger
from model.singer import build_singer
from nsf_cgmlp_vocos_multistage import MultiStageNSFCGMLPVocos


def small_multistage_args():
    return dict(dims=(8, 12, 16),
                num_front=2, num_blocks=(1, 1, 1),
                kernel_size=3, expansion=2.0, dropout=0.0)


class NSFMultistageTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)

    def make_model(self, **overrides):
        args = dict(n_mels=6, sample_rate=8000, hop_length=8, n_fft=16)
        args.update(small_multistage_args())
        args.update(overrides)
        return MultiStageNSFCGMLPVocos(**args)

    def assert_finite_nonzero_grad(self, parameter):
        self.assertIsNotNone(parameter.grad)
        self.assertTrue(torch.isfinite(parameter.grad).all().item())
        self.assertGreater(parameter.grad.abs().sum().item(), 0.)

    def check_backward(self, device='cpu', autocast=False):
        model = self.make_model().to(device)
        condition = torch.randn(2, 5, 6, device=device, requires_grad=True)
        f0 = torch.full((2, 5), 220., device=device, requires_grad=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16,
                            enabled=autocast):
            wave, aux = model(condition, f0, return_aux=True)
            loss = wave.square().mean()
        self.assertEqual(tuple(wave.shape), (2, 40))
        self.assertEqual(wave.dtype, torch.float32)
        self.assertTrue(torch.isfinite(wave).all().item())
        self.assertEqual(tuple(aux['source'].shape), (2, 40))
        self.assertEqual([tuple(x.shape) for x in aux['stage_features']],
                         [(2, 5, 8), (2, 5, 12), (2, 5, 16)])
        for key in ('source_spec', 'filter', 'excitation'):
            self.assertEqual(tuple(aux[key].shape), (2, 9, 5))
            self.assertTrue(torch.isfinite(aux[key]).all().item())
        torch.testing.assert_close(aux['excitation'], aux['source_spec'] * aux['filter'])
        self.assertEqual(tuple(aux['spec'].shape), (2, 9, 5))
        loss.backward()
        self.assert_finite_nonzero_grad(condition)
        self.assert_finite_nonzero_grad(f0)
        filters = [module for name, module in model.named_modules()
                   if name.rsplit('.', 1)[-1] == 'filter_head']
        self.assertEqual(len(filters), 1)
        for head in filters:
            self.assert_finite_nonzero_grad(head.weight)
        self.assert_finite_nonzero_grad(model.spectral_head.weight)
        self.assert_finite_nonzero_grad(model.front[0].fc1.weight)

    def test_cpu_forward_backward_and_stage_grids(self):
        self.check_backward()

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                         'CUDA BF16 is unavailable')
    def test_cuda_bf16_forward_backward(self):
        self.check_backward(device='cuda', autocast=True)

    def test_one_frame_and_zero_f0(self):
        model = self.make_model().eval()
        for frames in (1, 5):
            with self.subTest(frames=frames), torch.no_grad():
                wave = model(torch.randn(2, frames, 6), torch.zeros(2, frames))
                self.assertEqual(tuple(wave.shape), (2, frames * 8))
                self.assertTrue(torch.isfinite(wave).all().item())

    def test_final_spectrum_has_no_source_skip(self):
        model = self.make_model().eval()
        with torch.no_grad():
            model.spectral_head.weight.zero_()
            model.spectral_head.bias.zero_()
            condition = torch.randn(1, 5, 6)
            first = model(condition, torch.full((1, 5), 220.))
            second = model(condition, torch.full((1, 5), 330.))
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_head_nsf_path_survives_disabled_stage_injections(self):
        model = self.make_model().eval()
        # Isolate the extra head path: the residual streams no longer receive F0.
        with torch.no_grad():
            for stage in model.stages:
                stage.injection_scale.zero_()
        head_inputs = []
        handle = model.spectral_head.register_forward_pre_hook(
            lambda _module, args: head_inputs.append(args[0].detach().clone()))
        condition = torch.randn(1, 5, 6)
        f0 = torch.full((1, 5), 220., requires_grad=True)
        try:
            first, aux = model(condition, f0, return_aux=True)
            with torch.no_grad():
                second = model(condition, f0.detach() * 1.5)
        finally:
            handle.remove()
        self.assertEqual(tuple(head_inputs[0].shape), (1, 5, 16 + 18))
        torch.testing.assert_close(head_inputs[0][..., :16], head_inputs[1][..., :16],
                                   rtol=0, atol=0)
        expected_nsf = torch.view_as_real(
            aux['excitation'].transpose(1, 2).contiguous()
        ).reshape(1, 5, 18)
        torch.testing.assert_close(head_inputs[0][..., 16:], expected_nsf)
        self.assertGreater((first - second).abs().max().item(), 1e-7)
        first.square().mean().backward()
        self.assert_finite_nonzero_grad(f0)
        self.assert_finite_nonzero_grad(model.filter_head.weight)
        self.assertGreater(model.spectral_head.weight.grad[:, 16:].abs().sum().item(), 0.)

    def test_each_stage_injects_before_blocks(self):
        model = self.make_model().eval()
        observed = []
        handles = []
        for stage in model.stages:
            values = {}
            observed.append(values)
            handles.append(stage.register_forward_pre_hook(
                lambda _module, args, values=values: values.update(input=args[0].clone())))
            handles.append(stage.blocks.register_forward_pre_hook(
                lambda _module, args, values=values: values.update(block_input=args[0].clone())))
        try:
            with torch.no_grad():
                model(torch.randn(1, 5, 6), torch.full((1, 5), 220.))
        finally:
            for handle in handles:
                handle.remove()
        for values in observed:
            self.assertGreater((values['input'] - values['block_input']).abs().sum().item(), 0.)

    def test_channel_expansion_preserves_time_and_hop(self):
        model = self.make_model(hop_length=6).eval()
        observed = []
        handles = [stage.blocks.register_forward_pre_hook(
            lambda _module, args: observed.append(tuple(args[0].shape)))
            for stage in model.stages]
        try:
            with torch.no_grad():
                wave = model(torch.randn(1, 7, 6), torch.full((1, 7), 220.))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(observed, [(1, 7, 8), (1, 7, 12), (1, 7, 16)])
        self.assertEqual(tuple(wave.shape), (1, 42))
        self.assertEqual(model.source_stft.hop_length, 6)
        self.assertEqual(model.istft.hop_length, 6)

    def test_front_modulates_nsf_once_before_residual_stages(self):
        model = self.make_model().eval()
        # Identity initialization has zero head weights. Use a nontrivial filter
        # to test that its modulation objective backpropagates into front cgMLP.
        with torch.no_grad():
            model.filter_head.weight.normal_(0., .01)
        order, injections = [], []
        handles = [model.front.register_forward_hook(
            lambda *_: order.append('front')),
            model.source_stft.register_forward_hook(lambda *_: order.append('stft')),
            model.filter_head.register_forward_hook(lambda *_: order.append('filter'))]
        for stage in model.stages:
            def observe(_module, args):
                order.append('stage')
                injections.append(args[1])
            handles.append(stage.register_forward_pre_hook(observe))
        try:
            _, aux = model(torch.randn(1, 5, 6), torch.full((1, 5), 220.), return_aux=True)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(order, ['front', 'stft', 'filter', 'stage', 'stage', 'stage'])
        self.assertTrue(all(x is injections[0] for x in injections))
        aux['excitation'].abs().square().mean().backward()
        self.assert_finite_nonzero_grad(model.front[0].fc1.weight)
        self.assert_finite_nonzero_grad(model.filter_head.weight)

    def test_factory_uses_multistage_singer(self):
        config = {
            'backbone': 'nsf_cgmlp_vocos_multistage',
            'fs2_hidden_size': 16, 'fs2_lays': 1, 'fs2_heads': 2,
            'fs2_dim_head': 8, 'fs2_kernel_size': 3,
            'mixenvc_hidden_size': 16, 'mixenvc_lays': 1,
            'mixenvc_heads': 2, 'mixenvc_dim_head': 8,
            'upss': True, 'useposte': True,
            'audio_sample_rate': 8000, 'hop_size': 8, 'fft_size': 32,
            'nsf_multistage_args': small_multistage_args(),
        }
        singer = build_singer(config, vocab_size=6)
        self.assertIsInstance(singer, NSFVocosSinger)
        self.assertIsInstance(singer.vocos, MultiStageNSFCGMLPVocos)
        with torch.no_grad():
            _, wave, harmonic, noise = singer(
                f0=torch.full((1, 3), 220.),
                txt_tokens=torch.tensor([[1, 2]]),
                mel2ph=torch.tensor([[1, 1, 2]]))
        self.assertEqual(tuple(wave.shape), (1, 24))
        self.assertTrue(torch.isfinite(wave).all().item())
        self.assertIsNone(harmonic)
        self.assertIsNone(noise)
        self.assertEqual(singer.vocos.source_stft.n_fft, config['fft_size'])
        self.assertEqual(singer.vocos.istft.n_fft, config['fft_size'])
        self.assertEqual(singer.vocos.filter_head.out_features, config['fft_size'] + 2)
        self.assertEqual(singer.vocos.spectral_head.out_features, config['fft_size'] + 2)
        config['fft_size'] = 16
        smaller = build_singer(config, vocab_size=6)
        self.assertEqual(smaller.vocos.source_stft.n_fft, 16)
        self.assertEqual(smaller.vocos.istft.n_fft, 16)
        self.assertEqual(smaller.vocos.filter_head.out_features, 18)
        self.assertEqual(smaller.vocos.spectral_head.out_features, 18)


if __name__ == '__main__':
    unittest.main()
