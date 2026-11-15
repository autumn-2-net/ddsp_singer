"""Exercise the NSF task without reading dictionaries, data or checkpoints."""
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from test_nsf_vocos import small_config
from utils_model.ddsp_sinder_task import ddsp_ss


class FixedWave(torch.nn.Module):
    def __init__(self, wave):
        super().__init__()
        self.wave = torch.nn.Parameter(wave.clone())

    def forward(self, **kwargs):
        return None, self.wave, None, None


class NSFTaskTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)

    def make_task(self, **overrides):
        config = small_config()
        config.update(dict_path=['unused.txt'], data_index_path='unused',
                      win_size=32, fmin=0, fmax=4000,
                      audio_num_mel_bins=8, nsf_loss_fft_sizes=[32, 16])
        config.update(overrides)
        with patch('utils_model.ddsp_sinder_task.load_dict_list', return_value=['a', 'b']), \
                patch('utils_model.ddsp_sinder_task.SVS_Dataset', return_value=[]):
            task = ddsp_ss(config)
        task.build_losses_and_metrics()
        return task

    def sample(self, frames=5):
        alignment = torch.ones(2, frames, dtype=torch.long)
        if frames > 1:
            alignment[:, frames // 2:] = 2
        return {'ph_idx': torch.tensor([[1, 2], [2, 1]]),
                'mel2ph': alignment, 'f0': torch.full((2, frames), 220.),
                'wav': torch.randn(2, frames * 8) * .1,
                'wavmask': torch.ones(2, frames * 8, dtype=torch.bool)}

    def test_actual_task_weighted_loss_and_backward(self):
        task = self.make_task(lambda_l1=2.5, lambda_mstft=.4)
        sample = self.sample()
        with torch.no_grad():
            wave = task.run_model(sample, infer=True)
            expected = (2.5 * F.l1_loss(wave, sample['wav'])
                        + .4 * task.wav_loss(wave, sample['wav']))
        loss = task.run_model(sample, inx=True)['wav_loss']
        torch.testing.assert_close(loss, expected)
        loss.backward()
        for parameter in (task.model.fs2.txt_embed.weight,
                          task.model.vocos.filter_head.weight,
                          task.model.vocos.spectral_head.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())
            self.assertGreater(parameter.grad.abs().sum().item(), 0.)

    def test_padding_has_no_loss_or_gradient_contribution(self):
        task = self.make_task()
        sample = self.sample()
        sample['wavmask'][0, 24:] = False
        sample['wavmask'][1, 16:] = False
        task.model = FixedWave(torch.randn_like(sample['wav']))
        before = task.run_model(sample, inx=True)['wav_loss']
        before.backward()
        invalid = ~sample['wavmask']
        torch.testing.assert_close(task.model.wave.grad[invalid],
                                   torch.zeros_like(task.model.wave.grad[invalid]))
        with torch.no_grad():
            task.model.wave[invalid] = 10000.
        sample['wav'][invalid] = -10000.
        after = task.run_model(sample, inx=True)['wav_loss']
        torch.testing.assert_close(before, after)

    def test_masked_l1_divides_by_valid_samples(self):
        task = self.make_task(lambda_l1=3., lambda_mstft=0.)
        sample = self.sample()
        sample['wav'].fill_(1.)
        sample['wavmask'][:, 8:] = False
        prediction = torch.full_like(sample['wav'], 99.)
        prediction[:, :8] = 3.
        task.model = FixedWave(prediction)
        loss = task.run_model(sample, inx=True)['wav_loss']
        torch.testing.assert_close(loss, torch.tensor(6.))
        loss.backward()
        torch.testing.assert_close(task.model.wave.grad[:, :8],
                                   torch.full((2, 8), 3. / 16.))
        torch.testing.assert_close(task.model.wave.grad[:, 8:],
                                   torch.zeros(2, 32))

    def test_one_frame_shorter_than_fft_and_inference(self):
        task = self.make_task()
        self.assertEqual(task.lambda_l1, 1.)
        self.assertEqual(task.lambda_mstft, 1.)
        sample = self.sample(frames=1)
        wave = task.run_model(sample, infer=True)
        self.assertEqual(tuple(wave.shape), (2, 8))
        self.assertTrue(torch.isfinite(wave).all().item())
        loss = task.run_model(sample, inx=True)['wav_loss']
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(torch.isfinite(task.model.vocos.spectral_head.weight.grad).all().item())


if __name__ == '__main__':
    unittest.main()
