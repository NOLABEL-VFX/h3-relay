"""Reject malformed cached/VAE audio before constructing stereo guide slots."""
import importlib
from types import SimpleNamespace
import unittest

import torch
import module_fixture

m = importlib.import_module(module_fixture.PACKAGE + '.vendor.context_loop.nodes')


class ContextAudioShapeTests(unittest.TestCase):
    def test_encoder_must_return_stereo_h3_latent(self):
        waveform = {'waveform': torch.zeros(1, 1, 32000), 'sample_rate': 32000}
        for shape in ((1, 32, 1, 412), (1, 16, 2, 412), (1, 32, 2, 0)):
            vae = SimpleNamespace(audio_sample_rate=32000, encode=lambda _: torch.zeros(shape))
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, 'encoded context audio.*stereo'):
                m._encode_tail_audio(vae, waveform, 1)

    def test_valid_encoder_values_are_preserved(self):
        expected = torch.arange(1 * 32 * 2 * 40).reshape(1, 32, 2, 40).float()
        vae = SimpleNamespace(audio_sample_rate=32000, encode=lambda _: expected)
        actual, steps = m._encode_tail_audio(vae, {'waveform': torch.zeros(1, 2, 32000), 'sample_rate': 32000}, 1)
        self.assertIs(actual, expected)
        self.assertEqual(steps, 40)

    def test_cached_mono_latent_is_rejected_before_packing(self):
        latent = {'samples': [torch.zeros(1, 24, 72, 2, 2), torch.zeros(1, 32, 1, 412)]}
        with self.assertRaisesRegex(ValueError, 'context_latent audio.*stereo'):
            m._audio_tail_from_latent(latent, 18)

    def test_cached_stereo_and_unbatched_tail_remain_exact(self):
        audio = torch.arange(32 * 2 * 207).reshape(32, 2, 207).float()
        for data in (audio, audio.unsqueeze(0)):
            latent = {'samples': [torch.zeros(1, 24, 37, 2, 2), data]}
            tail, steps, _ = m._audio_tail_from_latent(latent, 18)
            self.assertEqual(steps, 30)
            self.assertTrue(torch.equal(tail, audio.unsqueeze(0)[..., -30:]))

    def test_unregistered_disk_helpers_are_not_shipped(self):
        self.assertEqual(set(m.NODE_CLASS_MAPPINGS), {'MiniMaxH3LoopTrim'})
        for name in ('_resolve_latent_path', 'MiniMaxH3MotionContextLoadLatent', 'MiniMaxH3MotionContextSaveLatent'):
            self.assertFalse(hasattr(m, name))


if __name__ == '__main__':
    unittest.main()
