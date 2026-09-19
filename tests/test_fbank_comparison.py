"""Numerical migration check; TorchAudio is a TEST-ONLY dependency.

Run from the repository root:
    python -m unittest discover -s tests -p 'test_fbank_comparison.py' -v
Optional real recordings (decoded once, first channel, no resampling):
    FBANK_TEST_AUDIO=/path/a.wav:/path/b.flac python -m unittest discover \
        -s tests -p 'test_fbank_comparison.py' -v

Both backends receive identical float32 samples and dither=0. We compare raw
features and the actual CAM++ temporal mean subtraction (each backend performs
its own mean). Tolerances are engineering regression criteria, not a guarantee
of unchanged embeddings or threshold decisions. Resamplers are not tested.
"""
import os
from pathlib import Path
import sys
import unittest

import numpy as np

# Allow direct execution as well as unittest discovery.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
    import torchaudio
    import kaldi_native_fbank as knf
except ImportError as exc:
    raise ImportError(
        'This comparison requires matching torch/torchaudio and kaldi-native-fbank '
        'in the TEST environment; do not add torch to production requirements.'
    ) from exc

from fbank import fbank as native_fbank
from torchaudio.compliance.kaldi import fbank as torch_fbank


class FbankComparisonTests(unittest.TestCase):
    ATOL = 1e-3
    RTOL = 1e-4
    records = []

    @classmethod
    def setUpClass(cls):
        cls.records = []
        print(f'\nVersions: torch={torch.__version__}, torchaudio={torchaudio.__version__}, '
              f'kaldi-native-fbank={knf.__version__}, numpy={np.__version__}', flush=True)
        print(f'Acceptance: abs(native-reference) <= {cls.ATOL} + '
              f'{cls.RTOL} * abs(reference)', flush=True)

    @classmethod
    def tearDownClass(cls):
        if cls.records:
            worst = max(cls.records, key=lambda row: row[1])
            print(f'\nWorst absolute error: {worst[1]:.8g} ({worst[0]})', flush=True)

    def compare(self, name, waveform, **overrides):
        options = dict(sample_frequency=16000, num_mel_bins=80,
                       dither=0.0, subtract_mean=False)
        options.update(overrides)
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 1:
            waveform = waveform[None, :]
        with torch.no_grad():
            reference = torch_fbank(torch.from_numpy(waveform.copy()), **options)
            reference_cmn = reference - reference.mean(dim=0, keepdim=True)
        native = native_fbank(waveform.copy(), **options)
        native_cmn = native - native.mean(axis=0, keepdims=True)
        for stage, expected, actual in (
            ('raw', reference.numpy(), native),
            ('cmn', reference_cmn.numpy(), native_cmn),
        ):
            with self.subTest(case=name, stage=stage):
                self.assertEqual(expected.shape, actual.shape)
                self.assertEqual(actual.dtype, np.float32)
                self.assertGreater(actual.size, 0)
                self.assertTrue(np.isfinite(expected).all())
                self.assertTrue(np.isfinite(actual).all())
                delta = actual.astype(np.float64) - expected.astype(np.float64)
                error = np.abs(delta)
                outside = error > self.ATOL + self.RTOL * np.abs(expected)
                index = np.unravel_index(np.argmax(error), error.shape)
                self.records.append((f'{name}/{stage}', float(error.max())))
                print(f'{name:28s} {stage:3s} shape={str(actual.shape):12s} '
                      f'max={error.max():.7g} mean={error.mean():.7g} '
                      f'p99={np.quantile(error, .99):.7g} '
                      f'rmse={np.sqrt(np.mean(delta**2)):.7g} '
                      f'outside={np.count_nonzero(outside)}/{actual.size}', flush=True)
                self.assertFalse(outside.any(),
                    f'{name}/{stage}: worst={index}, reference={expected[index]}, '
                    f'native={actual[index]}, max_abs={error.max():.9g}')

    def test_camplus_waveforms(self):
        rng = np.random.default_rng(20260915)
        t = np.arange(16000, dtype=np.float64) / 16000
        noise = rng.normal(0, .1, len(t)).astype(np.float32)
        tone = (.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        impulse = np.zeros(16000, dtype=np.float32)
        impulse[[0, 399, 8000, 15999]] = 1
        cases = {
            'silence': np.zeros(16000, dtype=np.float32),
            'dc': np.full(16000, .25, dtype=np.float32),
            'noise': noise,
            'quiet_noise': noise * 1e-5,
            'pcm16_scale': noise * 32768,
            'pure_440hz': tone,
            'tone_with_noise': tone + noise * .1,
            'impulses': impulse,
            'silence_then_noise': np.concatenate([np.zeros(8000, np.float32), noise]),
        }
        for name, waveform in cases.items():
            self.compare(name, waveform)

    def test_frame_boundaries_and_channels(self):
        rng = np.random.default_rng(9)
        for length in (400, 401, 559, 560, 16001):
            x = rng.normal(0, .1, length).astype(np.float32)
            for snip in (True, False):
                self.compare(f'length={length}/snip={snip}', x, snip_edges=snip)
        stereo = rng.normal(0, .1, (2, 16000)).astype(np.float32)
        self.compare('stereo_channel_1', stereo, channel=1)

    def test_supported_options(self):
        x = np.random.default_rng(10).normal(0, .1, 16000).astype(np.float32)
        cases = [
            ('bins23', dict(num_mel_bins=23)),
            ('energy_first', dict(use_energy=True)),
            ('energy_last', dict(use_energy=True, htk_compat=True)),
            ('windowed_energy', dict(use_energy=True, raw_energy=False)),
            ('magnitude', dict(use_power=False)),
            ('linear', dict(use_log_fbank=False)),
            ('no_dc_no_preemphasis', dict(remove_dc_offset=False, preemphasis_coefficient=0)),
            ('frequency_limits', dict(low_freq=80, high_freq=-200)),
            ('wrapper_subtract_mean', dict(subtract_mean=True)),
        ]
        cases += [(w, dict(window_type=w)) for w in ('hanning', 'hamming', 'blackman', 'rectangular')]
        for name, options in cases:
            self.compare(name, x, **options)
        for sr in (8000, 22050, 44100, 48000):
            self.compare(f'sample_rate={sr}', x, sample_frequency=sr, num_mel_bins=23)

    @unittest.skip('Native 1.21.1 exits the process for round_to_power_of_two=False; not used by CAM++')
    def test_fft_without_padding(self):
        x = np.random.default_rng(11).normal(0, .1, 16000).astype(np.float32)
        self.compare('no_fft_padding', x, round_to_power_of_two=False)

    @unittest.skipUnless(os.environ.get('FBANK_TEST_AUDIO'), 'Set FBANK_TEST_AUDIO for real recordings')
    def test_real_recordings(self):
        import soundfile as sf
        for filename in os.environ['FBANK_TEST_AUDIO'].split(os.pathsep):
            audio, sr = sf.read(filename, dtype='float32', always_2d=True)
            self.compare(f'file:{Path(filename).name}', audio[:, 0], sample_frequency=sr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
