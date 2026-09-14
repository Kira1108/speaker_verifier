import unittest
from unittest.mock import Mock, patch

import numpy as np

from enroller.pcm_vad import PcmVad


class PcmVadEnrollmentTests(unittest.TestCase):
    """Use controlled VAD probabilities; no model loading or downloads."""

    def make_vad(self, probabilities, sample_rate=16000,
                 return_seconds=True, **settings):
        calls = 0

        def infer(_output_names, inputs):
            nonlocal calls
            probability = probabilities[calls % len(probabilities)]
            calls += 1
            return np.array([[probability]], dtype=np.float32), inputs["state"]

        session = Mock()
        session.run.side_effect = infer
        config = {
            "min_speech_duration_ms": 32,
            "min_silence_duration_ms": 64,
            "speech_pad_ms": 30,
        }
        config.update(settings)
        with patch("enroller.pcm_vad.ort.InferenceSession", return_value=session):
            vad = PcmVad(None, sample_rate=sample_rate,
                         return_seconds=return_seconds, **config)
        return vad, session

    @staticmethod
    def make_pcm(samples):
        # Distinct samples expose incorrect offsets or reordered segments.
        return np.arange(samples, dtype="<i2").tobytes()

    def test_enroll_uses_one_pass_and_exact_slices_in_both_units(self):
        probabilities = [0.0] * 2 + [0.9] * 4 + [0.0] * 5 + [0.9] * 5 + [0.0] * 4
        for sample_rate in (8000, 16000):
            for return_seconds in (False, True):
                with self.subTest(sample_rate=sample_rate, seconds=return_seconds):
                    vad, session = self.make_vad(
                        probabilities, sample_rate, return_seconds,
                    )
                    frame = 256 if sample_rate == 8000 else 512
                    padding = sample_rate * 30 // 1000
                    pcm = self.make_pcm(len(probabilities) * frame)
                    expected = (
                        pcm[(2 * frame - padding) * 2:(6 * frame + padding) * 2]
                        + pcm[(11 * frame - padding) * 2:(16 * frame + padding) * 2]
                    )
                    duration = len(expected) / (sample_rate * 2)

                    result = vad.enroll(pcm, min_enroll_seconds=duration)

                    self.assertEqual(result.pcm, expected)
                    self.assertEqual(result.duration, duration)
                    self.assertIs(result.success, True)
                    self.assertEqual(len(result.pcm) % 2, 0)
                    self.assertEqual(session.run.call_count, len(probabilities))

    def test_compute_length_always_returns_seconds(self):
        for sample_rate in (8000, 16000):
            for return_seconds in (False, True):
                with self.subTest(sample_rate=sample_rate, seconds=return_seconds):
                    vad, _ = self.make_vad([0.9], sample_rate, return_seconds)
                    pcm = self.make_pcm(sample_rate // 2)
                    self.assertEqual(vad.compute_speech_length(pcm), 0.5)
                    self.assertEqual(vad.enroll(pcm).duration, 0.5)

    def test_public_timestamps_preserve_selected_units(self):
        probabilities = [0.0] * 2 + [0.9] * 4 + [0.0] * 4
        pcm = self.make_pcm(len(probabilities) * 512)
        expected = [{"start": 544, "end": 3552}]
        for return_seconds in (False, True):
            with self.subTest(seconds=return_seconds):
                vad, _ = self.make_vad(probabilities, return_seconds=return_seconds)
                timestamps = vad.timestamps(pcm)
                if return_seconds:
                    self.assertEqual(timestamps, [
                        {key: value / 16000 for key, value in expected[0].items()}
                    ])
                else:
                    self.assertEqual(timestamps, expected)

    def test_empty_audio_is_unsuccessful_without_inference(self):
        vad, session = self.make_vad([0.9])
        result = vad.enroll(b"")
        self.assertEqual(result.duration, 0.0)
        self.assertEqual(result.pcm, b"")
        self.assertIs(result.success, False)
        self.assertEqual(vad.compute_speech_length(b""), 0.0)
        self.assertEqual(vad.timestamps(b""), [])
        session.run.assert_not_called()

    def test_no_speech_returns_empty_audio(self):
        vad, session = self.make_vad([0.0])
        result = vad.enroll(self.make_pcm(5120))
        self.assertEqual(result.duration, 0.0)
        self.assertEqual(result.pcm, b"")
        self.assertIs(result.success, False)
        self.assertEqual(session.run.call_count, 10)

    def test_insufficient_duration_still_returns_retained_audio(self):
        vad, _ = self.make_vad([0.9])
        pcm = self.make_pcm(8000)
        result = vad.enroll(pcm, min_enroll_seconds=1.0)
        self.assertEqual(result.pcm, pcm)
        self.assertEqual(result.duration, 0.5)
        self.assertIs(result.success, False)

    def test_threshold_boundary_is_inclusive(self):
        vad, _ = self.make_vad([0.9])
        self.assertIs(vad.enroll(self.make_pcm(15999)).success, False)
        self.assertIs(vad.enroll(self.make_pcm(16000)).success, True)
        self.assertIs(vad.enroll(self.make_pcm(16001)).success, True)

    def test_eof_padding_is_not_included_in_output(self):
        vad, session = self.make_vad([0.9])
        pcm = self.make_pcm(1001)
        result = vad.enroll(pcm, min_enroll_seconds=0.01)
        self.assertEqual(result.pcm, pcm)
        self.assertEqual(result.duration, 1001 / 16000)
        self.assertEqual(session.run.call_count, 2)

    def test_short_internal_pause_is_retained(self):
        probabilities = [0.9] * 3 + [0.0] + [0.9] * 3
        vad, _ = self.make_vad(probabilities)
        pcm = self.make_pcm(len(probabilities) * 512)
        result = vad.enroll(pcm)
        self.assertEqual(result.pcm, pcm)
        self.assertEqual(result.duration, len(pcm) / 32000)

    def test_padding_does_not_duplicate_samples_in_short_gap(self):
        probabilities = [0.9] * 3 + [0.0] * 3 + [0.9] * 3
        vad, _ = self.make_vad(
            probabilities, return_seconds=False, speech_pad_ms=80,
        )
        pcm = self.make_pcm(len(probabilities) * 512)
        segments = vad.timestamps(pcm)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["end"], segments[1]["start"])
        result = vad.enroll(pcm)
        self.assertEqual(result.pcm, pcm)
        self.assertEqual(result.duration, len(pcm) / 32000)

    def test_invalid_minimum_duration_is_rejected_before_inference(self):
        vad, session = self.make_vad([0.9])
        for minimum in (0, -1, float("nan"), float("inf"), -float("inf"),
                        True, False, np.bool_(True)):
            with self.subTest(minimum=minimum):
                with self.assertRaises(ValueError):
                    vad.enroll(b"", min_enroll_seconds=minimum)
        session.run.assert_not_called()

    def test_incomplete_pcm_sample_is_rejected(self):
        vad, session = self.make_vad([0.9])
        for method in (vad.timestamps, vad.compute_speech_length, vad.enroll):
            with self.subTest(method=method.__name__):
                with self.assertRaises(ValueError):
                    method(b"\x00")
        session.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()