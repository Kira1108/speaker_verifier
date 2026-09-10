import asyncio
import unittest

from demo_mic_verify import (
    EnrollmentCompleted,
    SAMPLE_WIDTH,
    UtteranceSkipped,
    VerificationReady,
    VoiceSegmentCollector,
    verification_worker,
)
from vad.analyzer import VADAnalyzer, VADState


SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512


def pcm_chunk(value: int, samples: int = CHUNK_SAMPLES) -> bytes:
    return int(value).to_bytes(2, "little", signed=True) * samples


class VoiceSegmentCollectorTests(unittest.TestCase):
    def test_accumulates_enrollment_across_utterances(self):
        collector = VoiceSegmentCollector(enrollment_seconds=0.128)
        chunks = [pcm_chunk(value) for value in range(1, 9)]

        self.assertEqual(collector.process(chunks[0], VADState.STARTING), [])
        self.assertEqual(collector.process(chunks[1], VADState.SPEAKING), [])
        collector.process(chunks[2], VADState.STOPPING)
        collector.process(chunks[3], VADState.QUIET)
        collector.process(chunks[4], VADState.STARTING)
        events = collector.process(chunks[5], VADState.SPEAKING)

        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], EnrollmentCompleted)
        self.assertEqual(
            events[0].reference_audio,
            chunks[0] + chunks[1] + chunks[4] + chunks[5],
        )

    def test_discards_unconfirmed_starting_audio(self):
        collector = VoiceSegmentCollector(enrollment_seconds=0.064)
        rejected = pcm_chunk(9)
        first = pcm_chunk(1)
        second = pcm_chunk(2)

        collector.process(rejected, VADState.STARTING)
        collector.process(pcm_chunk(0), VADState.QUIET)
        collector.process(first, VADState.STARTING)
        events = collector.process(second, VADState.SPEAKING)

        self.assertEqual(events[0].reference_audio, first + second)
        self.assertNotIn(rejected, events[0].reference_audio)

    def test_splits_enrollment_inside_confirmed_audio(self):
        target_samples = 600
        collector = VoiceSegmentCollector(
            enrollment_seconds=target_samples / SAMPLE_RATE,
            min_verify_seconds=0.01,
        )
        first = pcm_chunk(1)
        second = pcm_chunk(2)
        third = pcm_chunk(3)

        collector.process(first, VADState.STARTING)
        events = collector.process(second, VADState.SPEAKING)
        completed = events[0]
        expected_all = first + second
        target_bytes = target_samples * SAMPLE_WIDTH

        self.assertEqual(completed.reference_audio, expected_all[:target_bytes])

        collector.process(third, VADState.SPEAKING)
        self.assertEqual(collector.process(pcm_chunk(0), VADState.STOPPING), [])
        endpoint_events = collector.process(pcm_chunk(0), VADState.QUIET)

        self.assertEqual(len(endpoint_events), 1)
        self.assertIsInstance(endpoint_events[0], VerificationReady)
        self.assertEqual(
            endpoint_events[0].audio,
            expected_all[target_bytes:] + third,
        )

    def test_submits_only_at_confirmed_endpoint(self):
        collector = VoiceSegmentCollector(
            enrollment_seconds=CHUNK_SAMPLES / SAMPLE_RATE,
            min_verify_seconds=CHUNK_SAMPLES / SAMPLE_RATE,
        )
        collector.process(pcm_chunk(1), VADState.SPEAKING)
        self.assertTrue(collector.is_enrolled)

        collector.process(pcm_chunk(2), VADState.SPEAKING)
        self.assertEqual(collector.process(pcm_chunk(0), VADState.STOPPING), [])
        events = collector.process(pcm_chunk(0), VADState.QUIET)

        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], VerificationReady)

    def test_does_not_emit_empty_utterance_after_enrollment(self):
        collector = VoiceSegmentCollector(
            enrollment_seconds=CHUNK_SAMPLES / SAMPLE_RATE,
        )

        collector.process(pcm_chunk(1), VADState.SPEAKING)
        collector.process(pcm_chunk(0), VADState.STOPPING)
        events = collector.process(pcm_chunk(0), VADState.QUIET)

        self.assertEqual(events, [])

    def test_skips_short_utterance(self):
        collector = VoiceSegmentCollector(
            enrollment_seconds=CHUNK_SAMPLES / SAMPLE_RATE,
            min_verify_seconds=1.0,
        )
        collector.process(pcm_chunk(1), VADState.SPEAKING)
        collector.process(pcm_chunk(2), VADState.SPEAKING)
        collector.process(pcm_chunk(0), VADState.STOPPING)
        events = collector.process(pcm_chunk(0), VADState.QUIET)

        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], UtteranceSkipped)
        self.assertAlmostEqual(events[0].duration, CHUNK_SAMPLES / SAMPLE_RATE)


class FakeVerifier:
    def __init__(self):
        self.calls = []

    async def verify_pcm(self, reference, utterance, **kwargs):
        self.calls.append((reference, utterance, kwargs))
        return {"score": 0.81234, "text": "yes"}


class VerificationWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_verifies_and_formats_result(self):
        verifier = FakeVerifier()
        queue = asyncio.Queue()
        output = []
        reference = pcm_chunk(1)
        utterance = pcm_chunk(2)
        await queue.put(utterance)
        await queue.put(None)

        await verification_worker(verifier, reference, queue, 0.31, output.append)

        self.assertEqual(len(verifier.calls), 1)
        self.assertEqual(verifier.calls[0][0], reference)
        self.assertEqual(verifier.calls[0][1], utterance)
        self.assertIn("同一说话人", output[0])
        self.assertIn("score=0.81234", output[0])


class FakeAnalyzer(VADAnalyzer):
    def num_frames_required(self):
        return CHUNK_SAMPLES

    def voice_confidence(self, buffer):
        return 0.0


class AnalyzerCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_cleanup_is_idempotent(self):
        analyzer = FakeAnalyzer(sample_rate=SAMPLE_RATE)
        analyzer.set_sample_rate(SAMPLE_RATE)

        await analyzer.cleanup()
        await analyzer.cleanup()

        self.assertTrue(analyzer._executor_shutdown)


if __name__ == "__main__":
    unittest.main()