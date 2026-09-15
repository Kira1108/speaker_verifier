"""Offline Silero ONNX VAD for mono signed 16-bit little-endian raw PCM.

Dependencies: onnxruntime and its NumPy dependency. No runtime downloads.
Model: snakers4/silero-vad, src/silero_vad/data/silero_vad.onnx.
Implements hysteresis segmentation without a maximum segment duration.
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from pydantic import BaseModel


DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "vad" / "models" / "silero_vad.onnx"


class EnrollResult(BaseModel):
    """音频准备结果，不代表声纹提取、保存或音频质量检查成功。

    duration: 保留 PCM 的时长（秒），包含边界保护和片段内的短暂停顿。
    pcm: 按原顺序拼接的单声道 s16le PCM；时长不足时也返回。
    success: 仅表示 duration 达到要求的最小时长。
    """

    duration: float
    pcm: bytes
    success: bool


class PcmVad:
    def __init__(self, model_path, sample_rate=16000, *, threshold=0.5,
                 min_speech_duration_ms=250, min_silence_duration_ms=100,
                 speech_pad_ms=30, return_seconds=True):
        """Configure CPU inference and segmentation for all calls.

        return_seconds=False selects integer sample offsets for exact slicing.
        """
        if sample_rate not in (8000, 16000):
            raise ValueError("PCM must already be sampled at 8000 or 16000 Hz")
        if not 0.15 <= threshold <= 1:
            raise ValueError("threshold must be between 0.15 and 1")
        durations = (min_speech_duration_ms, min_silence_duration_ms, speech_pad_ms)
        if any(not np.isfinite(v) or v < 0 for v in durations):
            raise ValueError("Durations must be finite and nonnegative")
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.return_seconds = return_seconds
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def timestamps(self, pcm):
        """Return [{start, end}, ...], with exclusive ends.

        pcm: bytes containing mono s16le samples, WITHOUT a WAV header.
        Each call is an independent recording; recurrent state is local.
        Units and segmentation settings are configured in the initializer.
        Retained segments include boundary padding and short internal pauses.
        """
        segments = self._timestamps_samples(pcm)
        if self.return_seconds:
            return [
                {"start": seg["start"] / self.sample_rate,
                 "end": seg["end"] / self.sample_rate}
                for seg in segments
            ]
        return segments

    def _timestamps_samples(self, pcm):
        """Detect retained segments once, using exact integer sample offsets."""
        sample_rate = self.sample_rate
        threshold = self.threshold
        min_speech_duration_ms = self.min_speech_duration_ms
        min_silence_duration_ms = self.min_silence_duration_ms
        speech_pad_ms = self.speech_pad_ms
        if len(pcm) % 2:
            raise ValueError("s16le PCM byte length must be divisible by 2")
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        length = len(audio)
        if not length:
            return []

        frame_size = 512 if sample_rate == 16000 else 256
        context_size = 64 if sample_rate == 16000 else 32
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, context_size), dtype=np.float32)
        sr = np.array(sample_rate, dtype=np.int64)
        minimum = sample_rate * min_speech_duration_ms / 1000
        silence = sample_rate * min_silence_duration_ms / 1000
        padding = int(sample_rate * speech_pad_ms / 1000)
        exit_threshold = max(0.01, threshold - 0.15)
        segments = []
        start = quiet_start = None

        for offset in range(0, length, frame_size):
            frame = np.zeros((1, frame_size), dtype=np.float32)
            part = audio[offset:offset + frame_size]
            frame[0, :len(part)] = part
            model_input = np.concatenate((context, frame), axis=1)
            probability, state = self.session.run(
                None, {"input": model_input, "state": state, "sr": sr}
            )
            context = model_input[:, -context_size:].copy()
            probability = float(probability.reshape(-1)[0])

            if probability >= threshold:
                quiet_start = None
                if start is None:
                    start = offset
            elif start is not None and probability < exit_threshold:
                if quiet_start is None:
                    quiet_start = offset
                if offset - quiet_start >= silence:
                    if quiet_start - start > minimum:
                        segments.append({"start": start, "end": quiet_start})
                    start = quiet_start = None

        # EOF closes an active segment at the REAL length, not the padded length.
        if start is not None and length - start > minimum:
            segments.append({"start": start, "end": length})

        # Add boundary padding; divide short gaps so segments never overlap.
        result = []
        for index, segment in enumerate(segments):
            left = max(0, segment["start"] - padding)
            right = min(length, segment["end"] + padding)
            if index:
                previous_end = segments[index - 1]["end"]
                left = max(left, (previous_end + segment["start"]) // 2)
            if index + 1 < len(segments):
                next_start = segments[index + 1]["start"]
                right = min(right, (segment["end"] + next_start) // 2)
            result.append({"start": left, "end": right})
        return result

    def compute_speech_length(self, pcm) -> float:
        """Return retained audio duration in seconds, regardless of return_seconds.

        Includes boundary padding and short pauses within retained segments.
        """
        segments = self._timestamps_samples(pcm)
        samples = sum(seg["end"] - seg["start"] for seg in segments)
        return samples / self.sample_rate

    def enroll(
        self,
        pcm: bytes,
        min_enroll_seconds: float = 1.0,
    ) -> EnrollResult:
        """提取注册用音频；success 仅表示保留音频时长达标。

        只执行一次 VAD，按整数样本位置拼接，不受 return_seconds 影响。
        min_enroll_seconds 必须是有限正数。时长不足仍返回提取的音频；
        没有保留片段时返回 duration=0.0、pcm=b""、success=False。
        """
        if (
            isinstance(min_enroll_seconds, (bool, np.bool_))
            or not np.isfinite(min_enroll_seconds)
            or min_enroll_seconds <= 0
        ):
            raise ValueError("min_enroll_seconds must be finite and positive")

        segments = self._timestamps_samples(pcm)
        # s16le 每个样本占 2 字节，直接用整数位置避免浮点换算及半样本切片。
        speech_pcm = b"".join(
            pcm[seg["start"] * 2:seg["end"] * 2] for seg in segments
        )
        duration = len(speech_pcm) / (self.sample_rate * 2)

        return EnrollResult(
            duration=duration,
            pcm=speech_pcm,
            success=bool(duration >= min_enroll_seconds),
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Local silero_vad.onnx path")
    parser.add_argument("pcm", help="Raw mono s16le PCM path")
    parser.add_argument("--sr", type=int, choices=[8000, 16000], default=16000)
    args = parser.parse_args()
    vad = PcmVad(args.model, sample_rate=args.sr)
    print(json.dumps(vad.timestamps(Path(args.pcm).read_bytes()), indent=2))
