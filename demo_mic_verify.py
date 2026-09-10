import argparse
import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from vad.analyzer import VADState


SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
VAD_FRAMES = 512
VAD_CHUNK_BYTES = VAD_FRAMES * CHANNELS * SAMPLE_WIDTH
VERIFY_QUEUE_SIZE = 4


@dataclass(frozen=True)
class EnrollmentCompleted:
    reference_audio: bytes


@dataclass(frozen=True)
class VerificationReady:
    audio: bytes


@dataclass(frozen=True)
class UtteranceSkipped:
    duration: float


CollectorEvent = EnrollmentCompleted | VerificationReady | UtteranceSkipped


class VoiceSegmentCollector:
    """Collect confirmed VAD speech for enrollment and later verification."""

    def __init__(
        self,
        enrollment_seconds: float = 10.0,
        min_verify_seconds: float = 1.0,
        sample_rate: int = SAMPLE_RATE,
    ):
        if enrollment_seconds <= 0:
            raise ValueError("enrollment_seconds 必须大于 0")
        if min_verify_seconds <= 0:
            raise ValueError("min_verify_seconds 必须大于 0")

        self.sample_rate = sample_rate
        self.bytes_per_second = sample_rate * CHANNELS * SAMPLE_WIDTH
        self.enrollment_target_bytes = (
            round(enrollment_seconds * sample_rate) * CHANNELS * SAMPLE_WIDTH
        )
        self.min_verify_bytes = (
            round(min_verify_seconds * sample_rate) * CHANNELS * SAMPLE_WIDTH
        )

        self._previous_state = VADState.QUIET
        self._pending_start = bytearray()
        self._pending_stop = bytearray()
        self._enrollment_audio = bytearray()
        self._reference_audio: bytes | None = None
        self._current_utterance = bytearray()

    @property
    def is_enrolled(self) -> bool:
        return self._reference_audio is not None

    @property
    def reference_audio(self) -> bytes | None:
        return self._reference_audio

    @property
    def enrollment_collected_seconds(self) -> float:
        return len(self._enrollment_audio) / self.bytes_per_second

    def process(self, pcm_chunk: bytes, state: VADState) -> list[CollectorEvent]:
        if len(pcm_chunk) % SAMPLE_WIDTH:
            raise ValueError("PCM 数据必须包含完整的 int16 样本")

        events: list[CollectorEvent] = []

        if state == VADState.STARTING:
            if self._previous_state != VADState.STARTING:
                self._pending_start.clear()
            self._pending_start.extend(pcm_chunk)
        elif state == VADState.SPEAKING:
            confirmed = bytearray()
            if self._previous_state == VADState.STARTING:
                confirmed.extend(self._pending_start)
            self._pending_start.clear()
            self._pending_stop.clear()
            confirmed.extend(pcm_chunk)
            events.extend(self._accept_confirmed(bytes(confirmed)))
        elif state == VADState.STOPPING:
            if self._previous_state != VADState.STOPPING:
                self._pending_stop.clear()
            self._pending_stop.extend(pcm_chunk)
        elif state == VADState.QUIET:
            if self._previous_state == VADState.STOPPING and self.is_enrolled:
                completed_utterance = self._finish_utterance()
                if completed_utterance is not None:
                    events.append(completed_utterance)
            self._pending_start.clear()
            self._pending_stop.clear()

        self._previous_state = state
        return events

    def _accept_confirmed(self, audio: bytes) -> list[CollectorEvent]:
        if self.is_enrolled:
            self._current_utterance.extend(audio)
            return []

        needed = self.enrollment_target_bytes - len(self._enrollment_audio)
        self._enrollment_audio.extend(audio[:needed])
        if len(self._enrollment_audio) < self.enrollment_target_bytes:
            return []

        self._reference_audio = bytes(self._enrollment_audio)
        remainder = audio[needed:]
        if remainder:
            self._current_utterance.extend(remainder)
        return [EnrollmentCompleted(self._reference_audio)]

    def _finish_utterance(self) -> CollectorEvent | None:
        audio = bytes(self._current_utterance)
        self._current_utterance.clear()
        if not audio:
            return None
        duration = len(audio) / self.bytes_per_second
        if len(audio) < self.min_verify_bytes:
            return UtteranceSkipped(duration)
        return VerificationReady(audio)


def format_verification_result(
    result: dict[str, Any], duration: float, threshold: float
) -> str:
    score = float(result["score"])
    model_decision = str(result.get("text", "")).lower()
    is_same = model_decision == "yes" if model_decision else score >= threshold
    decision = "同一说话人" if is_same else "非注册说话人"
    return (
        f"验证结果: {decision} | 时长={duration:.2f}s "
        f"| score={score:.5f} | threshold={threshold:.3f}"
    )


async def verification_worker(
    verifier: Any,
    reference_audio: bytes,
    queue: asyncio.Queue[bytes | None],
    threshold: float,
    output: Callable[[str], None] = print,
) -> None:
    while True:
        utterance = await queue.get()
        try:
            if utterance is None:
                return
            duration = len(utterance) / (SAMPLE_RATE * SAMPLE_WIDTH)
            result = await verifier.verify_pcm(
                reference_audio,
                utterance,
                sample_rate1=SAMPLE_RATE,
                pcm_format1="s16le",
            )
            output(format_verification_result(result, duration, threshold))
        except Exception as exc:
            output(f"说话人验证失败: {exc}")
        finally:
            queue.task_done()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 PyAudio、Silero VAD 和 CAM++ 测试实时说话人验证"
    )
    parser.add_argument("--list-devices", action="store_true", help="列出录音设备后退出")
    parser.add_argument("--device-index", type=int, help="PyAudio 输入设备编号")
    parser.add_argument(
        "--enrollment-seconds",
        type=float,
        default=10.0,
        help="注册所需的有效语音秒数，默认 10",
    )
    parser.add_argument(
        "--min-verify-seconds",
        type=float,
        default=1.0,
        help="提交验证的最短话段秒数，默认 1",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.31,
        help="CAM++ 同人判定阈值，默认 0.31",
    )
    return parser


def load_pyaudio():
    try:
        import pyaudio
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "缺少 PyAudio。macOS 可先执行 `brew install portaudio`，"
            "再执行 `pip install PyAudio`。"
        ) from exc
    return pyaudio


def list_input_devices(pyaudio_module) -> None:
    audio = pyaudio_module.PyAudio()
    try:
        print("可用输入设备:")
        for index in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(index)
            if int(info.get("maxInputChannels", 0)) > 0:
                print(
                    f"  [{index}] {info['name']} "
                    f"(channels={int(info['maxInputChannels'])}, "
                    f"default_rate={int(info['defaultSampleRate'])})"
                )
    finally:
        audio.terminate()


async def run_demo(args: argparse.Namespace, pyaudio_module) -> None:
    from speaker_verifier.cam_plus import SpeakerVerifier
    from vad.silero_vad import SileroVADAnalyzer

    collector = VoiceSegmentCollector(
        enrollment_seconds=args.enrollment_seconds,
        min_verify_seconds=args.min_verify_seconds,
    )

    print("正在加载 Silero VAD 和 CAM++ 模型，首次运行可能需要下载模型...")
    verifier = await SpeakerVerifier.create(threshold=args.threshold)
    analyzer = SileroVADAnalyzer(sample_rate=SAMPLE_RATE)
    analyzer.set_sample_rate(SAMPLE_RATE)

    loop = asyncio.get_running_loop()
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
    verification_queue: asyncio.Queue[bytes | None] | None = None
    worker_task: asyncio.Task[None] | None = None
    dropped_chunks = 0
    last_progress_second = -1
    pcm_buffer = bytearray()

    def enqueue_audio(chunk: bytes, status_flags: int) -> None:
        nonlocal dropped_chunks
        if status_flags:
            print(f"PyAudio 状态警告: {status_flags}")
        try:
            audio_queue.put_nowait(chunk)
        except asyncio.QueueFull:
            dropped_chunks += 1
            if dropped_chunks == 1 or dropped_chunks % 50 == 0:
                print(f"录音处理不及时，已丢弃 {dropped_chunks} 个音频块")

    def audio_callback(in_data, frame_count, time_info, status_flags):
        del frame_count, time_info
        loop.call_soon_threadsafe(enqueue_audio, bytes(in_data), status_flags)
        return (None, pyaudio_module.paContinue)

    audio = None
    stream = None
    try:
        audio = pyaudio_module.PyAudio()
        try:
            stream = audio.open(
                format=pyaudio_module.paInt16,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=args.device_index,
                frames_per_buffer=VAD_FRAMES,
                stream_callback=audio_callback,
            )
        except (OSError, ValueError) as exc:
            device = "系统默认输入设备" if args.device_index is None else args.device_index
            raise RuntimeError(
                f"无法以 16 kHz/单声道/int16 打开设备 {device}。"
                "请在 macOS 系统设置中授予终端或 VS Code 麦克风权限，"
                "并使用 --list-devices / --device-index 选择兼容设备。"
            ) from exc
        stream.start_stream()
        print("开始注册，请由主说话人持续说话；只累计 VAD 确认的语音。")

        while True:
            pcm_buffer.extend(await audio_queue.get())
            while len(pcm_buffer) >= VAD_CHUNK_BYTES:
                chunk = bytes(pcm_buffer[:VAD_CHUNK_BYTES])
                del pcm_buffer[:VAD_CHUNK_BYTES]
                state = await analyzer.analyze_audio(chunk)
                events = collector.process(chunk, state)

                if not collector.is_enrolled:
                    progress_second = int(collector.enrollment_collected_seconds)
                    if progress_second != last_progress_second:
                        last_progress_second = progress_second
                        print(
                            "注册进度: "
                            f"{collector.enrollment_collected_seconds:.1f}/"
                            f"{args.enrollment_seconds:.1f}s"
                        )

                for event in events:
                    if isinstance(event, EnrollmentCompleted):
                        print(
                            f"注册完成: 已记录 {args.enrollment_seconds:.1f}s 有效语音。"
                            "后续话段将在 VAD 尾点进行验证。"
                        )
                        verification_queue = asyncio.Queue(maxsize=VERIFY_QUEUE_SIZE)
                        worker_task = asyncio.create_task(
                            verification_worker(
                                verifier,
                                event.reference_audio,
                                verification_queue,
                                args.threshold,
                            )
                        )
                    elif isinstance(event, UtteranceSkipped):
                        print(
                            f"跳过过短话段: {event.duration:.2f}s "
                            f"< {args.min_verify_seconds:.2f}s"
                        )
                    elif isinstance(event, VerificationReady):
                        if verification_queue is None:
                            continue
                        try:
                            verification_queue.put_nowait(event.audio)
                            duration = len(event.audio) / (SAMPLE_RATE * SAMPLE_WIDTH)
                            print(f"检测到尾点，提交 {duration:.2f}s 话段进行验证...")
                        except asyncio.QueueFull:
                            print("验证队列已满，本话段已丢弃。请等待 CAM++ 推理完成。")
    finally:
        if stream is not None:
            if stream.is_active():
                stream.stop_stream()
            stream.close()
        if audio is not None:
            audio.terminate()

        if verification_queue is not None and worker_task is not None:
            await verification_queue.put(None)
            await verification_queue.join()
            await worker_task
        await analyzer.cleanup()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.enrollment_seconds <= 0:
        parser.error("--enrollment-seconds 必须大于 0")
    if args.min_verify_seconds <= 0:
        parser.error("--min-verify-seconds 必须大于 0")
    if not -1 <= args.threshold <= 1:
        parser.error("--threshold 必须在 [-1, 1] 范围内")

    try:
        pyaudio_module = load_pyaudio()
        if args.list_devices:
            list_input_devices(pyaudio_module)
            return
        asyncio.run(run_demo(args, pyaudio_module))
    except KeyboardInterrupt:
        print("\n已停止。")
    except Exception as exc:
        raise SystemExit(f"启动失败: {exc}") from exc


if __name__ == "__main__":
    main()