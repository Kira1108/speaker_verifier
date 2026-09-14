from dataclasses import dataclass
from vad.analyzer import VADState


SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
VAD_FRAMES = 512
VAD_CHUNK_BYTES = VAD_FRAMES * CHANNELS * SAMPLE_WIDTH
VERIFY_QUEUE_SIZE = 4


@dataclass(frozen=True)
class EnrollmentCompleted:
    """Enrollment has been completed and a reference audio sample is available."""
    reference_audio: bytes


@dataclass(frozen=True)
class VerificationReady:
    """A new utterance is ready for verification."""
    audio: bytes


@dataclass(frozen=True)
class UtteranceSkipped:
    """An utterance was skipped because of insufficient length."""
    duration: float


CollectorEvent = EnrollmentCompleted | VerificationReady | UtteranceSkipped


class VoiceSegmentCollector:
    """Collect confirmed VAD speech for enrollment and later verification.
    
    Args:
        enrollment_seconds: The number of seconds of confirmed speech to collect for enrollment.
        min_verify_seconds: The minimum number of seconds of confirmed speech to accept for verification.
        sample_rate: The audio sample rate in Hz (default: 16000).
    
    """

    def __init__(
        self,
        enrollment_seconds: float = 10.0,
        min_verify_seconds: float = 1.0,
        sample_rate: int = SAMPLE_RATE,
    ):
        
        # must record at least enrollment_seconds of confirmed speech before enrollment is complete.
        if enrollment_seconds <= 0:
            raise ValueError("enrollment_seconds 必须大于 0")
        
        # short utterances are skipped and not sent for verification.
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
        """Reference audio is available for verification."""
        return self._reference_audio is not None

    @property
    def reference_audio(self) -> bytes | None:
        """Get the reference audio collected during enrollment."""
        return self._reference_audio

    @property
    def enrollment_collected_seconds(self) -> float:
        """Get the number of seconds of confirmed speech collected for enrollment."""
        return len(self._enrollment_audio) / self.bytes_per_second

    def process(self, pcm_chunk: bytes, state: VADState) -> list[CollectorEvent]:
        
        if len(pcm_chunk) % SAMPLE_WIDTH:
            raise ValueError("PCM 数据必须包含完整的 int16 样本")

        events: list[CollectorEvent] = []

        if state == VADState.STARTING:
            
            # 刚刚识别到有声音（并未转入speaking状态）时，清空pending start队列，开始收集有效音频。
            if self._previous_state != VADState.STARTING:
                self._pending_start.clear()
            
            # 所有starting的音频都要积累
            self._pending_start.extend(pcm_chunk)
           
        elif state == VADState.SPEAKING:
            # 真正转入speaking的时候，已经积累一部分的有效音频
            confirmed = bytearray()
            
            # 如果转入speaking之前已经有音频，则appendingpending start音频
            if self._previous_state == VADState.STARTING:
                confirmed.extend(self._pending_start)  
             
            # 如果转入speaking之前是stopping状态，则说明之前的音频是有效的，应该把pending stop的音频加入到confirmed中   
            if self._previous_state == VADState.STOPPING:
                confirmed.extend(self._pending_stop)
                
            # 此时属于说话中，可以放心清空pending start
            self._pending_start.clear()
            
            # 另一种情况是从stopping转回speaking，这种情况也需要把pending stop的音频加入到confirmed中
            self._pending_stop.clear()
            confirmed.extend(pcm_chunk)
            events.extend(self._accept_confirmed(bytes(confirmed)))
            
            
        elif state == VADState.STOPPING:

            # 刚刚进入stopping的时候，做一次stopping队列的清理
            if self._previous_state != VADState.STOPPING:
                self._pending_stop.clear()
                
            # 所有stopping的音频都要积累
            self._pending_stop.extend(pcm_chunk)
            
        elif state == VADState.QUIET:
            
            # quiet有两种可能：
            # 1. stopping -> quiet 确认静音， 如果当前确认静音， 并且reference audio已经存在，则确认utternace即可
            # 2. starting -> quiet 启动失败， 这种情况， 不处理即可。
            if self._previous_state == VADState.STOPPING and self.is_enrolled:
                completed_utterance = self._finish_utterance()
                if completed_utterance is not None:
                    events.append(completed_utterance)
                    
            # 一旦到达quiet状态，清空pending start和pending stop队列
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
