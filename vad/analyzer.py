import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import logging
logger = logging.getLogger()    
from pydantic import BaseModel

from utils import (
    calculate_audio_volume, 
    exp_smoothing
)

VAD_CONFIDENCE = 0.7
VAD_START_SECS = 0.2
VAD_STOP_SECS = 0.2
VAD_MIN_VOLUME = 0.6


class VADState(Enum):
    """Voice Activity Detection states.

    Parameters:
        QUIET: No voice activity detected.
        STARTING: Voice activity beginning, transitioning from quiet.
        SPEAKING: Active voice detected and confirmed.
        STOPPING: Voice activity ending, transitioning to quiet.
    """

    QUIET = 1
    STARTING = 2
    SPEAKING = 3
    STOPPING = 4


class VADParams(BaseModel):
    """Configuration parameters for Voice Activity Detection.

    Parameters:
        confidence: Minimum confidence threshold for voice detection.
        start_secs: Duration to wait before confirming voice start.
        stop_secs: Duration to wait before confirming voice stop.
        min_volume: Minimum audio volume threshold for voice detection. （我们希望暴露一个参数，控制最小的声响）
    """

    confidence: float = VAD_CONFIDENCE
    start_secs: float = VAD_START_SECS
    stop_secs: float = VAD_STOP_SECS
    min_volume: float = VAD_MIN_VOLUME


class VADAnalyzer(ABC):
    """Abstract base class for Voice Activity Detection analyzers.

    Provides the framework for implementing VAD analysis with configurable
    parameters, state management, and audio processing capabilities.
    Subclasses must implement the core voice confidence calculation.
    """

    def __init__(self, *, sample_rate: int | None = None, params: VADParams | None = None):
        """Initialize the VAD analyzer.

        Args:
            sample_rate: Audio sample rate in Hz. If None, will be set later.
            params: VAD parameters for detection configuration.
        """
        self._init_sample_rate = sample_rate
        self._sample_rate = 0
        self._params = params or VADParams()
        self._num_channels = 1

        self._vad_buffer = b""

        # Volume exponential smoothing
        self._smoothing_factor = 0.2
        self._prev_volume = 0

        # Thread executor that will run the model. We only need one thread per
        # analyzer because one analyzer just handles one audio stream.
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._executor_shutdown = False

    @property
    def sample_rate(self) -> int:
        """Get the current sample rate.

        Returns:
            Current audio sample rate in Hz.
        """
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        """Get the number of audio channels.

        Returns:
            Number of audio channels (always 1 for mono).
        """
        return self._num_channels

    @property
    def params(self) -> VADParams:
        """Get the current VAD parameters.

        Returns:
            Current VAD configuration parameters.
        """
        return self._params

    @abstractmethod
    def num_frames_required(self) -> int:
        """Get the number of audio frames required for analysis.

        Returns:
            Number of frames needed for VAD processing.
        """
        pass

    @abstractmethod
    def voice_confidence(self, buffer: bytes) -> float:
        """Calculate voice activity confidence for the given audio buffer.

        Args:
            buffer: Audio buffer to analyze.

        Returns:
            Voice confidence score between 0.0 and 1.0.
        """
        pass

    def set_sample_rate(self, sample_rate: int):
        """Set the sample rate for audio processing.

        Args:
            sample_rate: Audio sample rate in Hz.
        """
        self._sample_rate = self._init_sample_rate or sample_rate
        self.set_params(self._params)

    def set_params(self, params: VADParams):
        """Set VAD parameters and recalculate internal values.

        Args:
            params: VAD parameters for detection configuration.
        """
        logger.debug(f"Setting VAD params to: {params}")
        self._params:VADParams = params
        
        # number of frames required for each analysis window
        self._vad_frames:int = self.num_frames_required()
        
        # number of bytes required for each analysis window
        self._vad_frames_num_bytes:int = self._vad_frames * self._num_channels * 2

        # number of analysis windows per second
        vad_frames_per_sec:float = self._vad_frames / self.sample_rate

        # number of frames required for audio starting detection
        self._vad_start_frames:int = round(self._params.start_secs / vad_frames_per_sec)
        
        # number of frames required for audio stopping detection
        self._vad_stop_frames:int = round(self._params.stop_secs / vad_frames_per_sec)
        
        # initialize VAD state counters and current state
        self._vad_starting_count:int = 0
        self._vad_stopping_count:int = 0
        
        # current VAD state
        self._vad_state: VADState = VADState.QUIET

    def _get_smoothed_volume(self, audio: bytes) -> float:
        """Calculate smoothed audio volume using exponential smoothing."""
        
        # calculate the current volume of the audio buffer
        volume = calculate_audio_volume(audio, self.sample_rate)
        
        # the volume is exponentially smoothed based on the previous volume
        return exp_smoothing(volume, self._prev_volume, self._smoothing_factor)

    
    async def analyze_audio(self, buffer: bytes) -> VADState:
        """Analyze audio buffer and return current VAD state.

        Processes incoming audio data, maintains internal state, and determines
        voice activity status based on confidence and volume thresholds.

        Args:
            buffer: Audio buffer to analyze.

        Returns:
            Current VAD state after processing the buffer.
        """
        loop = asyncio.get_running_loop()
        state = await loop.run_in_executor(self._executor, self._run_analyzer, buffer)
        return state

    def _run_analyzer(self, buffer: bytes) -> VADState:
        """Analyze audio buffer and return current VAD state.
        buffer: the current audio buffer (chunk) to analyze, as raw bytes (16-bit signed integers).
        """
        
        # append the incoming audio buffer to the internal VAD buffer
        self._vad_buffer += buffer
        
        # determine the number of bytes required for one analysis window
        num_required_bytes = self._vad_frames_num_bytes
        if len(self._vad_buffer) < num_required_bytes:
            return self._vad_state

        # when accumulated buffer length exceeds the required number of bytes for one analysis window, process the buffer in chunks.
        while len(self._vad_buffer) >= num_required_bytes:
            
            # get frames to be analyzed
            audio_frames = self._vad_buffer[:num_required_bytes]
            
            # remove the frames that have been analyzed from the buffer
            self._vad_buffer = self._vad_buffer[num_required_bytes:]

            # subclass specific method to determine the speaking confidence of the current audio frames.
            confidence = self.voice_confidence(audio_frames)

            # calculate volume and update prev_volume
            volume = self._get_smoothed_volume(audio_frames)
            self._prev_volume = volume

            # high confidence and sufficient volume indicate speaking
            # this is not the final state, it will be further refined by the state machine below.
            speaking = confidence >= self._params.confidence and volume >= self._params.min_volume

            # core state machine update (handle state change and buffer count)
            if speaking:
                match self._vad_state:
                    case VADState.QUIET:
                        self._vad_state = VADState.STARTING
                        self._vad_starting_count = 1
                    case VADState.STARTING:
                        self._vad_starting_count += 1
                    case VADState.STOPPING:
                        self._vad_state = VADState.SPEAKING
                        self._vad_stopping_count = 0
            else:
                match self._vad_state:
                    case VADState.STARTING:
                        self._vad_state = VADState.QUIET
                        self._vad_starting_count = 0
                    case VADState.SPEAKING:
                        self._vad_state = VADState.STOPPING
                        self._vad_stopping_count = 1
                    case VADState.STOPPING:
                        self._vad_stopping_count += 1

        # handle buffer triggered state transitions
        if (
            self._vad_state == VADState.STARTING
            and self._vad_starting_count >= self._vad_start_frames
        ):
            self._vad_state = VADState.SPEAKING
            self._vad_starting_count = 0
        
        if (
            self._vad_state == VADState.STOPPING
            and self._vad_stopping_count >= self._vad_stop_frames
        ):
            self._vad_state = VADState.QUIET
            self._vad_stopping_count = 0

        return self._vad_state

    async def cleanup(self):
        """Clean up resources.

        This method should be called when the object is no longer needed.
        It waits for all currently executing event handler tasks to finish
        before returning.
        """
        if self._executor_shutdown:
            return
        self._executor_shutdown = True
        await asyncio.to_thread(self._executor.shutdown, wait=True)