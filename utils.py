import numpy as np
import pyloudnorm as pyln

def normalize_value(value, min_value, max_value):
    """Normalize a value to the range [0, 1] and clamp it to bounds.
    This is really a simple min-max normalizer.
    
    Args:
        value: The value to normalize.
        min_value: The minimum value of the input range.
        max_value: The maximum value of the input range.

    Returns:
        Normalized value clamped to the range [0, 1].
    """
    normalized = (value - min_value) / (max_value - min_value)
    normalized_clamped = max(0, min(1, normalized))
    return normalized_clamped


def calculate_audio_volume(audio: bytes, sample_rate: int) -> float:
    """Calculate the loudness level of audio data using EBU R128 standard.

    Uses the pyloudnorm library to calculate integrated loudness according
    to the EBU R128 recommendation, then normalizes the result to [0, 1].

    Args:
        audio: Audio data as raw bytes (16-bit signed integers).
        sample_rate: Sample rate of the audio in Hz.

    Returns:
        Normalized loudness value between 0 (quiet) and 1 (loud).
    """
    audio_np = np.frombuffer(audio, dtype=np.int16)
    audio_float = audio_np.astype(np.float64)

    block_size = audio_np.size / sample_rate
    meter = pyln.Meter(sample_rate, block_size=block_size)
    loudness = meter.integrated_loudness(audio_float)

    # Loudness goes from -20 to 80 (more or less), where -20 is quiet and 80 is
    # loud.
    loudness = normalize_value(loudness, -20, 80)

    return loudness


def exp_smoothing(value: float, prev_value: float, factor: float) -> float:
    """Apply exponential smoothing to a value.

    Exponential smoothing is used to reduce noise in time-series data by
    giving more weight to recent values while still considering historical data.

    Args:
        value: The new value to incorporate.
        prev_value: The previous smoothed value.
        factor: Smoothing factor between 0 and 1. Higher values give more
                weight to the new value.

    Returns:
        The exponentially smoothed value.
    """
    return prev_value + factor * (value - prev_value)