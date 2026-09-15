#!/usr/bin/env python3
"""Kaldi fbank using NumPy + kaldi-native-fbank, with TorchAudio defaults.

Input: float waveform [samples] or [channels, samples]; output: float32 [T, F].
No automatic amplitude scaling, resampling, channel mixing, or CMVN.
Float32 numerical equivalence is intended, not bitwise identity. Nonzero dither
uses the native library's RNG, so samples will differ from TorchAudio's RNG.
Non-default VTLN warp is unsupported by the native online Python API.
"""
import warnings

import numpy as np
import kaldi_native_fbank as knf


def fbank(
    waveform, blackman_coeff=0.42, channel=-1, dither=0.0,
    energy_floor=1.0, frame_length=25.0, frame_shift=10.0,
    high_freq=0.0, htk_compat=False, low_freq=20.0, min_duration=0.0,
    num_mel_bins=23, preemphasis_coefficient=0.97, raw_energy=True,
    remove_dc_offset=True, round_to_power_of_two=True,
    sample_frequency=16000.0, snip_edges=True, subtract_mean=False,
    use_energy=False, use_log_fbank=True, use_power=True,
    vtln_high=-500.0, vtln_low=100.0, vtln_warp=1.0, window_type="povey",
):
    """Use TorchAudio parameter names; waveform must already have the right scale.

    Like TorchAudio's implementation, channel=-1 selects channel zero.
    Inputs shorter than one analysis window raise ValueError. min_duration
    rejection returns shape (0,), matching TorchAudio.
    """
    x = np.asarray(waveform)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("Expected [samples] or [channels, samples]")
    if not np.issubdtype(x.dtype, np.floating):
        raise TypeError("Use floating-point samples with the original amplitude scale")
    c = max(channel, 0)
    if c >= x.shape[0]:
        raise ValueError("Channel index is out of range")
    x = np.ascontiguousarray(x[c], dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Waveform contains NaN or infinity")
    if sample_frequency <= 0 or frame_length <= 0 or frame_shift <= 0:
        raise ValueError("Sample rate and frame sizes must be positive")
    size = int(sample_frequency * frame_length * 0.001)
    shift = int(sample_frequency * frame_shift * 0.001)
    if size < 2 or size > len(x) or shift < 1:
        raise ValueError("Need at least one full window and a positive sample shift")
    if not round_to_power_of_two and size % 2:
        raise ValueError("FFT length must be even")
    if not 0 <= preemphasis_coefficient <= 1 or dither < 0 or energy_floor < 0:
        raise ValueError("Invalid preemphasis, dither, or energy floor")
    if window_type not in {"povey", "hanning", "hamming", "rectangular", "blackman"}:
        raise ValueError("Unsupported window type")
    nyquist = sample_frequency / 2
    high = high_freq if high_freq > 0 else nyquist + high_freq
    if not 0 <= low_freq < high <= nyquist or num_mel_bins <= 3:
        raise ValueError("Invalid mel frequency limits or bin count")
    if vtln_warp != 1.0:
        raise NotImplementedError("OnlineFbank Python API supports only vtln_warp=1.0")
    if len(x) < min_duration * sample_frequency:
        return np.empty(0, dtype=np.float32)
    if dither:
        warnings.warn("Nonzero dither does not reproduce TorchAudio's random noise", stacklevel=2)

    opts = knf.FbankOptions()
    for name, value in {
        "samp_freq": sample_frequency, "frame_length_ms": frame_length,
        "frame_shift_ms": frame_shift, "dither": dither,
        "preemph_coeff": preemphasis_coefficient, "remove_dc_offset": remove_dc_offset,
        "window_type": window_type, "round_to_power_of_two": round_to_power_of_two,
        "blackman_coeff": blackman_coeff, "snip_edges": snip_edges,
    }.items():
        setattr(opts.frame_opts, name, value)
    for name, value in {
        "num_bins": num_mel_bins, "low_freq": low_freq, "high_freq": high_freq,
        "vtln_low": vtln_low, "vtln_high": vtln_high,
        "debug_mel": False, "htk_mode": False, "is_librosa": False,
    }.items():
        setattr(opts.mel_opts, name, value)
    for name, value in {
        "use_energy": use_energy, "energy_floor": energy_floor,
        "raw_energy": raw_energy, "htk_compat": htk_compat,
        "use_log_fbank": use_log_fbank, "use_power": use_power,
    }.items():
        setattr(opts, name, value)
    extractor = knf.OnlineFbank(opts)
    extractor.accept_waveform(sample_frequency, x.tolist())
    extractor.input_finished()  # Flush reflected end frames when snip_edges=False.
    result = np.empty((extractor.num_frames_ready, num_mel_bins + int(use_energy)), dtype=np.float32)
    for i in range(len(result)):
        result[i] = extractor.get_frame(i)
    if subtract_mean and len(result):
        result -= result.mean(axis=0, keepdims=True)
    return result


