"""ModelScope-style CAM++ callable with no PyTorch dependency.

Requires fbank.py to export a NumPy-compatible fbank function.
NumPy inputs are assumed to already use MODEL_CONFIG['sample_rate'].
Resampling uses SoXR HQ; it is not bitwise equivalent to SoX rate.
"""
import io
import os
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import onnxruntime as ort
import soundfile as sf
import soxr
from fbank import fbank


MODEL_PATH = Path(__file__).parent / 'campplus.onnx'
MODEL_CONFIG = {
    'sample_rate': 16000,
    'fbank_dim': 80,
    'emb_size': 192,
    'yesOrno_thr': 0.31,
}


class CampPlusONNX:
    def __init__(self, *, providers=None):
        self.config = MODEL_CONFIG.copy()
        self.thr = float(self.config['yesOrno_thr'])
        self.session = ort.InferenceSession(
            str(MODEL_PATH), providers=providers or ['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def _read(self, item):
        if isinstance(item, (str, os.PathLike)):
            location = str(item)
            if location.startswith(('http://', 'https://')):
                with urlopen(location, timeout=60) as response:
                    source = io.BytesIO(response.read())
            else:
                source = location
            audio, sr = sf.read(source, dtype='float32')
            if audio.ndim == 2:
                audio = audio[:, 0]  # Preserve first-channel selection.
            audio = np.ascontiguousarray(audio, dtype=np.float32)
            if audio.size == 0 or not np.isfinite(audio).all():
                raise ValueError('Audio must contain finite, nonempty samples.')
            if sr != self.config['sample_rate']:
                audio = soxr.resample(
                    audio, sr, self.config['sample_rate'], quality='HQ')
        elif isinstance(item, np.ndarray):
            if item.ndim != 1:
                raise ValueError('Each waveform must be a one-dimensional array.')
            if item.dtype in (np.int16, np.int32, np.int64):
                # Preserve the original convention even for int32/int64:
                # these arrays are interpreted as PCM16-scale values.
                item = item / (1 << 15)
            audio = np.asarray(item, dtype=np.float32)
        else:
            raise TypeError('Expected an audio path/URL or numpy waveform.')
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        min_samples = int(self.config['sample_rate'] * 0.025)
        if audio.size < min_samples or not np.isfinite(audio).all():
            raise ValueError(
                f'Audio must contain at least {min_samples} finite samples '
                f"at {self.config['sample_rate']} Hz.")
        return audio

    def __call__(self, in_audios, save_dir=None, output_emb=False, thr=None):
        if thr is not None:
            self.thr = float(thr)  # Preserve persistent threshold changes.
        if not -1 <= self.thr <= 1:
            raise ValueError('thr must be in [-1, 1].')
        if isinstance(in_audios, (str, os.PathLike)):
            raise TypeError('Wrap audio paths in a list: ["audio.wav"].')
        if len(in_audios) == 0:
            raise ValueError('in_audios must not be empty.')
        embeddings = []
        for item in in_audios:
            audio = self._read(item)
            features = fbank(
                audio[None, :],
                num_mel_bins=self.config['fbank_dim'],
                sample_frequency=self.config['sample_rate'],
                dither=0.0,
                subtract_mean=False,
            )
            features = np.asarray(features, dtype=np.float32)
            if (features.ndim != 2 or features.shape[0] == 0
                    or features.shape[1] != self.config['fbank_dim']
                    or not np.isfinite(features).all()):
                raise ValueError('fbank must return finite features with shape [frames, fbank_dim].')
            features = features - features.mean(axis=0, keepdims=True)
            value = np.ascontiguousarray(features[None, :, :], dtype=np.float32)
            embeddings.append(self.session.run(None, {self.input_name: value})[0])
        embs = np.concatenate(embeddings, axis=0)
        if save_dir is not None and isinstance(in_audios[0], (str, os.PathLike)):
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            for item, emb in zip(in_audios, embs):
                np.save(Path(save_dir) / (Path(item).stem + '.npy'), emb)
        if len(embs) == 2:
            a, b = embs
            score = float(np.sum(
                (a / max(float(np.linalg.norm(a)), 1e-6)) *
                (b / max(float(np.linalg.norm(b)), 1e-6))))
            score = round(score, 5)
            outputs = {'score': score, 'text': 'yes' if score >= self.thr else 'no'}
        else:
            outputs = {'text': 'No similarity score output'}
        return {'outputs': outputs, 'embs': embs} if output_emb else outputs
