"""ModelScope-style CAM++ callable; network inference uses ONNX Runtime."""
import io
import os
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio
from torchaudio.compliance.kaldi import fbank


MODEL_PATH = Path(__file__).parent / "campplus.onnx"


MODEL_CONFIG = {
  "sample_rate": 16000,
  "fbank_dim": 80,
  "emb_size": 192,
  "yesOrno_thr": 0.31
}

class CampPlusONNX:
    def __init__(self, *, providers=None):
        model = MODEL_PATH
        self.config = MODEL_CONFIG
        self.thr = float(self.config['yesOrno_thr'])
        self.session = ort.InferenceSession(
            str(model), providers=providers or ['CPUExecutionProvider'])
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
                audio = audio[:, 0]  # Original interface selects the first channel.
            audio = torch.from_numpy(audio.copy())
            if sr != self.config['sample_rate']:
                # Preserve original SoX resampling semantics.
                audio, _ = torchaudio.sox_effects.apply_effects_tensor(
                    audio.unsqueeze(0), sr,
                    effects=[['rate', str(self.config['sample_rate'])]])
                audio = audio.squeeze(0)
        elif isinstance(item, np.ndarray):
            if item.ndim != 1:
                raise ValueError('Each waveform must be a one-dimensional array.')
            if item.dtype in (np.int16, np.int32, np.int64):
                item = item / (1 << 15)
            audio = torch.from_numpy(item.astype(np.float32))
        else:
            raise TypeError('Expected an audio path/URL or numpy waveform.')
        if audio.numel() < 400 or not torch.isfinite(audio).all():
            raise ValueError('Audio must contain at least 400 finite samples at 16 kHz.')
        return audio

    def __call__(self, in_audios, save_dir=None, output_emb=False, thr=None):
        if thr is not None:
            self.thr = float(thr)  # Original interface persists threshold changes.
        if not -1 <= self.thr <= 1:
            raise ValueError('thr must be in [-1, 1].')
        if isinstance(in_audios, (str, os.PathLike)):
            raise TypeError('Wrap audio paths in a list: ["audio.wav"].')
        if len(in_audios) == 0:
            raise ValueError('in_audios must not be empty.')
        embeddings = []
        for item in in_audios:
            audio = self._read(item)
            features = fbank(audio.unsqueeze(0),
                             num_mel_bins=self.config['fbank_dim'],
                             sample_frequency=self.config['sample_rate'])
            features = features - features.mean(dim=0, keepdim=True)
            value = np.ascontiguousarray(features.unsqueeze(0).numpy())
            embeddings.append(self.session.run(None, {self.input_name: value})[0])
        embs = np.concatenate(embeddings, axis=0)
        if save_dir is not None and isinstance(in_audios[0], (str, os.PathLike)):
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            for item, emb in zip(in_audios, embs):
                np.save(Path(save_dir) / (Path(item).stem + '.npy'), emb)
        if len(embs) == 2:
            a, b = embs
            score = float(np.sum((a / max(float(np.linalg.norm(a)), 1e-6)) *
                                 (b / max(float(np.linalg.norm(b)), 1e-6))))
            score = round(score, 5)
            outputs = {'score': score, 'text': 'yes' if score >= self.thr else 'no'}
        else:
            outputs = {'text': 'No similarity score output'}
        return {'outputs': outputs, 'embs': embs} if output_emb else outputs
