import asyncio
from math import gcd
from threading import Lock
from typing import Optional

import numpy as np
from scipy.signal import resample_poly
from modelscope.pipelines import pipeline


class SpeakerVerifier:
    """支持 NumPy 和裸 PCM 输入的异步说话人验证。"""

    TARGET_SAMPLE_RATE = 16000

    PCM_FORMATS = {
        "s16le": "<i2",  # 16-bit 小端有符号整数
        "s32le": "<i4",  # 32-bit 小端有符号整数
        "f32le": "<f4",  # 32-bit 小端浮点数，幅值约 [-1, 1]
        "u8": "u1",      # 8-bit 无符号整数
    }

    def __init__(self, threshold: float = 0.31):
        self.threshold = self._validate_threshold(threshold)
        self._lock = Lock()

        # 只加载一次模型，后续调用复用
        self._pipeline = pipeline(
            task="speaker-verification",
            model="iic/speech_campplus_sv_zh-cn_16k-common",
            model_revision="v1.0.0",
        )

    @classmethod
    async def create(cls, threshold: float = 0.31):
        """异步创建实例，避免模型加载阻塞事件循环。"""
        return await asyncio.to_thread(cls, threshold=threshold)

    @staticmethod
    def _validate_threshold(threshold):
        threshold = float(threshold)
        if not -1 <= threshold <= 1:
            raise ValueError("threshold 必须在 [-1, 1] 范围内")
        return threshold

    @staticmethod
    def _validate_positive_int(value, name):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value <= 0
        ):
            raise ValueError(f"{name} 必须是正整数")
        return int(value)

    @classmethod
    def _prepare_numpy(cls, audio: np.ndarray, sample_rate: int):
        """
        转换为 16 kHz、单声道、float32。

        支持形状：
            (samples,)
            (samples, channels)

        支持类型：
            float：输入应已归一化到 [-1, 1] 附近
            int16 / int32：按整数位宽归一化
            uint8：以 128 为中心归一化
        """
        sample_rate = cls._validate_positive_int(
            sample_rate, "sample_rate"
        )
        audio = np.asarray(audio)

        if audio.ndim not in (1, 2) or audio.size == 0:
            raise ValueError(
                "audio 必须是非空的 (samples,) "
                "或 (samples, channels) 数组"
            )

        if audio.dtype.kind == "i" and audio.dtype.itemsize in (2, 4):
            scale = float(2 ** (audio.dtype.itemsize * 8 - 1))
            audio = audio.astype(np.float32) / scale
        elif audio.dtype == np.uint8:
            audio = (audio.astype(np.float32) - 128.0) / 128.0
        elif audio.dtype.kind == "f":
            audio = audio.astype(np.float32)
        else:
            raise TypeError("支持 float、int16、int32 或 uint8 音频")

        if not np.isfinite(audio).all():
            raise ValueError("audio 不能包含 NaN 或 Inf")

        # 多声道混合为单声道
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        # 数组输入需要自行重采样
        if sample_rate != cls.TARGET_SAMPLE_RATE:
            factor = gcd(sample_rate, cls.TARGET_SAMPLE_RATE)
            audio = resample_poly(
                audio,
                up=cls.TARGET_SAMPLE_RATE // factor,
                down=sample_rate // factor,
            )

        return np.ascontiguousarray(audio, dtype=np.float32)

    @classmethod
    def _decode_pcm(
        cls,
        pcm: bytes,
        channels: int = 1,
        pcm_format: str = "s16le",
    ):
        """解码裸 PCM；多声道数据按采样帧交错排列，不含 WAV 头。"""
        channels = cls._validate_positive_int(channels, "channels")

        if pcm_format not in cls.PCM_FORMATS:
            raise ValueError(
                f"pcm_format 必须是 {list(cls.PCM_FORMATS)}"
            )

        if not isinstance(pcm, (bytes, bytearray, memoryview)):
            raise TypeError(
                "pcm 必须是 bytes、bytearray 或 memoryview"
            )

        dtype = np.dtype(cls.PCM_FORMATS[pcm_format])
        frame_bytes = dtype.itemsize * channels
        byte_count = memoryview(pcm).nbytes

        if byte_count == 0:
            raise ValueError("PCM 数据不能为空")
        if byte_count % frame_bytes != 0:
            raise ValueError(
                "PCM 字节数必须是完整采样帧大小的整数倍"
            )

        return np.frombuffer(pcm, dtype=dtype).reshape(-1, channels)

    def _verify_numpy_sync(
        self,
        audio1,
        audio2,
        sample_rate1,
        sample_rate2,
        threshold,
        output_emb,
    ):
        """在线程中执行的同步实现。"""
        if sample_rate2 is None:
            sample_rate2 = sample_rate1

        thr = self._validate_threshold(
            self.threshold if threshold is None else threshold
        )
        wav1 = self._prepare_numpy(audio1, sample_rate1)
        wav2 = self._prepare_numpy(audio2, sample_rate2)

        # pipeline 内部有可变状态，同一个实例串行推理
        with self._lock:
            result = self._pipeline(
                [wav1, wav2],
                thr=thr,
                output_emb=output_emb,
            )

            # 复制返回值，避免后续调用改变已返回的结果
            if output_emb:
                return {
                    "outputs": dict(result["outputs"]),
                    "embs": result["embs"].copy(),
                }

            return dict(result)

    def _verify_pcm_sync(
        self,
        pcm1,
        pcm2,
        sample_rate1,
        sample_rate2,
        channels1,
        channels2,
        pcm_format1,
        pcm_format2,
        threshold,
        output_emb,
    ):
        if channels2 is None:
            channels2 = channels1
        if pcm_format2 is None:
            pcm_format2 = pcm_format1

        audio1 = self._decode_pcm(pcm1, channels1, pcm_format1)
        audio2 = self._decode_pcm(pcm2, channels2, pcm_format2)

        return self._verify_numpy_sync(
            audio1,
            audio2,
            sample_rate1,
            sample_rate2,
            threshold,
            output_emb,
        )

    async def verify_numpy(
        self,
        audio1: np.ndarray,
        audio2: np.ndarray,
        sample_rate1: int = 16000,
        sample_rate2: Optional[int] = None,
        *,
        threshold: Optional[float] = None,
        output_emb: bool = False,
    ):
        """
        异步比较两段 NumPy 音频。

        sample_rate2 不传时，沿用 sample_rate1。
        调用完成前，请勿修改传入的数组。
        """
        return await asyncio.to_thread(
            self._verify_numpy_sync,
            audio1,
            audio2,
            sample_rate1,
            sample_rate2,
            threshold,
            output_emb,
        )

    async def verify_pcm(
        self,
        pcm1: bytes,
        pcm2: bytes,
        sample_rate1: int = 16000,
        sample_rate2: Optional[int] = None,
        *,
        channels1: int = 1,
        channels2: Optional[int] = None,
        pcm_format1: str = "s16le",
        pcm_format2: Optional[str] = None,
        threshold: Optional[float] = None,
        output_emb: bool = False,
    ):
        """
        异步比较两段裸 PCM 音频。

        第二段的采样率、声道数、PCM 格式不传时，沿用第一段。
        如果传入可变缓冲区，调用完成前请勿修改。
        """
        return await asyncio.to_thread(
            self._verify_pcm_sync,
            pcm1,
            pcm2,
            sample_rate1,
            sample_rate2,
            channels1,
            channels2,
            pcm_format1,
            pcm_format2,
            threshold,
            output_emb,
        )


async def main():
    # 模型异步加载一次
    verifier = await SpeakerVerifier.create(threshold=0.31)

    rng = np.random.default_rng(42)

    # 两段 3 秒随机音频，采样率分别为 48 kHz 和 16 kHz
    audio1 = rng.uniform(
        -0.1, 0.1, size=48000 * 3
    ).astype(np.float32)

    audio2 = rng.uniform(
        -0.1, 0.1, size=16000 * 3
    ).astype(np.float32)

    # 测试 NumPy 入口
    import time
    start_time = time.perf_counter()
    result = await verifier.verify_numpy(
        audio1,
        audio2,
        sample_rate1=48000,
        sample_rate2=16000,
        output_emb=True,
    )
    print("NumPy 验证结果:", result["outputs"])
    print("Embedding shape:", result["embs"].shape)
    print("NumPy 验证耗时:", time.perf_counter() - start_time)

    # 转成 16-bit 小端裸 PCM
    pcm1 = (audio1 * 32767).astype("<i2").tobytes()
    pcm2 = (audio2 * 32767).astype("<i2").tobytes()

    # 测试 PCM 入口
    start_time = time.perf_counter()
    result = await verifier.verify_pcm(
        pcm1,
        pcm2,
        sample_rate1=48000,
        sample_rate2=16000,
        pcm_format1="s16le",
    )
    print("PCM 验证结果:", result)
    print("PCM 验证耗时:", time.perf_counter() - start_time)


if __name__ == "__main__":
    asyncio.run(main())