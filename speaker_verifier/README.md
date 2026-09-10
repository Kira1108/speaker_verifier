# Speaker Verifier 使用说明

`SpeakerVerifier` 使用 CAM++ 模型比较两段语音，判断它们是否来自同一个说话人。接口为异步调用，模型只需加载一次，后续可以复用同一个实例。

## 初始化

请从项目根目录运行代码：

```python
from speaker_verifier.cam_plus import SpeakerVerifier


verifier = await SpeakerVerifier.create(threshold=0.31)
```

推荐使用 `create()`，模型加载会在线程中执行，不会阻塞 asyncio 事件循环。`threshold` 的范围是 `[-1, 1]`；阈值越高，判定为同一说话人的条件越严格。

首次加载模型时，ModelScope 可能需要联网下载 `iic/speech_campplus_sv_zh-cn_16k-common`。

## 比较 NumPy 音频

输入支持以下形式：

- 形状为 `(samples,)` 的单声道数组
- 形状为 `(samples, channels)` 的多声道数组
- 数据类型为 `float`、`int16`、`int32` 或 `uint8`
- 浮点音频的幅值应在 `[-1, 1]` 附近

采样率可以不同，verifier 会自动混合为单声道并重采样到 16 kHz。

```python
import asyncio

import soundfile as sf

from speaker_verifier.cam_plus import SpeakerVerifier


async def main():
    audio1, sample_rate1 = sf.read("speaker_a_1.wav", always_2d=False)
    audio2, sample_rate2 = sf.read("speaker_a_2.wav", always_2d=False)

    verifier = await SpeakerVerifier.create(threshold=0.31)
    result = await verifier.verify_numpy(
        audio1,
        audio2,
        sample_rate1=sample_rate1,
        sample_rate2=sample_rate2,
    )

    print(result)


asyncio.run(main())
```

返回值是 ModelScope speaker-verification pipeline 的结果字典，其中包含相似度分数和基于阈值的验证结果。

## 比较裸 PCM 音频

`verify_pcm()` 接受不包含 WAV 文件头的 PCM 字节。支持的 `pcm_format` 为：

| 格式 | 含义 |
| --- | --- |
| `s16le` | 16 位小端有符号整数，默认值 |
| `s32le` | 32 位小端有符号整数 |
| `f32le` | 32 位小端浮点数 |
| `u8` | 8 位无符号整数 |

```python
import asyncio

from speaker_verifier.cam_plus import SpeakerVerifier


async def main():
    with open("voice_1.pcm", "rb") as file:
        pcm1 = file.read()
    with open("voice_2.pcm", "rb") as file:
        pcm2 = file.read()

    verifier = await SpeakerVerifier.create()
    result = await verifier.verify_pcm(
        pcm1,
        pcm2,
        sample_rate1=16000,
        sample_rate2=16000,
        channels1=1,
        channels2=1,
        pcm_format1="s16le",
        pcm_format2="s16le",
    )

    print(result)


asyncio.run(main())
```

如果两段 PCM 的采样率、声道数和格式相同，可以省略第二段对应的参数：

```python
result = await verifier.verify_pcm(
    pcm1,
    pcm2,
    sample_rate1=16000,
    channels1=1,
    pcm_format1="s16le",
)
```

## 获取说话人 Embedding

传入 `output_emb=True` 可以同时返回两段音频的 embedding：

```python
result = await verifier.verify_numpy(
    audio1,
    audio2,
    sample_rate1=16000,
    output_emb=True,
)

print(result["outputs"])
print(result["embs"].shape)
```

此时 `result["outputs"]` 是验证结果，`result["embs"]` 是两段音频的 embedding 数组。

## 调用注意事项

- 使用同一个 `SpeakerVerifier` 实例处理多次验证，避免重复加载模型。
- 同一实例支持并发调用，但模型推理会在内部串行执行。
- 调用完成前不要修改传入的 NumPy 数组或可变 PCM 缓冲区。
- 音频应包含清晰、长度足够的单人语音；过短、静音或噪声较强的片段会降低验证可靠性。
- 可以通过单次调用的 `threshold` 参数临时覆盖实例默认阈值。