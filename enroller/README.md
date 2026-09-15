# PcmVad 注册音频准备

`PcmVad` 使用本地 Silero ONNX VAD，从录音中提取语音片段、按原顺序拼接，并判断保留音频的时长是否满足注册要求。

适合在 Voice Agent 启动前，通过一次 HTTP 上传完成注册音频检查，不需要 WebSocket。

**这个模块只准备音频，不负责提取、保存声纹，也不判断说话人是否唯一或音频质量是否合格。**

## 1. 环境与输入格式

从项目根目录运行下面的命令和 Python 示例，以便导入 `enroller.pcm_vad`。

核心依赖为 `numpy`、`onnxruntime`、`pydantic`；下方 WAV 示例另外使用 `soundfile` 和 `scipy`。默认环境已安装时无需重复安装，否则可执行：

```bash
python -m pip install numpy onnxruntime pydantic soundfile scipy
```

模型默认使用仓库中的 [silero_vad.onnx](../vad/models/silero_vad.onnx)，通过 CPU 执行，不会在运行时下载模型。初始化时传入 `model_path=None` 即可选择这个默认模型。

### PCM 必须符合以下格式

| 属性 | 要求 |
| --- | --- |
| 采样率 | 8000 或 16000 Hz，必须与 `sample_rate` 配置一致 |
| 声道 | 单声道 |
| 样本格式 | 有符号 16 位小端整数，即 s16le / int16 little-endian |
| 数据 | 裸 PCM 字节，不包含 WAV 文件头 |
| 长度 | 必须为 2 字节的整数倍，不能包含半个样本 |

**建议统一使用 16 kHz。** `PcmVad` 本身不解码 WAV/WebM/Opus，不混合声道，也不重采样。不能直接把 WAV 文件的全部字节传给 `enroll()`；请参考下面的 WAV 示例先解码转换。

## 2. 直接处理 PCM

下面假设输入已经是 16 kHz、单声道、s16le 裸 PCM。把示例中的输入路径换成自己的录音路径。

```python
from pathlib import Path

from enroller.pcm_vad import PcmVad


# 初始化一次，后续录音可复用同一个实例。
vad = PcmVad(model_path=None, sample_rate=16000)

pcm = Path("input.pcm").read_bytes()
result = vad.enroll(pcm, min_enroll_seconds=5.0)

print(f"原始音频时长：{len(pcm) / (16000 * 2):.2f} 秒")
print(f"保留音频时长：{result.duration:.2f} 秒")
print(f"时长是否达标：{result.success}")

if result.success:
    # 示例仅保存注册参考音频；后续可交给说话人验证器提取声纹。
    Path("enrollment.pcm").write_bytes(result.pcm)
    print("音频时长达标，可以继续声纹提取和保存。")
else:
    print("保留音频不足 5 秒，请继续录制或重新录制。")
```

### 返回值：`EnrollResult`

返回的是 Pydantic `BaseModel`，可以通过属性读取：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `duration` | `float` | 实际返回 PCM 的时长，单位为秒 |
| `pcm` | `bytes` | 按原顺序拼接的保留音频，格式和采样率与输入一致 |
| `success` | `bool` | 仅表示 `duration >= min_enroll_seconds` |

- `duration` 包含边界保护音频和片段内的短暂停顿，不是严格逐帧的纯语音时长。
- 时长不足时仍返回已提取的 `pcm` 和 `duration`。
- 没有保留音频时，返回 `duration=0.0`、`pcm=b""`、`success=False`。
- 达标后不会裁剪到恰好 5 秒，而是返回所有保留片段。
- `enroll()` 默认最小时长为 **1 秒**；产品要求 5 秒时，需要显式传入 `min_enroll_seconds=5.0`。
- `min_enroll_seconds` 必须为有限正数；零、负数、NaN、无穷值和布尔值会被拒绝。

## 3. 使用 WAV 文件测试（完整代码）

下面的代码可以保存为一个 Python 脚本，在项目根目录执行。它会：

1. 使用 `soundfile` 解码 WAV，而不是将 WAV 文件头当作 PCM。
2. 将多声道取平均，转换成单声道。
3. 将输入重采样到 16 kHz（也可指定 8 kHz）。
4. 转换成 s16le 裸 PCM，然后调用 `enroll()`。
5. 打印检测结果，并将非空的保留音频写成可播放的 WAV。

```python
import argparse
from math import gcd
from pathlib import Path
import wave

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from enroller.pcm_vad import PcmVad


def load_wav_pcm(path: Path, target_rate: int) -> bytes:
    """将 WAV 解码为指定采样率、单声道、s16le 裸 PCM。"""
    audio, source_rate = sf.read(
        str(path), dtype="float32", always_2d=True,
    )
    if audio.shape[0] == 0:
        return b""
    if not np.isfinite(audio).all():
        raise ValueError("录音包含 NaN 或 Inf")

    # 输入形状为 (samples, channels)。
    mono = audio.mean(axis=1)
    if source_rate != target_rate:
        divisor = gcd(source_rate, target_rate)
        mono = resample_poly(
            mono,
            up=target_rate // divisor,
            down=source_rate // divisor,
        )

    # 重采样可能产生轻微过冲；取整并限幅，避免 int16 溢出。
    pcm_int16 = np.clip(
        np.rint(mono * 32768.0), -32768, 32767,
    ).astype("<i2")
    return pcm_int16.tobytes()


def main():
    parser = argparse.ArgumentParser(description="使用 WAV 测试注册音频提取")
    parser.add_argument("input_wav", type=Path, help="输入 WAV 路径")
    parser.add_argument("output_wav", type=Path, help="输出保留音频的 WAV 路径")
    parser.add_argument("--sample-rate", type=int, choices=(8000, 16000), default=16000)
    parser.add_argument("--min-seconds", type=float, default=5.0)
    args = parser.parse_args()

    if args.input_wav.resolve() == args.output_wav.resolve():
        parser.error("输入和输出路径不能相同")
    if args.output_wav.exists():
        parser.error("输出文件已存在，请选择新路径，避免覆盖")

    pcm = load_wav_pcm(args.input_wav, args.sample_rate)
    vad = PcmVad(model_path=None, sample_rate=args.sample_rate)
    result = vad.enroll(pcm, min_enroll_seconds=args.min_seconds)

    print(f"转换后原始时长：{len(pcm) / (args.sample_rate * 2):.3f} 秒")
    print(f"保留音频时长：{result.duration:.3f} 秒")
    print(f"要求最小时长：{args.min_seconds:.3f} 秒")
    print(f"时长是否达标：{result.success}")

    if not result.pcm:
        print("没有检测到满足分段条件的语音，不生成输出文件。")
        return

    # 即使总时长未达标，也保存非空结果，方便试听和检查。
    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output_wav), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(args.sample_rate)
        output.writeframes(result.pcm)

    print(f"已保存保留音频：{args.output_wav}")


if __name__ == "__main__":
    main()
```

执行方式如下；将脚本路径和录音路径换成实际路径。此脚本需自行保存，上面的文档示例不会自动创建脚本。

```bash
python your_wav_test.py input.wav retained.wav --min-seconds 5 --sample-rate 16000
```

> 多声道取平均适用于常见录音。如果不同声道录了不同说话人，应选择目标说话人所在的声道，而不是混合。重采样也不会恢复原始 8 kHz 音频中已经缺失的高频信息。

## 4. 只查询时长或片段位置

```python
from pathlib import Path

from enroller.pcm_vad import PcmVad


pcm = Path("input.pcm").read_bytes()
vad = PcmVad(model_path=None, sample_rate=16000, return_seconds=False)

# 始终返回秒，不受 return_seconds 影响。
duration = vad.compute_speech_length(pcm)
print("保留时长（秒）：", duration)

# return_seconds=False：返回整数样本位置；end 不包含在区间内。
# 默认 return_seconds=True：返回秒数。
segments = vad.timestamps(pcm)
print("保留片段（样本位置）：", segments)
```

这些方法每次调用都会独立运行 VAD。如果同时需要时长和提取后的 PCM，直接调用一次 `enroll()`，不要先调用 `compute_speech_length()` 再调用 `enroll()`。

## 5. 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `model_path` | 必须传入，可传 `None` | `None` 使用仓库内置模型，也可指定本地模型路径 |
| `sample_rate` | `16000` | 输入 PCM 的真实采样率，只支持 8000 / 16000 |
| `threshold` | `0.5` | VAD 语音进入阈值，范围为 `[0.15, 1]`，不是声纹相似度阈值 |
| `min_speech_duration_ms` | `250` | 分段保留的最小时长条件；当前实现要求片段时长严格大于此值 |
| `min_silence_duration_ms` | `100` | 确认片段结束所需的低概率持续时长，按分析帧判断 |
| `speech_pad_ms` | `30` | 每段两端的保护时长；会裁到录音范围内并避免相邻片段重叠 |
| `return_seconds` | `True` | 仅影响 `timestamps()` 的单位，不影响注册切片或返回时长 |

## 6. 接入注册页面后端

- 注册期间 Agent 不启动，提示用户在安静环境中独自说话。
- 可以先录制约 6–8 秒，再通过 HTTP 上传；保留音频未达到 5 秒时提示补录，不保证录满 5 秒就一定达标。
- 每次调用是独立的完整录音处理，不是流式累积接口。若需要累计补录，由后端按会话保留原始 PCM，在格式一致的前提下拼接，再提交检查。
- 对累计录音重新检查后，使用本次 `duration` 更新进度；不要把多次累计录音的结果时长相加。
- `PcmVad` 可以复用以避免重复加载模型；`enroll()` 是同步、CPU 推理调用。异步后端可将调用放到工作线程，例如 `await asyncio.to_thread(vad.enroll, pcm, min_enroll_seconds=5.0)`，并限制并发及上传时长。
- HTTP 响应通常只需要 `duration` 和 `success`。`pcm` 留在后端用于声纹提取或参考音频存储；不要直接把任意二进制 PCM 当作 UTF-8 文本塞进 JSON，确需传输时使用二进制响应或显式 Base64 编码。
- `success=True` 表示录音时长达标；页面的最终“注册成功”应在后续声纹提取和保存完成后显示。

## 7. 运行回归测试

从项目根目录执行：

```bash
python -m unittest discover -s tests -p 'test_pcm_vad.py' -v
```

测试使用受控 VAD 概率验证切片、时长、单位及边界情况，不代表真实录音的识别准确率。实际注册效果应使用单人语音、静音和含噪录音进一步检查。