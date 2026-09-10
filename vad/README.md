# Silero VAD 流式语音检测

本目录提供基于 Silero VAD ONNX 模型的流式语音活动检测。`SileroVADAnalyzer` 可以持续接收音频分片，在内部拼接数据并输出当前语音状态。

## 输入要求

- 单声道 PCM 音频
- 采样格式：有符号 16 位整数（`int16`）
- 采样率：`16000 Hz` 或 `8000 Hz`
- 推荐每个分片包含 `512` 帧（16 kHz）或 `256` 帧（8 kHz）

`analyze_audio()` 也接受任意大小的分片。数据不足一帧时会暂存在内部缓冲区，数据超过一帧时会连续处理。

首次运行时，如果未传入本地模型路径，程序会从 Silero VAD 官方仓库下载 `silero_vad.onnx` 到 `vad/models/`。

## 安装依赖

在项目根目录执行：

```bash
pip install -r requirements.txt onnxruntime pydantic
```

如果要运行下方的麦克风示例，还需要：

```bash
pip install sounddevice
```

## 状态说明

每次调用 `analyze_audio()` 都会返回一个 `VADState`：

| 状态 | 含义 |
| --- | --- |
| `QUIET` | 当前没有检测到语音 |
| `STARTING` | 检测到语音，正在等待持续时间达到启动阈值 |
| `SPEAKING` | 已确认正在说话 |
| `STOPPING` | 语音可能结束，正在等待持续时间达到停止阈值 |

默认情况下，语音置信度需要达到 `0.7`，并持续约 `0.2` 秒才进入 `SPEAKING`；低于阈值约 `0.2` 秒后回到 `QUIET`。检测还会同时应用音量阈值。

## 接入流式音频源

下面的 `pcm_source()` 代表 WebSocket、声卡、文件流或其他持续产生 PCM 字节的异步数据源：

```python
import asyncio

from vad.analyzer import VADParams, VADState
from vad.silero_vad import SileroVADAnalyzer


SAMPLE_RATE = 16000


async def pcm_source():
    """替换为实际音频源；每次 yield 一段单声道 int16 PCM 字节。"""
    raise NotImplementedError
    yield b""


async def main():
    analyzer = SileroVADAnalyzer(
        sample_rate=SAMPLE_RATE,
        params=VADParams(
            confidence=0.7,
            start_secs=0.2,
            stop_secs=0.2,
            min_volume=0.6,
        ),
    )

    # 当前实现需要显式调用，以初始化帧大小和状态机。
    analyzer.set_sample_rate(SAMPLE_RATE)

    previous_state = VADState.QUIET
    async for pcm_bytes in pcm_source():
        state = await analyzer.analyze_audio(pcm_bytes)
        if state != previous_state:
            print(f"{previous_state.name} -> {state.name}")
            previous_state = state


asyncio.run(main())
```

不要在送入分析器前把每个网络包强制裁剪成 512 帧；分析器自己的缓冲区可以处理网络分片边界。必须确保同一个分析器实例只对应一路音频流，因为模型状态和 PCM 缓冲区会跨调用保留。

## 麦克风实时示例

以下示例通过 `sounddevice.RawInputStream` 获取麦克风数据。程序只在状态变化时打印结果，按 `Ctrl+C` 停止：

```python
import asyncio

import sounddevice as sd

from vad.analyzer import VADState
from vad.silero_vad import SileroVADAnalyzer


SAMPLE_RATE = 16000
BLOCK_SIZE = 512


async def main():
    loop = asyncio.get_running_loop()
    audio_queue = asyncio.Queue()
    analyzer = SileroVADAnalyzer(sample_rate=SAMPLE_RATE)
    analyzer.set_sample_rate(SAMPLE_RATE)

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status)
        loop.call_soon_threadsafe(audio_queue.put_nowait, bytes(indata))

    previous_state = VADState.QUIET
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=1,
        dtype="int16",
        callback=audio_callback,
    ):
        print("正在监听麦克风，按 Ctrl+C 停止")
        while True:
            pcm_bytes = await audio_queue.get()
            state = await analyzer.analyze_audio(pcm_bytes)
            if state != previous_state:
                print(f"{previous_state.name} -> {state.name}")
                previous_state = state


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
```

请从项目根目录运行示例，以便 Python 能找到 `vad` 和根目录下的 `utils.py`。macOS 首次运行时需要允许终端或 VS Code 使用麦克风。

## 使用本地模型

如果运行环境不能联网，可以提前下载 ONNX 模型并显式指定路径：

```python
analyzer = SileroVADAnalyzer(
    sample_rate=16000,
    model_path="/path/to/silero_vad.onnx",
)
analyzer.set_sample_rate(16000)
```

## 调参建议

- 环境噪声导致误触发时，提高 `confidence` 或 `min_volume`。
- 说话开头经常被漏掉时，降低 `start_secs`；业务侧仍应保留一小段预录音频。
- 句间短暂停顿被误判为结束时，提高 `stop_secs`。
- 更换音频流、采样率或说话人会话时，建议新建一个 `SileroVADAnalyzer` 实例，避免继承上一条流的缓存和模型状态。