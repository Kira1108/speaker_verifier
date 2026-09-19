from pathlib import Path

import pyaudio


# 裸 PCM 不包含格式信息，必须与录音时的格式一致。
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # int16，小端
CHUNK_FRAMES = 1024
PCM_FILE_PATH = Path(__file__).resolve().parent / "d427117d-cd1e-4130-9967-f3be56230d8c.pcm"


def play_pcm(file_path: str | Path) -> None:
	"""读取并播放 16 kHz、单声道、s16le 裸 PCM，播放完成后返回。"""
	pcm = Path(file_path).read_bytes()
	if not pcm:
		raise ValueError("PCM 文件为空")
	frame_bytes = CHANNELS * SAMPLE_WIDTH
	if len(pcm) % frame_bytes:
		raise ValueError("PCM 文件必须包含完整的 int16 采样帧")

	audio = pyaudio.PyAudio()
	try:
		stream = audio.open(
			format=pyaudio.paInt16,
			channels=CHANNELS,
			rate=SAMPLE_RATE,
			output=True,
			frames_per_buffer=CHUNK_FRAMES,
		)
		try:
			chunk_bytes = CHUNK_FRAMES * frame_bytes
			for offset in range(0, len(pcm), chunk_bytes):
				stream.write(pcm[offset:offset + chunk_bytes])
		finally:
			try:
				stream.stop_stream()
			finally:
				stream.close()
	finally:
		audio.terminate()


if __name__ == "__main__":
	print(f"正在播放：{PCM_FILE_PATH}")
	play_pcm(PCM_FILE_PATH)
	print("播放完成。")
