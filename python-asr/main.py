from fastapi import FastAPI, WebSocket
from funasr import AutoModel
import asyncio
import numpy as np
import json
import os
import torch
import time

# 修复 PyTorch 2.6+ 安全限制
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

# M2 优化
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

app = FastAPI()

# 初始化模型
print("Loading FunASR model...")
model = AutoModel(
    model=os.getenv("FUNASR_MODEL", "paraformer-zh-streaming"),
    disable_update=True,
    device="cpu",
    disable_pbar=True,   # ✅ 新增：关闭 tqdm 进度条输出，保持日志清爽
)
print("FunASR model loaded!")

# ================== 配置参数 ==================
TARGET_DURATION = 2.3               # 目标音频块长度（秒）
SAMPLE_RATE = 16000                  # 采样率
TARGET_SAMPLES = int(TARGET_DURATION * SAMPLE_RATE)  # 24000 采样点
BYTES_PER_SAMPLE = 2                 # int16 每样本 2 字节
TARGET_BYTES = TARGET_SAMPLES * BYTES_PER_SAMPLE     # 48000 字节
# =============================================

@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    last_text = ""
    stream_cache = {}
    last_chunk = None                     # 记录最后处理的音频块（用于最终处理）
    audio_buffer = bytearray()            # 新增：字节缓冲区

    # 定义幻觉黑名单
    HALLUCINATION_BLACKLIST = [
        "谢谢观看", "订阅", "转发", "点赞", "打赏", "明镜", "点点",
        "重复输出", "普通话简体", "实时转录", "欢迎大家", "下期再见"
    ]

    async def process_audio(audio_chunk, is_final):
        nonlocal last_text
        try:
            audio_duration = len(audio_chunk) / SAMPLE_RATE
            start_calc = time.time()

            # FunASR generate() 支持直接传入 float32 waveform（16k）
            result = await asyncio.to_thread(
                model.generate,
                input=audio_chunk,
                cache=stream_cache,
                is_final=is_final,
                chunk_size=[5,10,5],   # ✅ 已使用官方推荐参数，确保完整输出
                encoder_chunk_look_back=4,
                decoder_chunk_look_back=1,
                use_itn=True,
            )

            text = ""
            if isinstance(result, list) and len(result) > 0:
                first = result[0]
                if isinstance(first, dict):
                    text = (first.get("text") or "").strip()
                elif isinstance(first, str):
                    text = first.strip()
            elif isinstance(result, dict):
                text = (result.get("text") or "").strip()

            calc_duration = time.time() - start_calc

            print(f"ASR Profiler: Audio {audio_duration:.2f}s | Calc {calc_duration:.3f}s | Speed {audio_duration/calc_duration:.1f}x")

            if len(text) < 2:
                return

            if any(bad_word in text for bad_word in HALLUCINATION_BLACKLIST):
                return

            if text == last_text:
                return

            last_text = text

            if websocket.client_state.name != "CONNECTED":
                return

            await websocket.send_text(json.dumps({
                "speaker": "Speaker 0",
                "text": text,
                "isFinal": is_final,
                "engine_latency": round(calc_duration, 3),
                "backend_latency": 0
            }))

        except Exception as e:
            print(f"ASR Error: {e}")

    try:
        while True:
            # 接收客户端发来的音频字节数据
            data = await websocket.receive_bytes()
            audio_buffer.extend(data)      # 追加到缓冲区

            # 当缓冲区累积达到目标长度时，切出一块进行处理
            while len(audio_buffer) >= TARGET_BYTES:
                # 取出目标长度的字节数据
                chunk_bytes = audio_buffer[:TARGET_BYTES]
                # 保留剩余部分
                audio_buffer = audio_buffer[TARGET_BYTES:]

                # 转换为 float32 归一化数组
                chunk = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                # 音频归一化（防削波）
                max_val = np.max(np.abs(chunk)) + 1e-6
                chunk = chunk / max_val

                last_chunk = chunk
                await process_audio(chunk, is_final=False)

    except Exception as e:
        print(f"WebSocket Closed: {e}")
    finally:
        # 处理缓冲区中剩余的数据（如有）
        if websocket.client_state.name == "CONNECTED":
            if len(audio_buffer) > 0:
                # 剩余数据不足一个目标块，作为最后一块处理
                remaining_bytes = audio_buffer
                chunk = np.frombuffer(remaining_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                max_val = np.max(np.abs(chunk)) + 1e-6
                chunk = chunk / max_val
                last_chunk = chunk
                await process_audio(chunk, is_final=False)

            # 发送最终结束信号（is_final=True）
            if last_chunk is not None:
                await process_audio(last_chunk, is_final=True)
            await websocket.close()
        # 如果连接已关闭，跳过 close() 调用，避免 RuntimeError

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)