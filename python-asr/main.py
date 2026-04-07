from fastapi import FastAPI, WebSocket
from faster_whisper import WhisperModel
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
print("Loading Whisper Model (small) with int8...")
model = WhisperModel("medium", device="cpu", compute_type="int8", cpu_threads=4)
print("Whisper Model loaded!")

@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = []
    is_processing = False
    last_text = ""
    
    # 定义幻觉黑名单
    HALLUCINATION_BLACKLIST = [
        "谢谢观看", "订阅", "转发", "点赞", "打赏", "明镜", "点点", 
        "重复输出", "普通话简体", "实时转录", "欢迎大家", "下期再见"
    ]

    async def process_audio(full_audio, start_time):
        nonlocal is_processing, last_text
        try:
            audio_duration = len(full_audio) / 16000
            start_calc = time.time()

            segments_gen, _ = await asyncio.to_thread(
                model.transcribe,
                full_audio,
                beam_size=2,
                language="zh",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=800),
                initial_prompt="",
                condition_on_previous_text=False,
                repetition_penalty=1.2,
                no_speech_threshold=0.6,
                temperature=0
            )

            segments = list(segments_gen)

            calc_duration = time.time() - start_calc

            print(f"ASR Profiler: Audio {audio_duration:.2f}s | Calc {calc_duration:.3f}s | Speed {audio_duration/calc_duration:.1f}x")

            for segment in segments:

                text = segment.text.strip()

                # 原有过滤
                if len(text) < 2 or segment.no_speech_prob > 0.8:
                    continue

                # 新增质量过滤
                if segment.avg_logprob < -1.0:
                    continue

                if segment.compression_ratio > 2.4:
                    continue

                if any(bad_word in text for bad_word in HALLUCINATION_BLACKLIST):
                    continue

                if text == last_text:
                    continue

                last_text = text

                if websocket.client_state.name != "CONNECTED":
                    return

                await websocket.send_text(json.dumps({
                    "speaker": "Speaker 0",
                    "text": text,
                    "isFinal": True,
                    "engine_latency": round(calc_duration, 3),
                    "backend_latency": 0
                }))

        except Exception as e:
            print(f"ASR Error: {e}")
        finally:
            is_processing = False

    try:
        while True:

            data = await websocket.receive_bytes()

            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            # 音频归一化
            max_val = np.max(np.abs(chunk)) + 1e-6
            chunk = chunk / max_val

            audio_buffer.append(chunk)

            # 缓冲区策略
            if len(audio_buffer) > 18:
                audio_buffer = audio_buffer[-12:]

            # 积压 2 秒触发
            if not is_processing and len(audio_buffer) >= 10:
                is_processing = True
                to_process = np.concatenate(audio_buffer)
                audio_buffer = audio_buffer[-4:]
                asyncio.create_task(process_audio(to_process, time.time()))

    except Exception as e:
        print(f"WebSocket Closed: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)