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

# 初始化 ASR 模型
print("Loading FunASR model...")
model = AutoModel(
    model=os.getenv("FUNASR_MODEL", "paraformer-zh-streaming"),
    disable_update=True,
    device="cpu",
    disable_pbar=True,
)
print("FunASR model loaded!")

# 初始化 VAD 模型
print("Loading FunASR VAD model...")
vad_model = AutoModel(
    model="fsmn-vad",
    model_revision="v2.0.4",
    disable_update=True,
    device="cpu",
    disable_pbar=True,
)
print("FunASR VAD model loaded!")

# ================== 配置参数 ==================
TARGET_DURATION = 0.6
SAMPLE_RATE = 16000
TARGET_SAMPLES = int(TARGET_DURATION * SAMPLE_RATE)
BYTES_PER_SAMPLE = 2
TARGET_BYTES = TARGET_SAMPLES * BYTES_PER_SAMPLE
# =============================================

@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(f"\n{'='*60}")
    print(f"🟢 WebSocket 连接建立")

    # ---- 核心状态 ----
    stream_cache = {}           # 每句结束后 clear()，避免 encoder 历史累积导致延迟上升
    vad_cache = {}
    last_text = ""              # 去重用：上一帧发送的 text
    audio_buffer = bytearray()

    # ---- VAD 断句状态 ----
    is_speaking = False
    silence_count = 0
    SILENCE_LIMIT = 1

    HALLUCINATION_BLACKLIST = [
        "谢谢观看", "订阅", "转发", "点赞", "打赏", "明镜", "点点",
        "重复输出", "普通话简体", "实时转录", "欢迎大家", "下期再见","嗯"
    ]

    # ---- Debug ----
    def ts():
        return time.strftime("%H:%M:%S", time.localtime())

    async def process_audio(audio_chunk, is_final, start_time=None):
        """送入 ASR，返回识别 text 并通过 WebSocket 推送给前端"""
        nonlocal last_text
        try:
            start_calc = time.time()

            result = await asyncio.to_thread(
                model.generate,
                input=audio_chunk,
                cache=stream_cache,
                is_final=is_final,
                chunk_size=[20, 20, 10],
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

            tag = "[FINAL]" if is_final else "[PART]"
            if not text:
                if is_final:
                    print(f"  {ts()} {tag} ⚪ 无文字")
                return

            if any(bad_word in text for bad_word in HALLUCINATION_BLACKLIST):
                print(f"  {ts()} {tag} 🚫 幻听: \"{text}\"")
                return

            # 去重
            if text == last_text:
                if is_final:
                    print(f"  {ts()} {tag} 🔁 dup: \"{text}\"")
                return

            prev = last_text

            # beam search 可能重解前缀 → 任意帧（PART 或 FINAL）都检测并拼接
            if prev and not text.startswith(prev):
                merged = prev + text
                print(f"  {ts()} {tag} 🔗 拼回: \"{prev}\" + \"{text}\" = \"{merged}\"")
                text = merged

            last_text = text

            if websocket.client_state.name != "CONNECTED":
                print(f"  {ts()} {tag} 🔌 断开丢弃: \"{text}\"")
                return

            if prev and not text.startswith(prev):
                print(f"  {ts()} {tag} 📤 跳变! \"{prev}\" -> \"{text}\"")
            elif prev:
                print(f"  {ts()} {tag} 📤 增长: \"{text}\"")
            else:
                print(f"  {ts()} {tag} 📤 新句: \"{text}\"")

            # 延迟：引擎耗时 + 后端总耗时（含 VAD、排队等），单位 ms
            backend_latency = round((time.time() - start_time) * 1000) if start_time else 0
            await websocket.send_text(json.dumps({
                "speaker": "Speaker 0",
                "text": text,
                "isFinal": is_final,
                "engine_latency": round(calc_duration * 1000),
                "backend_latency": backend_latency
            }, ensure_ascii=False))

        except Exception as e:
            print(f"  {ts()} [ERR] ASR: {e}")

    def _has_speech(vad_result):
        """VAD 结果中是否包含有效语音"""
        try:
            if isinstance(vad_result, list) and len(vad_result) > 0:
                first = vad_result[0]
                if isinstance(first, dict):
                    value = first.get("value", [])
                    return len(value) > 0 if isinstance(value, list) else bool(value)
        except Exception:
            pass
        return False

    chunk_seq = 0
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_buffer.extend(data)

            while len(audio_buffer) >= TARGET_BYTES:
                chunk_seq += 1
                chunk_bytes = audio_buffer[:TARGET_BYTES]
                audio_buffer = audio_buffer[TARGET_BYTES:]

                chunk = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                max_val = np.max(np.abs(chunk)) + 1e-6

                # 能量门控：峰值低于 -46dBFS 视为静音，不归一化（避免噪声放大导致 VAD 误检）
                is_silent = max_val <= 0.005
                if not is_silent:
                    chunk = chunk / max_val

                # ---- VAD 检测（静音跳过，省去 VAD 模型调用） ----
                if is_silent:
                    has_voice = False
                else:
                    vad_result = await asyncio.to_thread(
                        vad_model.generate,
                        input=chunk,
                        cache=vad_cache,
                        is_final=False,
                        chunk_size=200,
                    )
                    has_voice = _has_speech(vad_result)

                # ---- 断句状态机（VAD 仅控制断句，不控制音频是否送入 ASR） ----
                if has_voice:
                    if not is_speaking:
                        print(f"\n  {ts()} 🎤 VAD: 语音开始 (chunk#{chunk_seq})")
                        # VAD 漏检了前面的语音：先 flush 间隙文本为完整句，再清 encoder 开新句
                        if last_text:
                            dummy = np.zeros(TARGET_SAMPLES, dtype=np.float32)
                            await process_audio(dummy, is_final=True, start_time=time.time())
                            stream_cache.clear()
                            last_text = ""
                    is_speaking = True
                    silence_count = 0
                elif is_speaking:
                    silence_count += 1
                    bar = "█" * min(silence_count, SILENCE_LIMIT) + "░" * max(0, SILENCE_LIMIT - silence_count)
                    print(f"  {ts()} 🔇 VAD: 静音 [{bar}] {silence_count}/{SILENCE_LIMIT} (chunk#{chunk_seq})")
                # else: 句间静音/环境噪声 → VAD 未确认，走硬能量门限判断

                # 达到静音阈值 → 本帧以 is_final=True 送入，触发断句
                should_finalize = (
                    is_speaking and silence_count >= SILENCE_LIMIT
                )

                if should_finalize:
                    print(f"  {ts()} ✅ VAD: 断句! is_final=True (chunk#{chunk_seq})")

                # 送 ASR 条件：VAD 说在说话，或峰值能量 >0.01（约 -40dBFS，硬门限兜底 VAD 漏检）
                chunk_start = time.time()
                if is_speaking or max_val > 0.01:
                    await process_audio(chunk, is_final=should_finalize, start_time=chunk_start)

                # 句子结束：重置状态
                if should_finalize:
                    stream_cache.clear()     # 重置 encoder 历史，避免延迟累积
                    last_text = ""
                    is_speaking = False
                    silence_count = 0
                    print(f"  {ts()} ── 句子结束 ──")
                    print(f"{'='*60}")

    except Exception as e:
        print(f"\n🔴 WebSocket Closed: {e}")
    finally:
        print(f"\n  {ts()} ── finally 块进入 ──")
        print(f"  {ts()}    is_speaking={is_speaking}, conn={websocket.client_state.name}")
        if websocket.client_state.name == "CONNECTED" and is_speaking:
            print(f"  {ts()} ⚡ 强制 flush 最后一句话...")
            try:
                dummy = np.zeros(TARGET_SAMPLES, dtype=np.float32)
                await process_audio(dummy, is_final=True)
                print(f"  {ts()} ✅ flush 完成")
            except Exception as e:
                print(f"  {ts()} ❌ flush 失败: {e}")
        elif not is_speaking:
            print(f"  {ts()}   无需 flush (is_speaking=False)")
        else:
            print(f"  {ts()}   连接已断，跳过 flush")
        try:
            await websocket.close()
            print(f"  {ts()} 🔌 WebSocket 已关闭")
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
