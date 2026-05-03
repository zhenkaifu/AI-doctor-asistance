import { Injectable, OnModuleDestroy, Logger } from '@nestjs/common';
import { WebSocket as WSClient } from 'ws';

export interface AsrTranscriptionResult {
  speaker: string;
  text: string;
  start: number;
  end: number;
  isFinal: boolean;
}

export interface AsrRawResult {
  text: string;
  segments?: Array<{
    start: number;
    end: number;
    text: string;
    speaker_id?: string | number;
    is_final?: boolean;
  }>;
}

@Injectable()
export class AsrService implements OnModuleDestroy {
  private readonly logger = new Logger(AsrService.name);
  private readonly asrUrl = 'ws://localhost:8001/asr'; // 连接到 Python ASR 服务
  private connections = new Map<string, WSClient>();
  private audioBuffers = new Map<string, Buffer[]>(); // 临时缓冲区，存储连接就绪前的音频

  /**
   * 初始化与 ASR 的连接，返回 Promise 确保连接就绪后再返回
   */
  async createSession(clientId: string, onResult: (result: AsrTranscriptionResult) => void): Promise<void> {
    if (this.connections.has(clientId)) {
      this.closeSession(clientId);
    }

    return new Promise((resolve, reject) => {
      const ws = new WSClient(this.asrUrl);
      this.audioBuffers.set(clientId, []);

      // 连接超时处理 (5秒)
      const timeout = setTimeout(() => {
        ws.terminate();
        reject(new Error(`ASR connection timeout for client ${clientId}`));
      }, 5000);

      ws.on('open', () => {
        clearTimeout(timeout);
        this.logger.log(`Connected to ASR for client: ${clientId}`);
        this.connections.set(clientId, ws);
        
        // 连接建立后，立即发送缓冲区中的数据
        const buffer = this.audioBuffers.get(clientId) || [];
        buffer.forEach(chunk => ws.send(chunk));
        this.audioBuffers.delete(clientId);
        
        resolve();
      });

      ws.on('message', (data: Buffer) => {
        const receiveTime = Date.now(); // 后端收到结果的时间
        try {
          const raw: any = JSON.parse(data.toString());
          // 注入后端处理时间戳
          raw.backend_receive_at = receiveTime;
          
          if (raw.segments && raw.segments.length > 0) {
            raw.segments.forEach((seg) => {
              onResult({
                ...seg,
                speaker: seg.speaker_id ? `Speaker ${seg.speaker_id}` : 'Unknown',
                engine_latency: raw.engine_latency,
                backend_latency: Date.now() - receiveTime // 后端解析耗时
              } as any);
            });
          } else {
            onResult({
              speaker: raw.speaker || 'Unknown',
              text: raw.text,
              start: raw.start || 0,
              end: raw.end || 0,
              isFinal: raw.isFinal ?? false,
              engine_latency: raw.engine_latency,
              backend_latency: Date.now() - receiveTime
            } as any);
          }
        } catch (e) {
          this.logger.error(`Error parsing ASR message: ${(e as Error).message}`);
        }
      });

      ws.on('error', (err) => {
        clearTimeout(timeout);
        this.logger.error(`ASR connection error for client ${clientId}: ${err.message}`);
        this.audioBuffers.delete(clientId);
        reject(err);
      });

      ws.on('close', () => {
        this.logger.log(`ASR connection closed for client: ${clientId}`);
        this.connections.delete(clientId);
        this.audioBuffers.delete(clientId);
      });
    });
  }

  /**
   * 转发音频数据，如果连接尚未就绪则存入缓冲区
   */
  sendAudio(clientId: string, buffer: Buffer) {
    const ws = this.connections.get(clientId);
    if (ws && ws.readyState === WSClient.OPEN) {
      ws.send(buffer);
    } else {
      // 如果连接还在建立中，先存入缓冲区
      const queue = this.audioBuffers.get(clientId);
      if (queue) {
        queue.push(buffer);
        // 限制缓冲区大小防止内存溢出
        if (queue.length > 50) queue.shift();
      } else {
        this.logger.warn(`Cannot send/buffer audio: No active session for client ${clientId}`);
      }
    }
  }

  closeSession(clientId: string) {
    const ws = this.connections.get(clientId);
    if (ws) {
      ws.close();
      this.connections.delete(clientId);
    }
    this.audioBuffers.delete(clientId);
  }

  onModuleDestroy() {
    this.connections.forEach((ws) => ws.close());
    this.connections.clear();
    this.audioBuffers.clear();
  }
}
