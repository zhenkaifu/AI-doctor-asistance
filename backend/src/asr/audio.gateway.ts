import {
  WebSocketGateway,
  WebSocketServer,
  OnGatewayConnection,
  OnGatewayDisconnect,
  SubscribeMessage,
  MessageBody,
  ConnectedSocket,
} from '@nestjs/websockets';
import { Server, Socket } from 'socket.io';
import { AsrService, AsrTranscriptionResult } from './asr.service';
import { Logger } from '@nestjs/common';

@WebSocketGateway({
  cors: {
    origin: '*', // 开发环境下允许所有源
  },
})
export class AudioGateway implements OnGatewayConnection, OnGatewayDisconnect {
  private readonly logger = new Logger(AudioGateway.name);

  @WebSocketServer()
  server: Server;

  constructor(private readonly asrService: AsrService) {}

  /**
   * 当客户端连接时，建立与 ASR 的会话
   */
  async handleConnection(client: Socket) {
    this.logger.log(`Client connected: ${client.id}`);
    
    try {
      // 等待与 ASR 的连接真正建立
      await this.asrService.createSession(client.id, (result: AsrTranscriptionResult) => {
        // 实时回传转写结果给前端
        client.emit('transcription', result);
      });
      
      // 通知前端：所有链路已就绪
      client.emit('ready');
      this.logger.log(`Session ready for client: ${client.id}`);
    } catch (err) {
      this.logger.error(`Failed to create session for client ${client.id}: ${(err as Error).message}`);
      client.emit('error', 'Failed to initialize AI engine connection');
      client.disconnect();
    }
  }

  /**
   * 处理音频切片
   */
  @SubscribeMessage('audio-chunk')
  handleAudioChunk(@MessageBody() data: Buffer, @ConnectedSocket() client: Socket) {
    if (Buffer.isBuffer(data)) {
      this.asrService.sendAudio(client.id, data);
    } else {
      this.logger.warn(`Received non-buffer data from client ${client.id}`);
    }
  }

  /**
   * 客户端断开连接时，关闭与 ASR 的会话
   */
  handleDisconnect(client: Socket) {
    this.logger.log(`Client disconnected: ${client.id}`);
    this.asrService.closeSession(client.id);
  }
}
