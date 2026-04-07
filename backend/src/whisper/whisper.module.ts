import { Module } from '@nestjs/common';
import { WhisperService } from './whisper.service';
import { AudioGateway } from './audio.gateway';

@Module({
  providers: [WhisperService, AudioGateway],
  exports: [WhisperService],
})
export class WhisperModule {}
