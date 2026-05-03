import { Module } from '@nestjs/common';
import { AsrService } from './asr.service';
import { AudioGateway } from './audio.gateway';

@Module({
  providers: [AsrService, AudioGateway],
  exports: [AsrService],
})
export class AsrModule {}
