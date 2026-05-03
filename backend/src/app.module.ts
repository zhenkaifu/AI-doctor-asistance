import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { AsrModule } from './asr/asr.module';

@Module({
  imports: [AsrModule],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
