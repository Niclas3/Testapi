import requests
from http import HTTPStatus
import dashscope
from dashscope.audio.asr import Recognition
import time
import LogHelper

class AudioProcessor:
    def __init__(self, api_key,audioFormat,rate,logger):
        dashscope.api_key = api_key
        self.log = logger
        modelStr="paraformer-realtime-8k-v1"
        if rate==16000:
            modelStr="paraformer-realtime-v2" 
        self.recognition = Recognition(model=modelStr,
                          format=audioFormat,
                          sample_rate=rate,
                          callback=None)    

    def process_audio(self, audio_file):
        start_time = time.time()
        result = self.recognition.call(audio_file)
        obj=None
        if result.status_code == HTTPStatus.OK:
            obj= result.get_sentence()
        else:
            self.log.logger.error('Error : ' + str(result.message))
            
        end_time = time.time()
        elapsed_time = end_time - start_time        
        print(f"代码执行耗时: {elapsed_time} 秒")
        return obj,elapsed_time