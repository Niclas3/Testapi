import requests
from http import HTTPStatus
import dashscope
from dashscope.audio.asr import Recognition
import time
import LogHelper
import sys  #for let systemd know OS error happend
import re

class ASRProcessor:
    def __init__(self, api_key, prompt,logger):
        dashscope.api_key = api_key
        self.prompt=prompt
        self.log = logger
        self.recognition = Recognition(model='paraformer-realtime-8k-v1',
                          format='wav',
                          sample_rate=8000,
                          callback=None)    

    def process_audio(self, audio_file):
        start_time = time.time()
        result = self.recognition.call(audio_file)
        
        txts = ''
        if result.status_code == HTTPStatus.OK:
            for sentence in result.get_sentence():
                txts += sentence['text']
            self.log.logger.info('Recognition done!'+txts)    
        else:
            self.log.logger.error('Error: ', result.message)
            pattern = r"Errno\s*24"
            matches = re.findall(pattern, result.message)
            if matches != []:
                sys.exit(1) # some bad things happened server need to restart NOW!
        messages = [{'role': 'system', 'content':self.prompt},
                    {'role': 'user', 'content': '录音对话内容如下：'+txts}]

        response = dashscope.Generation.call(
            dashscope.Generation.Models.qwen_plus,
            messages=messages,
            temperature=0.2,
            result_format='message',  # 将返回结果格式设置为 message
        )
        end_time = time.time()
        elapsed_time = end_time - start_time        
        print(f"代码执行耗时: {elapsed_time} 秒")
        
        if response.status_code == HTTPStatus.OK:
            return response['output']['choices'][0]['message']['content'],txts,elapsed_time
        else:
            return None,txts,elapsed_time

