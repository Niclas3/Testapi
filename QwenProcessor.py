from functools import total_ordering
import requests
from http import HTTPStatus
import dashscope
import time
import LogHelper

class QwenProcessor:
    def __init__(self, api_key, prompt,modelType):
        dashscope.api_key = api_key
        self.prompt=prompt
        self.modelType=modelType

    def process_audio(self, content):
        start_time = time.time()
        messages = [{'role': 'system', 'content':self.prompt},
                    {'role': 'user', 'content': content}]

        model=dashscope.Generation.Models.qwen_turbo
        if self.modelType==1:
            model= dashscope.Generation.Models.qwen_plus
        elif self.modelType==2:
            model= dashscope.Generation.Models.qwen_max   
            
        response = dashscope.Generation.call(
            model,
            messages=messages,
            temperature=0.3,
            result_format='message',  # 将返回结果格式设置为 message
        )
        end_time = time.time()
        elapsed_time = end_time - start_time        
        print(f"代码执行耗时: {elapsed_time} 秒")
        
        if response.status_code == HTTPStatus.OK:
            total_token= response['usage']['total_tokens']
            return response['output']['choices'][0]['message']['content'],total_token,elapsed_time
        else:
            return None,0,elapsed_time

