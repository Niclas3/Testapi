from math import log
from fastapi import FastAPI,Request
from PIL import Image
from io import BytesIO
import os
import uvicorn
import shutil
import numpy as np
from pydantic import BaseModel
from ASRProcessor import ASRProcessor
from AudioProcessor import AudioProcessor
import LogHelper
import base64
from fastapi.responses import JSONResponse
import json
import re

from QwenProcessor import QwenProcessor

app = FastAPI()
logger = None  # 日志

class Item(BaseModel):
    img_b64: str = None
    
@app.post('/remove/watermark')
def remove_water_api(request_data:Item):
    try:
        logger.logger.info('获得图片数据')
        imgPath=None
        raw_image = base64.b64decode(request_data.img_b64.encode('utf8'))
        img = Image.open(BytesIO(raw_image))
        img = levelsDeal(img,108,164)
        img_res = Image.fromarray(img.astype('uint8'))
        logger.logger.info('图片处理完毕')
        
        # 将处理后的图像转换为base64编码字符串
        buffered = BytesIO()
        img_res.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    except Exception as e:
        logger.logger.error('Error : ' + str(e))
        
@app.post('/ai/voiceToFactor')
def voice_factor(localFileUrl,key,prompt):
    code=0
    message="成功"
    datajson=None
    data=None
    try:
        asr= ASRProcessor(key,prompt,logger)
        data=asr.process_audio(localFileUrl)
        try:
            dText=remove_comments(extract_json_string(data[0]))
            datajson= json.loads(dText)
            code=0;
        except Exception as e:
            datajson= data
            code=10
            message="解析json结果异常"
    except Exception as e: 
        code=-1
        message=f"调用异常,{str(e)}"
        logger.logger.error('Error : ' + str(e))
        exit_service()
    
    data1=None
    data2=None
    if data is not None:
        data1 =data[1]
        data2=data[2]
        
    response = JSONResponse(content={
            "code": code,
            "message": message,
            "data": datajson,
            "voiceToText": data1,
            "times":data2
        })
    return response

@app.post('/audio/voiceToText')
def voice_text(localFileUrl,key,audioFormat,rate:int):
    code=0
    message="成功"
    asr= AudioProcessor(key,audioFormat,rate,logger)
    data=asr.process_audio(localFileUrl)

    response = JSONResponse(content={
            "code": code,
            "message": message,
            "data": data[0],
            "times":data[1]
        })
    return response     

@app.post('/ai/chat')
async def chat(request: Request):
    body = await request.json()
    key = body.get("key")
    prompt = body.get("prompt")
    content = body.get("content")
    mtype = body.get("mtype")
    
    code=0
    message="成功"
    qwen= QwenProcessor(key,prompt,mtype)
    data=qwen.process_audio(content)

    response = JSONResponse(content={
            "code": code,
            "message": message,
            "data": data[0],
            "tokenCount": data[1],
            "times":data[2]
        })
    return response    

def extract_json_string(text):
    pattern = r'```json(.*?)```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

def remove_comments(json_string):
    # 使用正则表达式移除行内注释
    json_string = re.sub(r'//.*|#.*', '', json_string)
    return json_string

#图片处理
def imgDeal(img_path,save_path):
    img = Image.open(img_path)
    img = levelsDeal(img,108,164)
    img_res = Image.fromarray(img.astype('uint8'))
    print(u'图片[' + img_path + u']处理完毕')
    img_res.save(save_path)

#图像矩阵处理
def levelsDeal(img, black,white):
    if white > 255:
        white = 255
    if black < 0:
        black = 0
    if black >= white:
        black = white - 2
    img_array = np.array(img, dtype = int)
    cRate = -(white - black) /255.0 * 0.05
    rgb_diff = img_array - black
    rgb_diff = np.maximum(rgb_diff, 0)
    img_array = rgb_diff * cRate
    img_array = np.around(img_array, 0)
    img_array = img_array.astype(int)
    return img_array

def exit_service():
    pattern = r"Errno\s*24"
    matches = re.findall(pattern, result.message)
    if matches != []:
        sys.exit(1) # some bad things happened server need to restart NOW!

def main():
    global logger
    logger = LogHelper.LogHelper("watermarkapi.log")
    logger.logger.info('启动执行')
     
    name = "waterApi"
    if os.path.exists(name):
        shutil.rmtree(name)
    os.makedirs(name)  
    uvicorn.run(app=app,
	            host="0.0.0.0",
	            port="5123",
	            workers=1)	

if __name__ == "__main__":
    main()
