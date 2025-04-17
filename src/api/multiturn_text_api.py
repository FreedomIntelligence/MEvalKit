import requests
import base64
import os
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
from dotenv import load_dotenv
from typing import Dict, List, Any
class MultiturnTextAPI:
    def __init__(self, model_name: str, system_prompt: str, user_prompt: str, temperature: float, conversation_id: str):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.temperature = temperature
        self.conversation_history : Dict[str, List[Dict[str, Any]]] = {}
        self.conversation_id = conversation_id

    def generate_response(self) -> str:
        """
        进行多轮对话，保存对话历史
        
        参数:
            message: 用户消息
            conversation_id: 会话ID，用于区分不同的对话
            
        返回:
            模型的回复
        """
        load_dotenv()
        url = os.environ['OPENAI_API_BASE'] + "/chat/completions"
        
        # 初始化会话历史（如果不存在）
        if self.conversation_id not in self.conversation_history:
            self.conversation_history[self.conversation_id] = [
                {"role": "system", "content": self.system_prompt}
            ]
        
        # 添加用户消息到历史记录
        self.conversation_history[self.conversation_id].append(
            {"role": "user", "content": self.user_prompt}
        )
        
        # 准备请求
        payload = {
            "model": self.model_name,
            "messages": self.conversation_history[self.conversation_id],
            "stream": False,
            "max_tokens": 1024,
            "stop": None,
            "temperature": self.temperature,
            "top_p": 0.7,
            "frequency_penalty": 0.5,
            "n": 1,
            "response_format": {
                "type": "text"
            }
        }
        
        headers = {
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json"
        }
        
        # 发送请求
        response = requests.request("POST", url, headers=headers, json=payload)
        response_data = response.json()
        print(response_data)
        # 获取助手回复
        assistant_message = response_data['choices'][0]['message']['content']
        
        # 将助手回复添加到历史记录
        self.conversation_history[self.conversation_id].append(
            {"role": "assistant", "content": assistant_message}
        )
        
        return assistant_message
    
    
    
    