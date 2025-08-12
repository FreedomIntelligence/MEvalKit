import requests
import base64
import os
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional, Union
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ssl
import time

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ConversationAPI:
    def __init__(self, model_name: str, system_prompt: str, user_prompt: str, 
                 image_input: Optional[Union[str, Image.Image, np.ndarray, bytes]] = None, 
                 temperature: float = 0.7, conversation_id: Optional[str] = None, 
                 model_key: str = "", api_base: str = "", enable_history: bool = True):
        """
        综合的模型对话API，支持单轮/多轮对话和纯文本/多模态对话
        
        参数:
            model_name: 模型名称
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            image_input: 图像输入（可选），支持多种格式
            temperature: 温度参数
            conversation_id: 会话ID，用于多轮对话
            model_key: API密钥
            api_base: API基础URL
            enable_history: 是否启用对话历史（True为多轮，False为单轮）
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.image_input = image_input
        self.temperature = temperature
        self.conversation_id = conversation_id or "default"
        self.enable_history = enable_history
        self.conversation_history: Dict[str, List[Dict[str, Any]]] = {}
        
        if model_key and model_key.strip():
            os.environ['OPENAI_API_KEY'] = model_key
        if api_base and api_base.strip():
            os.environ['OPENAI_API_BASE'] = api_base

    def handle_image_url(self):
        """处理图像输入，支持多种格式"""
        if self.image_input is None:
            return None
            
        if isinstance(self.image_input, str) and self.image_input.startswith(("http://", "https://")):
            return self.image_input
        if isinstance(self.image_input, str) and os.path.exists(self.image_input):
            return self.encode_local_image()
        if isinstance(self.image_input, Image.Image):
            return self.encode_pil_input()
        if isinstance(self.image_input, np.ndarray):
            return self.encode_cv2_input()
        if isinstance(self.image_input, bytes):
            return self.encode_bytes_input()
        raise ValueError("Invalid image input")

    def encode_local_image(self):
        """编码本地图像文件"""
        mime_type = {
            '.jpg': 'jpeg', '.jpeg': 'jpeg',
            '.png': 'png', '.webp': 'webp'
        }.get(os.path.splitext(self.image_input)[1].lower(), 'jpeg')

        with open(self.image_input, 'rb') as f:
            img = Image.open(f)
            buffered = BytesIO()
            img.save(buffered, format=mime_type.upper())
            b64 = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/{mime_type};base64,{b64}"

    def encode_pil_input(self):
        """编码PIL图像对象"""
        self.image_input = self.image_input.convert("RGB")
        buffered = BytesIO()
        self.image_input.save(buffered, format='JPEG')
        b64 = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"

    def encode_cv2_input(self):
        """编码OpenCV图像数组"""
        self.image_input = cv2.cvtColor(self.image_input, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(self.image_input)
        original_input = self.image_input
        self.image_input = pil_image
        result = self.encode_pil_input()
        self.image_input = original_input
        return result

    def encode_bytes_input(self):
        """处理字节流格式的图像数据"""
        try:
            img = Image.open(BytesIO(self.image_input))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            buffered = BytesIO()
            img.save(buffered, format='JPEG')
            b64 = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/jpeg;base64,{b64}"
        except Exception as e:
            raise ValueError(f"Invalid bytes image input: {e}")

    def generate_response(self) -> str:
        """
        生成回复，支持单轮/多轮对话和纯文本/多模态对话
        
        返回:
            模型的回复
        """
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                load_dotenv()
                url = os.environ['OPENAI_API_BASE'] + "/chat/completions"
                
                session = requests.Session()
                session.proxies = {'http': None, 'https': None}
                session.trust_env = False
                session.verify = False
                
                retry_strategy = Retry(
                    total=3,
                    backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504],
                )
                adapter = HTTPAdapter(max_retries=retry_strategy)
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                
                # 构建消息列表
                messages = []
                
                if self.enable_history:
                    # 多轮对话：使用历史记录
                    if self.conversation_id not in self.conversation_history:
                        self.conversation_history[self.conversation_id] = [
                            {"role": "system", "content": self.system_prompt}
                        ]
                    messages = self.conversation_history[self.conversation_id].copy()
                    
                    # 添加当前用户消息
                    user_message = self._build_user_message()
                    messages.append(user_message)
                    self.conversation_history[self.conversation_id].append(user_message)
                else:
                    # 单轮对话：不使用历史记录
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        self._build_user_message()
                    ]
                
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "max_tokens": 16384,
                    "stop": None,
                    "temperature": self.temperature,
                    "top_p": 0.7,
                    "frequency_penalty": 0.5,
                    "n": 1,
                    "response_format": {"type": "text"}
                }
                
                headers = {
                    "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                    "Content-Type": "application/json"
                }
                
                response = session.post(url, headers=headers, json=payload, timeout=3000)
                #print(response.json())
                
                if response.status_code == 200:
                    response_data = response.json()
                    #print(response_data)
                    if 'error' in response_data:
                        print("Error: ", response_data)
                        return "Neglected"
                    else:
                        assistant_message = response_data['choices'][0]['message']['content']
                        #print("Assistant Message: ", assistant_message)
                        if assistant_message == None:
                            print("Assistant Message is None: ", response_data)
                        if self.enable_history:
                            # 多轮对话：保存助手回复到历史记录
                            self.conversation_history[self.conversation_id].append({
                                "role": "assistant", 
                                "content": assistant_message
                            })
                        
                        return assistant_message
                else:
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(2)
                    continue
                    
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, 
                   requests.exceptions.Timeout, Exception) as e:
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(2)
                continue
        
        return "Neglected"

    def _build_user_message(self) -> Dict[str, Any]:
        """构建用户消息"""
        image_url = self.handle_image_url()
        
        if image_url is not None:
            # 多模态消息：包含图像和文本
            return {
                "role": "user",
                "content": [
                    {
                        "image_url": {
                            "detail": "auto",
                            "url": image_url
                        },
                        "type": "image_url"
                    },
                    {
                        "text": self.user_prompt,
                        "type": "text"
                    }
                ]
            }
        else:
            # 纯文本消息
            return {
                "role": "user",
                "content": self.user_prompt
            }

    def clear_conversation(self, conversation_id: Optional[str] = None):
        """清除指定会话的历史记录"""
        if conversation_id is None:
            conversation_id = self.conversation_id
        
        if conversation_id in self.conversation_history:
            del self.conversation_history[conversation_id]

    def get_conversation_history(self, conversation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取指定会话的历史记录"""
        if conversation_id is None:
            conversation_id = self.conversation_id
        
        return self.conversation_history.get(conversation_id, [])

    def update_prompt(self, user_prompt: str, image_input: Optional[Union[str, Image.Image, np.ndarray, bytes]] = None):
        """更新提示词和图像输入"""
        self.user_prompt = user_prompt
        self.image_input = image_input 