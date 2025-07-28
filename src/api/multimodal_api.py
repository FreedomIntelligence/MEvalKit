import requests
import base64
import os
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
from dotenv import load_dotenv
import time
import random
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ssl

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class MultimodalAPI:
    def __init__(self, model_name: str, system_prompt: str, user_prompt: str, image_input, temperature: float, model_key: str, api_base: str = ""):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.image_input = image_input
        self.temperature = temperature
        os.environ['OPENAI_API_KEY'] = model_key if model_key != "" else os.environ['OPENAI_API_KEY']
        os.environ['OPENAI_API_BASE'] = api_base if api_base != "" else os.environ['OPENAI_API_BASE']

    def handle_image_url(self):
        if isinstance(self.image_input, str) and self.image_input.startswith(("http://", "https://")):
            return self.image_input
        if isinstance(self.image_input, str) and os.path.exists(self.image_input):
            return self.encode_local_image()
        if isinstance(self.image_input, Image.Image):
            return self.encode_pil_input()
        if isinstance(self.image_input, np.ndarray):
            return self.encode_cv2_input()
        # 添加对字节流的处理支持
        if isinstance(self.image_input, bytes):
            return self.encode_bytes_input()
        raise ValueError("Invalid image input")

    def encode_local_image(self):
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
        self.image_input = self.image_input.convert("RGB")
        buffered = BytesIO()
        self.image_input.save(buffered, format='JPEG')
        b64 = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"

    def encode_cv2_input(self):
        self.image_input = cv2.cvtColor(self.image_input, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(self.image_input)
        # 临时替换self.image_input为PIL对象
        original_input = self.image_input
        self.image_input = pil_image
        result = self.encode_pil_input()
        # 恢复原始输入
        self.image_input = original_input
        return result

    def encode_bytes_input(self):
        """处理字节流格式的图像数据"""
        try:
            # 从字节流创建PIL Image对象
            img = Image.open(BytesIO(self.image_input))
            # 确保图像是RGB模式
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 转换为base64编码
            buffered = BytesIO()
            img.save(buffered, format='JPEG')
            b64 = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/jpeg;base64,{b64}"
        except Exception as e:
            print(f"处理字节流图像时出错: {e}")
            raise ValueError(f"Invalid bytes image input: {e}")

    
    def generate_response(self):
        # 多模态API请求处理
        retry_count = 0
        max_retries = 3
        
        # 尝试不同的网络配置
        proxy_configs = [
            {'use_proxy': False, 'verify_ssl': False},  # 不使用代理，不验证SSL
            {'use_proxy': True, 'verify_ssl': False},   # 使用代理，不验证SSL
            {'use_proxy': False, 'verify_ssl': True},   # 不使用代理，验证SSL
        ]
        
        for config in proxy_configs:
            retry_count = 0
            print(f"\n=== 多模态API尝试配置: 代理={config['use_proxy']}, SSL验证={config['verify_ssl']} ===")
            
            while retry_count < max_retries:
                try:
                    load_dotenv()
                    url = os.environ['OPENAI_API_BASE'] + "/chat/completions"
                    
                    # 创建一个会话对象并配置SSL
                    session = requests.Session()
                    
                    # 配置代理设置
                    if not config['use_proxy']:
                        # 禁用代理
                        session.proxies = {
                            'http': None,
                            'https': None
                        }
                        # 确保环境变量代理也被忽略
                        session.trust_env = False
                    
                    # 配置SSL验证
                    session.verify = config['verify_ssl']
                    
                    # 设置重试策略
                    retry_strategy = Retry(
                        total=3,
                        backoff_factor=1,
                        status_forcelist=[429, 500, 502, 503, 504],
                    )
                    adapter = HTTPAdapter(max_retries=retry_strategy)
                    session.mount("http://", adapter)
                    session.mount("https://", adapter)
                    
                    payload = {
                        "model": self.model_name,
                        "messages": [
                            {
                                "role": "system",
                                "content": self.system_prompt
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "image_url": {
                                            "detail": "auto",
                                            "url": self.handle_image_url()
                                        },
                                        "type": "image_url"
                                    },
                                    {
                                        "text": self.user_prompt,
                                        "type": "text"
                                    }
                                ]
                            }
                        ],
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
                    
                    proxy_status = "无代理" if not config['use_proxy'] else "使用代理"
                    ssl_status = "验证SSL" if config['verify_ssl'] else "跳过SSL验证"
                    print(f"尝试多模态请求 (第{retry_count + 1}次) - {proxy_status}, {ssl_status}: {url}")
                    
                    #使用session发送请求，设置超时时间
                    response = session.post(url, headers=headers, json=payload, timeout=3000)
                    
                    print(f"响应状态码: {response.status_code}")
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        print(f"✓ 多模态请求成功！使用配置: {proxy_status}, {ssl_status}")
                        print(f"响应数据: {response_data}")
                        
                        if 'error' in response_data:
                            print(f"API返回错误: {response_data}")
                            return "Neglected"
                        else:
                            return response_data['choices'][0]['message']['content']
                    else:
                        print(f"HTTP错误: {response.status_code}, {response.text}")
                        retry_count += 1
                        if retry_count < max_retries:
                            print(f"等待2秒后重试...")
                            time.sleep(2)
                        continue
                        
                except requests.exceptions.SSLError as ssl_err:
                    print(f"SSL错误 (第{retry_count + 1}次尝试): {ssl_err}")
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"等待3秒后重试...")
                        time.sleep(3)
                    else:
                        print(f"当前配置SSL连接失败，尝试下一个配置")
                        break
                        
                except requests.exceptions.ConnectionError as conn_err:
                    print(f"连接错误 (第{retry_count + 1}次尝试): {conn_err}")
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"等待3秒后重试...")
                        time.sleep(3)
                    else:
                        print(f"当前配置连接失败，尝试下一个配置")
                        break
                        
                except requests.exceptions.Timeout as timeout_err:
                    print(f"请求超时 (第{retry_count + 1}次尝试): {timeout_err}")
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"等待2秒后重试...")
                        time.sleep(2)
                    else:
                        print(f"当前配置请求超时，尝试下一个配置")
                        break
                        
                except Exception as e:
                    print(f"其他错误 (第{retry_count + 1}次尝试): {type(e).__name__}: {e}")
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"等待2秒后重试...")
                        time.sleep(2)
                    else:
                        print(f"当前配置请求失败，尝试下一个配置")
                        break
        
        print("所有网络配置和重试都失败了")
        return "Neglected"
