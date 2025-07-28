import requests
import base64
import os
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
from dotenv import load_dotenv
from typing import Dict, List, Any
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ssl
import time

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class MultiturnTextAPI:
    def __init__(self, model_name: str, system_prompt: str, user_prompt: str, temperature: float, conversation_id: str, model_key: str, api_base: str):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.temperature = temperature
        self.conversation_history : Dict[str, List[Dict[str, Any]]] = {}
        self.conversation_id = conversation_id
        if model_key and model_key.strip():
            os.environ['OPENAI_API_KEY'] = model_key
        if api_base and api_base.strip():
            os.environ['OPENAI_API_BASE'] = api_base

    def generate_response(self) -> str:
        """
        进行多轮对话，保存对话历史
        
        参数:
            message: 用户消息
            conversation_id: 会话ID，用于区分不同的对话
            
        返回:
            模型的回复
        """
        # 多轮对话API请求处理
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
            #print(f"\n=== 多轮对话API尝试配置: 代理={config['use_proxy']}, SSL验证={config['verify_ssl']} ===")
            
            while retry_count < max_retries:
                try:
                    url = os.environ['OPENAI_API_BASE'] + "/chat/completions"
                    #print(os.environ['OPENAI_API_BASE'])
                    #print(os.environ['OPENAI_API_KEY'])
                    
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
                    
                    proxy_status = "无代理" if not config['use_proxy'] else "使用代理"
                    ssl_status = "验证SSL" if config['verify_ssl'] else "跳过SSL验证"
                    print(f"尝试多轮对话请求 (第{retry_count + 1}次) - {proxy_status}, {ssl_status}: {url}")
                    
                    # 使用session发送请求，设置超时时间
                    response = session.post(url, headers=headers, json=payload, timeout=3000)
                    
                    print(f"响应状态码: {response.status_code}")
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        print(f"✓ 多轮对话请求成功！使用配置: {proxy_status}, {ssl_status}")
                        print(f"响应数据: {response_data}")
                        
                        if "error" in response_data:
                            print(f"API返回错误: {response_data}")
                            return "Neglected"
                        #print(response_data)
                        # 获取助手回复
                        assistant_message = response_data['choices'][0]['message']['content']
                        
                        # 将助手回复添加到历史记录
                        self.conversation_history[self.conversation_id].append(
                            {"role": "assistant", "content": assistant_message}
                        )
                        #print(self.conversation_history[self.conversation_id])
                        return assistant_message
                    else:
                        print(f"HTTP错误: {response.status_code}, {response.text}")
                        retry_count += 1
                        if retry_count < max_retries:
                            #print(f"等待2秒后重试...")
                            time.sleep(2)
                        continue
                        
                except requests.exceptions.SSLError as ssl_err:
                    print(f"SSL错误 (第{retry_count + 1}次尝试): {ssl_err}")
                    retry_count += 1
                    if retry_count < max_retries:
                        #print(f"等待3秒后重试...")
                        time.sleep(3)
                    else:
                        #print(f"当前配置SSL连接失败，尝试下一个配置")
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
            print(self.conversation_history[self.conversation_id])
        print("所有网络配置和重试都失败了")
        return "Neglected"
    
    
    
    