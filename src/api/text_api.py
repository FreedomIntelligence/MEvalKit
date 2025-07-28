"""
文本API接口模块

该模块提供了与文本大语言模型API的交互接口，支持多种网络配置和错误重试机制。
主要功能包括：
- 标准OpenAI接口调用
- 多种网络配置支持（代理、SSL验证）
- 自动重试和错误处理
- 响应格式化和验证

作者: MEvalKit Team
版本: 1.0.0
"""

import requests
import base64
import os
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
from dotenv import load_dotenv
import random
import time
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ssl

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class TextAPI:
    """
    文本API接口类
    
    提供与文本大语言模型的标准化接口，支持多种网络配置和错误处理机制。
    
    属性:
        model_name: 模型名称
        system_prompt: 系统提示词
        user_prompt: 用户提示词
        temperature: 生成温度参数
        model_key: API密钥
        api_base: API基础URL
    """
    
    def __init__(self, model_name: str, system_prompt: str, user_prompt: str, temperature: float, model_key: str, api_base: str = ""):
        """
        初始化文本API接口
        
        参数:
            model_name: 要使用的模型名称
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            temperature: 生成温度参数（控制随机性）
            model_key: API访问密钥
            api_base: API基础URL（可选）
        """
        load_dotenv()
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.temperature = temperature
        # 设置环境变量，优先使用传入的参数
        os.environ['OPENAI_API_KEY'] = model_key if model_key != "" else os.environ['OPENAI_API_KEY']
        os.environ['OPENAI_API_BASE'] = api_base if api_base != "" else os.environ['OPENAI_API_BASE']

    def generate_response(self):
        """
        生成模型响应
        
        该方法实现了多种网络配置的自动尝试机制，包括：
        - 不同的代理设置
        - SSL验证选项
        - 自动重试机制
        - 错误处理和恢复
        
        返回:
            str: 模型的响应文本，如果所有尝试都失败则返回错误信息
        """
        # 调用标准OpenAI接口
        retry_count = 0
        max_retries = 3
        
        # 尝试不同的网络配置
        # 这些配置涵盖了不同的网络环境需求
        proxy_configs = [
            {'use_proxy': False, 'verify_ssl': False},  # 不使用代理，不验证SSL（内网环境）
            {'use_proxy': True, 'verify_ssl': False},   # 使用代理，不验证SSL（代理环境）
            {'use_proxy': False, 'verify_ssl': True},   # 不使用代理，验证SSL（标准环境）
        ]
        
        for config in proxy_configs:
            retry_count = 0
            print(f"\n=== 尝试配置: 代理={config['use_proxy']}, SSL验证={config['verify_ssl']} ===")
            
            while retry_count < max_retries:
                try:
                    load_dotenv()
                    base_url = os.environ['OPENAI_API_BASE']
                    url = base_url + "/chat/completions"
                    #print(url)
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
                    # 针对常见的网络错误进行自动重试
                    retry_strategy = Retry(
                        total=3,  # 最多重试3次
                        backoff_factor=1,  # 退避因子
                        status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的HTTP状态码
                    )
                    adapter = HTTPAdapter(max_retries=retry_strategy)
                    session.mount("http://", adapter)
                    session.mount("https://", adapter)
                    
                    # 构建API请求载荷
                    payload = {
                            "model": self.model_name,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": self.system_prompt
                                },
                                {
                                    "role": "user",
                                    "content": self.user_prompt
                                }
                            ],
                            "max_tokens": 1024,  # 最大生成token数
                            "temperature": self.temperature,  # 温度参数
                            "top_p": 0.7,  # 核采样参数
                            "frequency_penalty": 0.5,  # 频率惩罚
                            "n": 1,  # 生成数量
                            "response_format": {
                                "type": "text"  # 响应格式为纯文本
                            }
                    }
                    headers = {
                            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                            "Content-Type": "application/json"
                        }
                    
                    proxy_status = "无代理" if not config['use_proxy'] else "使用代理"
                    ssl_status = "验证SSL" if config['verify_ssl'] else "跳过SSL验证"
                    print(f"尝试请求 (第{retry_count + 1}次) - {proxy_status}, {ssl_status}: {url}")
                    
                    # 使用session发送请求，设置超时时间
                    response = session.post(url, headers=headers, json=payload, timeout=3000)
                    
                    print(f"响应状态码: {response.status_code}")
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        print(f"✓ 请求成功！使用配置: {proxy_status}, {ssl_status}")
                        print(f"响应数据: {response_data}")
                        
                        if "error" in response_data:
                            print(f"API返回错误: {response_data}")
                            return "Neglected"
                        else:
                            return response_data['choices'][0]['message']['content']
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


if __name__ == "__main__":
    system_prompt = "你的任务是：给你一道医学题，你需要判断该题目所涉及的科室是什么。\
        注意，一道题目仅涉及一个科室。\
        备选项如下：\
            - 中医科：Traditional Chinese Medicine \
            - 产科：Obstetrics \
            - 介入外科（血管与肿瘤）：Interventional \
            - 儿科：Pediatrics \
            - 全科医学科：General \
            - 内分泌代谢科：Endocrinology \
            - 口腔科：Stomatology \
            - 呼吸与危重症医学科：Respiratory \
            - 妇科：Gynecology \
            - 康复医学科：Rehabilitation \
            - 心胸外科：Cardiothoracic \
            - 心血管内科：Cardiovascular \
            - 感染性疾病科：Infectious \
            - 普外科：GeneralOut \
            - 泌尿外科：Urology \
            - 消化内科：Gastroenterology \
            - 烧伤整形科：Burn \
            - 甲状腺乳腺外科：Thyroid \
            - 疼痛科：Pain \
            - 皮肤科：Dermatology \
            - 眼科：Ophthalmology \
            - 神经内科：Neurology \
            - 神经外科：Neurosurgery \
            - 精神心理科：Psychiatry \
            - 老年医学科：Geriatrics \
            - 耳鼻喉科：Otolaryngology \
            - 肛肠外科：Anorectal \
            - 肾内风湿科：Nephrology \
            - 肿瘤科门诊：Oncology \
            - 营养科：Nutrition \
            - 血液内科：Hematology \
            - 血管外科：Vascular \
            - 骨科：Orthopedics \
        输出方式：唯一的一行，内容为“科室为：<科室名称（中文）>”，不要有任何其他内容。\
        输出示例： \
            科室为：心血管内科"
    user_prompt = "题目：张某，女，43岁，1个月来干咳，胸闷憋气，呼吸困难，夜间明显，影响睡眠，既往有类似发作病史，查体:双肺可闻及哮鸣音。治疗应首选的药物是"
    temperature = 0
    api = TextAPI(model_name="gpt-4o", system_prompt=system_prompt, user_prompt=user_prompt, temperature=temperature, model_key="", api_base="https://api.huatuogpt.cn/v1")
    response = api.generate_response()
    print(response)

