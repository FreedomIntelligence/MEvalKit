#!/usr/bin/env python3
"""
调试API响应的脚本
"""

import requests
import urllib3
from requests.exceptions import SSLError, ConnectionError

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def debug_api_response():
    """调试API响应"""
    print("调试API响应...")
    
    # 测试参数
    url = "https://api.huatuogpt.cn/v1/chat/completions"
    headers = {
        "Authorization": "Bearer sk-fPz5uPZn2ubb9Qexx62yWcFl55Z46iRdBfdlvnjufQ6o0BVo",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "doubao-1.5-pro-32k",
        "messages": [
            {
                "role": "system",
                "content": "你是一个有用的助手。"
            },
            {
                "role": "user",
                "content": "你好"
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.7
    }
    
    print(f"测试URL: {url}")
    print(f"请求头: {headers}")
    print(f"请求体: {payload}")
    
    # 测试HTTPS
    print("\n1. 测试HTTPS连接...")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        print(f"响应内容: {response.text[:500]}...")
        
        if response.status_code == 200:
            try:
                json_data = response.json()
                print(f"JSON解析成功: {json_data}")
            except Exception as e:
                print(f"JSON解析失败: {e}")
        else:
            print(f"HTTP错误: {response.status_code}")
            
    except SSLError as e:
        print(f"SSL错误: {e}")
    except Exception as e:
        print(f"其他错误: {e}")
    
    # 测试HTTP
    print("\n2. 测试HTTP连接...")
    http_url = url.replace("https://", "http://")
    try:
        response = requests.post(http_url, headers=headers, json=payload, timeout=30, verify=False)
        print(f"状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        print(f"响应内容: {response.text[:500]}...")
        
        if response.status_code == 200:
            try:
                json_data = response.json()
                print(f"JSON解析成功: {json_data}")
            except Exception as e:
                print(f"JSON解析失败: {e}")
        else:
            print(f"HTTP错误: {response.status_code}")
            
    except Exception as e:
        print(f"HTTP连接错误: {e}")

if __name__ == "__main__":
    debug_api_response() 