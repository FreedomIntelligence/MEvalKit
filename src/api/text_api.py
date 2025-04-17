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
class TextAPI:
    def __init__(self, model_name: str, system_prompt: str, user_prompt: str, temperature: float):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.temperature = temperature


    def generate_response(self):
        retry_count = 0
        try:
            load_dotenv()
            url = os.environ['OPENAI_API_BASE'] + "/chat/completions"
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
            response = requests.request("POST", url, headers=headers, json=payload)
            response_data = response.json()
            print(response_data)
            if "error" in response_data:
                return "Neglected"
            else:
                return response_data['choices'][0]['message']['content']
        except Exception as e:
            return "Neglected"




