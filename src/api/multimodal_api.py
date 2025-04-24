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

class MultimodalAPI:
    def __init__(self, model_name: str, system_prompt: str, user_prompt: str, image_input, temperature: float):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.image_input = image_input
        self.temperature = temperature

        self.max_retries = 5
        self.initial_retry_delay = 1.0
        self.max_retry_delay = 30.0

    def handle_image_url(self):
        if isinstance(self.image_input, str) and self.image_input.startswith(("http://", "https://")):
            return self.image_input
        if isinstance(self.image_input, str) and os.path.exists(self.image_input):
            return self.encode_local_image()
        if isinstance(self.image_input, Image.Image):
            return self.encode_pil_input()
        if isinstance(self.image_input, np.ndarray):
            return self.encode_cv2_input()
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
        return self.encode_pil_input(pil_image)

    
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
            response = requests.request("POST", url, headers=headers, json=payload)
            response_data = response.json()
            #print(response_data)
            if 'error' in response_data:
                return "Neglected"
            else:
                return response_data['choices'][0]['message']['content']
        except Exception as e:
            return "Neglected"
