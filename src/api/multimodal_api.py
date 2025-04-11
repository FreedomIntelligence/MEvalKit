import requests
import base64
import os
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
from dotenv import load_dotenv

class MultimodalAPI:
    def __init__(self, model_name: str, assistant_prompt: str, user_prompt: str):
        self.model_name = model_name
        self.assistant_prompt = assistant_prompt
        self.user_prompt = user_prompt

    def handle_image_url(self, image_input):
        if isinstance(image_input, str) and image_input.startswith(("http://", "https://")):
            return image_input
        if isinstance(image_input, str) and os.path.exists(image_input):
            return self.encode_local_image(image_input)
        if isinstance(image_input, Image.Image):
            return self.encode_pil_input(image_input)
        if isinstance(image_input, np.ndarray):
            return self.encode_cv2_input(image_input)
        raise ValueError("Invalid image input")

    def encode_local_image(self, image_input):
        mime_type = {
        '.jpg': 'jpeg', '.jpeg': 'jpeg',
        '.png': 'png', '.webp': 'webp'
    }.get(os.path.splitext(image_input)[1].lower(), 'jpeg')

        with open(image_input, 'rb') as f:
            img = Image.open(f)
            buffered = BytesIO()
            img.save(buffered, format=mime_type.upper())
            b64 = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/{mime_type};base64,{b64}"

    def encode_pil_input(self, image_input):
        image_input = image_input.convert("RGB")
        buffered = BytesIO()
        image_input.save(buffered, format='JPEG')
        b64 = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"

    def encode_cv2_input(self, image_input):
        image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_input)
        return self.encode_pil_input(pil_image)


    def generate_response(self, image_input):
        load_dotenv()
        url = "https://api.siliconflow.cn/v1/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": self.assistant_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "image_url": {
                                "detail": "auto",
                                "url": self.handle_image_url(image_input)
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
            "temperature": 0,
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
        return response.json()['choices'][0]['message']['content']
        

