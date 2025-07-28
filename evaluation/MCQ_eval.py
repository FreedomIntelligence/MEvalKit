import sys
import os
import json
import requests
from pathlib import Path
from openai import BadRequestError, AuthenticationError
from typing import Optional

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.MCQ import *
from src.api.text_api import *
from src.utils.MCQ_constants import *
from src.utils.default_prompts import *
from src.utils.model_and_dataset import *
from tqdm import tqdm
import concurrent.futures
from typing import List, Tuple, Dict, Any, Literal
from dotenv import load_dotenv
import re

def extract_answer(response: str, dataset_name: str):
    """
    提取单选题答案
    
    参数:
        response: 模型的响应文本
        dataset_name: 数据集名称，用于确定答案格式
        
    返回:
        提取的答案选项（如A、B、C、D），如果未找到则返回None
    """
    if response == "Neglected":
        return response
    max_letter, PATTERNS = build_patterns(dataset_name)
    for pattern in PATTERNS:
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    return None

def extract_multi_answer(response: str, dataset_name: str):
    """
    提取多选题答案
    
    参数:
        response: 模型的响应文本
        dataset_name: 数据集名称，用于确定答案格式
        
    返回:
        提取的答案选项列表（如['A', 'B', 'C']），如果未找到则返回None
    """