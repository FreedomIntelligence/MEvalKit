import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from datasets import load_dataset, Dataset
from utils.MCQ_constants import *
from utils.utils_loading import *
import random
import json
from tqdm import tqdm

DATASET_INFO = load_yaml_dataset_info("dataset_info/MCQ_config.yaml")

class MCQ:

    def __init__(self, dataset_name: str):
        self.dataset_info = DATASET_INFO[dataset_name]
        self.language = self.dataset_info['language']
        self.max_score = self.dataset_info['max_score']
        self.question = self.load_table_component("question", "key")
        self.question_type_key = self.dataset_info['question'].get('question_type_key', None)
        if self.question_type_key is None:
            self.question_type_list = ['single'] * len(self.question)
        else:
            self.question_type_list = self.load_table_component("question", "question_type_key")
        self.background = self.load_text_content("background")
        self.case = self.load_table_component("case", "key")
        self.image = self.load_table_component("image", "key")
        self.answer = self.load_table_component("answer", "key")
        self.hint = self.load_table_component("hint", "key")
        self.answer_type = self.dataset_info['answer'].get('answer_type', 'choice') if self.dataset_info['answer'] != {} else 'choice'
        self.choice = self.load_table_component("choices", "key")

    def load_table_component(self, type: str, key_name: str):
        data = []
        information = self.dataset_info.get(type, None)
        if information is None:
            return None
        loading_way = information.get('loading_way', "")
        key = information.get(key_name, "")
        sub_key = information.get("sub_key", "")
        if loading_way == "":
            return None
        raw_data = load_dataset_compile(information, loading_way)
        
        # 特殊处理choices类型：如果有多个key，需要将它们组合成一个选项列表
        if type == "choices" and len(key) > 1:
            for d in raw_data:
                # 将多个字段组合成一个选项列表
                choice_list = []
                for k in key:
                    choice_list.append(d[k])
                data.append(choice_list)
        else:
            # 原有逻辑：单个字段或其他类型
            for d in raw_data:
                for k in key:
                    data.append(d[k])
        return data
    
    def load_text_content(self, type: str):
        information = self.dataset_info[type]
        if information is None:
            return None
        loading_way = information["loading_way"]
        if loading_way == "content":
            return information["path"]
        elif loading_way == "txt":
            path = information["path"]
            with open(path, "r") as f:
                lines = f.readlines()
            content = "\n".join(lines)
            return content