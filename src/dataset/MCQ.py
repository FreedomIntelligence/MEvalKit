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

class MCQ():

    def __init__(self, dataset_name: str):
        self.dataset_info = DATASET_INFO[dataset_name]
        self.language = self.dataset_info['language']
        self.max_score = self.dataset_info['max_score']
        self.question = self.load_table_component("question", "key")
        self.background = self.load_text_content("background")
        self.case = self.load_table_component("case", "key")
        if self.case is None:
            self.case = [None] * len(self.question)
        self.image = self.load_table_component("image", "key")
        if self.image is None:
            self.image = [None] * len(self.question)
        self.answer = self.load_table_component("answer", "key")
        if self.answer is None:
            self.answer = [None] * len(self.question)
        self.choices = self.load_table_component("choices", "key")
        if self.choices is None:
            self.choices = [None] * len(self.question)

    def load_table_component(self, type: str, key_name: str):
        information = self.dataset_info[type]
        if information is None:
            return None
        loading_way = information['loading_way']
        key = information.get(key_name, "")
        sub_key = information.get("sub_key", "")
        data = load_dataset_compile(information, loading_way)
        result = []
        for d in data:
            if isinstance(key, str):
                # 如果key为空字符串，跳过这个字段
                if key == "":
                    continue
                if sub_key == "":
                    result.append(d[key])
                elif isinstance(sub_key, str): 
                    result.append(d[key][sub_key])
                elif isinstance(sub_key, list):
                    elements = []
                    for k in sub_key:
                        if k in d[key]:
                            elements.append(d[key][k])
                    result.append(elements)
            elif isinstance(key, list):
                elements = []
                for k in key:
                    elements.append(d[k])
                result.append(elements)
        return result
    
    def load_text_content(self, type: str):
        information = self.dataset_info[type]
        if information == {}:
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