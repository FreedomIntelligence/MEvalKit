import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from datasets import load_dataset, Dataset
from src.utils.utils_loading import *

from typing import List, Tuple

DATASET_INFO = load_dataset_info("dataset_info/QA_config.yaml")

class QA:

    def __init__(self, dataset_name: str):
        self.dataset_info = DATASET_INFO[dataset_name]
        self.language = self.dataset_info['language']
        self.max_score = self.dataset_info['max_score']
        self.background = self.load_text_content("background")
        self.question = self.load_table_component("question", "key") or []
        self.reference_answer = self.load_table_component("reference_answer", "key")
        self.scoring_criteria = self.dataset_info.get('scoring_criteria', 'llmjudge')
        self.judge_prompt = self.load_text_content("judge_prompt")
        self.judge_prompt_with_reference = self.load_text_content("judge_prompt_with_reference")
        self.case = self.load_table_component("case", "key") or []
        

    def load_table_component(self, type: str, key_name: str):
        
        information = self.dataset_info[type]
        if information is None:
            return None
        table_component = {
            'data': [],
            "prompt_template": None,
            "keys": None
        }
        loading_way = information['loading_way']
        key = information[key_name]
        sub_key = information.get('sub_key', "")
        if loading_way == "":
            return None
        raw_data = load_dataset_compile(information, loading_way)
        data = []
        for d in raw_data:
            if isinstance(key, str):
                if key == "":
                    continue
                if sub_key == "":
                    data.append(d.get(key, None))
                elif isinstance(sub_key, str):
                    data.append(d.get(key, {}).get(sub_key, None))
                elif isinstance(sub_key, list):
                    dat = []
                    for k in sub_key:
                        if k in d.get(key, {}):
                            dat.append(d.get(key, {}).get(k, None))
                    data.append(dat)
            elif isinstance(key, list):
                if len(key) == 1:
                    # 单个key的情况，直接取值不再包装
                    data.append(d.get(key[0], None))
                else:
                    # 多个key的情况，返回数组
                    dat = []
                    for k in key:
                        dat.append(d.get(k, None))
                    data.append(dat)
        table_component['data'] = data
        table_component['prompt_template'] = information.get("prompt_template", None)
        table_component['keys'] = key
        return table_component
    
    def load_text_content(self, type: str):
        information = self.dataset_info.get(type, None)
        if information is None:
            return None
        loading_way = information.get("loading_way", "")
        if loading_way == "content":
            return information.get("path", None)
        elif loading_way == "txt":
            path = information.get("path", None)
            if path is None:
                return None
            with open(path, "r") as f:
                lines = f.readlines()
            content = "\n".join(lines)
            return content
        return None
    