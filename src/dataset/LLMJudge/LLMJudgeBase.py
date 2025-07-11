import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from datasets import load_dataset, Dataset
from utils.utils_loading import *

from typing import List, Tuple
DATASET_INFO = load_dataset_info("dataset_info/LLMJudge_dataset.json")


class LLMJudgeBase:
    
    def __init__(self, dataset_name: str):
        self.dataset_info = DATASET_INFO[dataset_name]
        self.language = self.dataset_info.get('language', 'en')
        self.max_score = self.dataset_info.get('max_score', 10)
        self.background = self.load_text_content("background")
        self.questions = self.load_table_component("question") or []
        self.answers = self.load_table_component("reference_answer") or []
        self.judge_prompt = self.load_text_content("judge_prompt")
        self.judge_prompt_with_reference = self.load_text_content("judge_prompt_with_reference")
        self.case = self.load_table_component("case") or []
        
    def load_table_component(self, type: str):
        information = self.dataset_info.get(type, {})
        if information == {}:
            return None
        loading_way = information.get('loading_way', '')
        key = information.get('key', '')
        sub_key = information.get('sub_key', "")
        data = loading_map[loading_way](information)
        result = []
        for d in data:
            if isinstance(key, str):
                if key == "":
                    continue
                if sub_key == "":
                    result.append(d.get(key, None))
                elif isinstance(sub_key, str):
                    result.append(d.get(key, {}).get(sub_key, None))
                elif isinstance(sub_key, list):
                    choices = []
                    for k in sub_key:
                        if k in d.get(key, {}):
                            choices.append(d[key][k])
                    result.append(choices)
            elif isinstance(key, list):
                choices = []
                for k in key:
                    choices.append(d.get(k, None))
                result.append(choices)
        return result
    
    def load_text_content(self, type: str):
        information = self.dataset_info.get(type, {})
        if information == {}:
            return None
        loading_way = information.get("loading_way", "")
        if loading_way == "content":
            path = information.get("path", "")
            if isinstance(path, list):
                path = path[0] if path else ""
            return path
        elif loading_way == "txt":
            path = information.get("path", "")
            if isinstance(path, list):
                path = path[0] if path else ""
            with open(path, "r") as f:
                lines = f.readlines()
            content = "\n".join(lines)
            return content
        return None
    
        

    def load_response(self):
        information = self.dataset_info.get("model_response", {})
        if information == {}:
            return None, None
        loading_way = information['loading_way']
        models = information['models']
        result = {}
        data = loading_map[loading_way](information)
        for model in models:
            result[model] = []
            for d in data:
                result[model].append(d[model])
        return models, result

    def load_prompt(self, type: str):
        information = self.dataset_info[type]
        if information == {}:
            return None
        loading_way = information['loading_way']
        if loading_way == 'content':
            return information['path']
        else:
            path = information['path']
            with open(path, 'r') as f:
                lines = f.readlines()
            prompt = '\n'.join(lines)
            return prompt


    

    



if __name__ == "__main__":
    dataset = LLMJudgeBase("MedEthicsMatrixCase")
    print(dataset.model_responses['claude-3.7-sonnet-20250219'])