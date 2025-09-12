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
        self.question = self.load_table_component("question", "key")
        self.reference_answer = self.load_table_component("reference_answer", "key")
        self.scoring_criteria = self.dataset_info.get('scoring_criteria', 'llmjudge')
        self.judge_prompt = self.load_text_content("judge_prompt")
        self.judge_prompt_with_reference = self.load_text_content("judge_prompt_with_reference")
        self.case = self.load_table_component("case", "key") or []
        self.image = self.load_table_component("image", "key")
        self.reference = self.load_table_component("reference", "key")
        

        

    def load_table_component(self, type: str, key_name: str):
        
        information = self.dataset_info[type]
        table_component = {
            'data': None,
            "prompt_template": None,
            "keys": None
        }
        if information is None:
            return table_component
        loading_way = information['loading_way']
        key = information[key_name]
        sub_key = information.get('sub_key', None)
        if loading_way == "":
            return None
        raw_data = load_dataset_compile(information, loading_way)
        data = []
        for d in raw_data:
            if isinstance(key, list):
                if len(key) == 1:
                    # 单个key的情况，直接取值
                    datum = d.get(key[0], None)
                    if sub_key and isinstance(sub_key, list):
                        # 如果有sub_key，从datum中提取对应的值
                        if datum and isinstance(datum, list) and len(datum) > 0:
                            # datum是一个list，从第一个元素中提取sub_key
                            first_item = datum[0]
                            if isinstance(first_item, dict) and len(sub_key) == 1:
                                extracted_value = first_item.get(sub_key[0], "")
                                data.append(str(extracted_value))
                            else:
                                data.append(str(first_item))
                        else:
                            data.append("")
                    else:
                        # 无sub_key，直接处理datum
                        if isinstance(datum, list):
                            # 如果datum是列表（如MT-Bench的turns），返回嵌套列表
                            # 每个元素转换为字符串
                            question_list = [str(item) for item in datum]
                            data.append(question_list)

                        elif datum is None:
                            data.append(None)
                        else:
                            # 将其他类型转换为字符串
                            data.append(str(datum))
                else:
                    # 多个key的情况，组合成嵌套列表
                    values = []
                    for k in key:
                        val = d.get(k, "")
                        values.append(str(val))
                    data.append(values)
            else:
                # key不是list的情况
                datum = d.get(key, None)
                if datum is None:
                    data.append("")
                else:
                    data.append(str(datum))
                    
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

if __name__ == "__main__":
    dataset = QA("MT-Bench")
    print(dataset.reference_answer['data'][0:50])