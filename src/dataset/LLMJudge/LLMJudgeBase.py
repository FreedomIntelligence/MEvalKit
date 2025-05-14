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
        self.questions = self.load("question")
        self.answers = self.load("answer")
        
    def load(self, type: str):
        information = self.dataset_info[type]
        loading_way = information['loading_way']
        key = information['key']

        result = []
        data = loading_map[loading_way](information)
        for d in data:
            try:
                result.append(d[key])   
            except KeyError as e:
                result.append("")
        return result

    

    



if __name__ == "__main__":
    dataset = LLMJudgeBase("MT-Bench")
    #print(dataset.questions)
    print(dataset.answers)