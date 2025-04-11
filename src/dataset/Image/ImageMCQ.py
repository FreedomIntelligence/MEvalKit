import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
print(project_root)
sys.path.append(str(project_root))

from utils.utils_loading import *
from utils.MCQ_constants import *
from utils.utils_stategraph import *
from utils.utils_loading import *

DATASET_INFO = load_dataset_info("dataset_info/image_dataset.json")

class ImageMCQ:
    def __init__(self, dataset_name: str):
        self.dataset_info = DATASET_INFO[dataset_name]
        self.questions, self.question_type_list = self.load_and_convert_question()
        self.answers, self.answer_type = self.load_and_convert_answer()
        self.choices = self.load_and_convert_choices()
        self.hints = self.load_and_convert_hint()
        self.language = self.dataset_info['language']
        self.image_list = self.load_image()

    def load_and_convert_question(self):
        q_info = self.dataset_info['question']
        loading_way = q_info['loading_way']
        question_type_key = q_info['question_type_key']
        
        key = q_info['key']
        data = loading_map[loading_way](q_info)
        result = []
        question_type_list = []
        for d in data:
            result.append(d[key])
            if question_type_key == "":
                question_type = "single"
            else:
                question_type = d[question_type_key]
            question_type_list.append(question_type)
        return result, question_type_list
    
    def load_and_convert_choices(self):
        c_info = self.dataset_info['choices']
        if c_info == {}:
            return None
        loading_way = c_info['loading_way']
        key = c_info['key']
        sub_key = c_info['sub_key']
        data = loading_map[loading_way](c_info)
        result = []
        for d in data:
            if isinstance(key, str):
                if sub_key == "":
                    result.append(d[key])
                elif isinstance(sub_key, str):
                    result.append(d[key][sub_key])
                elif isinstance(sub_key, list):
                    choices = []
                    for k in sub_key:
                        choices.append(d[key][k])
                    result.append(choices)
        return result
    
    def load_and_convert_hint(self):
        h_info = self.dataset_info['hint']
        if h_info == {}:
            return None
        else: 
            loading_way = h_info['loading_way']
            key = h_info['key']
            data = loading_map[loading_way](h_info)
            result = []
            for d in data:
                result.append(d[key])
            return result
        
    def load_and_convert_answer(self):
        a_info = self.dataset_info['answer']
        if a_info == {}:
            return None
        loading_way = a_info['loading_way']
        key = a_info['key']
        data = loading_map[loading_way](a_info)
        result = []
        for d in data:
            result.append(d[key])
        answer_type = a_info['answer_type']
        return result, answer_type
    
    def load_image(self):
        i_info = self.dataset_info['image']
        loading_way = i_info['loading_way']
        key = i_info['key']
        data = loading_map[loading_way](i_info)
        result = []
        for d in data:
            result.append(d[key])
        return result
            

    