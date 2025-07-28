import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from utils.utils_loading import *
from utils.MCQ_constants import *
from utils.utils_loading import *

DATASET_INFO = load_dataset_info("dataset_info/image_dataset.json")

class ImageMCQ:
    def __init__(self, dataset_name: str):
        self.dataset_info = DATASET_INFO[dataset_name]
        self.background = self.load_text_content("background")
        self.case = self.load_table_component("case", "key")
        self.questions = self.load_table_component("question", "key")
        question_type_key = self.dataset_info['question'].get('question_type_key', '')
        if question_type_key == '':
            self.question_type_list = ['single'] * len(self.questions)
        else:
            self.question_type_list = self.load_table_component("question", "question_type_key")
        self.answers = self.load_table_component("answer", "key")
        self.answer_type = self.dataset_info['answer'].get('answer_type', 'choice') if self.dataset_info['answer'] != {} else 'choice'
        self.choices = self.load_table_component("choices", "key")
        self.hints = self.load_table_component("hint", "key")
        if self.case is None:
            self.case = [None] * len(self.questions)
        #self.hints = self.load_and_convert_hint()
        self.language = self.dataset_info['language']
        self.max_score = self.dataset_info['max_score']
        self.image_list = self.load_image()

    def load_table_component(self, type: str, key_name: str):
        information = self.dataset_info[type]
        if information == {}:
            return None
        loading_way = information['loading_way']
        key = information[key_name]
        sub_key = information.get('sub_key', "")
        data = load_dataset_compile(information, loading_way)
        result = []
        for d in data:
            if isinstance(key, str):
                if key == "":
                    continue
                if sub_key == "":
                    result.append(d[key])
                elif isinstance(sub_key, str): 
                    result.append(d[key][sub_key])
                elif isinstance(sub_key, list):
                    choices = []
                    for k in sub_key:
                        if k in d[key]:
                            choices.append(d[key][k])
                    result.append(choices)
            elif isinstance(key, list):
                choices = []
                for k in key:
                    choices.append(d[k])
                result.append(choices)
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

    def load_and_convert_question(self):
        q_info = self.dataset_info['question']
        loading_way = q_info['loading_way']
        question_type_key = q_info.get('question_type_key', '')
        
        key = q_info['key']
        data = load_dataset_compile(q_info, loading_way)
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
        sub_key = c_info.get('sub_key', "")
        data = load_dataset_compile(c_info, loading_way)
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
                        if k in d[key]:
                            choices.append(d[key][k])
                    result.append(choices)
            elif isinstance(key, list):
                choices = []
                for k in key:
                    choices.append(d[k])
                result.append(choices)
        return result
    
    def load_and_convert_hint(self):
        h_info = self.dataset_info['hint']
        if h_info == {}:
            return None
        else: 
            loading_way = h_info['loading_way']
            key = h_info['key']
            data = load_dataset_compile(h_info, loading_way)
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
        data = load_dataset_compile(a_info, loading_way)
        result = []
        for d in data:
            result.append(d[key])
        answer_type = a_info['answer_type']
        return result, answer_type
    
    def load_image(self):
        i_info = self.dataset_info.get('image', {})
        if i_info == {}:
            return None
        loading_way = i_info['loading_way']
        key = i_info['key']
        data = load_dataset_compile(i_info, loading_way)
        result = []
        for d in data:
            result.append(d[key])
        return result

    def load_system_prompt(self):
        s_info = self.dataset_info.get('system_prompt', {})
        if s_info == {}:
            return None
        # 这里可以添加system_prompt的加载逻辑
        return None

    def load_response(self):
        information = self.dataset_info.get("model_response", {})
        if information == {}:
            return None, None
        loading_way = information['loading_way']
        models = information['models']
        result = {}
        data = load_dataset_compile(information, loading_way)
        for model in models:
            result[model] = []
            for d in data:
                result[model].append(d[model])
        return models, result        

    