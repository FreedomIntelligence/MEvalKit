import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from datasets import load_dataset, Dataset
from utils.MCQ_constants import *
from utils.utils_loading import *
import random
import json
from tqdm import tqdm

DATASET_INFO = load_dataset_info("dataset_info/text_dataset.json")

class TextMCQ():

    def __init__(self, dataset_name: str):
        self.dataset_info = DATASET_INFO[dataset_name]
        self.background = self.load_text_content("background")
        self.case = self.load_table_component("case", "key")
        self.questions = self.load_table_component("question", "key")
        # 修复：如果question_type_key为空，则生成默认类型列表
        question_type_key = self.dataset_info['question'].get('question_type_key', '')
        if question_type_key == '':
            self.question_type_list = ['single'] * len(self.questions)
        else:
            self.question_type_list = self.load_table_component("question", "question_type_key")
        self.answers = self.load_table_component("answer", "key")
        # answer_type直接从配置中读取，不是数据字段
        self.answer_type = self.dataset_info['answer'].get('answer_type', 'choice') if self.dataset_info['answer'] != {} else 'choice'
        self.choices = self.load_table_component("choices", "key")
        self.hints = self.load_table_component("hint", "key")
        # case为None时初始化为[None]*len(self.questions)
        if self.case is None:
            self.case = [None] * len(self.questions)
        #self.questions, self.question_type_list = self.load_and_convert_question()
        #self.answers, self.answer_type = self.load_and_convert_answer()
        #self.choices = self.load_and_convert_choices()
        #self.hints = self.load_and_convert_hint()
        self.language = self.dataset_info['language']
        self.max_score = self.dataset_info['max_score']



    def load_table_component(self, type: str, key_name: str):
        information = self.dataset_info[type]
        if information == {}:
            return None
        loading_way = information['loading_way']
        key = information[key_name]
        # 检查是否有sub_key字段，如果没有则设为空字符串
        sub_key = information.get('sub_key', "")
        data = loading_map[loading_way](information)
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
    
    # 根据config内容，提取评测集的题目内容及题目类型
    def load_and_convert_question(self):
        # 获取评测集中
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
                if question_type in SINGLE_CHOICE_LIST:
                    question_type = "single"
                elif question_type in MULTIPLE_CHOICE_LIST:
                    question_type = "multiple"
            question_type_list.append(question_type)
        return result, question_type_list
    
    # 根据config内容，提取评测集的选项
    def load_and_convert_choices(self):
        c_info = self.dataset_info['choices']
        if c_info == {}:
            return None
        loading_way = c_info['loading_way']
        key = c_info['key']
        # 检查是否有sub_key字段，如果没有则设为空字符串
        sub_key = c_info.get('sub_key', "")
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
                        if k in d[key]:
                            choices.append(d[key][k])
                    result.append(choices)
            elif isinstance(key, list):
                choices = []
                for k in key:
                    choices.append(d[k])
                result.append(choices)

        return result
    
    # 根据config内容，提取评测集每个问题的提示
    def load_and_convert_hint(self):
        # 若config中未提供提示相关信息，则返回None
        h_info = self.dataset_info['hint']
        if h_info == {}:
            return None
        else: 
            # 读取
            loading_way = h_info['loading_way']
            key = h_info['key']
            data = loading_map[loading_way](h_info)
            result = []
            for d in data:
                result.append(d[key])
            return result
    
    # 提取评测集中的答案列表，并记录答案类型（选项形式/内容形式）
    def load_and_convert_answer(self):
        # 若config中未提供答案相关信息，则全部返回None
        a_info = self.dataset_info['answer']
        if a_info == {}:
            return None, None
        loading_way = a_info['loading_way']
        key = a_info['key']
        data = loading_map[loading_way](a_info)
        result = []
        for d in data:
            result.append(d[key])
        answer_type = a_info['answer_type']
        return result, answer_type
    
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
    
    def load_background(self):
        information = self.dataset_info["background"]


    
if __name__ == "__main__":
    dataset = TextMCQ("CMMLU_Med")

    #print(dataset.answers)
    print(len(dataset.questions))