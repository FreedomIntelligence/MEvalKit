import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
print(project_root)
sys.path.append(str(project_root))

from datasets import load_dataset, Dataset
from utils.utils_stategraph import *
from utils.constants import *
from utils.utils_loading import *
import random
import json
from tqdm import tqdm

DATASET_INFO = load_dataset_info("WindaEvalKit/dataset_info/text_dataset.json")

class TextMCQ:
    
    def __init__(self, dataset_name: str):
        self.dataset_info = DATASET_INFO[dataset_name]
        self.dataset = self.convert_dataset()

    def convert_dataset(self):

        dataset_loading_way = self.dataset_info['loading_way']
        dataset_information = self.dataset_info['dataset']
        dataset = loading_map[dataset_loading_way](dataset_information)
        # if dataset_loading_way == 'huggingface':
        #     dataset_path = dataset_information['dataset_path']
        #     subset_name = dataset_information['subset_name']
        #     split_name = dataset_information['split_name']
        #     dataset = load_dataset(path=dataset_path, name=subset_name, split=split_name, trust_remote_code=True)
        # elif dataset_loading_way == 'csv':
        #     dataset_path = dataset_information['dataset_path']
        #     data_files = os.path.normpath(dataset_path)
        #     dataset = load_dataset(
        #         'csv', 
        #         data_files={'test': data_files},
        #         delimiter=','
        #     )['test']
        # elif dataset_loading_way == "json":
        #     dataset_path = dataset_information['dataset_path']
        #     dataset = load_dataset(
        #         'json',
        #         data_files={'test': dataset_path}
        #     )['test']
        question_information = self.dataset_info['question']
        answer_information = self.dataset_info['answer']
        choices_information = self.dataset_info['choices']
        hint_information = self.dataset_info['hint']

        new_dataset = []
        for example in dataset:
            new_example = {
                "question": "",
                "choices": [],
                "answer": "",
                "hint": "",
                "question_type": ""
            }

            question_key = question_information['question_key']
            question = example[question_key]
            question_type_key = question_information['question_type_key']
            question_type = example[question_type_key] if question_type_key != '' else 'single'
            hint_key = hint_information['hint_key']
            hint = ''
            if hint_key != '':
                hint = example[hint_key]


            if not answer_information['answer_existence']:
                answer = ""
            else:
                answer_key = answer_information['answer_key']
                answer = example[answer_key]
                answer_type = answer_information['answer_type']
            

            choices_key = choices_information['choices_key']
            sub_key = choices_information['sub_key']
            if isinstance(choices_key, str):
                if sub_key == "":
                    choices = example[choices_key]
                elif isinstance(sub_key, str):
                    choices = example[choices_key][sub_key]
                elif isinstance(sub_key, list):
                    choices = []
                    for key in sub_key:
                        choices.append(example[choices_key][key])
            elif isinstance(choices_key, list):
                choices = []
                for key in choices_key:
                    choices.append(example[key])
            choices = [choice for choice in choices if choice != '']

            if answer != '' and answer_type == 'choice':
                if isinstance(answer, int):
                    number_index = answer
                else:
                    if 'A' <= answer <= 'Z' and answer.isupper():
                        number_index = ord(answer) - ord('A')
                    elif 'a' <= answer <= 'z' and answer.islower():
                        number_index = ord(answer) - ord('a')
                    elif '0' <= answer <= '9':
                        number_index = int(answer)
                answer = choices[number_index]
            
                random.shuffle(choices)
                answer_index = chr(choices.index(answer) + 65)
                new_example['answer'] = answer_index
            
            new_example['question'] = question
            new_example['choices'] = choices
            new_example['hint'] = hint
            if question_type == '' or question_type in SINGLE_CHOICE_LIST:
                new_example['question_type'] = 'single'
            elif question_type in MULTIPLE_CHOICE_LIST:
                new_example['question_type'] = 'multiple'
            new_dataset.append(new_example)
        
        new_dataset = Dataset.from_list(new_dataset)
        return new_dataset

    def build_single_prompt(self, line_number: int):
        result_prompt = {
            "prompt": "",
            "type": ""
        }
        example = self.dataset[line_number]
        question = example['question']
        choices = example['choices']
        hint = example['hint']
        question_type = example['question_type']
        if self.dataset_info['language'] == 'en':
            choice_prompt = "The choices are: "
            choice_prompt += "\n"
            for i, choice in enumerate(choices):
                choice_prompt += f"{chr(65 + i)}. {choice} "
            choice_prompt += "\n"
            hint_prompt = ""

            if hint != '':
                hint_prompt = f"The hint is: {hint}"
                hint_prompt += "\n"
            prompt = MCQ_TEMPLATE_EN
            prompt += "\n"
            prompt += f"Now answer the following question based on the choices provided.\nQuestion: {question}\n{choice_prompt}\n{hint_prompt}\n"

            if question_type == 'single':
                prompt += SINGLE_MCQ_TEMPLATE_EN
            elif question_type == 'multiple':
                prompt += MULTIPLE_MCQ_TEMPLATE_EN
        elif self.dataset_info['language'] == 'zh':
            choice_prompt = "选项："
            choice_prompt += "\n"
            for i, choice in enumerate(choices):
                choice_prompt += f"{chr(65 + i)}. {choice} "
            choice_prompt += "\n"
            hint_prompt = ""

            if hint != '':
                hint_prompt = f"提示：{hint}"
                hint_prompt += "\n"
            prompt = MCQ_TEMPLATE_ZH
            prompt += "\n"
            prompt += f"现在根据所提供的选项回答以下问题。\n问题：{question}\n{choice_prompt}\n{hint_prompt}\n"

            if question_type == 'single':
                prompt += SINGLE_MCQ_TEMPLATE_ZH
            elif question_type == 'multiple':
                prompt += MULTIPLE_MCQ_TEMPLATE_ZH
        result_prompt['prompt'] = prompt
        result_prompt['type'] = question_type
        return result_prompt

    
    
    def build_prompts(self):
        prompt_list = []
        length = len(self.dataset)
        for i in range(length):
            prompt = self.build_single_prompt(i)
            prompt_list.append(prompt)
        return prompt_list
    

if __name__ == "__main__":
    # data_path = "WindaEvalKit/data/gpqa_main.csv"
    # data_files = os.path.normpath(data_path)
    # dataset = load_dataset('csv', data_files=data_files, delimiter=',')
    # print(dataset[0]['Correct Answer'])
    dataset = TextMCQ("GPQA")
    print(dataset.dataset[0])

    



