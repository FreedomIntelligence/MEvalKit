from datasets import load_dataset, Dataset
from utils.utils_stategraph import *

DATASET_INFO = load_dataset_info("WindaEvalKit/dataset_info/text_dataset.json")

class TextBase:
    def __init__(self, dataset_name: str):
        self.dataset_info = DATASET_INFO[dataset_name]
        self.dataset = self.convert_dataset()

    def convert_dataset(self):
        dataset_information = self.dataset_info['dataset']
        dataset_loading_way = self.dataset_info['dataset_loading_way']
        if dataset_loading_way == 'huggingface':
            dataset_path = dataset_information['dataset_path']
            subset_name = dataset_information['subset_name']
            split_name = dataset_information['split_name']
            dataset = load_dataset(dataset_path, subset_name, split_name)
        question_information = self.dataset_info['question']
        answer_information = self.dataset_info['answer']
        hint_information = self.dataset_info['hint']

        new_dataset = []
        for example in dataset:
            new_example = {
                "question": "",
                "answer": "",
                "hint": ""
            }

            question_key = question_information['question_key']
            question = example[question_key]

            hint_key = hint_information['hint_key']
            hint = ''
            if hint_key != '':
                hint = example[hint_key]

            answer_key = answer_information['answer_key']
            answer = example[answer_key]
            
            new_example['question'] = question
            new_example['answer'] = answer
            new_example['hint'] = hint
            new_dataset.append(new_example)
        
        new_dataset = Dataset.from_list(new_dataset)
        return new_dataset



    def build_single_prompt(self, line_number: int):
        example = self.dataset[line_number]
        question = example['question']
        hint = example['hint']
        prompt = f"Question: {question}\nHint: {hint}"
        return prompt

    def build_prompts(self):
        prompt_list = []
        length = len(self.dataset)
        for i in range(length):
            prompt = self.build_single_prompt(i)
            prompt_list.append(prompt)
        return prompt_list