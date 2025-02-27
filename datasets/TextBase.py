from datasets import load_dataset, Dataset
from utils import load_dataset_info
from transformers import AutoTokenizer, AutoModelForCausalLM
DATASET_INFO = load_dataset_info("/mnt/nvme1n1/yuang_workspace/EvaluatorKit/datasets/dataset_info.json")

class TextBaseDataset:
    def __init__(self, dataset_name):
        self.dataset_info = DATASET_INFO[dataset_name]
        self.dataset = self.convert_dataset()
        #self.dataset = load_dataset(path=self.dataset_info["dataset"]["dataset_path"], name=self.dataset_info["dataset"]["subset_name"], split=self.dataset_info["dataset"]["split_name"])
        self.system_prompt = self.dataset_info["system_prompt"]

    def convert_dataset(self):
        dataset_information = self.dataset_info["dataset"]
        dataset_loading_way = dataset_information["dataset_loading_way"]
        if dataset_loading_way == "huggingface":
            dataset_path = dataset_information["dataset_path"]
            subset_name = dataset_information["subset_name"]
            split_name = dataset_information["split_name"]
            dataset = load_dataset(path=dataset_path, name=subset_name, split=split_name)
        question_information = self.dataset_info["question"]
        answer_information = self.dataset_info["answer"]
        hint_information = self.dataset_info["hint"]

        new_dataset = []
        for example in dataset:
            new_example = {
                "question": "",
                "answer": "",
                "hint": ""
            }
            question_key = question_information["question_key"]
            question = example[question_key]
            hint_key = hint_information["hint_key"]
            hint = ""
            if hint_key != "":
                hint = example[hint_key]
            answer_key = answer_information["answer_key"]
            answer = example[answer_key]
            new_example["question"] = question
            new_example["hint"] = hint
            new_example["answer"] = answer
            new_dataset.append(new_example)
        new_dataset = Dataset.from_list(new_dataset)
        return new_dataset
    
    def build_single_prompt(self, line_number: int):
        example = self.dataset[line_number]
        question = example["question"]
        prompt = f"Answer the following question:\n{question}\n"
        message = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': prompt}
        ]
        return message
    
    def load_tokenizer_and_model(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        return tokenizer, model



            