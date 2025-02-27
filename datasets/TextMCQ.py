from datasets import load_dataset, Dataset, get_dataset_config_names
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import os
import json
from tqdm import tqdm
from utils import load_dataset_info
import random
from TextBase import TextBaseDataset

PATTERNS = [
        # 直接匹配
        r'^\s*([A-D])\s*$',                    # 单个字母
        r'[^A-D]*([A-D])[^A-D]*$',             # 句子中的单个字母
        
        # 中文关键词
        r'答案[是为：:\s]*([A-D])',             # "答案是A"，"答案为B"
        r'选[择项]?\s*([A-D])',                 # "选A"，"选择B"，"选项C"
        r'([A-D])\s*选[择项]',                  # "A选项"，"B选择"
        r'正确[的答案]?[是为：:\s]*([A-D])',     # "正确答案是A"，"正确的是B"
        r'应[该当]?[是选择：:\s]*([A-D])',       # "应该是A"，"应当选择B"
        r'我[的认为]?[选择认为]([A-D])',         # "我选A"，"我认为B"
        
        # 英文关键词
        r'[Tt]he\s+answer\s+is\s+([A-D])',     # "The answer is A"
        r'[Cc]hoose\s+([A-D])',                # "Choose A"
        r'[Ss]elect\s+([A-D])',                # "Select A"
        r'[Oo]ption\s+([A-D])',                # "Option A"
        r'[Tt]he\s+correct\s+[answer\s+]?is\s+([A-D])',  # "The correct answer is A"
        r'[Ii]\s+[choose\s+]?([A-D])',         # "I choose A"
        
        # 带括号的格式
        r'\(([A-D])\)',                        # "(A)"
        r'（([A-D])）',                         # "（A）"
        r'【([A-D])】',                         # "【A】"
        r'\[([A-D])\]',                        # "[A]"
        
        # 特殊格式
        r'[选择项]\s*([A-D])\s*[选择项]',        # "选项A选项"
        r'答案?对应\s*([A-D])',                 # "答对应A"，"答案对应B"
        r'([A-D])\s*[是为]正确[的答案]?',        # "A是正确答案"
        r'最终[选择答案]\s*[为是：:\s]*([A-D])',  # "最终选择为A"
        
        # 模糊匹配（作为最后的尝试）
        r'.*?([A-D]).*?(?:正确|对|yes|correct)',  # 包含"正确"或"对"的句子中的选项
        r'.*?最终.*?([A-D])',                    # 包含"最终"的句子中的选项
        r'.*?([A-D]).*?(?:选择|选定|确定)',       # 包含"选择"相关词的句子中的选项
    ]
DATASET_INFO = load_dataset_info("/mnt/nvme1n1/yuang_workspace/EvaluatorKit/datasets/dataset_info.json")


class TextMCQDataset(TextBaseDataset):
    def __init__(self, dataset_name):
        #super().__init__(dataset_name)
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
            dataset = load_dataset(path=dataset_path, name=subset_name, split=split_name, trust_remote_code=True)
        question_information = self.dataset_info["question"]
        answer_information = self.dataset_info["answer"]
        choices_information = self.dataset_info["choices"]
        hint_information = self.dataset_info["hint"]
        

        new_dataset = []
        for example in dataset:
            new_example = {
            "question": "",
            "choices": [],
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
            answer_type = answer_information["answer_type"]
            choices_type = choices_information["choices_type"]
            choices_key = choices_information["choices_key"]
            sub_key = choices_information["sub_key"]
            if choices_type == "single":

                if sub_key == "":
                    choices = example[choices_key]
                else:
                    choices = example[choices_key][sub_key]
            elif choices_type == "multiple":
                choices = []
                for choice_key in choices_key:
                    choices.append(example[choice_key])
            choices = [choice for choice in choices if choice != ""]
            if answer_type == "choice":
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
            answer_index = chr(65 + choices.index(answer))
            new_example["question"] = question
            new_example["choices"] = choices
            new_example["answer"] = answer_index
            new_example["hint"] = hint
            new_dataset.append(new_example)
        new_dataset = Dataset.from_list(new_dataset)
        return new_dataset

                


        
    def build_single_prompt(self, line_number: int):
        example = self.dataset[line_number]
        questions = example['question']
        choices = example["choices"]
        choice_prompt = "The choices are: "
        for i, choice in enumerate(choices):
            choice_prompt += f"{chr(65 + i)}. {choice} "
        choice_prompt += "\n"
        prompt = f"Answer the following question based on the choices provided.\nQuestion: {questions}\n{choice_prompt}\n"
        message = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': prompt}
        ]
        return message

    @torch.inference_mode()
    def single_inference(self, line_number: int, tokenizer, model):
        prompt = self.build_single_prompt(line_number)
        print(prompt)
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 获取输入的token长度
        input_length = inputs["input_ids"].shape[1]
        
        generate_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=2048,
            # 添加以下参数
            return_dict_in_generate=True,
            output_scores=False
        )
        
        # 只解码新生成的token
        response = tokenizer.batch_decode(
            generate_ids.sequences[:, input_length:],
            skip_special_tokens=True
        )[0]
        
        return response
    
    def load_tokenizer_and_model(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        return tokenizer, model
    


    def evaluate(self, model_path=None, workers=4):
        if model_path is None:
            raise ValueError("model is required")
        
        tokenizer, model = self.load_tokenizer_and_model(model_path)
        
        total = len(self.dataset)
        correct = 0
        results = []
        
        # 使用tqdm显示进度
        for i in tqdm(range(total), desc="Evaluating"):
            # 获取真实答案
            true_answer = self.dataset[i]["answer"]
            
            # 获取模型预测
            response = self.single_inference(i, tokenizer, model)
            pred_answer = self.extract_answer(response)
            
            # 记录结果
            is_correct = (pred_answer == true_answer)
            correct += int(is_correct)
            
            result = {
                "index": i,
                "question": self.dataset[i]["question"],
                "choices": self.dataset[i]["choices"],
                "true_answer": true_answer,
                "pred_answer": pred_answer,
                "model_response": response,
                "is_correct": is_correct
            }
            results.append(result)
            
            # 每100个样本输出一次当前准确率
            if (i + 1) % 100 == 0:
                current_accuracy = correct / (i + 1)
                print(f"\nCurrent accuracy: {current_accuracy:.4f}")
        
        # 计算最终准确率
        final_accuracy = correct / total
        
        # 保存评测结果
        evaluation_result = {
            "model_name": model_path,
            "total_samples": total,
            "correct_predictions": correct,
            "accuracy": final_accuracy,
            "detailed_results": results
        }
        
        # 创建结果目录
        os.makedirs("evaluation_results", exist_ok=True)
        
        # 保存结果到JSON文件
        result_file = f"evaluation_results/{model_path.replace('/', '_')}_results.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
        
        print(f"\nFinal Accuracy: {final_accuracy:.4f}")
        print(f"Results saved to: {result_file}")
        
        return final_accuracy, results


    def extract_answer(self, response):
            for pattern in PATTERNS:
                match = re.search(pattern, response)
                if match:
                    return match.group(1)
            return None



if __name__ == "__main__":
    dataset = TextMCQDataset("C-Eval")
    tokenizer, model = dataset.load_tokenizer_and_model("Qwen/Qwen2-7B-Instruct")
    for i in range(10):
        #print(dataset.dataset[i]["question"])
        answer = dataset.single_inference(i, tokenizer, model)
        print("Question:", i)
        #print(answer)
        print("Correct Answer:", dataset.dataset[i]["answer"])
        print("Predicted Answer:", dataset.extract_answer(answer))
    #dataset.evaluate(model_path="Qwen/Qwen2-7B-Instruct")
    torch.cuda.empty_cache()
    
    #print(dataset.dataset[0])
    #print(dataset.build_single_prompt(0))