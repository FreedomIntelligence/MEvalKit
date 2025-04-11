import sys
import os
from pathlib import Path
from openai import BadRequestError, AuthenticationError

project_root = Path(__file__).resolve().parent.parent
print(project_root)
sys.path.append(str(project_root))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.Image.ImageMCQ import *
from src.api.multimodal_api import *
from src.utils.MCQ_constants import *
from tqdm import tqdm
import concurrent.futures
from typing import List, Tuple
import re
import random

def extract_answer(response: str, dataset_name: str):
    max_letter, PATTERNS = build_patterns(dataset_name)
    for pattern in PATTERNS:
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    return None

def extract_multi_answer(response: str, dataset_name: str) -> List[str]:
    """提取多选题答案"""
    max_letter, PATTERNS_MULTI = build_patterns_multi(dataset_name)
    # 预处理：移除多余空格，统一逗号格式
    response = response.strip().replace('，', ',')
    
    # 尝试所有模式匹配
    for pattern in PATTERNS_MULTI:
        matches = re.findall(pattern, response)
        if matches:
            # 提取所有选项并去重
            answers = []
            for match in matches:
                # 提取A-D的字母
                options = re.findall(f'[A-{max_letter}]', match)
                answers.extend(options)
            
            # 去重并排序
            answers = sorted(list(set(answers)))
            return answers
    
    # 如果没有匹配到完整格式，尝试提取单个选项
    single_options = re.findall(f'[A-{max_letter}]', response)
    if single_options:
        return sorted(list(set(single_options)))
    
    return None

def shuffle_and_convert(dataset: ImageMCQ):
    answers = dataset.answers
    answer_type = dataset.answer_type
    choices = dataset.choices
    if choices is None:
        return None, answers
    if choices is not None and answers is None:
        new_choices = []
        for choice_list in choices:
            random.shuffle(choice_list)
            new_choices.append(choice_list)
        return new_choices, None
    new_choices = []
    new_answer = []

    for choice_list, answer in zip(choices, answers):
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
            answer = choice_list[number_index]
                
        random.shuffle(choice_list)
        answer_index = chr(choice_list.index(answer) + 65)
        new_choices.append(choice_list)
        new_answer.append(answer_index)
    return new_choices, new_answer

def process_image_question(args):
    """处理单个图像问题"""
    i, question, dataset_name, image, choices, answer, hint, language, model_name, assistant_prompt = args
    
    try:
        question_prompt = question + "\n"
        if choices is not None:
            for choice in choices:
                question_prompt += f"{chr(65 + choices.index(choice))}. {choice}" + " "
        if hint != "":
            question_prompt += f"\nHint: {hint}"
        
        api = MultimodalAPI(model_name, assistant_prompt, question_prompt)
        response = api.generate_response(image)
        response = extract_answer(response, dataset_name)
        return i, response, (answer == response) if answer != "" else None
    except Exception as e:
        print(f"处理问题 {i} 时出错: {str(e)}")
        return i, f"Error: {str(e)}", False

def evaluate(dataset_name: str, model_name: str, max_workers=64):
    """并行评估图像问题"""
    dataset = ImageMCQ(dataset_name)
    dataset.choices, dataset.answers = shuffle_and_convert(dataset)
    language = dataset.language
    
    response_list = []
    correct_count = 0
    total_count = 0
    
    # 准备参数列表
    args_list = []
    for i in range(len(dataset.questions)):
        question = dataset.questions[i]
        question_type = dataset.question_type_list[i]
        image = dataset.image_list[i]
        choices = None if dataset.choices is None else dataset.choices[i]
        answer = "" if dataset.answers is None else dataset.answers[i]
        hint = "" if dataset.hints is None else dataset.hints[i]
        
        # 选择适当的提示模板
        if language == 'en' and question_type == 'single':
            assistant_prompt = MCQ_TEMPLATE_SINGLE_EN
        elif language == "en" and question_type == "multiple":
            assistant_prompt = MCQ_TEMPLATE_MULTIPLE_EN
        elif language == "zh" and question_type == "single":
            assistant_prompt = MCQ_TEMPLATE_SINGLE_ZH
        elif language == "zh" and question_type == "multiple":
            assistant_prompt = MCQ_TEMPLATE_MULTIPLE_ZH
        
        args_list.append((i, question, dataset_name, image, choices, answer, hint, language, model_name, assistant_prompt))
    
    # 并行执行
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_image_question, args): args[0] for args in args_list}
        
        # 使用tqdm显示进度
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="处理图像问题"):
            idx, response, is_correct = future.result()
            response_list.append(response)
            
            print(f"问题 {idx+1}: {response}")
            
            if is_correct is not None:
                total_count += 1
                if is_correct:
                    correct_count += 1
    
    # 计算准确率
    accuracy = correct_count / total_count if total_count > 0 else None
    if accuracy is not None:
        print(f"准确率: {accuracy:.4f}")
    
    return response_list, accuracy
            
if __name__ == "__main__":
    evaluate("MMStar", "Pro/Qwen/Qwen2.5-VL-7B-Instruct")
