import sys
import os
import json
from pathlib import Path
from openai import BadRequestError, AuthenticationError

project_root = Path(__file__).resolve().parent.parent
print(project_root)
sys.path.append(str(project_root))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.Image.ImageMCQ import *
from src.api.multimodal_api import *
from src.utils.MCQ_constants import *
from src.utils.map_constants import *
from tqdm import tqdm
import concurrent.futures
from typing import List, Tuple, Dict, Any, Literal
import re
import random
from dotenv import load_dotenv

def extract_answer(response: str, dataset_name: str):
    if response == "Neglected":
        return response
    max_letter, PATTERNS = build_patterns(dataset_name)
    for pattern in PATTERNS:
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    return None

def extract_multi_answer(response: str, dataset_name: str) -> List[str]:
    if response == "Neglected":
        return response
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
    question_type_list = dataset.question_type_list
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

    for choice_list, answer, question_type in zip(choices, answers, question_type_list):
        if question_type == "multiple":
            new_choices.append(choice_list)
            new_answer.append(answer)
            continue
        else:
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

def write_json_file(data, file_path):
    """将数据写入JSON文件"""
    try:
        # 确保目录存在
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        #print(f"数据已成功写入: {file_path}")
        return True
    except Exception as e:
        print(f"写入JSON文件时出错: {str(e)}")
        return False

def read_json_file(file_path):
    """从JSON文件读取数据"""
    try:
        if not os.path.exists(file_path):
            return None
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"读取JSON文件时出错: {str(e)}")
        return None

def process_image_question(args):
    """处理单个图像问题"""
    i, question, dataset_name, image, choices, answer, hint, language, model_name, assistant_prompt, question_type = args
    
    try:
        question_prompt = question + "\n"
        if choices is not None:
            for choice in choices:
                question_prompt += f"{chr(65 + choices.index(choice))}. {choice}" + " "
        if hint != "":
            question_prompt += f"\nHint: {hint}"
            
        api = MultimodalAPI(model_name, assistant_prompt, question_prompt, image, 0)
        response = api.generate_response()
        
        extracted_response = None
        if question_type in SINGLE_CHOICE_LIST:
            extracted_response = extract_answer(response, dataset_name)
        elif question_type in MULTIPLE_CHOICE_LIST:
            extracted_response = extract_multi_answer(response, dataset_name)
            
        return i, extracted_response, answer
    except Exception as e:
        print(f"处理问题 {i} 时出错: {str(e)}")
        return i, "Neglected", answer

def evaluate_imagemcq(dataset_name: str, model_name: str, max_workers=64,
                     evaluate_mode: Literal["start_from_beginning", "resume_from_checkpoint"] = "start_from_beginning"):
    """并行评估图像问题"""
    # 准备文件路径
    result_file = f"{dataset_name}_{model_name}_result.json"
    accuracy_file = f"{dataset_name}_{model_name}_result_accuracy.json"
    
    model = model_map[model_name]
    # 加载数据集
    dataset = ImageMCQ(dataset_name)
    dataset.choices, dataset.answers = shuffle_and_convert(dataset)
    language = dataset.language
    
    # 初始化或加载结果
    results = []
    
    if evaluate_mode == "start_from_beginning" or not os.path.exists(result_file):
        # 从头开始评测：初始化所有题目的结果为"Neglected"
        results = [{"id": i, "response": "Neglected"} for i in range(len(dataset.questions))]
        # 写入初始结果文件
        write_json_file(results, result_file)
    else:
        # 从断点处评测：加载现有结果
        existing_results = read_json_file(result_file)
        if existing_results:
            results = existing_results
        else:
            # 如果文件存在但无法读取，则从头开始
            results = [{"id": i, "response": "Neglected"} for i in range(len(dataset.questions))]
            write_json_file(results, result_file)
    
    # 准备参数列表
    args_list = []
    for i in range(len(dataset.questions)):
        # 检查是否需要处理此题
        if evaluate_mode == "resume_from_checkpoint" and results[i]["response"] != "Neglected":
            print(f"跳过已完成的问题 {i+1}")
            continue
            
        question = dataset.questions[i]
        question_type = dataset.question_type_list[i]
        image = dataset.image_list[i]
        choices = None if dataset.choices is None else dataset.choices[i]
        answer = "" if dataset.answers is None else dataset.answers[i]
        hint = "" if dataset.hints is None else dataset.hints[i]
        
        # 选择适当的提示模板
        if language == 'en' and question_type in SINGLE_CHOICE_LIST:
            assistant_prompt = MCQ_TEMPLATE_SINGLE_EN
        elif language == "en" and question_type in MULTIPLE_CHOICE_LIST:
            assistant_prompt = MCQ_TEMPLATE_MULTIPLE_EN
        elif language == "zh" and question_type in SINGLE_CHOICE_LIST:
            assistant_prompt = MCQ_TEMPLATE_SINGLE_ZH
        elif language == "zh" and question_type in MULTIPLE_CHOICE_LIST:
            assistant_prompt = MCQ_TEMPLATE_MULTIPLE_ZH
        
        args_list.append((i, question, dataset_name, image, choices, answer, hint, language, model, assistant_prompt, question_type))
    
    # 如果没有需要处理的问题，直接计算准确率
    if not args_list:
        print("没有需要处理的问题，直接计算准确率")
        return calculate_accuracy(results, dataset.answers, accuracy_file, dataset.question_type_list)
    
    # 并行执行
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_image_question, args): args[0] for args in args_list}
        
        # 使用tqdm显示进度
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="处理图像问题"):
            idx, response, answer = future.result()
            #print(response)
            # 更新结果
            results[idx]["response"] = response
            
            # 每完成一题就更新结果文件
            write_json_file(results, result_file)
            
            #print(f"问题 {idx+1}: 回答: {response}")
    
    # 计算准确率并写入文件
    return calculate_accuracy(results, dataset.answers, accuracy_file, dataset.question_type_list)

def calculate_accuracy(results, answers, accuracy_file, question_type_list=None, neglected_threshold=0.05):
    """
    计算准确率并写入文件
    
    参数:
        results: 模型回答结果列表
        answers: 标准答案列表
        accuracy_file: 准确率结果文件路径
        question_type_list: 问题类型列表，用于区分单选和多选题
        neglected_threshold: Neglected题目的最大比例阈值，超过此阈值则不计算准确率
    """
    # 计算Neglected题目的比例
    total_questions = len(results)
    neglected_count = sum(1 for result in results if result["response"] == "Neglected")
    neglected_ratio = neglected_count / total_questions if total_questions > 0 else 0
    
    # 检查Neglected题目比例是否超过阈值
    if neglected_ratio > neglected_threshold:
        print(f"Neglected题目比例 ({neglected_ratio:.2%}) 超过阈值 ({neglected_threshold:.2%})，暂不计算准确率")
        return results, None
    
    # 如果没有答案，无法计算准确率
    if answers is None:
        print("数据集没有标准答案，无法计算准确率")
        write_json_file({"accuracy": None, "message": "数据集没有标准答案"}, accuracy_file)
        return results, None
    
    # 计算准确率（排除Neglected题目）
    correct_count = 0
    valid_count = 0
    
    for i, result in enumerate(results):
        if i < len(answers) and result["response"] is not None and result["response"] != "Neglected":
            valid_count += 1
            
            # 获取模型回答和标准答案
            model_response = result["response"]
            correct_answer = answers[i]
            
            # 判断是否为多选题
            is_multiple_choice = False
            if question_type_list and i < len(question_type_list):
                question_type = question_type_list[i]
                is_multiple_choice = question_type in MULTIPLE_CHOICE_LIST
            
            # 多选题判断逻辑
            if is_multiple_choice:
                # 确保两者都是列表类型
                if isinstance(model_response, list) and isinstance(correct_answer, list):
                    # 排序后比较，要求完全一致
                    if sorted(model_response) == sorted(correct_answer):
                        correct_count += 1
                # 如果模型回答是字符串（如"ABC"），将其转换为列表再比较
                elif isinstance(model_response, str) and isinstance(correct_answer, list):
                    model_answers = [c for c in model_response if 'A' <= c <= 'Z']
                    if sorted(model_answers) == sorted(correct_answer):
                        correct_count += 1
                # 如果标准答案是字符串（如"ABC"），将其转换为列表再比较
                elif isinstance(model_response, list) and isinstance(correct_answer, str):
                    correct_answers = [c for c in correct_answer if 'A' <= c <= 'Z']
                    if sorted(model_response) == sorted(correct_answers):
                        correct_count += 1
                # 如果两者都是字符串，直接比较字符集合
                elif isinstance(model_response, str) and isinstance(correct_answer, str):
                    model_answers = set([c for c in model_response if 'A' <= c <= 'Z'])
                    correct_answers = set([c for c in correct_answer if 'A' <= c <= 'Z'])
                    if model_answers == correct_answers:
                        correct_count += 1
            # 单选题判断逻辑
            else:
                if model_response == correct_answer:
                    correct_count += 1
    
    # 计算准确率（基于有效题目数量）
    accuracy = correct_count / valid_count if valid_count > 0 else 0
    print(f"准确率: {accuracy:.4f} ({correct_count}/{valid_count})")
    print(f"有效题目: {valid_count}/{total_questions} (Neglected: {neglected_count}题, {neglected_ratio:.2%})")
    
    # 写入准确率文件
    accuracy_data = {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "valid_count": valid_count,
        "total_count": total_questions,
        "neglected_count": neglected_count,
        "neglected_ratio": neglected_ratio
    }
    write_json_file(accuracy_data, accuracy_file)
    
    return results, accuracy
            
if __name__ == "__main__":
    load_dotenv()
    # 从头开始评测
    # responses, accuracy = evaluate_imagemcq("MMStar", "Pro/Qwen/Qwen2.5-VL-7B-Instruct", evaluate_mode="start_from_beginning")
    
    # 从断点处继续评测
    responses, accuracy = evaluate_imagemcq("MMStar", "Pro/Qwen/Qwen2-VL-7B-Instruct", evaluate_mode="resume_from_checkpoint")
