import sys
import os
import json
from pathlib import Path
from openai import BadRequestError, AuthenticationError

project_root = Path(__file__).resolve().parent.parent
print(project_root)
sys.path.append(str(project_root))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.Text.TextMCQ import *
from src.api.text_api import *
from src.utils.MCQ_constants import *
from tqdm import tqdm
import concurrent.futures
from typing import List, Tuple, Dict, Any, Literal
from dotenv import load_dotenv
import re

def extract_answer(response: str, dataset_name: str):
    """
    提取单选题答案
    
    参数:
        response: 模型的响应文本
        dataset_name: 数据集名称，用于确定答案格式
        
    返回:
        提取的答案选项（如A、B、C、D），如果未找到则返回None
    """
    if response == "Neglected":
        return response
    max_letter, PATTERNS = build_patterns(dataset_name)
    for pattern in PATTERNS:
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    return None

def extract_multi_answer(response: str, dataset_name: str) -> List[str]:
    """
    提取多选题答案
    
    参数:
        response: 模型的响应文本
        dataset_name: 数据集名称，用于确定答案格式
        
    返回:
        提取的答案选项列表（如['A', 'B', 'C']），如果未找到则返回None
    """
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

def shuffle_and_convert(dataset: TextMCQ):
    """
    随机打乱选项顺序，并找到打乱后答案的索引
    
    参数:
        dataset: TextMCQ数据集实例
        
    返回:
        打乱后的选项列表和对应的答案列表
    """
    answers = dataset.answers
    answer_type = dataset.answer_type
    choices = dataset.choices
    question_type_list = dataset.question_type_list
    
    # 如果没有选项，直接返回
    if choices is None:
        return None, answers
    
    # 如果有选项但没有答案，只打乱选项
    if choices is not None and answers is None:
        new_choices = []
        for choice_list in choices:
            random.shuffle(choice_list)
            new_choices.append(choice_list)
        return new_choices, None
    
    # 如果既有选项又有答案，打乱选项并更新答案
    new_choices = []
    new_answer = []

    for choice_list, answer, question_type in zip(choices, answers, question_type_list):
        # 多选题不打乱选项顺序
        if question_type == "multiple":
            new_choices.append(choice_list)
            new_answer.append(answer)
            continue
        else:
            # 单选题处理：先找到正确答案对应的选项内容
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
                    
            # 打乱选项顺序
            random.shuffle(choice_list)
            # 找到打乱后正确答案的新位置
            answer_index = chr(choice_list.index(answer) + 65)
            new_choices.append(choice_list)
            new_answer.append(answer_index)
    
    return new_choices, new_answer

def process_question(args):
    """
    处理单个文本问题
    
    参数:
        args: 包含问题信息的元组，包括索引、数据集名称、问题、问题类型、选项、答案、提示、语言和模型名称
        
    返回:
        问题索引、模型回答和正确答案的元组
    """
    i, dataset_name, question, question_type, choices, answer, hint, language, model_name = args
    
    # 统一问题类型格式
    if question_type in SINGLE_CHOICE_LIST:
        question_type = "single"
    elif question_type in MULTIPLE_CHOICE_LIST:
        question_type = "multiple"
        
    # 构建问题提示
    question_prompt = question + "\n"
    for choice in choices:
        question_prompt += f"{chr(65 + choices.index(choice))}. {choice}" + " "
    if hint != "":
        question_prompt += f"\nHint: {hint}"
        
    # 选择适当的提示模板
    if language == 'en' and question_type == 'single':
        system_prompt = MCQ_TEMPLATE_SINGLE_EN
    elif language == "en" and question_type == "multiple":
        system_prompt = MCQ_TEMPLATE_MULTIPLE_EN
    elif language == "zh" and question_type == "single":
        system_prompt = MCQ_TEMPLATE_SINGLE_ZH
    elif language == "zh" and question_type == "multiple":
        system_prompt = MCQ_TEMPLATE_MULTIPLE_ZH

    # 获取回答
    try:
        chat = TextAPI(model_name, system_prompt, question_prompt, 0)
        response = chat.generate_response()
        extracted_response = None
        if question_type == "single":
            extracted_response = extract_answer(response, dataset_name)
        elif question_type == "multiple":
            extracted_response = extract_multi_answer(response, dataset_name)
    except Exception as e:
        print(f"处理问题 {i} 时出错: {str(e)}")
        extracted_response = "Neglected"
        
    return i, extracted_response, answer

def write_json_file(data, file_path):
    """
    将数据写入JSON文件
    
    参数:
        data: 要写入的数据
        file_path: 文件路径
        
    返回:
        写入是否成功
    """
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
    """
    从JSON文件读取数据
    
    参数:
        file_path: 文件路径
        
    返回:
        读取的数据，如果文件不存在或读取失败则返回None
    """
    try:
        if not os.path.exists(file_path):
            return None
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"读取JSON文件时出错: {str(e)}")
        return None

def evaluate_mcq(dataset_name: str, model_name: str, max_workers=64, 
                 evaluate_mode: Literal["start_from_beginning", "resume_from_checkpoint"] = "start_from_beginning"):
    """
    并行评估文本问题
    
    参数:
        dataset_name: 数据集名称
        model_name: 模型名称
        max_workers: 最大并行工作线程数
        evaluate_mode: 评估模式，"start_from_beginning"从头开始，"resume_from_checkpoint"从断点继续
        
    返回:
        评估结果和准确率
    """
    # 准备文件路径
    result_file = f"results/{dataset_name}_{model_name}_result.json"
    accuracy_file = f"results/{dataset_name}_{model_name}_result_accuracy.json"
    
    # 加载数据集
    dataset = TextMCQ(dataset_name)
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
        choices = dataset.choices[i]
        answer = ""
        hint = ""
        if dataset.answers is not None:
            answer = dataset.answers[i]
        if dataset.hints is not None:
            hint = dataset.hints[i]
        
        args_list.append((i, dataset_name, question, question_type, choices, answer, hint, language, model_name))
    
    # 如果没有需要处理的问题，直接计算准确率
    if not args_list:
        print("没有需要处理的问题，直接计算准确率")
        return calculate_accuracy(results, dataset.answers, accuracy_file)
    
    # 并行执行
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_question, args): args[0] for args in args_list}
        
        # 使用tqdm显示进度
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="处理文本问题"):
            idx, response, answer = future.result()
            
            # 更新结果
            results[idx]["response"] = response
            
            # 每完成一题就更新结果文件
            write_json_file(results, result_file)
            
            #print(f"问题 {idx+1}: 回答: {response}")
    
    # 计算准确率并写入文件
    return calculate_accuracy(results, dataset.answers, accuracy_file)

def calculate_accuracy(results, answers, accuracy_file, question_type_list=None, neglected_threshold=0.05):
    """
    计算准确率并写入文件
    
    参数:
        results: 模型回答结果列表
        answers: 标准答案列表
        accuracy_file: 准确率结果文件路径
        question_type_list: 问题类型列表，用于区分单选和多选题
        neglected_threshold: Neglected题目的最大比例阈值，超过此阈值则不计算准确率
        
    返回:
        结果列表和准确率
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
    # 评估MMLU数据集
    results, accuracy = evaluate_mcq("MMLU", "gpt-3.5-turbo", evaluate_mode="resume_from_checkpoint")
    print(accuracy)
            
            