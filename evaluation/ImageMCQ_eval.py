import sys
import os
import json
import time
from pathlib import Path
from openai import BadRequestError, AuthenticationError

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.Image.ImageMCQ import *
from src.api.multimodal_api import *
from src.utils.MCQ_constants import *
from src.utils.model_and_dataset import *
from src.utils.default_prompts import *
from src.utils.model_and_dataset import *
from tqdm import tqdm
import concurrent.futures
from typing import List, Tuple, Dict, Any, Literal
import re
import random
from dotenv import load_dotenv

def extract_answer(response: str, dataset_name: str):
    """
    从模型响应中提取单选题答案
    
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
    从模型响应中提取多选题答案
    
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

def shuffle_and_convert(dataset: ImageMCQ):
    """
    随机打乱选项顺序，并找到打乱后答案的索引
    
    参数:
        dataset: ImageMCQ数据集实例
    
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
            number_index = 0  # 默认值
            if answer != '' and answer_type == 'choice':
                if isinstance(answer, int):
                    number_index = answer
                elif isinstance(answer, str):
                    if 'A' <= answer <= 'Z' and answer.isupper():
                        number_index = ord(answer) - ord('A')
                    elif 'a' <= answer <= 'z' and answer.islower():
                        number_index = ord(answer) - ord('a')
                    elif '0' <= answer <= '9':
                        number_index = int(answer)
                    else:
                        # 如果答案不是标准格式，尝试直接匹配
                        try:
                            number_index = choice_list.index(answer)
                        except ValueError:
                            number_index = 0  # 如果找不到，使用默认值
                else:
                    number_index = 0  # 其他类型使用默认值
            else:
                number_index = 0  # 如果答案为空或类型不匹配，使用默认值
            
            # 确保number_index在有效范围内
            if number_index >= len(choice_list):
                number_index = 0
            
            answer = choice_list[number_index]
            
            # 打乱选项顺序
            random.shuffle(choice_list)
            # 找到打乱后正确答案的新位置
            answer_index = chr(choice_list.index(answer) + 65)
            new_choices.append(choice_list)
            new_answer.append(answer_index)
    
    return new_choices, new_answer

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

def process_image_question(args):
    """
    处理单个图像问题
    
    参数:
        args: 包含处理问题所需参数的元组
    
    返回:
        问题索引、提取的答案和正确答案
    """
    i, dataset_name, image, background, case, question, question_type, choices, answer, hint, language, model_name, model_key, api_base = args
    
    try:
        if model_name == "stressTest":
            extracted_response = "A"
            time.sleep(10)
        else:
            case_prompt = ""
            if case is not None:
                if language == "en":
                    case_prompt = f"Case of the question: {case}"
                elif language == "zh":
                    case_prompt = f"问题背景：{case}"
            question_prompt = case_prompt + question + "\n"

            if choices is not None:
                for choice in choices:
                    question_prompt += f"{chr(65 + choices.index(choice))}. {choice}" + " "
            if hint != "":
                question_prompt += f"\nHint: {hint}"
            
            if background is not None and language == "en":
                system_prompt = system_prompt + f"\nBackground: {background}"
            elif background is not None and language == "zh":
                system_prompt = system_prompt + f"\n任务背景：{background}"

            if language == "en" and question_type == 'single':
                system_prompt = MCQ_TEMPLATE_SINGLE_EN
            elif language == "en" and question_type == "multiple":
                system_prompt = MCQ_TEMPLATE_MULTIPLE_EN
            elif language == "zh" and question_type == "single":
                system_prompt = MCQ_TEMPLATE_SINGLE_ZH
            elif language == "zh" and question_type == "multiple":
                system_prompt = MCQ_TEMPLATE_MULTIPLE_ZH
            api = MultimodalAPI(model_name, system_prompt, question_prompt, image, 0, model_key, api_base)
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


def evaluate_imagemcq_manual(
        user_id: str = "",
        dataset_name: str = "MMStar",
        model_name: str = "gpt-4o",
        business_id: str = "",
        question_limitation: int = 100,
        response_url: str = "",
):
    dataset = ImageMCQ(dataset_name)
    dataset.max_score = dataset.dataset_info['max_score']
    language = dataset.language


    result_file = f"results/{user_id}/{business_id}_manual_result.json"
    score_file = f"results/{user_id}/{business_id}_manual_score.json"

    try:
        response = requests.get(response_url, timeout=60)
        response = response.json()
    except Exception as e:
        print(f"获取响应时出错: {str(e)}")
        return None

    result = []
    
    for i in range(question_limitation):
        result.append({
            "id": i,
            "response": response[i]["response"]
        })

    write_json_file(result, result_file)

    return calculate_valid_ratio_and_score(result_file, dataset.answers, score_file, max_score=dataset.max_score, business_id=business_id)
            
    

def evaluate_imagemcq_automatic(
        user_id: str = "",
        dataset_name: str = "MMStar",
        model_name: str = "gpt-3.5-turbo",
        model_key: str = "",
        api_base: str = "",
        business_id: str = "",
        question_limitation: int = 100,
        max_workers: int = 64
        ):
    """
    并行评估图像问题
    
    参数:
        dataset_name: 数据集名称
        model_name: 模型名称
        max_workers: 最大并行工作线程数
        evaluate_mode: 评估模式，"automatic"自动评估，"manual"手动评估
    
    返回:
        评估结果和准确率
    """
    # 准备文件路径
    result_file = f"results/{user_id}/{business_id}_result.json"
    accuracy_file = f"results/{user_id}/{business_id}_score.json"
    
    # 加载数据集
    dataset = ImageMCQ(dataset_name)
    dataset.choices, dataset.answers = shuffle_and_convert(dataset)
    language = dataset.language
    background = dataset.background

    # 初始化结果
    existing_results = read_json_file(result_file)
    if not existing_results:
        existing_results = [{"id": i, "response": "Neglected"} for i in range(question_limitation)]
        write_json_file(existing_results, result_file)
    args_list = []
    for i in range(question_limitation):
        if existing_results[i]['response'] != "Neglected":
            continue
        question = dataset.questions[i] if dataset.questions is not None else None
        question_type = dataset.question_type_list[i] if dataset.question_type_list is not None else None
        case = dataset.case[i] if dataset.case is not None else None
        image = dataset.image_list[i] if dataset.image_list is not None else None
        choices = dataset.choices[i] if dataset.choices is not None else None
        answer = ""
        hint = ""
        if dataset.answers is not None:
            answer = dataset.answers[i]
        if dataset.hints is not None:
            hint = dataset.hints[i]
        args_list.append((i, dataset_name, image, background, case, question, question_type, choices, answer, hint, language, model_name, model_key, api_base))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_image_question, args): args[0] for args in args_list}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"评测中"):
            idx, response, answer = future.result()
            existing_results[idx]["response"] = response
            write_json_file(existing_results, result_file)
    
    return calculate_valid_ratio_and_score(result_file, dataset.answers, accuracy_file, max_score=dataset.max_score, business_id=business_id)


def calculate_valid_ratio_and_score(result_file, answers, accuracy_file, question_type_list=None, neglected_threshold=0.05, max_score=1, business_id=""):
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
    results = read_json_file(result_file)
    total_questions = len(results)
    neglected_count = sum(1 for result in results if result["response"] == "Neglected")
    neglected_ratio = neglected_count / total_questions if total_questions > 0 else 0
    valid_ratio = 1 - neglected_ratio
    
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
    raw_score = correct_count / valid_count * 100 if valid_count > 0 else 0
    score = raw_score / max_score * 100
    
    # 写入准确率文件
    accuracy_data = {
        "business_id": business_id,
        "raw_score": raw_score,
        "score": score,
        "valid_ratio": valid_ratio
    }
    write_json_file(accuracy_data, accuracy_file)
    
    return valid_ratio, score
            
if __name__ == "__main__":
    load_dotenv()
    # 从头开始评测
    # responses, accuracy = evaluate_imagemcq("MMStar", "Pro/Qwen/Qwen2.5-VL-7B-Instruct", evaluate_mode="start_from_beginning")
    
    # 从断点处继续评测
    responses, accuracy = evaluate_imagemcq_automatic("MMStar", "Pro/Qwen/Qwen2-VL-7B-Instruct", evaluate_mode="give_answers")
