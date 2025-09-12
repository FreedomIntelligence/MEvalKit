import sys
import os
import json
import requests
from pathlib import Path
from openai import BadRequestError, AuthenticationError
from typing import Optional
from jinja2 import Template, Environment, FileSystemLoader

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.MCQ import *
from src.api.ConversationAPI import *
from src.utils.MCQ_constants import *
from src.utils.default_prompts import *
from src.utils.model_and_dataset import *
from tqdm import tqdm
import concurrent.futures
from typing import List, Tuple, Dict, Any, Literal
from dotenv import load_dotenv
import re
from datetime import datetime

# 添加MySQL数据库支持
from src.database.mysql_db import (
    save_evaluation_result, 
    load_evaluation_result,
    save_evaluation_score, 
    load_evaluation_score,
    initialize_database
)


def sanitize_filename(filename):
    """
    将文件名中的所有不安全字符替换为下划线
    """
    return re.sub(r'[\\/:*?"<>|]', '_', filename).strip(' .') or 'unknown_model'

def generate_business_id(dataset, model_name):
    """
    生成business_id：{dataset}_{safe_model}_{当前时间}
    """
    current_time = datetime.now().strftime("%Y%m%d%H%M")
    safe_model_name = sanitize_filename(model_name)
    return f"{dataset}_{safe_model_name}_{current_time}"

def extract_answer(response: str, dataset_name: str):
    """
    提取单选题答案
    
    参数:
        response: 模型的响应文本
        dataset_name: 数据集名称，用于确定答案格式
        
    返回:
        提取的答案选项（如A、B、C、D），如果未找到则返回"Neglected"
    """
    if response == "Neglected":
        return response
    max_letter, PATTERNS = build_patterns(dataset_name)
    for pattern in PATTERNS:
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    return "Neglected"

def extract_multi_answer(response: str, dataset_name: str):
    """
    提取多选题答案
    
    参数:
        response: 模型的响应文本
        dataset_name: 数据集名称，用于确定答案格式
        
    返回:
        提取的答案选项列表（如['A', 'B', 'C']），如果未找到则返回"Neglected"
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
    
    return "Neglected"

def extract_json_answer(response: str, question_type: str):
    """
    从JSON格式的响应中提取答案
    
    参数:
        response: 模型的响应文本
        question_type: 问题类型（"single"或"multiple"）
        
    返回:
        extracted_answer: 提取的答案
    """
    if response == "Neglected":
        return response
    
    # 首先尝试最严格的JSON解析
    json_candidates = []
    
    # 方法1: 提取```json代码块
    json_match = re.search(r'```json\s*\n(.*?)\n```', response, re.DOTALL)
    if json_match:
        json_candidates.append(json_match.group(1))
    
    # 方法2: 提取{}对象（寻找最大的完整JSON对象）
    brace_matches = []
    start_pos = 0
    while True:
        start = response.find('{', start_pos)
        if start == -1:
            break
        
        brace_count = 0
        end = start
        in_string = False
        escaped = False
        
        for i in range(start, len(response)):
            char = response[i]
            if escaped:
                escaped = False
                continue
            if char == '\\' and in_string:
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i
                        break
        
        if brace_count == 0:
            candidate = response[start:end+1]
            if len(candidate) > 20:  # 过滤太短的候选
                brace_matches.append(candidate)
        
        start_pos = start + 1
    
    # 按长度排序，优先尝试最长的
    brace_matches.sort(key=len, reverse=True)
    json_candidates.extend(brace_matches)
    
    if not json_candidates:
        # 如果没有找到JSON候选，直接尝试简单提取
        simple_answer = extract_simple_answer(response, question_type)
        return simple_answer
    
    # 尝试解析每个JSON候选
    for json_str in json_candidates:
        try:
            # 清理JSON字符串
            cleaned_json = clean_json_string(json_str)
            
            # 尝试解析
            data = json.loads(cleaned_json)
            
            answer = data.get("answer")
            
            if answer is None:
                continue
            
            # 验证答案格式
            if question_type == "single":
                if isinstance(answer, str) and len(answer) == 1 and 'A' <= answer <= 'Z':
                    return answer
            elif question_type == "multiple":
                if isinstance(answer, list) and all(isinstance(opt, str) and len(opt) == 1 and 'A' <= opt <= 'Z' for opt in answer):
                    return sorted(answer)
                    
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            # 如果这个候选解析失败，尝试下一个
            print(f"JSON候选解析失败: {e}")
            continue
    
    # 如果所有JSON候选都解析失败，尝试简单提取
    print("所有JSON候选都解析失败，尝试简单提取")
    simple_answer = extract_simple_answer(response, question_type)
    return simple_answer

def clean_json_string(json_str: str) -> str:
    """
    清理JSON字符串，修复常见的格式问题
    """
    # 移除Unicode BOM和零宽字符
    json_str = json_str.replace('\ufeff', '').replace('\u200b', '').replace('\u00a0', ' ')
    
    # 处理常见的Unicode字符
    unicode_replacements = {
        '\u27a5': '->',
        '\u2192': '->',
        '\u2190': '<-',
        '\u2194': '<->',
        '\u00b0': 'degrees',
        '\u00b1': '±',
        '\u00d7': 'x',
        '\u00f7': '/',
        '\u2013': '-',
        '\u2014': '--',
        '\u2026': '...',
        '\u2605': '*',  # ⭐
        '\u25b6': '>',  # ▶
        '\u25c0': '<',  # ◀
        '\u2022': '*',  # •
        '\u25cf': '*',  # ●
    }
    
    for unicode_char, replacement in unicode_replacements.items():
        json_str = json_str.replace(unicode_char, replacement)
    
    # 移除或替换损坏的字符序列
    # 移除所有非ASCII可打印字符（除了换行、回车、制表符）
    json_str = re.sub(r'[^\x20-\x7E\n\r\t]+', ' ', json_str)
    
    # 修复常见的LaTeX转义问题
    json_str = json_str.replace('\\\\', '\\')
    
    # 修复JSON中的控制字符问题
    # 替换JSON字符串中的未转义控制字符
    json_str = re.sub(r'[\x00-\x1F\x7F]', ' ', json_str)
    
    # 尝试修复常见的JSON语法错误
    # 修复可能的换行符问题
    json_str = json_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    
    return json_str

def extract_simple_answer(response: str, question_type: str) -> str:
    """
    当JSON解析失败时，尝试多种方式提取答案
    """
    # 方法1: 尝试在JSON块中找到answer字段
    answer_patterns = [
        r'"answer":\s*"([A-Z])"',
        r'"answer":\s*([A-Z])',
        r'"answer"\s*:\s*"([A-Z])"',
        r'answer["\']?\s*:\s*["\']?([A-Z])["\']?',
    ]
    
    for pattern in answer_patterns:
        answer_match = re.search(pattern, response, re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).upper()
            if question_type == "single":
                if len(answer) == 1 and 'A' <= answer <= 'Z':
                    return answer
    
    # 方法2: 对于多选题，尝试找到列表格式
    if question_type == "multiple":
        list_patterns = [
            r'"answer":\s*\[([^\]]+)\]',
            r'"answer"\s*:\s*\[([^\]]+)\]',
            r'answer["\']?\s*:\s*\[([^\]]+)\]',
        ]
        
        for pattern in list_patterns:
            list_match = re.search(pattern, response, re.IGNORECASE)
            if list_match:
                try:
                    list_content = list_match.group(1)
                    # 提取所有大写字母
                    options = re.findall(r'["\']?([A-Z])["\']?', list_content)
                    if options and all(len(opt) == 1 and 'A' <= opt <= 'Z' for opt in options):
                        return sorted(list(set(options)))
                except:
                    continue
    
    # 方法3: 寻找常见的答案模式（fallback）
    # 寻找类似 "The answer is X" 的模式
    fallback_patterns = [
        r'(?:answer|选择|选项|correct)\s*(?:is|为|是)\s*(?:clearly\s*)?([A-Z])',
        r'(?:therefore|因此|所以).{0,50}([A-Z])\s*[。.]',
        r'选项\s*([A-Z])',
        r'答案\s*([A-Z])',
        r'\b([A-Z])\s*(?:is|为|是)\s*(?:correct|正确)',
        r'(?:based on|根据).{0,30}(?:answer|答案).{0,10}([A-Z])',
    ]
    
    for pattern in fallback_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            answer = match.group(1).upper()
            if len(answer) == 1 and 'A' <= answer <= 'Z':
                if question_type == "single":
                    return answer
                elif question_type == "multiple":
                    return [answer]
    
    return "Neglected"

def shuffle_and_convert(dataset: MCQ, shuffle: bool = True, seed: str = None):
    """
    随机打乱选项顺序，并找到打乱后答案的索引
    
    参数:
        dataset: TextMCQ数据集实例
        
    返回:
        打乱后的选项列表和对应的答案列表
    """
    answer = dataset.answer
    answer_type = dataset.answer_type
    choice = dataset.choice
    question_type_list = dataset.question_type_list
    # 如果没有选项，直接返回
    if choice is None:
        return None, answer
    

    
    # 如果有选项但没有答案，只打乱选项
    if choice is not None and answer is None:
        new_choices = []
        for i, choice_list in enumerate(choice):
            if seed is not None:
                # 使用种子和题目索引生成一个特定的随机状态
                temp_random = random.Random(f"{seed}_{i}")
                temp_random.shuffle(choice_list)
            else:
                random.shuffle(choice_list)
            new_choices.append(choice_list)
        return new_choices, None
    
    # 如果既有选项又有答案，打乱选项并更新答案
    new_choices = []
    new_answer = []



    for i, (choice_list, answer, question_type) in enumerate(zip(choice, answer, question_type_list)):

        if isinstance(choice_list, dict):
            choice_list = [value for value in choice_list.values()]
        #print("choice_list", choice_list)
        #print("answer", answer)
        if question_type in SINGLE_CHOICE_LIST:
            question_type = "single"
        elif question_type in MULTIPLE_CHOICE_LIST:
            question_type = "multiple"
        
        # 多选题不打乱选项顺序
        if question_type == "multiple":
            new_choices.append(choice_list)
            new_answer.append(answer)
            continue
        elif isinstance(choice_list, str):
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
            if shuffle:
                # 如果提供了种子，使用种子确保一致性
                if seed is not None:
                    # 使用种子和题目索引生成一个特定的随机状态
                    temp_random = random.Random(f"{seed}_{i}")
                    temp_random.shuffle(choice_list)
                else:
                    random.shuffle(choice_list)
            # 找到打乱后正确答案的新位置
            answer_index = chr(choice_list.index(answer) + 65)
            new_choices.append(choice_list)
            new_answer.append(answer_index)
    
    return new_choices, new_answer

def write_json_file(data, file_path, business_id=None, dataset_name=None, model_name=None):
    """写入JSON文件并同时保存到数据库"""
    try:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        # 同时保存到数据库
        if business_id and dataset_name and model_name:
            save_evaluation_result(business_id, dataset_name, model_name, data)
        
        return True
    except Exception as e:
        print(f"写入JSON文件时出错: {str(e)}")
        return False

def read_json_file(file_path, business_id=None):
    """读取JSON文件，如果文件不存在则尝试从数据库加载"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif business_id:
            # 如果文件不存在，尝试从数据库加载
            return load_evaluation_result(business_id)
        return None
    except Exception as e:
        print(f"读取JSON文件时出错: {str(e)}")
        # 如果文件读取失败，尝试从数据库加载
        if business_id:
            return load_evaluation_result(business_id)
        return None


def process_question(args):
    i, input_data, language, background, dataset_name, model_name, model_key, api_base = args
    case = input_data['case']
    question = input_data['question']
    question_type = input_data['question_type']
    choices = input_data['choices']
    answer = input_data['answer']
    #print("1:", answer)
    hint = input_data['hint']
    image = input_data['image']


    system_prompt = ""
    question_prompt = ""
    
    if question_type in SINGLE_CHOICE_LIST:
        question_type = "single"
    elif question_type in MULTIPLE_CHOICE_LIST:
        question_type = "multiple"

    if background is None or background == "":
        # 使用JSON格式的prompt
        if language == "zh" and question_type == "single":
            system_prompt = MCQ_JSON_TEMPLATE_SINGLE_ZH
        elif language == "zh" and question_type == "multiple":
            system_prompt = MCQ_JSON_TEMPLATE_MULTIPLE_ZH
        elif language == "en" and question_type == "single":
            system_prompt = MCQ_JSON_TEMPLATE_SINGLE_EN
        elif language == "en" and question_type == "multiple":
            system_prompt = MCQ_JSON_TEMPLATE_MULTIPLE_EN
    else:
        system_prompt = background

    if case is not None and case != "":
        question_prompt += f"Case: {case}"
        question_prompt += "\n"
    
    if question is not None and question != "":
        question_prompt += f"Question: {question}"
        question_prompt += "\n"

    if choices is not None and choices != "":
        for choice in choices:
            question_prompt += f"{chr(65 + choices.index(choice))}. {choice}" + " "
        question_prompt += "\n"

    if hint is not None and hint != "":
        question_prompt += f"Hint: {hint}"

    
    chat = ConversationAPI(model_name, 
                                    system_prompt, 
                                    question_prompt, 
                                    image, 
                                    temperature=0, 
                                    conversation_id=None, 
                                    model_key=model_key, 
                                    api_base=api_base, 
                                    enable_history=False)
    response = chat.generate_response()
    #print(response)
    
    # 使用JSON格式的提取逻辑
    extracted_response = extract_json_answer(response, question_type)
    #print(extracted_response)
    
    return i, extracted_response, response, answer

    


                
            


        


def evaluate_all_mcq_automatic(
    user_id: str = "",
    dataset_name: str = "MMLU",
    model_name: str = "gpt-3.5-turbo",
    model_key: str = "",
    api_base: str = "",
    question_limitation: int = 100,
    max_workers: int = 64,
    business_id: str = None
):
    dataset = MCQ(dataset_name)
    background = dataset.background
    language = dataset.language
    max_score = dataset.max_score
    cases = dataset.case
    questions = dataset.question
    question_type_list = dataset.question_type_list
    hints = dataset.hint
    images = dataset.image
    if question_limitation >= len(questions):
        question_limitation = len(questions)
    
    if business_id is None:
        business_id = generate_business_id(dataset_name, model_name)
        result_file = f"results/{user_id}/{business_id}_result.json"
    else:
        # 检查是否存在指定business_id的结果文件
        import glob
        pattern = f"results/{user_id}/*{business_id}_result.json"
        matching_files = glob.glob(pattern)
        if not matching_files:
            raise FileNotFoundError(f"找不到business_id为{business_id}的结果文件")
        # 如果找到多个匹配文件，使用第一个
        if len(matching_files) > 1:
            print(f"找到多个匹配文件，使用第一个: {matching_files[0]}")
        # 使用实际找到的文件路径
        result_file = matching_files[0]
    
    # 确定business_id后再进行打乱
    choices, answers = shuffle_and_convert(dataset, shuffle=True, seed=business_id)
    #print(answers)
    
    # 初始化数据库
    initialize_database()
    
    existing_results = read_json_file(result_file, business_id)
    if not existing_results:
        existing_results = [{"id": i, "response": "Neglected", "answer": "Neglected"} for i in range(question_limitation)]
        write_json_file(existing_results, result_file, business_id, dataset_name, model_name)
    else:
        current_length = len(existing_results)
        if current_length < question_limitation:
            for i in range(current_length, question_limitation):
                existing_results.append({"id": i, "response": "Neglected", "answer": "Neglected"})
            write_json_file(existing_results, result_file, business_id, dataset_name, model_name)
        for i in range(question_limitation):
            if existing_results[i] is None:
                existing_results[i] = {"id": i, "response": "Neglected", "answer": "Neglected"}
            elif existing_results[i].get('response') is None:
                existing_results[i]['response'] = "Neglected"
            elif existing_results[i].get('answer') is None:
                existing_results[i]['answer'] = "Neglected"
        write_json_file(existing_results, result_file, business_id, dataset_name, model_name)
        
    args_list = []
    for i in range(question_limitation):
        if existing_results[i]['response'] != "Neglected":
            continue
        case = cases[i] if cases is not None else None
        question = questions[i] if questions is not None else None
        question_type = question_type_list[i]
        choice_list = choices[i] if choices is not None else None
        answer = answers[i] if answers is not None else None
        print("0:", answer)
        hint = hints[i] if hints is not None else None
        image = images[i] if images is not None else None
        input_data = {
            'case': case,
            'question': question,
            'question_type': question_type,
            'choices': choice_list,
            'answer': answer,
            'hint': hint,
            'image': image
        }
        args_list.append((i, input_data, language, background, dataset_name, model_name, model_key, api_base))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_question, args): args[0] for args in args_list}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"评测中"):
            idx, extracted_response, response, answer = future.result()
            existing_results[idx]["response"] = response
            existing_results[idx]["extracted_response"] = extracted_response
            existing_results[idx]["answer"] = answer
            #print("3:", answer)
            # 同时保存到文件（兼容性）和数据库
            write_json_file(existing_results, result_file, business_id, dataset_name, model_name)
    
    score_file = f"results/{user_id}/{business_id}_score.json"
    valid_ratio, score = calculate_valid_ratio_and_score(result_file, answers, score_file, question_type_list, neglected_threshold=0.05, max_score=max_score, business_id=business_id, dataset_name=dataset_name, model_name=model_name)
    return valid_ratio, score


def calculate_valid_ratio_and_score(result_file, answers, accuracy_file, question_type_list=None, neglected_threshold=0.05, max_score=1, business_id="", dataset_name="", model_name=""):
    """
    计算准确率并写入文件和数据库
    
    参数:
        results: 模型回答结果列表
        answers: 标准答案列表
        accuracy_file: 准确率结果文件路径
        question_type_list: 问题类型列表，用于区分单选和多选题
        neglected_threshold: Neglected题目的最大比例阈值，超过此阈值则不计算准确率
        business_id: 业务ID，用于数据库存储
        dataset_name: 数据集名称
        model_name: 模型名称
        
    返回:
        结果列表和准确率
    """
    results = read_json_file(result_file, business_id)
    total_questions = len(results)
    # 修复：同时统计 "Neglected" 和 None 两种无效响应
    neglected_count = sum(1 for result in results if result["extracted_response"] == "Neglected" or result["extracted_response"] is None)
    neglected_ratio = neglected_count / total_questions if total_questions > 0 else 0
    valid_ratio = 1 - neglected_ratio

    # if neglected_ratio > neglected_threshold:
    #     print(f"无效题目比例 ({neglected_ratio:.2%}) 超过阈值 ({neglected_threshold:.2%})，评测无效，不生成score文件")
    #     return valid_ratio, None
    
    if answers is None:
        print("数据集没有标准答案，无法计算准确率")
        accuracy_data = {"accuracy": None, "message": "数据集没有标准答案"}
        write_json_file(accuracy_data, accuracy_file)
        return results, None
    
    correct_count = 0
    valid_count = 0
    for i, result in enumerate(results):
        if i < len(answers) and result["extracted_response"] is not None and result["extracted_response"] != "Neglected":
            valid_count += 1
            model_response = result["extracted_response"]
            correct_answer = answers[i]
            #print("4:", correct_answer)

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
    
    raw_score = correct_count / valid_count * 100 if valid_count > 0 else 0
    score = raw_score / max_score * 100
    
    accuracy_data = {
        "business_id": business_id,
        "raw_score": raw_score,
        "score": score,
        "valid_ratio": valid_ratio
    }
    
    # 同时保存到文件和数据库
    write_json_file(accuracy_data, accuracy_file)
    if business_id and dataset_name and model_name:
        save_evaluation_score(business_id, dataset_name, model_name, score)
    
    return valid_ratio, score

if __name__ == "__main__":
    score = evaluate_all_mcq_automatic(
        user_id="test",
        dataset_name="MMStar",
        model_name="Pro/Qwen/Qwen2.5-VL-7B-Instruct",
        model_key="sk-fPz5uPZn2ubb9Qexx62yWcFl55Z46iRdBfdlvnjufQ6o0BVo",
        api_base="https://api.huatuogpt.cn/v1",
        business_id=None,
        question_limitation=100,  # 测试50题
        max_workers=64
    )



