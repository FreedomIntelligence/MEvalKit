"""
文本多选题评测模块

该模块提供了对文本多选题数据集进行评测的功能，支持自动模式和手动模式。
主要功能包括：
- 文本多选题的自动评测（实时调用API）
- 文本多选题的手动评测（使用预生成响应）
- 答案提取和验证
- 结果计算和存储

作者: MEvalKit Team
版本: 1.0.0
"""

import sys
import os
import json
import requests
from pathlib import Path
from openai import BadRequestError, AuthenticationError
from typing import Optional

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入相关模块
from src.dataset.Text.TextMCQ import *
from src.api.text_api import *
from src.utils.MCQ_constants import *
from src.utils.default_prompts import *
from src.utils.model_and_dataset import *
from tqdm import tqdm
import concurrent.futures
from typing import List, Tuple, Dict, Any, Literal
from dotenv import load_dotenv
import re

# 导入数据库模块
from src.database.repository import evaluation_repo, task_repo
from secure_database import SecureDatabase

# 通过环境变量自动认证
load_dotenv()
username = os.environ.get("DB_USER")
password = os.environ.get("DB_PASS")
db = SecureDatabase("mevalkit_secure.db")
if not db.authenticate(username, password):
    print("认证失败，程序退出。")
    exit(1)

def save_to_secure_database(
    business_id, user_id, dataset_name, model_name, evaluation_mode, eval_type,
    result_data=None, response_data=None, is_completed=False,
    score=None, raw_score=None, valid_ratio=None, total_questions=None, valid_questions=None
):
    """
    保存评测结果到加密数据库
    
    该函数将评测结果加密后存储到SQLite数据库中，支持upsert操作。
    
    参数:
        business_id: 业务ID，用于标识评测任务
        user_id: 用户ID
        dataset_name: 数据集名称
        model_name: 模型名称
        evaluation_mode: 评测模式（automatic/manual）
        eval_type: 评测类型
        result_data: 评测结果数据
        response_data: 模型响应数据
        is_completed: 是否完成评测
        score: 最终得分
        raw_score: 原始得分
        valid_ratio: 有效问题比例
        total_questions: 总问题数
        valid_questions: 有效问题数
        
    返回:
        bool: 保存是否成功
    """
    try:
        encrypted_result = db.encrypt_data(json.dumps(result_data)) if result_data else None
        encrypted_response = db.encrypt_data(json.dumps(response_data)) if response_data else None
        import sqlite3
        conn = sqlite3.connect("mevalkit_secure.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO evaluation_results_secure (
                business_id, user_id, dataset_name, model_name, evaluation_mode, eval_type,
                result_data_encrypted, response_data_encrypted, is_completed,
                score, raw_score, valid_ratio, total_questions, valid_questions
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(business_id, user_id) DO UPDATE SET
                result_data_encrypted=excluded.result_data_encrypted,
                response_data_encrypted=excluded.response_data_encrypted,
                is_completed=excluded.is_completed,
                score=excluded.score,
                raw_score=excluded.raw_score,
                valid_ratio=excluded.valid_ratio,
                total_questions=excluded.total_questions,
                valid_questions=excluded.valid_questions
        """, (
            business_id, user_id, dataset_name, model_name, evaluation_mode, eval_type,
            encrypted_result, encrypted_response, int(is_completed),
            score, raw_score, valid_ratio, total_questions, valid_questions
        ))
        conn.commit()
        conn.close()
        print(f"[加密数据库] business_id={business_id} 写入成功 is_completed={is_completed} score={score}")
        return True
    except Exception as e:
        print(f"[加密数据库] business_id={business_id} 写入失败: {str(e)}")
        return False

def extract_answer(response: str, dataset_name: str):
    """
    提取单选题答案
    
    从模型的响应文本中提取单选题的答案选项（如A、B、C、D）。
    支持多种答案格式的正则表达式匹配。
    
    参数:
        response: 模型的响应文本
        dataset_name: 数据集名称，用于确定答案格式
        
    返回:
        str: 提取的答案选项（如A、B、C、D），如果未找到则返回None
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
    
    从模型的响应文本中提取多选题的答案选项列表。
    支持多种答案格式，包括逗号分隔、空格分隔等。
    
    参数:
        response: 模型的响应文本
        dataset_name: 数据集名称，用于确定答案格式
        
    返回:
        List[str]: 提取的答案选项列表（如['A', 'B', 'C']），如果未找到则返回None
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

def shuffle_and_convert(dataset: TextMCQ, shuffle: bool = True):
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
    i, dataset_name, background, case, question, question_type, choices, answer, hint, language, model_path, model_key, api_base = args

    
    # 统一问题类型格式
    if question_type in SINGLE_CHOICE_LIST:
        question_type = "single"
    elif question_type in MULTIPLE_CHOICE_LIST:
        question_type = "multiple"

    case_prompt = ""
    if case is not None:
        if language == "en":
            case_prompt = f"Case of the question: {case}"
        elif language == "zh":
            case_prompt = f"问题背景：{case}"
        
    # 构建问题提示
    question_prompt = question + "\n"
    for choice in choices:
        question_prompt += f"{chr(65 + choices.index(choice))}. {choice}" + " "
    if hint != "":
        question_prompt += f"\nHint: {hint}"
        
    question_prompt = case_prompt + question_prompt

    # 选择适当的提示模板
    if language == 'en' and question_type == 'single':
        system_prompt = MCQ_TEMPLATE_SINGLE_EN
    elif language == "en" and question_type == "multiple":
        system_prompt = MCQ_TEMPLATE_MULTIPLE_EN
    elif language == "zh" and question_type == "single":
        system_prompt = MCQ_TEMPLATE_SINGLE_ZH
    elif language == "zh" and question_type == "multiple":
        system_prompt = MCQ_TEMPLATE_MULTIPLE_ZH
    
    if background is not None and language == "en":
        system_prompt = system_prompt + f"\nBackground: {background}"
    elif background is not None and language == "zh":
        system_prompt = system_prompt + f"\n任务背景：{background}"


    # 获取回答
    try:
        if model_path == "stressTest":
            extracted_response = "A"
            time.sleep(1)
        else:
            chat = TextAPI(model_path, system_prompt, question_prompt, 0, model_key, api_base)
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
    将数据写入JSON文件（保留兼容性）
    
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
    从JSON文件读取数据（保留兼容性）
    
    参数:
        file_path: 文件路径
        
    返回:
        读取的数据
    """
    try:
        if not os.path.exists(file_path):
            return None
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"读取JSON文件时出错: {str(e)}")
        return None

def save_to_database(business_id: str, user_id: str, dataset_name: str, 
                    model_name: str, evaluation_mode: str, eval_type: str,
                    result_data: List[Dict[str, Any]] = None, 
                    response_data: List[Dict[str, Any]] = None,
                    is_completed: bool = False) -> bool:
    """保存评测结果到数据库"""
    try:
        result_info = {
            'business_id': business_id,
            'user_id': user_id,
            'dataset_name': dataset_name,
            'model_name': model_name,
            'evaluation_mode': evaluation_mode,
            'eval_type': eval_type,
            'result_data': result_data,
            'response_data': response_data,
            'is_completed': is_completed
        }
        
        # 如果评测已完成，计算统计信息
        if is_completed and result_data:
            total_questions = len(result_data)
            valid_questions = sum(1 for item in result_data if item.get("response") != "Neglected")
            valid_ratio = valid_questions / total_questions if total_questions > 0 else 0
            
            if valid_ratio >= 0.95:
                # 计算准确率
                correct_count = 0
                for i, result in enumerate(result_data):
                    if result.get("response") != "Neglected":
                        # 这里需要根据具体的数据集来计算准确率
                        # 暂时使用简单的统计
                        pass
                
                result_info.update({
                    'total_questions': total_questions,
                    'valid_questions': valid_questions,
                    'valid_ratio': valid_ratio
                })
        
        return evaluation_repo.save_evaluation_result(result_info) is not None
    except Exception as e:
        print(f"保存到数据库失败: {str(e)}")
        return False

def get_from_database(business_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """从数据库获取评测结果"""
    try:
        result = evaluation_repo.get_evaluation_result(business_id, user_id)
        return result.to_dict() if result else None
    except Exception as e:
        print(f"从数据库获取结果失败: {str(e)}")
        return None

def evaluate_mcq_manual(
        user_id: str = "",
        dataset_name: str = "MMStar",
        model_name: str = "gpt-4o",
        business_id: str = "",
        question_limitation: int = 100,
        response_url: str = "",
):
    dataset = TextMCQ(dataset_name)
    dataset.choices, dataset.answers = shuffle_and_convert(dataset, shuffle=False)
    dataset.max_score = dataset.dataset_info['max_score']
    language = dataset.language

    if question_limitation >= len(dataset.questions):
        question_limitation = len(dataset.questions)

    try:
        response = requests.get(response_url, timeout=60)
        response = response.json()
    except Exception as e:
        print(f"获取响应时出错: {str(e)}")
        return None

    if len(response) < question_limitation:
        question_limitation = len(response)

    result = []
    
    for i in range(question_limitation):
        result.append({
            "id": i,
            "response": response[i]["response"]
        })

    # 保存到数据库
    save_to_secure_database(business_id, user_id, dataset_name, model_name, "manual", "textmcq", result, None, True)
    
    # 同时保存到文件（兼容性）
    result_file = f"results/{user_id}/{business_id}_manual_result.json"
    score_file = f"results/{user_id}/{business_id}_manual_score.json"
    write_json_file(result, result_file)

    return calculate_valid_ratio_and_score(result_file, dataset.answers, score_file, max_score=dataset.max_score, business_id=business_id)

def evaluate_mcq_automatic(
        user_id: str = "",
        dataset_name: str = "MMLU",
        model_name: str = "gpt-3.5-turbo",
        model_key: str = "",
        api_base: str = "",
        business_id: str = "",
        question_limitation: int = 100,
        max_workers: int = 64
):
    """
    并行评估文本问题
    
    参数:
        dataset_name: 数据集名称
        model_name: 模型名称
        max_workers: 最大并行工作线程数
        evaluate_mode: 评估模式，"start_from_beginning"从头开始，"give_answers"使用已有答案
        
    返回:
        评估结果和准确率
    """
    # 加载数据集
    dataset = TextMCQ(dataset_name)
    dataset.choices, dataset.answers = shuffle_and_convert(dataset, shuffle=True)
    background = dataset.background
    dataset.max_score = dataset.dataset_info['max_score']
    language = dataset.language

    if question_limitation >= len(dataset.questions):
        question_limitation = len(dataset.questions)

    # 尝试从数据库获取现有结果
    existing_db_result = get_from_database(business_id, user_id)
    
    if existing_db_result and existing_db_result.get('result_data'):
        existing_results = existing_db_result['result_data']
    else:
        # 从文件读取（兼容性）
        result_file = f"results/{user_id}/{business_id}_result.json"
        response_file = f"results/{user_id}/{business_id}_response.json"
    existing_results = read_json_file(result_file)
    
    if not existing_results:
        existing_results = [{"id": i, "response": "Neglected"} for i in range(question_limitation)]
        # 保存到数据库
        save_to_secure_database(business_id, user_id, dataset_name, model_name, "automatic", "textmcq", existing_results, existing_results, False)
        # 同时保存到文件（兼容性）
        write_json_file(existing_results, result_file)
        write_json_file(existing_results, response_file)
    else:
        # 如果existing_results存在但长度不足，需要扩展到question_limitation长度
        current_length = len(existing_results)
        if current_length < question_limitation:
            for i in range(current_length, question_limitation):
                existing_results.append({"id": i, "response": "Neglected"})
            # 保存到数据库
            save_to_secure_database(business_id, user_id, dataset_name, model_name, "automatic", "textmcq", existing_results, existing_results, False)
            # 同时保存到文件（兼容性）
            write_json_file(existing_results, result_file)
            write_json_file(existing_results, response_file)
        # 处理existing_results中可能存在的None情况，统一处理成Neglected
        for i in range(question_limitation):
            if existing_results[i] is None:
                existing_results[i] = {"id": i, "response": "Neglected"}
            elif existing_results[i].get('response') is None:
                existing_results[i]['response'] = "Neglected"
        write_json_file(existing_results, result_file)
        write_json_file(existing_results, response_file)

    args_list = []
    for i in range(question_limitation):
        if existing_results[i]['response'] != "Neglected":
            continue
        question = dataset.questions[i] if dataset.questions is not None else None
        question_type = dataset.question_type_list[i] if dataset.question_type_list is not None else None
        choices = dataset.choices[i] if dataset.choices is not None else None
        case = dataset.case[i] if dataset.case is not None else None
        answer = ""
        hint = ""
        if dataset.answers is not None:
            answer = dataset.answers[i]
        if dataset.hints is not None:
            hint = dataset.hints[i]
        args_list.append((i, dataset_name, background, case, question, question_type, choices, answer, hint, language, model_name, model_key, api_base))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_question, args): args[0] for args in args_list}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"评测中"):
            idx, response, answer = future.result()
            existing_results[idx]["response"] = response
            # 保存到数据库
            save_to_secure_database(business_id, user_id, dataset_name, model_name, "automatic", "textmcq", existing_results, existing_results, False)
            # 同时保存到文件（兼容性）
            write_json_file(existing_results, result_file)
            write_json_file(existing_results, response_file)
        
    # 生成评分摘要（兼容性）
    accuracy_file = f"results/{user_id}/{business_id}_score.json"
    valid_ratio, score = calculate_valid_ratio_and_score(result_file, dataset.answers, accuracy_file, max_score=dataset.max_score, business_id=business_id)
    
    # 读取计算出的分数并更新数据库
    try:
        with open(accuracy_file, 'r', encoding='utf-8') as f:
            score_data = json.load(f)
        
        # 计算统计信息
        total_questions = len(existing_results)
        valid_questions = sum(1 for item in existing_results if item.get("response") != "Neglected")
        
        # 准备完整的最终结果数据
        final_result = {
            'total_questions': total_questions,
            'valid_questions': valid_questions,
            'valid_ratio': score_data.get('valid_ratio', 0.0),
            'raw_score': score_data.get('raw_score', 0.0),
            'score': score_data.get('score', 0.0),
            'result_data': existing_results,
            'response_data': existing_results
        }
        
        # 完成评测，保存最终结果（包含分数）
        save_to_secure_database(business_id, user_id, dataset_name, model_name, "automatic", "textmcq", existing_results, existing_results, True,
                                score=final_result['score'], raw_score=final_result['raw_score'], valid_ratio=final_result['valid_ratio'],
                                total_questions=final_result['total_questions'], valid_questions=final_result['valid_questions'])
        
        # 使用complete_evaluation方法确保分数被正确保存
        from src.database.repository import evaluation_repo
        evaluation_repo.complete_evaluation(business_id, user_id, final_result)
        
    except Exception as e:
        print(f"更新数据库分数失败: {str(e)}")
        # 即使分数更新失败，也要保存基本结果
        save_to_secure_database(business_id, user_id, dataset_name, model_name, "automatic", "textmcq", existing_results, existing_results, True)
    
    return valid_ratio, score

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
    print(results)
    print(answers)
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
    # 评估MMLU数据集

    # response_url = "http://47.110.252.218:1995/admin-api/infra/file/31/get/evaluation/answer/20250717/mmlu_manualresponsetest_2_response_1752738413234.json"
    # response = requests.get(response_url, timeout=60)
    # response = response.json()
    # print(response)
    dataset = TextMCQ("GPQA")
    dataset.choices, dataset.answers = shuffle_and_convert(dataset, shuffle=False)
    print(dataset.answers)
            