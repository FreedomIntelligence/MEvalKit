import sys
import os
import json
import re
import time
from pathlib import Path
from openai import BadRequestError
import concurrent.futures
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.dataset.LLMJudge.LLMJudgeBase import *
from src.api.text_api import *
from src.api.multiturn_text_api import *
from src.utils.default_prompts import *
from src.utils.model_and_dataset import *
from typing import List, Literal, Tuple, Dict, Any, Optional, Union
from dotenv import load_dotenv

# 导入数据库模块
from src.database.repository import evaluation_repo, task_repo
from secure_database import SecureDatabase
import getpass
import json

# 通过环境变量自动认证
load_dotenv()
username = os.environ.get("DB_USER")
password = os.environ.get("DB_PASS")
db = SecureDatabase("mevalkit_secure.db")
if not db.authenticate(username, password):
    print("认证失败，程序退出。")
    exit(1)

def write_json_file(data, file_path):
    """将数据写入JSON文件（保留兼容性）"""
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
    """从JSON文件读取数据（保留兼容性）"""
    try:
        if not os.path.exists(file_path):
            return None
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"读取JSON文件时出错: {str(e)}")
        return None

# 支持upsert和分数字段的加密数据库写入函数
def save_to_secure_database(
    business_id, user_id, dataset_name, model_name, evaluation_mode, eval_type,
    result_data=None, response_data=None, is_completed=False,
    score=None, raw_score=None, valid_ratio=None, total_questions=None, valid_questions=None
):
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

def get_from_database(business_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """从数据库获取评测结果"""
    try:
        result = evaluation_repo.get_evaluation_result(business_id, user_id)
        return result.to_dict() if result else None
    except Exception as e:
        print(f"从数据库获取结果失败: {str(e)}")
        return None

def extract_scores(evaluate_response: str) -> Optional[Union[int, float]]:
    """
    从评估响应中提取分数
    
    评分规则要求第一行必须是1-10的整数分数
    处理各种可能的格式错误：
    1. 第一行是```等markdown标记
    2. 第一行包含额外文本
    3. 分数不在第一行
    4. 分数格式不规范
    """
    if not evaluate_response or not evaluate_response.strip():
        print("评估响应为空")
        return 0
    
    try:
        # 按行分割并清理
        lines = [line.strip() for line in evaluate_response.strip().split('\n') if line.strip()]
        
        if not lines:
            print("评估响应没有有效行")
            return 0
        
        # 查找包含分数的行
        score_line = None
        for line in lines:
            # 跳过markdown标记行
            if line.startswith('```') or line.startswith('`') or line.startswith('#'):
                continue
            
            # 跳过空行或只包含标点的行
            if not line or line in ['', ' ', '\t']:
                continue
            
            # 尝试从行中提取数字
            # 匹配1-10的整数，可能前后有空格或其他字符
            import re
            score_match = re.search(r'\b([1-9]|10)\b', line)
            if score_match:
                score_line = line
                break
        
        if not score_line:
            print(f"未找到有效分数行，响应内容：{evaluate_response[:200]}...")
            return 0
        
        # 从找到的行中提取分数
        # 支持多种格式：纯数字、数字+逗号、数字+其他文本
        import re
        
        # 提取所有1-10的数字
        scores = re.findall(r'\b([1-9]|10)\b', score_line)
        
        if not scores:
            print(f"从行中无法提取分数：{score_line}")
            return 0
        
        # 转换为整数
        score_values = [int(score) for score in scores]
        
        # 返回第一个分数（通常是最准确的）
        result_score = score_values[0]
        
        # 验证分数是否在合理范围内
        if result_score < 1 or result_score > 10:
            print(f"分数超出范围(1-10)：{result_score}")
            return 0
        
        return result_score
        
    except Exception as e:
        print(f"提取分数时出错：{e}")
        print(f"响应内容：{evaluate_response[:200]}...")
        return 0

def process_single_question_automatic(args):
    """
    处理单个问题
    
    参数:
        args: (idx, question, background, reference_answer, model_response, generate_model_name, evaluate_model_name, temperature, generate_prompt, judge_prompt, judge_prompt_with_reference)
        
    返回:
        (idx, result): 问题索引和处理结果
    """
    idx, language, background, case, questions, reference_answer, result, resp, model_name, model_key, api_base, judge_prompt, judge_prompt_with_reference, temperature = args
    evaluate_model = "gpt-4o"
    
    background_prompt_zh = f"任务背景：{background}" if background is not None and language == "zh" else f"任务背景：无"
    background_prompt_en = f"Background of the task: {background}" if background is not None and language == "en" else "Background of the task: None"

    if isinstance(case, str):
        case = [case]
    if isinstance(questions, str):
        questions = [questions]

    if language == "zh":
        system_prompt = DEFAULT_GENERATE_SYSTEM_PROMPT_ZH + background_prompt_zh
    else:
        system_prompt = DEFAULT_GENERATE_SYSTEM_PROMPT_EN + background_prompt_en

    if language == "en":
        system_judge_prompt = DEFAULT_JUDGE_SYSTEM_PROMPT_WITH_GIVEN_EN + background_prompt_en if judge_prompt is None \
            else DEFAULT_JUDGE_SYSTEM_PROMPT_WITH_GIVEN_EN + judge_prompt + background_prompt_en
        system_judge_prompt_with_reference = DEFAULT_JUDGE_SYSTEM_PROMPT_WITH_GIVEN_EN + background_prompt_en if judge_prompt_with_reference is None \
            else DEFAULT_JUDGE_SYSTEM_PROMPT_WITH_GIVEN_EN + judge_prompt_with_reference + background_prompt_en
    else:
        system_judge_prompt = DEFAULT_JUDGE_SYSTEM_PROMPT_WITH_GIVEN_ZH + background_prompt_zh if judge_prompt is None \
            else DEFAULT_JUDGE_SYSTEM_PROMPT_WITH_GIVEN_ZH + judge_prompt + background_prompt_zh
        system_judge_prompt_with_reference = DEFAULT_JUDGE_SYSTEM_PROMPT_WITH_GIVEN_ZH + background_prompt_zh if judge_prompt_with_reference is None \
            else DEFAULT_JUDGE_SYSTEM_PROMPT_WITH_GIVEN_ZH + judge_prompt_with_reference + background_prompt_zh


    if model_name == "stressTest":
        result["generate_response"] = "A"
        result["evaluate_response"] = "A"
        result["score"] = 1
        resp["response"] = "A"
        time.sleep(1)
        return idx, result
    else:
        evaluate_responses = []
        generate_responses = []
        scores = []
        generate_chat = MultiturnTextAPI(model_name, system_prompt, "", temperature, f"GenerateAgent_{idx}", model_key, api_base)
        evaluate_chat = MultiturnTextAPI(evaluate_model, system_judge_prompt, "", 0.7, f"JudgeAgent_{idx}", model_key, api_base)
        for i, question in enumerate(questions):
            # 构建问题提示语
            if case is None:
                if language == "en":
                    question_prompt = f"Question: {question}"
                else:
                    question_prompt = f"问题：{question}"
            else:
                case_text = case[0] if len(case) == 1 else case[i]
                if language == "en":
                    question_prompt = f"Case: {case_text}\n\nQuestion: {question}"
                else:
                    question_prompt = f"案例：{case_text}\n\n问题：{question}"
            generate_chat.user_prompt = question_prompt
            model_response = generate_chat.generate_response()
            generate_responses.append(model_response)


            # 构建评估提示语
            if reference_answer is not None:
                if language == "en":
                    evaluate_prompt = f"{question_prompt}\n\nGenerate Response: {model_response}\n\nReference Answer: {reference_answer}"
                    evaluate_chat.system_prompt = system_judge_prompt_with_reference
                    evaluate_chat.user_prompt = evaluate_prompt
                else:
                    evaluate_prompt = f"{question_prompt}\n\n模型回答：{model_response}\n\n参考答案：{reference_answer}"
                    evaluate_chat.system_prompt = system_judge_prompt_with_reference
                    evaluate_chat.user_prompt = evaluate_prompt
            else:
                if language == "en":
                    evaluate_prompt = f"{question_prompt}\n\nGenerate Response: {model_response}"
                    evaluate_chat.system_prompt = system_judge_prompt
                    evaluate_chat.user_prompt = evaluate_prompt
                else:
                    evaluate_prompt = f"{question_prompt}\n\n模型回答：{model_response}"
                    evaluate_chat.system_prompt = system_judge_prompt
                    evaluate_chat.user_prompt = evaluate_prompt

            # 获取评估结果
            evaluate_response = evaluate_chat.generate_response()
            score = extract_scores(evaluate_response)
            scores.append(score)
            evaluate_responses.append(evaluate_response)

        # 保存结果
        result["generate_response"] = generate_responses
        result["evaluate_response"] = evaluate_responses
        result["score"] = sum(scores) / len(scores)
        resp["response"] = generate_responses
        return idx, result, resp


def evaluate_llmjudge_automatic(
        user_id: str = "test",
        dataset_name: str = "MT-Bench",
        model_name: str = "gpt-3.5-turbo",
        model_key: str = "",
        api_base: str = "",
        business_id: str = "",
        question_limitation: int = 100,
        max_workers: int = 1
):
    """
    并行评估LLM Judge
    
    参数:
    """
    dataset = LLMJudgeBase(dataset_name)
    language = dataset.language
    background = dataset.background
    case_list = dataset.case
    question_list = dataset.questions
    reference_answer_list = dataset.answers
    max_score = dataset.max_score
    judge_prompt = dataset.judge_prompt
    judge_prompt_with_reference = dataset.judge_prompt_with_reference

    if question_limitation > len(question_list):
        question_limitation = len(question_list)

    # 修复：如果case_list为空，初始化为与问题数量相同的空值列表
    if not case_list:
        case_list = [None] * len(question_list)

    # 尝试从数据库获取现有结果
    existing_db_result = get_from_database(business_id, user_id)
    
    if existing_db_result and existing_db_result.get('result_data'):
        existing_results = existing_db_result['result_data']
        response = existing_db_result.get('response_data', [])
    else:
        # 从文件读取（兼容性）
        result_file = f"results/{user_id}/{business_id}_result.json"
    response_file = f"results/{user_id}/{business_id}_response.json"
    existing_results = read_json_file(result_file)
    response = read_json_file(response_file)
    
    if not existing_results:
        existing_results = [{"id": i, "reference_answer": "None", "generate_response": "Neglected", "judge_response": "Neglected", "score": -1} for i in range(question_limitation)]
        response = [{"id": i, "response": "Neglected"} for i in range(question_limitation)]
        # 保存到数据库
        save_to_secure_database(business_id, user_id, dataset_name, model_name, "automatic", "llmjudge", existing_results, response, False)
        # 同时保存到文件（兼容性）
        write_json_file(existing_results, result_file)
        write_json_file(response, response_file)
    else:
        current_length = len(existing_results)
        if current_length < question_limitation:
            for i in range(current_length, question_limitation):
                existing_results.append({"id": i, "reference_answer": "None", "generate_response": "Neglected", "judge_response": "Neglected", "score": -1})
                response.append({"id": i, "response": "Neglected"})
            write_json_file(existing_results, result_file)
            write_json_file(response, response_file)

    args_list = []
    for i in range(question_limitation):
        # 如果结果已经存在且分数大于0，则跳过
        #print(existing_results[i])
        if existing_results[i]['score'] >= 0 and existing_results[i]['score'] <= max_score:
            continue
        result = existing_results[i]
        resp = response[i]
        cases = case_list[i] if (case_list is not None and i < len(case_list)) else None
        questions = question_list[i] if (question_list is not None and i < len(question_list)) else None
        reference_answer = reference_answer_list[i] if (reference_answer_list is not None and i < len(reference_answer_list)) else None
        temperature = 0 if reference_answer is not None else 0.7
        args_list.append((i, language, background, cases, questions, reference_answer, result, resp, model_name, model_key, api_base, judge_prompt, judge_prompt_with_reference, temperature))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_question_automatic, args): args[0] for args in args_list}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"评测中"):
            idx, result, resp = future.result()
            existing_results[idx] = result
            response[idx] = resp
            # 保存到数据库
            save_to_secure_database(business_id, user_id, dataset_name, model_name, "automatic", "llmjudge", existing_results, response, False)
            # 同时保存到文件（兼容性）
            write_json_file(existing_results, result_file)
            write_json_file(response, response_file)
    
    # 生成评分摘要（兼容性）
    accuracy_file = f"results/{user_id}/{business_id}_score.json"
    generate_score_summary(existing_results, accuracy_file, max_score=max_score)
    
    # 读取计算出的分数并更新数据库
    try:
        with open(accuracy_file, 'r', encoding='utf-8') as f:
            score_data = json.load(f)
        
        # 计算统计信息
        total_questions = len(existing_results)
        valid_questions = sum(1 for item in existing_results if item.get("score", -1) >= 0 and item.get("score", -1) <= max_score)
        valid_ratio = valid_questions / total_questions if total_questions > 0 else 0
        
        # 准备完整的最终结果数据
        final_result = {
            'total_questions': total_questions,
            'valid_questions': valid_questions,
            'valid_ratio': valid_ratio,
            'raw_score': score_data.get('raw_score', 0.0),
            'score': score_data.get('score', 0.0),
            'result_data': existing_results,
            'response_data': response
        }
        
        # 完成评测，保存最终结果（包含分数）
        save_to_secure_database(business_id, user_id, dataset_name, model_name, "automatic", "llmjudge", existing_results, response, True, score=final_result['score'], raw_score=final_result['raw_score'], valid_ratio=final_result['valid_ratio'], total_questions=final_result['total_questions'], valid_questions=final_result['valid_questions'])
        
        # 使用complete_evaluation方法确保分数被正确保存
        from src.database.repository import evaluation_repo
        evaluation_repo.complete_evaluation(business_id, user_id, final_result)
        
    except Exception as e:
        print(f"更新数据库分数失败: {str(e)}")
        # 即使分数更新失败，也要保存基本结果
        save_to_secure_database(business_id, user_id, dataset_name, model_name, "automatic", "llmjudge", existing_results, response, True)
    
    return existing_results
        

    
def evaluate_llmjudge_manual(
        user_id: str = "",
        dataset_name: str = "MMStar",
        model_name: str = "gpt-4o",
        business_id: str = "",
        question_limitation: int = 100,
        response_url: str = "",
        max_workers: int = 64,
        ):
    """
    评估LLM Judge
    
    参数:
        dataset_name: 数据集名称
        generate_model_name: 生成回答的模型名称
        evaluate_model_name: 评估回答的模型名称
        max_workers: 并行处理的最大工作线程数
        evaluate_mode: 评估模式，"start_from_beginning"从头开始，"give_answers"使用已有答案
        
    返回:
        评估结果列表
    """
    # 加载数据集
    dataset = LLMJudgeBase(dataset_name)
    # 加载必要元素
    language = dataset.language
    #print(language)
    background = dataset.background
    case_list = dataset.case
    question_list = dataset.questions
    reference_answer_list = dataset.answers

    max_score = dataset.max_score
    judge_prompt = dataset.judge_prompt
    judge_prompt_with_reference = dataset.judge_prompt_with_reference

    # 修复：如果case_list为空，初始化为与问题数量相同的空值列表
    if not case_list:
        case_list = [None] * len(question_list)
    
    try:
        response = requests.get(response_url, timeout=60)
        response = response.json()
        print(response)
    except Exception as e:
        print(f"获取响应时出错: {str(e)}")
        return None

    if len(response) < question_limitation:
        question_limitation = len(response)
    
    if question_limitation >= len(question_list):
        question_limitation = len(question_list)

    # 尝试从数据库获取现有结果
    existing_db_result = get_from_database(business_id, user_id)
    
    if existing_db_result and existing_db_result.get('result_data'):
        existing_results = existing_db_result['result_data']
    else:
        # 从文件读取（兼容性）
        result_file = f"results/{user_id}/{business_id}_manual_result.json"
        existing_results = read_json_file(result_file)
    
    if not existing_results:
        existing_results = [{"id": i, "reference_answer": "None", "generate_response": "Neglected", "judge_response": "Neglected", "score": -1} for i in range(question_limitation)]
        # 保存到数据库
        save_to_secure_database(business_id, user_id, dataset_name, model_name, "manual", "llmjudge", existing_results, None, False)
        # 同时保存到文件（兼容性）
        write_json_file(existing_results, result_file)
    else:
        # 如果existing_results存在但长度不足，需要扩展到question_list长度
        current_length = len(existing_results)
        if current_length < question_limitation:
            for i in range(current_length, question_limitation):
                existing_results.append({"id": i, "reference_answer": "None", "generate_response": "Neglected", "judge_response": "Neglected", "score": -1})
            # 保存到数据库
            save_to_secure_database(business_id, user_id, dataset_name, model_name, "manual", "llmjudge", existing_results, None, False)
            # 同时保存到文件（兼容性）
            write_json_file(existing_results, result_file)


        
    args_list = []
    for i in range(question_limitation):
        if existing_results[i]['score'] >= 0 and existing_results[i]['score'] <= max_score:
            continue
        result = existing_results[i]
        cases = case_list[i] if (case_list is not None and i < len(case_list)) else None
        questions = question_list[i] if (question_list is not None and i < len(question_list)) else None
        reference_answer = reference_answer_list[i] if (reference_answer_list is not None and i < len(reference_answer_list)) else None
        model_responses = response[i]['response']
        load_dotenv()
        args_list.append((i, language, background, cases, questions, model_responses, reference_answer, result, model_name, judge_prompt, judge_prompt_with_reference, os.getenv("MODEL_KEY"), os.getenv("API_BASE")))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_question_manual, args): args[0] for args in args_list}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"评测中"):
            idx, result = future.result()
            existing_results[idx] = result
            # 保存到数据库
            save_to_secure_database(business_id, user_id, dataset_name, model_name, "manual", "llmjudge", existing_results, None, False)
            # 同时保存到文件（兼容性）
            write_json_file(existing_results, result_file)
        
    # 完成评测，保存最终结果
    save_to_secure_database(business_id, user_id, dataset_name, model_name, "manual", "llmjudge", existing_results, None, True)
    
    # 生成评分摘要（兼容性）
    score_file = f"results/{user_id}/{business_id}_manual_score.json"
    generate_score_summary(existing_results, score_file, max_score=max_score)
    return existing_results

def process_single_question_manual(args):
    """
    处理单个问题
    
    参数:
        args: (idx, language, background, cases, questions, model_responses, reference_answer, result, model, judge_prompt, judge_prompt_with_reference, temperature)
    """
    idx, language, background, cases, questions, model_responses, reference_answer, result, model, judge_prompt, judge_prompt_with_reference, model_key, api_base = args
    background_prompt_zh = f"任务背景：{background}" if background is not None and language == "zh" else f"任务背景：无"
    background_prompt_en = f"Background of the task: {background}" if background is not None and language == "en" else "Background of the task: None"

    if language == "en":
        system_judge_prompt = DEFAULT_JUDGE_SYSTEM_PROMPT_WITH_GIVEN_EN + background_prompt_en if judge_prompt is None \
            else DEFAULT_JUDGE_SYSTEM_PROMPT_WITH_GIVEN_EN + judge_prompt + background_prompt_en
        system_judge_prompt_with_reference = DEFAULT_JUDGE_SYSTEM_PROMPT_WITH_GIVEN_EN + background_prompt_en if judge_prompt_with_reference is None \
            else DEFAULT_JUDGE_SYSTEM_PROMPT_WITH_GIVEN_EN + judge_prompt_with_reference + background_prompt_en
    else:
        system_judge_prompt = DEFAULT_JUDGE_SYSTEM_PROMPT_WITH_GIVEN_ZH + background_prompt_zh if judge_prompt is None \
            else DEFAULT_JUDGE_SYSTEM_PROMPT_WITH_GIVEN_ZH + judge_prompt + background_prompt_zh
        system_judge_prompt_with_reference = DEFAULT_JUDGE_SYSTEM_PROMPT_WITH_GIVEN_ZH + background_prompt_zh if judge_prompt_with_reference is None \
            else DEFAULT_JUDGE_SYSTEM_PROMPT_WITH_GIVEN_ZH + judge_prompt_with_reference + background_prompt_zh
    
    if isinstance(cases, str):
        cases = [cases]
    if isinstance(questions, str):
        questions = [questions]
    if isinstance(model_responses, str):
        model_responses = [model_responses]
    
    evaluate_responses = []
    generate_responses = []
    scores = []
    evaluate_chat = MultiturnTextAPI("gpt-4o", system_judge_prompt, "", 0.7, f"JudgeAgent_{idx}", model_key, api_base)
    for i, question in enumerate(questions):
        if cases is None:
            if language == "en":
                question_prompt = f"Question: {question}"
            else:
                question_prompt = f"问题：{question}"
        else:
            case_text = cases[0] if len(cases) == 1 else cases[i]
            if language == "en":
                question_prompt = f"Case: {case_text}\n\nQuestion: {question}"
            else:
                question_prompt = f"案例：{case_text}\n\n问题：{question}"

        model_response = model_responses[i]
        if reference_answer is not None:
            if language == "en":
                evaluate_prompt = f"{question_prompt}\n\nGenerate Response: {model_response}\n\nReference Answer: {reference_answer}"
            else:
                evaluate_prompt = f"{question_prompt}\n\n模型回答：{model_response}\n\n参考答案：{reference_answer}"
            evaluate_chat.system_prompt = system_judge_prompt_with_reference
            evaluate_chat.user_prompt = evaluate_prompt

        else:
            if language == "en":
                evaluate_prompt = f"{question_prompt}\n\nGenerate Response: {model_response}"
            else:
                evaluate_prompt = f"{question_prompt}\n\n模型回答：{model_response}"
            evaluate_chat.system_prompt = system_judge_prompt
            evaluate_chat.user_prompt = evaluate_prompt

        evaluate_response = evaluate_chat.generate_response()
        score = extract_scores(evaluate_response)
        scores.append(score)
        evaluate_responses.append(evaluate_response)

    result["generate_response"] = generate_responses
    result["evaluate_response"] = evaluate_responses
    result["score"] = sum(scores) / len(scores)
    return idx, result


def generate_score_summary(all_results, score_file, completion_threshold=0.95, max_score=10):
    """
    生成评分摘要并写入文件
    
    参数:
        all_results: 所有评估结果
        score_file: 评分摘要文件路径
        completion_threshold: 完成评分的题目比例阈值
    """
    # 统计所有问题和有效评分
    total_questions = 0
    valid_scores = []
    
    # 检查all_results的结构
    if isinstance(all_results, list) and len(all_results) > 0:
        # 如果是LLMJudge格式（直接是结果列表）
        if "score" in all_results[0]:
            for result in all_results:
                total_questions += 1
                if result["score"] >= 0 and result["score"] <= max_score:
                    valid_scores.append(result["score"])
        # 如果是其他格式（包含question_set）
        elif "results" in all_results[0]:
            for question_set in all_results:
                for result in question_set["results"]:
                    total_questions += 1
                    if result["score"] >= 0 and result["score"] <= max_score:
                        valid_scores.append(result["score"])
    
    # 计算完成率
    completion_ratio = len(valid_scores) / total_questions if total_questions > 0 else 0
    
    # 如果完成率达到阈值，生成评分摘要
    if completion_ratio >= completion_threshold:
        raw_score = sum(valid_scores) / len(valid_scores)
        score = raw_score / max_score * 100
        
        # 创建摘要数据
        summary_data = {
            "completion_ratio": completion_ratio,
            "raw_score": raw_score,
            "score": score,
            "total_questions": total_questions,
            "valid_questions": len(valid_scores)
        }
        
        # 写入摘要文件
        write_json_file(summary_data, score_file)
        print(f"评分摘要已生成: {score_file}")
    else:
        print(f"完成率 ({completion_ratio:.2%}) 未达到阈值 ({completion_threshold:.2%})，暂不生成评分摘要")



if __name__ == "__main__":
    response_url = "http://47.110.252.218:1995/admin-api/infra/file/31/get/evaluation/answer/20250718/case_1752823934098.json"
    response = requests.get(response_url, timeout=60)
    response = response.json()
    print(response)
    
