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
    idx, language, background, case, questions, reference_answer, result, model_name, model_key, api_base, judge_prompt, judge_prompt_with_reference, temperature = args
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
        time.sleep(1)
        return idx, result
    else:
        evaluate_responses = []
        generate_responses = []
        scores = []

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

            # 生成回答
            generate_chat = MultiturnTextAPI(model_name, system_prompt, question_prompt, temperature, f"GenerateAgent_{idx}", model_key, api_base)
            model_response = generate_chat.generate_response()
            generate_responses.append(model_response)

            # 构建评估提示语
            if reference_answer is not None:
                if language == "en":
                    evaluate_prompt = f"{question_prompt}\n\nGenerate Response: {model_response}\n\nReference Answer: {reference_answer}"
                    evaluate_chat = MultiturnTextAPI(evaluate_model, system_judge_prompt_with_reference, evaluate_prompt, 0.7, f"JudgeAgent_{idx}", model_key, api_base)
                else:
                    evaluate_prompt = f"{question_prompt}\n\n模型回答：{model_response}\n\n参考答案：{reference_answer}"
                    evaluate_chat = MultiturnTextAPI(evaluate_model, system_judge_prompt_with_reference, evaluate_prompt, 0.7, f"JudgeAgent_{idx}", model_key, api_base)
            else:
                if language == "en":
                    evaluate_prompt = f"{question_prompt}\n\nGenerate Response: {model_response}"
                    evaluate_chat = MultiturnTextAPI(evaluate_model, system_judge_prompt, evaluate_prompt, 0.7, f"JudgeAgent_{idx}", model_key, api_base)
                else:
                    evaluate_prompt = f"{question_prompt}\n\n模型回答：{model_response}"
                    evaluate_chat = MultiturnTextAPI(evaluate_model, system_judge_prompt, evaluate_prompt, 0.7, f"JudgeAgent_{idx}", model_key, api_base)

            # 获取评估结果
            evaluate_response = evaluate_chat.generate_response()
            score = extract_scores(evaluate_response)
            scores.append(score)
            evaluate_responses.append(evaluate_response)

        # 保存结果
        result["generate_response"] = generate_responses
        result["evaluate_response"] = evaluate_responses
        result["score"] = sum(scores) / len(scores)
        return idx, result


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

    result_file = f"results/{user_id}/{business_id}_result.json"
    accuracy_file = f"results/{user_id}/{business_id}_score.json"
    
    existing_results = read_json_file(result_file)
    if not existing_results:
        existing_results = [{"id": i, "reference_answer": "None", "generate_response": "Neglected", "judge_response": "Neglected", "score": -1} for i in range(question_limitation)]
        write_json_file(existing_results, result_file)

    args_list = []
    for i in range(question_limitation):
        if existing_results[i]['score'] >= 0 and existing_results[i]['score'] <= max_score:
            continue
        result = existing_results[i]
        cases = case_list[i] if i < len(case_list) else None
        questions = question_list[i] if i < len(question_list) else None
        reference_answer = reference_answer_list[i] if i < len(reference_answer_list) else None
        temperature = 0 if reference_answer is not None else 0.7
        args_list.append((i, language, background, cases, questions, reference_answer, result, model_name, model_key, api_base, judge_prompt, judge_prompt_with_reference, temperature))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_question_automatic, args): args[0] for args in args_list}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"评测中"):
            idx, result = future.result()
            existing_results[idx] = result
            write_json_file(existing_results, result_file)
    
    final_results = read_json_file(result_file)
    generate_score_summary(final_results, accuracy_file, max_score=max_score)
    return final_results
        

    
def evaluate_llmjudge_manual(
        user_id: str = "",
        dataset_name: str = "MMStar",
        model_name: str = "gpt-4o",
        business_id: str = "",
        question_limitation: int = 100,
        response_url: str = "",
        model_key: str = "",
        api_base: str = "",
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
    except Exception as e:
        print(f"获取响应时出错: {str(e)}")
        return None

    result_file = f"results/{user_id}/{business_id}_manual_result.json"
    score_file = f"results/{user_id}/{business_id}_manual_score.json"
    existing_results = read_json_file(result_file)
    if not existing_results:
        existing_results = [{"id": i, "reference_answer": "None", "generate_response": "Neglected", "judge_response": "Neglected", "score": -1} for i in range(len(question_list))]
        write_json_file(existing_results, result_file)

    if question_limitation >= len(question_list):
        question_limitation = len(question_list)
        
    args_list = []
    for i in range(len(question_list)):
        if existing_results[i]['score'] >= 0 and existing_results[i]['score'] <= max_score:
            continue
        result = existing_results[i]
        cases = case_list[i]
        questions = question_list[i]
        reference_answer = reference_answer_list[i]
        model_responses = response[i]['response']
        args_list.append((i, language, background, cases, questions, model_responses, reference_answer, result, model_name, judge_prompt, judge_prompt_with_reference, model_key, api_base))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_question_manual, args): args[0] for args in args_list}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"评测中"):
                idx, result = future.result()
                existing_results[idx] = result
                write_json_file(existing_results, result_file)
        
        final_results = read_json_file(result_file)
        generate_score_summary(final_results, score_file, max_score=max_score)
    return final_results

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
            evaluate_chat = MultiturnTextAPI("gpt-4o", system_judge_prompt_with_reference, evaluate_prompt, 0.7, f"JudgeAgent_{idx}", model_key, api_base)
        else:
            if language == "en":
                evaluate_prompt = f"{question_prompt}\n\nGenerate Response: {model_response}"
            else:
                evaluate_prompt = f"{question_prompt}\n\n模型回答：{model_response}"
            evaluate_chat = MultiturnTextAPI("gpt-4o", system_judge_prompt, evaluate_prompt, 0.7, f"JudgeAgent_{idx}", model_key, api_base)

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





