import sys
import os
import json
import re
import time
import math
from pathlib import Path
from openai import BadRequestError
import concurrent.futures
from tqdm import tqdm
from jinja2 import Template
import re
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.dataset.QA import *
from src.api.multiturn_multimodal_api import *
from src.utils.default_prompts import *
from src.utils.model_and_dataset import *
from typing import List, Literal, Tuple, Dict, Any, Optional, Union
from dotenv import load_dotenv

# 添加MySQL数据库支持
from src.database.mysql_db import (
    save_evaluation_result, 
    load_evaluation_result,
    save_evaluation_score, 
    load_evaluation_score,
    initialize_database
)

# 导入数据库模块
#from src.database.repository import evaluation_repo, task_repo
#from secure_database import SecureDatabase
from datetime import datetime
import getpass

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

def write_json_file(data, file_path, business_id=None, dataset_name=None, model_name=None):
    """将数据写入JSON文件并同时保存到数据库（保留兼容性）"""
    try:
        # 确保目录存在
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
    """从JSON文件读取数据，如果文件不存在则尝试从数据库加载（保留兼容性）"""
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

def extract_scores_from_evaluate_response(evaluate_response: str) -> Optional[Union[int, float]]:
    if not evaluate_response or not evaluate_response.strip():
        return 0
    
    try:
        lines = [line.strip() for line in evaluate_response.strip().split('\n') if line.strip()]
        
        if not lines:
            return 0
        
        score_line = None
        for line in lines:
            if line.startswith('```') or line.startswith('`') or line.startswith('#'):
                continue
            if not line or line in ['', ' ', '\t']:
                continue
            import re
            score_match = re.search(r'\b([1-9]|10)\b', line)
            if score_match:
                score_line = line
                break
        if not score_line:
            return 0
        
        import re
        scores = re.findall(r'\b([1-9]|10)\b', score_line)
        if not scores:
            return 0
        
        score_values = [int(score) for score in scores]
        result_score = score_values[0]
        if result_score < 1 or result_score > 10:
            return 0
        
        return result_score
    
    except Exception as e:
        return 0

def extract_scores(evaluate_response: str, max_score: int = 10) -> Optional[Union[int, float]]:
    """
    从评估响应中提取分数
    
    评分规则要求第一行必须是1-max_score的整数分数
    处理各种可能的格式错误：
    1. 第一行是```等markdown标记
    2. 第一行包含额外文本
    3. 分数不在第一行
    4. 分数格式不规范
    """
    if not evaluate_response or not evaluate_response.strip():
        return 0
    
    try:
        # 按行分割并清理
        lines = [line.strip() for line in evaluate_response.strip().split('\n') if line.strip()]
        
        if not lines:
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
            
            # 尝试从行中提取数字，支持动态最大分数
            if max_score <= 10:
                score_match = re.search(r'\b([1-9]|10)\b' if max_score >= 10 else fr'\b([1-{max_score}])\b', line)
            else:
                score_match = re.search(fr'\b([1-9]|[1-9][0-9]|{max_score})\b', line)
            
            if score_match:
                score_line = line
                break
        
        if not score_line:
            return 0
        
        # 从找到的行中提取分数
        # 支持多种格式：纯数字、数字+逗号、数字+其他文本
        if max_score <= 10:
            scores = re.findall(r'\b([1-9]|10)\b' if max_score >= 10 else fr'\b([1-{max_score}])\b', score_line)
        else:
            scores = re.findall(fr'\b([1-9]|[1-9][0-9]|{max_score})\b', score_line)
        
        if not scores:
            return 0
        
        # 转换为整数
        score_values = [int(score) for score in scores]
        
        # 返回第一个分数（通常是最准确的）
        result_score = score_values[0]
        
        # 验证分数是否在合理范围内
        if result_score < 1 or result_score > max_score:
            return 0
        
        return result_score
        
    except Exception as e:
        return 0

def generate_score_summary(all_results, score_file, completion_threshold=0.95, max_score=10, business_id=None, dataset_name=None, model_name=None):
    """
    生成评分摘要并写入文件和数据库
    
    参数:
        all_results: 所有评估结果
        score_file: 评分摘要文件路径
        completion_threshold: 完成评分的题目比例阈值
        max_score: 最大分数
        business_id: 业务ID
        dataset_name: 数据集名称
        model_name: 模型名称
    """
    # 统计所有问题和有效评分
    total_questions = 0
    valid_scores = []
    
    # 检查all_results的结构
    if isinstance(all_results, list) and len(all_results) > 0:
        # LLMJudge格式（直接是结果列表）
        if "score" in all_results[0]:
            for result in all_results:
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
        
        # 写入摘要文件和数据库
        write_json_file(summary_data, score_file, business_id, dataset_name, model_name)
        return summary_data
    else:
        return None

        
def scoring(generate_response_list: List[List[str]], reference_answer_list: List[List[str]], scoring_criteria: str, **kwargs) -> float:
    if scoring_criteria == "accuracy":
        total_questions = 0
        correct_questions = 0
        for i, generate_responses in enumerate(generate_response_list):
            if i >= len(reference_answer_list):
                continue
            reference_answers = reference_answer_list[i]
            # 确保reference_answers是列表
            if not isinstance(reference_answers, list):
                reference_answers = [reference_answers]
            
            for j, generate_response in enumerate(generate_responses):
                if j < len(reference_answers) and generate_response == reference_answers[j]:
                    correct_questions += 1
                total_questions += 1
        score = correct_questions / total_questions * 100 if total_questions > 0 else 0
        return score
    
    elif scoring_criteria == "bleu":
        total_bleu = 0
        total_pairs = 0
        smoothing = SmoothingFunction().method1
        
        for i, generate_responses in enumerate(generate_response_list):
            if i >= len(reference_answer_list):
                continue
            reference_answers = reference_answer_list[i]
            # 确保reference_answers是列表
            if not isinstance(reference_answers, list):
                reference_answers = [reference_answers]
                
            for j, generate_response in enumerate(generate_responses):
                if j >= len(reference_answers):
                    continue
                # 分词处理
                candidate_tokens = generate_response.split()
                reference_tokens = [reference_answers[j].split()]
                
                # 计算BLEU分数
                if candidate_tokens and reference_tokens[0]:
                    bleu_score = sentence_bleu(reference_tokens, candidate_tokens, 
                                             smoothing_function=smoothing)
                    total_bleu += bleu_score
                total_pairs += 1
        
        score = (total_bleu / total_pairs) * 100 if total_pairs > 0 else 0
        return score
    
    elif scoring_criteria == "rouge-l":
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        total_rouge_l = 0
        total_pairs = 0
        
        for i, generate_responses in enumerate(generate_response_list):
            if i >= len(reference_answer_list):
                continue
            reference_answers = reference_answer_list[i]
            # 确保reference_answers是列表
            if not isinstance(reference_answers, list):
                reference_answers = [reference_answers]
                
            for j, generate_response in enumerate(generate_responses):
                if j >= len(reference_answers):
                    continue
                if generate_response.strip() and reference_answers[j].strip():
                    try:
                        scores = scorer.score(reference_answers[j], generate_response)
                        rouge_l_f = scores['rougeL'].fmeasure
                        total_rouge_l += rouge_l_f
                    except:
                        # 如果计算失败，该对得0分
                        pass
                total_pairs += 1
        
        score = (total_rouge_l / total_pairs) * 100 if total_pairs > 0 else 0
        return score
    
    elif scoring_criteria == "f1":
        def compute_f1(prediction: str, ground_truth: str) -> float:
            pred_tokens = prediction.lower().split()
            gt_tokens = ground_truth.lower().split()
            
            if len(pred_tokens) == 0 and len(gt_tokens) == 0:
                return 1.0
            if len(pred_tokens) == 0 or len(gt_tokens) == 0:
                return 0.0
            
            pred_counter = Counter(pred_tokens)
            gt_counter = Counter(gt_tokens)
            
            # 计算交集
            common = pred_counter & gt_counter
            num_same = sum(common.values())
            
            if num_same == 0:
                return 0.0
            
            precision = num_same / len(pred_tokens)
            recall = num_same / len(gt_tokens)
            f1 = 2 * precision * recall / (precision + recall)
            return f1
        
        total_f1 = 0
        total_pairs = 0
        
        for i, generate_responses in enumerate(generate_response_list):
            if i >= len(reference_answer_list):
                continue
            reference_answers = reference_answer_list[i]
            # 确保reference_answers是列表
            if not isinstance(reference_answers, list):
                reference_answers = [reference_answers]
                
            for j, generate_response in enumerate(generate_responses):
                if j >= len(reference_answers):
                    continue
                f1_score = compute_f1(generate_response, reference_answers[j])
                total_f1 += f1_score
                total_pairs += 1
        
        score = (total_f1 / total_pairs) * 100 if total_pairs > 0 else 0
        return score
    
    elif scoring_criteria == "llmjudge":
        pass
    #     # 从 kwargs 获取 LLM Judge 需要的参数
    #     language = kwargs.get('language', 'en')
    #     background = kwargs.get('background')
    #     question_list = kwargs.get('question_list', [])
    #     max_score = kwargs.get('max_score', 10)
    #     judge_prompt = kwargs.get('judge_prompt')
    #     judge_prompt_with_reference = kwargs.get('judge_prompt_with_reference')
        
    #     # LLM Judge评分逻辑 - 并行处理
    #     def process_single_judge(args):
    #         """处理单个评判任务"""
    #         i, j, generate_response, reference_answers, question, system_judge_prompt, evaluate_prompt = args
    #         evaluate_model = "gpt-4o"  # 默认使用 gpt-4o 作为评判模型
            
    #         # 创建评判API
    #         load_dotenv()
    #         evaluate_chat = ConversationAPI(
    #             model_name=evaluate_model,
    #             system_prompt=system_judge_prompt,
    #             user_prompt=evaluate_prompt,
    #             temperature=0.7,
    #             conversation_id=f"JudgeAgent_{i}_{j}",
    #             model_key=os.getenv("MODEL_KEY"),
    #             api_base="https://api.huatuogpt.cn/v1"
    #         )
            
    #         try:
    #             # 获取评判结果
    #             evaluate_response = evaluate_chat.generate_response()
    #             score = extract_scores(evaluate_response, max_score)
    #             return score if score is not None and score > 0 else 0
    #         except Exception as e:
    #             return 0
        
    #     # 准备并行任务
    #     judge_tasks = []
    #     for i, generate_responses in enumerate(generate_response_list):
    #         if i >= len(reference_answer_list) if reference_answer_list else len(generate_response_list):
    #             continue
                
    #         reference_answers = reference_answer_list[i] if reference_answer_list else None
            
    #         # 确保generate_responses是列表
    #         if not isinstance(generate_responses, list):
    #             generate_responses = [generate_responses]
                
    #         for j, generate_response in enumerate(generate_responses):
    #             # 构建评判prompt
    #             system_judge_prompt = DEFAULT_JUDGE_SYSTEM_PROMPT_WITH_GIVEN_ZH if language == "zh" else DEFAULT_JUDGE_SYSTEM_PROMPT_WITH_GIVEN_EN
                
    #             # 添加背景信息
    #             if background:
    #                 if language == "zh":
    #                     system_judge_prompt = system_judge_prompt + f"\n任务背景：{background}"
    #                 else:
    #                     system_judge_prompt = system_judge_prompt + f"\nBackground: {background}"
                
    #             # 获取问题文本
    #             if i < len(question_list):
    #                 question = question_list[i][j] if isinstance(question_list[i], list) and j < len(question_list[i]) else question_list[i] if isinstance(question_list[i], str) else ""
    #             else:
    #                 question = ""
                
    #             # 选择合适的评判prompt模板
    #             if reference_answers and judge_prompt_with_reference:
    #                 system_judge_prompt = system_judge_prompt + f"\n{judge_prompt_with_reference}"
    #                 # 构建包含参考答案的评判内容
    #                 ref_answer = reference_answers[j] if isinstance(reference_answers, list) and j < len(reference_answers) else reference_answers
    #                 if language == "zh":
    #                     evaluate_prompt = f"问题：{question}\n\n模型回答：{generate_response}\n\n参考答案：{ref_answer}"
    #                 else:
    #                     evaluate_prompt = f"Question: {question}\n\nModel Response: {generate_response}\n\nReference Answer: {ref_answer}"
    #             elif judge_prompt:
    #                 system_judge_prompt = system_judge_prompt + f"\n{judge_prompt}"
    #                 # 构建不含参考答案的评判内容
    #                 if language == "zh":
    #                     evaluate_prompt = f"问题：{question}\n\n模型回答：{generate_response}"
    #                 else:
    #                     evaluate_prompt = f"Question: {question}\n\nModel Response: {generate_response}"
    #             else:
    #                 # 使用默认评判方式
    #                 if language == "zh":
    #                     evaluate_prompt = f"问题：{question}\n\n模型回答：{generate_response}"
    #                 else:
    #                     evaluate_prompt = f"Question: {question}\n\nModel Response: {generate_response}"
                
    #             judge_tasks.append((i, j, generate_response, reference_answers, question, system_judge_prompt, evaluate_prompt))
        
    #     # 并行执行评判任务
    #     total_scores = 0
    #     total_pairs = len(judge_tasks)
        
    #     if total_pairs > 0:
    #         with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    #             futures = [executor.submit(process_single_judge, task) for task in judge_tasks]
    #             for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="LLM Judge评分中"):
    #                 score = future.result()
    #                 total_scores += score
        
    #     # 计算平均分并转换为百分制
    #     raw_score = (total_scores / total_pairs) if total_pairs > 0 else 0
    #     score = (raw_score / max_score) * 100 if max_score > 0 else 0
    #     return score
    
    else:
        raise ValueError(f"Unsupported scoring criteria: {scoring_criteria}")





def process_question(args):
    question_idx, language, background, input_data, temperature, dataset_name, model_name, model_key, api_base, scoring_criteria, judge_prompt, judge_prompt_with_reference, max_score = args
    
    case_list, case_template = input_data['case'] if input_data['case'] else (None, None)
    question_list, question_template = input_data['question']
    reference_answer_list, reference_answer_template = input_data['reference_answer'] if input_data['reference_answer'] else (None, None)
    

    system_prompt = DEFAULT_GENERATE_SYSTEM_PROMPT_ZH if language == "zh" else DEFAULT_GENERATE_SYSTEM_PROMPT_EN
    question_prompt = ""

    if background is not None and background != "":
        if language == "zh":
            system_prompt = system_prompt + f"\n任务背景：{background}"
        else:
            system_prompt = system_prompt + f"\nBackground: {background}"
    
    generate_responses = []
    generate_chat = ConversationAPI(
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=question_prompt,
        temperature=temperature,
        conversation_id=f"GenerateAgent_{question_idx}",
        model_key=model_key,
        api_base=api_base
    )
    if isinstance(question_list, str):
        question_list = [question_list]
    
    for i, question in enumerate(question_list):
        case_prompt = ""
        question_prompt = ""
        reference_answer_prompt = ""
        
        # 使用jinja2处理case prompt
        
        # 处理case_list
        case_data = None
        case_prompt = ""
        
        if isinstance(case_list, str):
            case_data = case_list  # 对所有question都使用同一个case
        elif case_list and i < len(case_list):
            case_data = case_list[i]
            
        if case_data:
            
            if case_template:
                # 将{variable}格式转换为{{variable}}格式
                jinja_template = re.sub(r'{([^{}]+)}', lambda m: '{{' + m.group(1) + '}}', case_template)
                template = Template(jinja_template)
                
                if isinstance(case_data, list):
                    # 对于多键数据如DxBench，按顺序映射到模板占位符
                    var_names = re.findall(r'{([^{}]+)}', case_template)
                    case_context = {}
                    for j, var_name in enumerate(var_names):
                        if j < len(case_data):
                            case_context[var_name] = case_data[j]
                    case_prompt = template.render(**case_context)
                elif isinstance(case_data, dict):
                    case_prompt = template.render(**case_data)
                else:
                    # 单个值，获取第一个变量名
                    var_names = re.findall(r'{([^{}]+)}', case_template)
                    if var_names:
                        case_prompt = template.render(**{var_names[0]: case_data})
            else:
                case_prompt = str(case_data) if case_data else ""
            
        # 使用jinja2处理question prompt
        
        if question and question_template:
            jinja_template = re.sub(r'{([^{}]+)}', lambda m: '{{' + m.group(1) + '}}', question_template)
            template = Template(jinja_template)
            if isinstance(question, list):
                var_names = re.findall(r'{([^{}]+)}', question_template)
                question_context = {}
                for j, var_name in enumerate(var_names):
                    if j < len(question):
                        question_context[var_name] = question[j]
                question_prompt = template.render(**question_context)
            elif isinstance(question, dict):
                question_prompt = template.render(**question)
            else:
                var_names = re.findall(r'{([^{}]+)}', question_template)
                if var_names:
                    question_prompt = template.render(**{var_names[0]: question})
        else:
            question_prompt = str(question) if question else ""
        
        # 使用jinja2处理reference answer prompt
        # if reference_answer_list and i < len(reference_answer_list):
        #     ref_data = reference_answer_list[i]
        #     if reference_answer_template:
        #         jinja_template = re.sub(r'{([^{}]+)}', lambda m: '{{' + m.group(1) + '}}', reference_answer_template)
        #         template = Template(jinja_template)
        #         if isinstance(ref_data, list):
        #             var_names = re.findall(r'{([^{}]+)}', reference_answer_template)
        #             ref_context = {}
        #             for j, var_name in enumerate(var_names):
        #                 if j < len(ref_data):
        #                     ref_context[var_name] = ref_data[j]
        #             reference_answer_prompt = template.render(**ref_context)
        #         elif isinstance(ref_data, dict):
        #             reference_answer_prompt = template.render(**ref_data)
        #         else:
        #             var_names = re.findall(r'{([^{}]+)}', reference_answer_template)
        #             if var_names:
        #                 reference_answer_prompt = template.render(**{var_names[0]: ref_data})
        #     else:
        #         reference_answer_prompt = str(ref_data) if ref_data else ""
        
        # 组合完整的prompt
        full_prompt = ""
        if 'case_prompt' in locals() and case_prompt:
            full_prompt += case_prompt + "\n\n"
        if question_prompt:
            full_prompt += question_prompt
        
        # 生成回答
        try:
            # 更新prompt并生成响应
            generate_chat.update_prompt(full_prompt.strip())
            
            response = generate_chat.generate_response()
            
            generate_responses.append(response)
        except Exception as e:
            generate_responses.append("")
    
    # LLM Judge 评分逻辑
    if scoring_criteria == "llmjudge":
        def process_single_judge_with_response(args):
            """处理单个评判任务，返回评分和评判响应"""
            question_idx_inner, i, generate_response, reference_answer, question, system_judge_prompt, evaluate_prompt, max_score_inner = args
            evaluate_model = "gpt-4o"  # 默认使用 gpt-4o 作为评判模型
            
            # 创建评判API
            load_dotenv()
            evaluate_chat = ConversationAPI(
                model_name=evaluate_model,
                system_prompt=system_judge_prompt,
                user_prompt=evaluate_prompt,
                temperature=0.7,
                conversation_id=f"JudgeAgent_{question_idx_inner}_{i}",
                model_key=model_key,
                api_base="https://api.huatuogpt.cn/v1"
            )
            
            try:
                # 获取评判结果
                evaluate_response = evaluate_chat.generate_response()
                
                # 从数据集配置获取 max_score，这里需要传递正确的值
                score = extract_scores(evaluate_response, max_score_inner)
                
                return (score if score is not None and score > 0 else 0, evaluate_response)
            except Exception as e:
                return (0, f"评判失败: {str(e)}")
        
        # 准备并行评判任务 
        judge_tasks = []
        # 从函数参数中获取 max_score
        max_score_param = max_score
        
        for i, generate_response in enumerate(generate_responses):
            # 构建评判prompt
            system_judge_prompt = DEFAULT_JUDGE_SYSTEM_PROMPT_WITH_GIVEN_ZH if language == "zh" else DEFAULT_JUDGE_SYSTEM_PROMPT_WITH_GIVEN_EN
            
            # 添加背景信息
            if background:
                if language == "zh":
                    system_judge_prompt = system_judge_prompt + f"\n任务背景：{background}"
                else:
                    system_judge_prompt = system_judge_prompt + f"\nBackground: {background}"
            
            # 选择合适的评判prompt模板
            if reference_answer_list and judge_prompt_with_reference:
                system_judge_prompt = system_judge_prompt + f"\n{judge_prompt_with_reference}"
                # 构建包含参考答案的评判内容
                ref_answer = reference_answer_list[i] if isinstance(reference_answer_list, list) and i < len(reference_answer_list) else reference_answer_list
                if language == "zh":
                    evaluate_prompt = f"问题：{question_prompt}\n\n模型回答：{generate_response}\n\n参考答案：{ref_answer}"
                else:
                    evaluate_prompt = f"Question: {question_prompt}\n\nModel Response: {generate_response}\n\nReference Answer: {ref_answer}"
            elif judge_prompt:
                system_judge_prompt = system_judge_prompt + f"\n{judge_prompt}"
                # 构建不含参考答案的评判内容
                if language == "zh":
                    evaluate_prompt = f"问题：{question_prompt}\n\n模型回答：{generate_response}"
                else:
                    evaluate_prompt = f"Question: {question_prompt}\n\nModel Response: {generate_response}"
            else:
                # 使用默认评判方式
                if language == "zh":
                    evaluate_prompt = f"问题：{question_prompt}\n\n模型回答：{generate_response}"
                else:
                    evaluate_prompt = f"Question: {question_prompt}\n\nModel Response: {generate_response}"
            
            judge_tasks.append((question_idx, i, generate_response, reference_answer_list, question_prompt, system_judge_prompt, evaluate_prompt, max_score_param))
        
        # 并行执行评判任务
        judge_responses = []
        scores = []
        
        if judge_tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  # 降低并发数到1
                futures = [executor.submit(process_single_judge_with_response, task) for task in judge_tasks]
                for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                    score, judge_response = future.result()
                    scores.append(score)
                    judge_responses.append(judge_response)
        
        # 计算平均分
        avg_score = sum(scores) / len(scores) if scores else -1
        
        return question_idx, reference_answer_list, generate_responses, judge_responses, avg_score
    
    # 返回结果
    return question_idx, reference_answer_list, generate_responses

    

def evaluate_qa_automatic(
    user_id: str = "test",
    dataset_name: str = "MT-Bench",
    model_name: str = "gpt-3.5-turbo",
    model_key: str = "",
    api_base: str = "",
    question_limitation: int = 100,
    max_workers: int = 64,
    business_id: str = None
):
    dataset = QA(dataset_name)
    language = dataset.language
    max_score = dataset.max_score
    background = dataset.background
    cases = dataset.case
    questions = dataset.question
    reference_answers = dataset.reference_answer
    scoring_criteria = dataset.scoring_criteria
    judge_prompt = dataset.judge_prompt
    judge_prompt_with_reference = dataset.judge_prompt_with_reference

    if question_limitation >= len(questions['data']):
        question_limitation = len(questions['data'])
    
    if business_id is None:
        business_id = generate_business_id(dataset_name, model_name)
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

    # 初始化数据库
    initialize_database()

    result_file = f"results/{user_id}/{business_id}_result.json"
    existing_results = read_json_file(result_file, business_id)

    if not existing_results:
        if scoring_criteria == "llmjudge":
            existing_results = [{"id": i, "reference_answer": "None", "generate_response": "Neglected", "judge_response": "Neglected", "score": -1} for i in range(question_limitation)]
        else:
            existing_results = [{"id": i, "reference_answer": "None", "generate_response": "Neglected"} for i in range(question_limitation)]
        write_json_file(existing_results, result_file, business_id, dataset_name, model_name)
    else:
        current_length = len(existing_results)
        if current_length < question_limitation:
            if scoring_criteria == "llmjudge":
                for i in range(current_length, question_limitation):
                    existing_results.append({"id": i, "reference_answer": "None", "generate_response": "Neglected", "judge_response": "Neglected", "score": -1})
            else:
                for i in range(current_length, question_limitation):
                    existing_results.append({"id": i, "reference_answer": "None", "generate_response": "Neglected"})
            write_json_file(existing_results, result_file, business_id, dataset_name, model_name)
    
    args_list = []
    for i in range(question_limitation):
        if scoring_criteria == "llmjudge":
            if existing_results[i]['score'] >= 0 and existing_results[i]['score'] <= max_score:
                continue
        else:
            if existing_results[i]['generate_response'] != "Neglected":
                continue
        case = cases['data'][i] if cases is not None and 'data' in cases and i < len(cases['data']) else None
        question = questions['data'][i] if questions is not None and 'data' in questions and i < len(questions['data']) else None
        reference_answer = reference_answers['data'][i] if reference_answers is not None and 'data' in reference_answers and i < len(reference_answers['data']) else None
        temperature = 0 if reference_answer is not None else 0.7
        input_data = {
            'case': (case, cases['prompt_template']) if cases else None,
            'question': (question, questions['prompt_template']) if questions else None,
            'reference_answer': (reference_answer, reference_answers['prompt_template']) if reference_answers else None
        }
        args_list.append((i, language, background, input_data, temperature, dataset_name, model_name, model_key, api_base, scoring_criteria, judge_prompt, judge_prompt_with_reference, max_score))
    generate_response_list = []
    valid_questions = 0
    total_questions = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_question, args): args[0] for args in args_list}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"评测中"):
            result = future.result()
            if scoring_criteria == "llmjudge" and len(result) == 5:
                # LLM Judge 返回: idx, reference_answer_list, generate_responses, judge_responses, avg_score
                idx, reference_answer_list, generate_responses, judge_responses, avg_score = result
                existing_results[idx]['reference_answer'] = reference_answer_list
                existing_results[idx]['generate_response'] = generate_responses
                existing_results[idx]['judge_response'] = judge_responses
                existing_results[idx]['score'] = avg_score
            else:
                # 其他评分标准返回: idx, reference_answer_list, generate_responses
                idx, reference_answer_list, generate_responses = result
                existing_results[idx]['reference_answer'] = reference_answer_list
                existing_results[idx]['generate_response'] = generate_responses
            
            write_json_file(existing_results, result_file, business_id, dataset_name, model_name)
            generate_response_list.append(generate_responses)
            if "Neglected" not in generate_responses:
                valid_questions += 1
            total_questions += 1
    if total_questions > 0 and valid_questions / total_questions >= 0.9 and scoring_criteria != "llmjudge":

        构建reference_answer_list用于scoring函数
        reference_answer_list = []
        for result in existing_results:
            if result['reference_answer'] != "None":
                reference_answer_list.append(result['reference_answer'])
        

            
            score = scoring(
                generate_response_list, 
                reference_answer_list, 
                scoring_criteria,
                language=language,
                background=background,
                question_list=question_list_for_scoring,
                max_score=max_score,
                judge_prompt=judge_prompt,
                judge_prompt_with_reference=judge_prompt_with_reference
            )
            print("score:", score)

    else:
        score = 0
    
    # 如果是 LLM Judge 评分，检查是否需要生成分数摘要
    if scoring_criteria == "llmjudge" and total_questions > 0:
        score_file = f"results/{user_id}/{business_id}_score.json"
        summary_data = generate_score_summary(existing_results, score_file, completion_threshold=0.9, max_score=max_score, business_id=business_id, dataset_name=dataset_name, model_name=model_name)
        if summary_data:
            score = summary_data['score']  # 使用摘要中计算的分数
            # 同时保存到数据库
            if business_id and dataset_name and model_name:
                save_evaluation_score(business_id, dataset_name, model_name, score)
    
    return score
    
if __name__ == "__main__":
    score = evaluate_qa_automatic(
        user_id="test",
        dataset_name="MT-Bench",
        model_name="doubao-1.5-pro-32k",
        model_key="sk-fPz5uPZn2ubb9Qexx62yWcFl55Z46iRdBfdlvnjufQ6o0BVo",
        api_base="https://api.huatuogpt.cn/v1",
        question_limitation=10,
        max_workers=64
    )
    
