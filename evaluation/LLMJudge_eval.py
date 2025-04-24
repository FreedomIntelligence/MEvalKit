import sys
import os
import json
import re
from pathlib import Path
from openai import BadRequestError
import concurrent.futures
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
print(project_root)
sys.path.append(str(project_root))

from src.dataset.LLMJudge.LLMJudgeBase import *
from src.api.text_api import *
from src.api.multiturn_text_api import *
from src.utils.LLMJudge_constants import *
from typing import List, Tuple, Dict, Any, Optional, Union
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
        print(f"数据已成功写入: {file_path}")
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

def extract_score(evaluate_response: str) -> Optional[Union[int, float]]:
    """从评估响应中提取分数"""
    if not evaluate_response or evaluate_response == "Error" or evaluate_response == "Neglected":
        return None
    
    # 定义多种可能的评分模式
    score_patterns = [
        r'评分[:：]\s*(\d+(?:\.\d+)?)',  # 中文格式
        r'[Ss]core[:：]\s*(\d+(?:\.\d+)?)',  # 英文格式
        r'(\d+(?:\.\d+)?)\s*/\s*\d+',  # X/10 格式
        r'评分为\s*(\d+(?:\.\d+)?)',  # "评分为X" 格式
        r'[Rr]ating[:：]?\s*(\d+(?:\.\d+)?)',  # Rating: X 格式
        r'[Gg]rade[:：]?\s*(\d+(?:\.\d+)?)',  # Grade: X 格式
        r'[Pp]oints[:：]?\s*(\d+(?:\.\d+)?)',  # Points: X 格式
        r'(\d+(?:\.\d+)?)\s*[Pp]oints',  # X Points 格式
        r'得分[:：]?\s*(\d+(?:\.\d+)?)',  # 得分: X 格式
        r'分数[:：]?\s*(\d+(?:\.\d+)?)'   # 分数: X 格式
    ]  
    
    # 尝试所有模式匹配
    for pattern in score_patterns:
        match = re.search(pattern, evaluate_response)
        if match:
            try:
                score = float(match.group(1))
                if score.is_integer():
                    return int(score)
                return score
            except ValueError:
                continue
    
    return None

def process_single_question(args):
    """
    处理单个问题的评估
    
    参数:
        args: 包含问题信息的元组
        
    返回:
        包含评估结果的字典
    """
    idx, question, reference_answer, generate_model_name, evaluate_model_name, temperature = args
    
    result = {
        "question": question,
        "reference_answer": reference_answer,
        "generate_response": "Neglected",
        "evaluate_response": "Neglected",
        "score": None
    }
    
    try:
        # 使用生成模型回答问题
        generate_chat = MultiturnTextAPI(generate_model_name, GENERATE_SYSTEM_PROMPT, question, temperature, f"GenerateAgent_{idx}")
        generate_response = generate_chat.generate_response()
        result["generate_response"] = generate_response
        
        try:
            # 使用评估模型评估回答
            if reference_answer is not None:
                evaluate_prompt = f"Question: {question}\n\nGenerate Response: {generate_response}\n\nReference Answer: {reference_answer}"
                evaluate_chat = TextAPI(evaluate_model_name, JUDGE_SYSTEM_PROMPT_REASONING, evaluate_prompt, 0.7)
            else:
                evaluate_prompt = f"Question: {question}\n\nGenerate Response: {generate_response}"
                evaluate_chat = TextAPI(evaluate_model_name, JUDGE_SYSTEM_PROMPT, evaluate_prompt, 0.7)
            
            evaluate_response = evaluate_chat.generate_response()
            result["evaluate_response"] = evaluate_response
            
            # 提取评分
            score = extract_score(evaluate_response)
            result["score"] = score
            
        except Exception as e:
            # 评估模型出错，但生成模型正常
            print(f"评估问题 {idx} 时出错: {str(e)}")
            result["evaluate_response"] = "Neglected"
            result["score"] = None
    
    except Exception as e:
        # 生成模型出错
        print(f"生成问题 {idx} 的回答时出错: {str(e)}")
        result["generate_response"] = "Neglected"
        result["evaluate_response"] = "Neglected"
        result["score"] = None
    
    return idx, result

def evaluate_llmjudge(dataset_name: str, generate_model_name: str, evaluate_model_name: str, 
                     max_workers: int = 4, evaluate_mode: str = "start_from_beginning"):
    """
    评估LLM Judge
    
    参数:
        dataset_name: 数据集名称
        generate_model_name: 生成回答的模型名称
        evaluate_model_name: 评估回答的模型名称
        max_workers: 并行处理的最大工作线程数
        evaluate_mode: 评估模式，"start_from_beginning"从头开始，"resume_from_checkpoint"从断点继续
        
    返回:
        评估结果列表
    """
    # 加载数据集
    dataset = LLMJudgeBase(dataset_name)
    
    # 创建结果目录
    result_dir = Path("results") / "LLMJudge"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置结果文件路径
    result_file = result_dir / f"{dataset_name}_{generate_model_name}_{evaluate_model_name}_result.json"
    score_file = result_dir / f"{dataset_name}_{generate_model_name}_{evaluate_model_name}_score.json"
    
    # 初始化或加载结果
    if evaluate_mode == "start_from_beginning" or not result_file.exists():
        # 从头开始评估：初始化所有问题的结果
        all_results = []
        for question_number in range(len(dataset.questions)):
            questions = dataset.questions[question_number]
            answers = dataset.answers[question_number]
            
            # 确保问题是列表格式
            if isinstance(questions, str):
                questions = [questions]
            
            # 初始化问题结果
            question_results = []
            for i, question in enumerate(questions):
                reference_answer = answers[i] if answers is not None else None
                question_results.append({
                    "question": question,
                    "reference_answer": reference_answer,
                    "generate_response": "Neglected",
                    "evaluate_response": "Neglected",
                    "score": None
                })
            
            all_results.append({
                "dataset_name": dataset_name,
                "generate_model_name": generate_model_name,
                "evaluate_model_name": evaluate_model_name,
                "question_number": question_number,
                "results": question_results
            })
        
        # 写入初始结果文件
        write_json_file(all_results, result_file)
    else:
        # 从断点处继续评估：加载现有结果
        all_results = read_json_file(result_file)
        if not all_results:
            # 如果文件存在但无法读取，则从头开始
            return evaluate_llmjudge(dataset_name, generate_model_name, evaluate_model_name, 
                                    max_workers, "start_from_beginning")
    
    # 处理每个问题集
    for question_number in range(len(dataset.questions)):
        questions = dataset.questions[question_number]
        answers = dataset.answers[question_number]
        
        # 确保问题是列表格式
        if isinstance(questions, str):
            questions = [questions]
        
        # 根据是否有参考答案设置温度参数
        temperature = 0 if answers is not None else 0.7
        
        # 准备参数列表
        args_list = []
        for i, question in enumerate(questions):
            # 获取当前问题的结果
            current_result = all_results[question_number]["results"][i]
            
            # 检查是否需要处理此问题
            if evaluate_mode == "resume_from_checkpoint":
                # 如果生成和评估都已完成，跳过
                if (current_result["generate_response"] != "Neglected" and 
                    current_result["evaluate_response"] != "Neglected"):
                    continue
                # 如果只有生成完成但评估未完成，只进行评估
                if (current_result["generate_response"] != "Neglected" and 
                    current_result["evaluate_response"] == "Neglected"):
                    # 这种情况在process_single_question中处理
                    pass
            
            reference_answer = answers[i] if answers is not None else None
            args_list.append((i, question, reference_answer, generate_model_name, evaluate_model_name, temperature))
        
        # 如果没有需要处理的问题，跳过此问题集
        if not args_list:
            print(f"问题集 {question_number} 已全部完成，跳过")
            continue
        
        # 并行处理问题
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_question, args): args[0] for args in args_list}
            
            # 使用tqdm显示进度
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), 
                              desc=f"处理问题集 {question_number}"):
                idx, result = future.result()
                
                # 更新结果
                all_results[question_number]["results"][idx] = result
                
                # 每完成一题就更新结果文件
                write_json_file(all_results, result_file)
    
    # 生成评分摘要
    generate_score_summary(all_results, score_file)
    
    return all_results

def generate_score_summary(all_results, score_file, completion_threshold=0.95):
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
    
    for question_set in all_results:
        for result in question_set["results"]:
            total_questions += 1
            if result["score"] is not None:
                valid_scores.append(result["score"])
    
    # 计算完成率
    completion_ratio = len(valid_scores) / total_questions if total_questions > 0 else 0
    
    # 如果完成率达到阈值，生成评分摘要
    if completion_ratio >= completion_threshold:
        # 计算统计数据
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
            max_score = max(valid_scores)
            min_score = min(valid_scores)
            
            # 创建评分分布
            score_distribution = {}
            for score in valid_scores:
                score_str = str(score)
                if score_str in score_distribution:
                    score_distribution[score_str] += 1
                else:
                    score_distribution[score_str] = 1
        else:
            avg_score = None
            max_score = None
            min_score = None
            score_distribution = {}
        
        # 创建摘要数据
        summary_data = {
            "dataset_name": all_results[0]["dataset_name"],
            "generate_model_name": all_results[0]["generate_model_name"],
            "evaluate_model_name": all_results[0]["evaluate_model_name"],
            "total_questions": total_questions,
            "valid_scores": len(valid_scores),
            "completion_ratio": completion_ratio,
            "average_score": avg_score,
            "max_score": max_score,
            "min_score": min_score,
            "score_distribution": score_distribution
        }
        
        # 写入摘要文件
        write_json_file(summary_data, score_file)
        print(f"评分摘要已生成: {score_file}")
    else:
        print(f"完成率 ({completion_ratio:.2%}) 未达到阈值 ({completion_threshold:.2%})，暂不生成评分摘要")

if __name__ == "__main__":
    # 加载环境变量
    load_dotenv()
    
    # 从头开始评估
    results = evaluate_llmjudge("MT-Bench", "gpt-3.5-turbo", "gpt-4o", max_workers=4, evaluate_mode="start_from_beginning")
    
    # 从断点处继续评估
    #results = evaluate_llmjudge("MT-Bench", "gpt-3.5-turbo", "gpt-4o", max_workers=4, evaluate_mode="resume_from_checkpoint")



