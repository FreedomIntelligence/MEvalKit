"""
MEvalKit 命令行评测入口文件

该文件提供了命令行界面来运行模型评测任务，支持自动模式和手动模式两种评测方式。
自动模式：实时调用模型API进行评测
手动模式：使用预生成的响应文件进行评测

作者: MEvalKit Team
版本: 1.0.0
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# 导入评估模块
from evaluation.MCQ_eval import evaluate_all_mcq_automatic
from evaluation.QA_eval import evaluate_qa_automatic

from src.utils.model_and_dataset import *


def parse_evaluation_mode():
    """
    第一层参数解析：解析evaluation_mode参数
    
    该函数使用两层参数解析机制，首先解析evaluation_mode参数，
    然后根据模式解析相应的参数。这种设计允许不同模式有不同的参数集。
    
    返回:
        tuple: (evaluation_mode, unknown_args)
            - evaluation_mode: 评测模式 ("automatic" 或 "manual")
            - unknown_args: 剩余未解析的参数列表
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--evaluation_mode", type=str, required=False, default="automatic",
                        help="评估模式：automatic（自动模式）或manual（手动模式）")
    # 解析已知参数，忽略未知参数
    args, unknown = parser.parse_known_args()
    return args.evaluation_mode, unknown


def parse_automatic_args(unknown_args):
    """
    第二层参数解析：automatic模式的参数
    
    自动模式需要实时调用模型API，因此需要API密钥、接口地址等参数。
    
    参数:
        unknown_args: 从第一层解析中剩余的未知参数列表
        
    返回:
        argparse.Namespace: 解析后的参数对象，包含以下字段：
            - user_id: 用户ID
            - dataset: 数据集名称
            - model_name: 模型名称
            - api_base: API接口地址
            - model_key: API密钥
            - business_id: 业务ID
            - question_limitation: 评测问题数量限制
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", type=str, required=False, default="test",
                        help="用户id")
    parser.add_argument("--dataset", type=str, required=False, default="GPQA",
                        help="数据集名称，例如MMLU、GPQA等")
    # parser.add_argument("--model_type", type=str, required=False, default="default",
    #                     help="模型类型，例如vllm、openai等")
    parser.add_argument("--model_name", type=str, required=False, default="gpt-oss-20b",
                        help="准备进行评测的模型名称")
    parser.add_argument("--api_base", type=str, required=False, default="http://aistation.sribd.cn:30001/v1",
                        help="API接口地址")
    parser.add_argument("--model_key", type=str, required=False, default="",
                        help="模型key")
    parser.add_argument("--business_id", type=str, required=False, default=None,
                        help="业务id，如果不提供则自动生成新的business_id")
    # parser.add_argument("--evaluate_mode", type=str, required=False, default="start_from_beginning")
    parser.add_argument("--question_limitation", type=int, required=False, default=100,
                        help="评测的问题数量")

    return parser.parse_args(unknown_args)


def parse_manual_args(unknown_args):
    """
    第二层参数解析：manual模式的参数
    
    手动模式使用预生成的响应文件，因此需要响应文件URL等参数。
    
    参数:
        unknown_args: 从第一层解析中剩余的未知参数列表
        
    返回:
        argparse.Namespace: 解析后的参数对象，包含以下字段：
            - user_id: 用户ID
            - dataset: 数据集名称
            - model_name: 模型名称
            - business_id: 业务ID
            - question_limitation: 评测问题数量限制
            - response_url: 响应文件URL
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", type=str, required=False, default="test",
                        help="用户id")
    parser.add_argument("--dataset", type=str, required=False, default="MedEthicsMatrixMCQ",
                        help="数据集名称，例如MMStar等")
    parser.add_argument("--model_name", type=str, required=False, default="我的测试1",
                        help="准备进行评测的模型名称")
    parser.add_argument("--business_id", type=str, required=False, default=None,
                        help="业务id，如果不提供则自动生成新的business_id")
    parser.add_argument("--question_limitation", type=int, required=False, default=100,
                        help="评测的问题数量")
    parser.add_argument("--response_url", type=str, required=False, default="http://47.110.252.218:1995/admin-api/infra/file/31/get/evaluation/answer/20250721/mcq_1753066731583.json",
                        help="响应url")
    return parser.parse_args(unknown_args)


def main():
    """
    主函数：程序入口点
    
    该函数实现了两层参数解析机制：
    1. 首先解析evaluation_mode参数
    2. 根据模式解析相应的参数
    3. 调用相应的评测函数
    
    支持的模式：
    - automatic: 自动模式，实时调用API
    - manual: 手动模式，使用预生成响应
    """
    # 加载环境变量
    load_dotenv()
    
    # 第一层：解析evaluation_mode
    evaluation_mode, unknown_args = parse_evaluation_mode()
    
    print(f"评估模式: {evaluation_mode}")
    
    # 第二层：根据模式解析相应参数
    if evaluation_mode == "automatic":
        args = parse_automatic_args(unknown_args)
        print(f"评估数据集: {args.dataset}")
        print(f"使用模型: {args.model_name}")
        
        import json
        results = {}
        accuracy_result = {}
        # 调用相应的评估函数
        # MCQ型数据集（多选题）
        if args.dataset in TEXT_DATASETS or args.dataset in MULTIMODAL_DATASETS:
            print(f"执行MCQ多选题评估，并行工作线程数: {64}")
            valid_ratio, score = evaluate_all_mcq_automatic(
                user_id=args.user_id,
                dataset_name=args.dataset,
                model_name=args.model_name,
                model_key=args.model_key,
                api_base=args.api_base,
                business_id=args.business_id,
                question_limitation=args.question_limitation,
                max_workers=32
            )
            results[args.dataset] = {
                "valid_ratio": valid_ratio,
                "score": score
            }
            accuracy_result[args.dataset] = score
        # QA型数据集（问答题）
        elif args.dataset in LLMJUDGE_DATASETS:
            print(f"执行QA问答题评估，并行工作线程数: {1}")
            score = evaluate_qa_automatic(
                user_id=args.user_id,
                dataset_name=args.dataset,
                model_name=args.model_name,
                model_key=args.model_key,
                api_base=args.api_base,
                business_id=args.business_id,
                question_limitation=args.question_limitation,
                max_workers=32
            )
            results[args.dataset] = {
                "score": score
            }
            accuracy_result[args.dataset] = score
        else:
            print(f"不支持的数据集: {args.dataset}")
            return
        
        # 输出结果
        print(f"评测结果: {json.dumps(results, indent=2, ensure_ascii=False)}")
        print(f"准确率结果: {json.dumps(accuracy_result, indent=2, ensure_ascii=False)}")
        
    elif evaluation_mode == "manual":
        args = parse_manual_args(unknown_args)
        print(f"评估数据集: {args.dataset}")
        print(f"使用模型: {args.model_name}")
        
        import json
        results = {}
        accuracy_result = {}
        print("手动模式暂不支持，请使用automatic模式")
        return
        
        # 输出结果
        print(f"评测结果: {json.dumps(results, indent=2, ensure_ascii=False)}")
        print(f"准确率结果: {json.dumps(accuracy_result, indent=2, ensure_ascii=False)}")
        
    else:
        print(f"不支持的评估模式: {evaluation_mode}")
        print("支持的评估模式: automatic, manual")


if __name__ == "__main__":
    main()





