import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# 导入评估模块
from evaluation.TextMCQ_eval import evaluate_mcq_manual, evaluate_mcq_automatic
from evaluation.ImageMCQ_eval import evaluate_imagemcq_manual, evaluate_imagemcq_automatic
from evaluation.LLMJudge_eval import evaluate_llmjudge_automatic, evaluate_llmjudge_manual

from src.utils.model_and_dataset import *


def parse_evaluation_mode():
    """
    第一层参数解析：解析evaluation_mode参数
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
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", type=str, required=False, default="test",
                        help="用户id")
    parser.add_argument("--dataset", type=str, required=False, default="CMB",
                        help="数据集名称，例如MMLU、GPQA等")
    # parser.add_argument("--model_type", type=str, required=False, default="default",
    #                     help="模型类型，例如vllm、openai等")
    parser.add_argument("--model_name", type=str, required=False, default="gpt-4o",
                        help="准备进行评测的模型名称")
    parser.add_argument("--api_base", type=str, required=False, default="",
                        help="API接口地址")
    parser.add_argument("--model_key", type=str, required=False, default="",
                        help="模型key")
    parser.add_argument("--business_id", type=str, required=False, default="CMB_4o_a6769",
                        help="业务id")
    # parser.add_argument("--evaluate_mode", type=str, required=False, default="start_from_beginning")
    parser.add_argument("--question_limitation", type=int, required=False, default=100,
                        help="评测的问题数量")

    return parser.parse_args(unknown_args)


def parse_manual_args(unknown_args):
    """
    第二层参数解析：manual模式的参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", type=str, required=False, default="test",
                        help="用户id")
    parser.add_argument("--dataset", type=str, required=False, default="MMStar",
                        help="数据集名称，例如MMStar等")
    parser.add_argument("--model_name", type=str, required=False, default="gpt-4o",
                        help="准备进行评测的模型名称")
    parser.add_argument("--api_base", type=str, required=False, default="",
                        help="API接口地址")
    parser.add_argument("--model_key", type=str, required=False, default="",
                        help="模型key")
    parser.add_argument("--business_id", type=str, required=False, default="CMB_4o_a6769",
                        help="业务id")
    parser.add_argument("--question_limitation", type=int, required=False, default=100,
                        help="评测的问题数量")
    parser.add_argument("--response_url", type=str, required=False, default="",
                        help="响应url")
    return parser.parse_args(unknown_args)


def main():
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
        if args.dataset in TEXT_DATASETS:
            print(f"执行文本多选题评估，并行工作线程数: {1}")
            valid_ratio, score = evaluate_mcq_automatic(
                user_id=args.user_id,
                dataset_name=args.dataset,
                model_name=args.model_name,
                api_base=args.api_base,
                model_key=args.model_key,
                business_id=args.business_id,
                question_limitation=args.question_limitation,
                max_workers=64
            )
        elif args.dataset in MULTIMODAL_DATASETS:
            print(f"执行图像多选题评估，并行工作线程数: {1}")
            valid_ratio, score = evaluate_imagemcq_automatic(
                user_id=args.user_id,
                dataset_name=args.dataset,
                model_name=args.model_name,
                api_base=args.api_base,
                model_key=args.model_key,
                business_id=args.business_id,
                question_limitation=args.question_limitation,
                max_workers=64
            )
        elif args.dataset in LLMJUDGE_DATASETS:
            results = evaluate_llmjudge_automatic(
                args.user_id,
                args.dataset,
                args.model_name,
                args.model_key,
                args.api_base,
                max_workers=64,
                question_limitation=args.question_limitation,
                business_id=args.business_id
            )
        else:
            print(f"不存在/尚未支持的数据集类型: {args.dataset}")
            return
            
    else:  # manual模式
        args = parse_manual_args(unknown_args)
        print(f"评估数据集: {args.dataset}")
        
        # 调用manual模式的评估函数
        if args.dataset in TEXT_DATASETS:
            print(f"执行文本多选题手动评估")
            valid_ratio, score = evaluate_mcq_manual(
                user_id=args.user_id,
                dataset_name=args.dataset,
                model_name=args.model_name,
                business_id=args.business_id,
                question_limitation=args.question_limitation,
                response_url=args.response_url
            )
        elif args.dataset in MULTIMODAL_DATASETS:
            print(f"执行图像多选题手动评估")
            valid_ratio, score = evaluate_imagemcq_manual(
                user_id=args.user_id,
                dataset_name=args.dataset,
                model_name=args.model_name,
                business_id=args.business_id,
                question_limitation=args.question_limitation,
                response_url=args.response_url
            )
        elif args.dataset in LLMJUDGE_DATASETS:
            print(f"执行LLMJudge手动评估")
            final_results = evaluate_llmjudge_manual(
                user_id=args.user_id,
                dataset_name=args.dataset,
                model_name=args.model_name,
                business_id=args.business_id,
                question_limitation=args.question_limitation,
                response_url=args.response_url,
                model_key=args.model_key,
                api_base=args.api_base,
                max_workers=64
            )
        else:
            print(f"不存在/尚未支持的数据集类型: {args.dataset}")
            return


if __name__ == "__main__":
    main()





