import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# 导入评估模块
from evaluation.TextMCQ_eval import evaluate_mcq
from evaluation.ImageMCQ_eval import evaluate_imagemcq
from evaluation.LLMJudge_eval import evaluate_llmjudge





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=False, default="knowledgeQA_2500",
                        help="数据集名称，例如MMLU、GPQA等")
    parser.add_argument("--model", type=str, required=False, default="gpt-3.5-turbo",
                        help="准备进行评测的模型名称")
    parser.add_argument("--judgment_model", type=str, required=False, default="gpt-4o",
                        help="LLMJudge中用于评判的模型名称")
    parser.add_argument("--workers", type=int, required=False, default=64,
                        help="并行处理的工作线程数量")
    parser.add_argument("--evaluate_mode", type=str, required=False, default="start_from_beginning")
    parser.add_argument("--question_limitation", type=int, required=False, default=100,
                        help="评测的问题数量")
    return parser.parse_args()

def main():
    # 加载环境变量
    load_dotenv()
    
    # 解析命令行参数
    args = parse_args()
    
    print(f"评估数据集: {args.dataset}")
    print(f"使用模型: {args.model}")
    
    # 根据数据集类型选择评估函数
    # 文本多选题数据集
    text_mcq_datasets = ["MMLU", "CMB", "GPQA", "knowledgeQA_2500"]
    # 图像多选题数据集
    image_mcq_datasets = ["MMStar"]
    # LLMJudge数据集
    llm_judge_datasets = ["MT-Bench"]

    
    import json
    results = {}
    accuracy_result = {}
    # 调用相应的评估函数
    if args.dataset in text_mcq_datasets:
        print(f"执行文本多选题评估，并行工作线程数: {args.workers}")
        responses, accuracy = evaluate_mcq(args.dataset, args.model, max_workers=args.workers, evaluate_mode=args.evaluate_mode, question_limitation=args.question_limitation)
    elif args.dataset in image_mcq_datasets:
        print(f"执行图像多选题评估，并行工作线程数: {args.workers}")
        responses, accuracy = evaluate_imagemcq(args.dataset, args.model, max_workers=args.workers, evaluate_mode=args.evaluate_mode, question_limitation=args.question_limitation)
    elif args.dataset in llm_judge_datasets:
        responses = evaluate_llmjudge(args.dataset, args.model, args.judgment_model, evaluate_mode=args.evaluate_mode, question_limitation=args.question_limitation)
        
            
    else:
        print(f"不存在/尚未支持的数据集类型: {args.dataset}")
        return


if __name__ == "__main__":
    main()





