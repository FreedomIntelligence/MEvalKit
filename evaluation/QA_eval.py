import sys
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from evaluation.QA_response_generator import QA_answer_generator
from evaluation.QA_scorer import LLMJudge_scorer, Accuracy_scorer, Rubric_scorer
from src.dataset.QA import QA
from src.utils.config import config

QA_SCORER_CONFIG_PATH = "dataset_info/QA_scorer_config.yaml"

def load_scorer_config():
    """加载QA评分器配置"""
    with open(QA_SCORER_CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class QA_evaluator:
    def __init__(self,
                 user_id: str,
                 dataset_name: str,
                 model_name: str,
                 model_key: str,
                 api_base: str,
                 business_id: str = None,
                 question_limitation: int = None,
                 max_workers: int = 4,
                 judge_model: str = None,
                 judge_key: str = None,
                 judge_api_base: str = None):
        
        self.user_id = user_id
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.model_key = model_key
        self.api_base = api_base
        self.business_id = business_id
        self.question_limitation = question_limitation
        self.max_workers = max_workers
        
        # 加载数据集和配置
        self.dataset = QA(dataset_name)
        self.scorer_config = load_scorer_config()
        
        # 获取评分标准
        self.criteria = self.get_scoring_criteria()
        print(f"数据集 {dataset_name} 使用评分标准: {self.criteria}")
        
        # 设置评分模型
        if self.criteria in ['llmjudge', 'rubrics']:
            self.judge_model = judge_model or self.scorer_config.get(dataset_name, {}).get('judge_model', 'gpt-4o')
            self.judge_key = judge_key or model_key
            self.judge_api_base = judge_api_base or api_base
        else:
            self.judge_model = None
            self.judge_key = None
            self.judge_api_base = None
    
    def get_scoring_criteria(self) -> str:
        """根据数据集配置获取评分标准"""
        if self.dataset_name in self.scorer_config:
            return self.scorer_config[self.dataset_name].get('criteria', 'accuracy')
        else:
            # 如果数据集不在scorer配置中，检查是否有scoring_criteria字段
            if hasattr(self.dataset, 'scoring_criteria') and self.dataset.scoring_criteria:
                return self.dataset.scoring_criteria
            else:
                print(f"警告: 数据集 {self.dataset_name} 未找到评分标准配置，使用默认的 'accuracy'")
                return 'accuracy'
    
    def generate_responses(self) -> str:
        """生成模型响应"""
        print(f"开始生成响应 - 数据集: {self.dataset_name}, 模型: {self.model_name}")
        
        generator = QA_answer_generator(
            user_id=self.user_id,
            dataset_name=self.dataset_name,
            model_name=self.model_name,
            model_key=self.model_key,
            api_base=self.api_base,
            business_id=self.business_id,
            question_limitation=self.question_limitation,
            max_workers=self.max_workers
        )
        
        business_id = generator.generate_responses()
        self.business_id = business_id
        return business_id
    
    def score_responses(self) -> str:
        """对响应进行评分"""
        if not self.business_id:
            raise ValueError("需要先生成响应或提供business_id")
        
        print(f"开始评分 - 评分标准: {self.criteria}")
        
        if self.criteria == 'accuracy':
            scorer = Accuracy_scorer(
                dataset_name=self.dataset_name,
                user_id=self.user_id,
                business_id=self.business_id
            )
        elif self.criteria == 'llmjudge':
            scorer = LLMJudge_scorer(
                dataset_name=self.dataset_name,
                user_id=self.user_id,
                business_id=self.business_id,
                # judge_model=self.judge_model,
                # judge_key=self.judge_key,
                # judge_api_base=self.judge_api_base,
                # max_workers=self.max_workers
            )
        elif self.criteria == 'rubrics':
            scorer = Rubric_scorer(
                dataset_name=self.dataset_name,
                user_id=self.user_id,
                business_id=self.business_id,
                # judge_model=self.judge_model,
                # judge_key=self.judge_key,
                # judge_api_base=self.judge_api_base,
                # max_workers=self.max_workers
            )
        else:
            raise ValueError(f"不支持的评分标准: {self.criteria}")
        
        return scorer.scoring()
    
    def run_full_evaluation(self) -> Dict[str, str]:
        """运行完整的评估流程：生成响应 -> 评分"""
        print(f"开始完整评估流程")
        print(f"数据集: {self.dataset_name}")
        print(f"模型: {self.model_name}")
        print(f"评分标准: {self.criteria}")
        print(f"问题数量限制: {self.question_limitation}")
        print("-" * 50)
        
        # 步骤1: 生成响应
        business_id = self.generate_responses()
        
        # 步骤2: 评分响应
        score_business_id = self.score_responses()
        
        result = {
            "response_business_id": business_id,
            "score_business_id": score_business_id,
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "criteria": self.criteria,
            "question_limitation": self.question_limitation
        }
        
        print(f"完整评估完成！")
        print(f"响应结果ID: {business_id}")
        print(f"评分结果ID: {score_business_id}")
        
        return result


def main():
    """主函数示例"""
    # 使用配置模块获取API密钥，而不是硬编码
    try:
        api_key = config.get_api_key_safe()
        api_base = config.get_api_base_safe()
    except ValueError as e:
        print(f"配置错误: {e}")
        print("请设置环境变量OPENAI_API_KEY和OPENAI_API_BASE")
        return
    
    evaluator = QA_evaluator(
        user_id="test",
        dataset_name="DotaBench",
        model_name="doubao-1.5-pro-32k",
        model_key=api_key,
        api_base=api_base,
        business_id=None,
        question_limitation=5,
        max_workers=4,
        judge_model="gpt-4o",
        judge_key=api_key,
        judge_api_base=api_base
    )
    
    result = evaluator.run_full_evaluation()
    print(f"\n最终结果: {result}")


if __name__ == "__main__":
    main()