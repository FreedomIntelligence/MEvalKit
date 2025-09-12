import sys
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from evaluation.Agent_response_generator import Agent_response_generator
from evaluation.Agent_scorer import Scorer
from src.dataset.Agent import Agent

AGENT_SCORER_CONFIG_PATH = "dataset_info/Agent_scorer_config.yaml"

def load_agent_scorer_config():
    """加载Agent评分器配置"""
    with open(AGENT_SCORER_CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class Agent_evaluator:
    def __init__(self,
                 user_id: str,
                 dataset_name: str,
                 agent_1_model: str = "gpt-4o",
                 agent_2_model: str = None,
                 response_agent_model: str = None,
                 model_key: str = None,
                 api_base: str = None,
                 business_id: str = None,
                 question_limitation: int = None,
                 max_workers: int = 4,
                 judge_model: str = None,
                 judge_key: str = None,
                 judge_api_base: str = None):
        
        self.user_id = user_id
        self.dataset_name = dataset_name
        self.agent_1_model = agent_1_model
        self.agent_2_model = agent_2_model if agent_2_model else agent_1_model
        self.response_agent_model = response_agent_model if response_agent_model else agent_1_model
        self.model_key = model_key
        self.api_base = api_base
        self.business_id = business_id
        self.question_limitation = question_limitation
        self.max_workers = max_workers
        
        # 加载数据集和配置
        self.dataset = Agent(dataset_name)
        self.scorer_config = load_agent_scorer_config()
        
        # 获取评分标准
        self.criteria = self.get_scoring_criteria()
        print(f"数据集 {dataset_name} 使用评分标准: {self.criteria}")
        
        # 设置评分模型（根据criteria决定是否需要）
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
        """生成Agent对话响应"""
        print(f"开始生成Agent对话响应")
        print(f"数据集: {self.dataset_name}")
        print(f"Agent_1模型: {self.agent_1_model}")
        print(f"Agent_2模型: {self.agent_2_model}")
        print(f"Response Agent模型: {self.response_agent_model}")
        
        generator = Agent_response_generator(
            user_id=self.user_id,
            dataset_name=self.dataset_name,
            agent_1_model=self.agent_1_model,
            agent_2_model=self.agent_2_model,
            response_agent_model=self.response_agent_model,
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
        """对Agent响应进行评分"""
        if not self.business_id:
            raise ValueError("需要先生成响应或提供business_id")
        
        print(f"开始评分Agent响应 - 评分标准: {self.criteria}")
        
        if self.criteria == 'accuracy':
            # 使用精确匹配评分
            scorer = AccuracyScorer(
                dataset_name=self.dataset_name,
                user_id=self.user_id,
                business_id=self.business_id
            )
        elif self.criteria == 'llmjudge':
            # 使用LLM评判评分 (未来可扩展)
            raise NotImplementedError(f"LLM评判评分尚未实现，数据集: {self.dataset_name}")
        elif self.criteria == 'rubrics':
            # 使用评分标准评分 (未来可扩展)
            raise NotImplementedError(f"评分标准评分尚未实现，数据集: {self.dataset_name}")
        else:
            raise ValueError(f"不支持的评分标准: {self.criteria}")
        
        return scorer.score_responses()
    
    def run_full_evaluation(self) -> Dict[str, str]:
        """运行完整的Agent评估流程：生成对话响应 -> 评分"""
        print(f"开始完整Agent评估流程")
        print(f"数据集: {self.dataset_name}")
        print(f"Agent_1模型: {self.agent_1_model}")
        print(f"Agent_2模型: {self.agent_2_model}")
        print(f"Response Agent模型: {self.response_agent_model}")
        print(f"评分标准: {self.criteria}")
        print(f"问题数量限制: {self.question_limitation}")
        print("-" * 50)
        
        # 步骤1: 生成Agent对话响应
        business_id = self.generate_responses()
        
        # 步骤2: 评分响应
        score_business_id = self.score_responses()
        
        result = {
            "response_business_id": business_id,
            "score_business_id": score_business_id,
            "dataset_name": self.dataset_name,
            "agent_1_model": self.agent_1_model,
            "agent_2_model": self.agent_2_model,
            "response_agent_model": self.response_agent_model,
            "criteria": self.criteria,
            "question_limitation": self.question_limitation
        }
        
        print(f"完整Agent评估完成！")
        print(f"响应结果ID: {business_id}")
        print(f"评分结果ID: {score_business_id}")
        
        return result


class AccuracyScorer(Scorer):
    """精确匹配评分器，用于Agent响应评分"""
    
    def __init__(self, dataset_name: str, user_id: str, business_id: str):
        super().__init__(dataset_name, user_id, business_id)
        # 加载Agent数据集以获取参考答案
        self.agent_dataset = Agent(dataset_name)
    
    def score_responses(self):
        """对Agent响应进行精确匹配评分"""
        result_file = self.get_result_file_path()
        score_file = self.get_score_file_path()
        
        # 读取响应结果
        results = read_json_file(result_file)
        if not results:
            raise FileNotFoundError(f"找不到结果文件: {result_file}")
        
        total_count = len(results)
        correct_count = 0
        scores = []
        
        print(f"开始评分 {total_count} 个Agent响应...")
        
        for result in results:
            score_item = {
                "id": result["id"],
                "response": result["response"],
                "answer": result["answer"],
                "score": 0
            }
            
            # 精确匹配评分
            if result["response"] and result["answer"]:
                response = str(result["response"]).strip()
                answer = str(result["answer"]).strip()
                
                if response == answer:
                    score_item["score"] = 1
                    correct_count += 1
                else:
                    # 检查是否包含正确答案（部分匹配）
                    if answer in response:
                        score_item["score"] = 0.5
                        correct_count += 0.5
            
            scores.append(score_item)
        
        # 计算总体得分
        accuracy = correct_count / total_count if total_count > 0 else 0
        final_score = {
            "total_count": total_count,
            "correct_count": correct_count,
            "accuracy": accuracy,
            "valid_ratio": 1.0,  # Agent评估中所有响应都被认为是有效的
            "score": accuracy * 100,  # 转换为百分制
            "detailed_scores": scores
        }
        
        # 保存评分结果
        write_json_file(final_score, score_file)
        print(f"Agent评分完成！准确率: {accuracy:.2%}, 保存到: {score_file}")
        
        return self.business_id


def main():
    """主函数示例"""
    evaluator = Agent_evaluator(
        user_id="test",
        dataset_name="IOR-Dynamic",
        agent_1_model="gpt-4o",
        agent_2_model="doubao-1.5-pro-32k",
        response_agent_model="doubao-1.5-pro-32k",
        model_key="sk-fPz5uPZn2ubb9Qexx62yWcFl55Z46iRdBfdlvnjufQ6o0BVo",
        api_base="https://api.huatuogpt.cn/v1",
        business_id=None,
        question_limitation=5,
        max_workers=4
    )
    
    result = evaluator.run_full_evaluation()
    print(f"\n最终结果: {result}")


if __name__ == "__main__":
    main()