#!/usr/bin/env python3
"""
DotaBench 评测脚本
用于评价模型在医疗多轮对话诊断任务中的能力
"""

import json
import re
from typing import Dict, List, Any, Tuple
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import jieba
from rouge import Rouge
from bert_score import score as bert_score


class DotaBenchEvaluator:
    """DotaBench数据集评测器"""
    
    def __init__(self, dataset_name: str = "FreedomIntelligence/DotaBench"):
        """
        初始化评测器
        
        Args:
            dataset_name: 数据集名称
        """
        self.dataset_name = dataset_name
        self.dataset = None
        self.rouge = Rouge()
        self.load_dataset()
    
    def load_dataset(self):
        """加载DotaBench数据集"""
        try:
            self.dataset = load_dataset(self.dataset_name)['test']
            print(f"✓ 成功加载DotaBench数据集，共{len(self.dataset)}条样本")
        except Exception as e:
            print(f"✗ 加载数据集失败: {e}")
            raise
    
    def call_model(self, question: str, conversation_history: List[str] = None) -> str:
        """
        调用模型API进行推理
        
        Args:
            question: 当前问题
            conversation_history: 对话历史
        
        Returns:
            模型回答
        
        Note: 这里需要替换为您的模型API调用
        """
        # TODO: 替换为您的模型API调用
        # 示例：
        # response = your_model.generate(question, history=conversation_history)
        # return response
        
        # 临时示例回答（实际使用时请替换）
        return "这是模型的示例回答，请替换为实际的模型API调用"
    
    def evaluate_single_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        评测单个案例
        
        Args:
            case: 数据集中的单个案例
            
        Returns:
            评测结果
        """
        case_id = case['id']
        results = {
            'case_id': case_id,
            'turns': [],
            'overall_score': 0.0
        }
        
        conversation_history = []
        total_score = 0.0
        
        # 三轮对话评测
        for turn in range(1, 4):
            question_key = f'turn_{turn}_question'
            question = case[question_key]
            
            # 调用模型获取回答
            model_answer = self.call_model(question, conversation_history)
            
            # 获取参考答案
            reference_answers = case['reference']
            if turn <= len(reference_answers):
                reference = reference_answers[turn-1]['answer']
                ref_question = reference_answers[turn-1]['question']
            else:
                reference = ""
                ref_question = ""
            
            # 计算当前轮次得分
            turn_score = self.calculate_turn_score(model_answer, reference)
            
            # 更新对话历史
            conversation_history.append(f"问题{turn}: {question}")
            conversation_history.append(f"回答{turn}: {model_answer}")
            
            # 保存轮次结果
            turn_result = {
                'turn': turn,
                'question': question,
                'model_answer': model_answer,
                'reference_answer': reference,
                'reference_question': ref_question,
                'score': turn_score
            }
            results['turns'].append(turn_result)
            total_score += turn_score
        
        # 计算整体得分
        results['overall_score'] = total_score / 3
        
        return results
    
    def calculate_turn_score(self, model_answer: str, reference: str) -> float:
        """
        计算单轮对话得分
        
        Args:
            model_answer: 模型答案
            reference: 参考答案
            
        Returns:
            得分 (0-1)
        """
        if not model_answer or not reference:
            return 0.0
        
        scores = []
        
        # 1. ROUGE得分
        try:
            rouge_score = self.rouge.get_scores(model_answer, reference)[0]
            rouge_f1 = rouge_score['rouge-l']['f']
            scores.append(rouge_f1)
        except:
            scores.append(0.0)
        
        # 2. 关键词匹配得分
        keyword_score = self.calculate_keyword_score(model_answer, reference)
        scores.append(keyword_score)
        
        # 3. 医疗术语准确性得分
        medical_term_score = self.calculate_medical_term_score(model_answer, reference)
        scores.append(medical_term_score)
        
        # 综合得分
        return np.mean(scores)
    
    def calculate_keyword_score(self, model_answer: str, reference: str) -> float:
        """计算关键词匹配得分"""
        # 分词
        model_words = set(jieba.cut(model_answer))
        ref_words = set(jieba.cut(reference))
        
        # 计算交集
        intersection = model_words & ref_words
        union = model_words | ref_words
        
        if not union:
            return 0.0
        
        # Jaccard相似度
        return len(intersection) / len(union)
    
    def calculate_medical_term_score(self, model_answer: str, reference: str) -> float:
        """计算医疗术语准确性得分"""
        # 医疗关键词模式
        medical_patterns = [
            r'诊断[:：]?([^。，,\n]+)',
            r'病因[:：]?([^。，,\n]+)',
            r'症状[:：]?([^。，,\n]+)',
            r'治疗[:：]?([^。，,\n]+)',
            r'疾病[:：]?([^。，,\n]+)',
        ]
        
        model_terms = set()
        ref_terms = set()
        
        for pattern in medical_patterns:
            model_matches = re.findall(pattern, model_answer)
            ref_matches = re.findall(pattern, reference)
            
            model_terms.update(model_matches)
            ref_terms.update(ref_matches)
        
        if not ref_terms:
            return 1.0 if not model_terms else 0.5
        
        # 计算医疗术语匹配度
        intersection = model_terms & ref_terms
        return len(intersection) / len(ref_terms)
    
    def evaluate_all(self, max_samples: int = None) -> Dict[str, Any]:
        """
        评测所有样本
        
        Args:
            max_samples: 最大评测样本数（None表示全部）
            
        Returns:
            完整评测结果
        """
        if max_samples:
            dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        else:
            dataset = self.dataset
        
        print(f"开始评测 {len(dataset)} 个样本...")
        
        all_results = []
        total_scores = []
        
        for i, case in enumerate(dataset):
            print(f"正在评测样本 {i+1}/{len(dataset)}...")
            
            try:
                result = self.evaluate_single_case(case)
                all_results.append(result)
                total_scores.append(result['overall_score'])
                
                print(f"  样本 {case['id']} 得分: {result['overall_score']:.3f}")
            except Exception as e:
                print(f"  样本 {case['id']} 评测失败: {e}")
                continue
        
        # 计算统计信息
        stats = self.calculate_statistics(all_results, total_scores)
        
        # 生成报告
        report = {
            'dataset_info': {
                'name': self.dataset_name,
                'total_samples': len(dataset),
                'evaluated_samples': len(all_results),
                'evaluation_time': datetime.now().isoformat()
            },
            'results': all_results,
            'statistics': stats
        }
        
        return report
    
    def calculate_statistics(self, all_results: List[Dict], total_scores: List[float]) -> Dict[str, Any]:
        """计算统计信息"""
        if not total_scores:
            return {}
        
        # 总体统计
        stats = {
            'overall': {
                'mean_score': np.mean(total_scores),
                'median_score': np.median(total_scores),
                'std_score': np.std(total_scores),
                'min_score': np.min(total_scores),
                'max_score': np.max(total_scores),
            },
            'turn_wise': {},
            'score_distribution': {}
        }
        
        # 按轮次统计
        for turn in range(1, 4):
            turn_scores = []
            for result in all_results:
                if len(result['turns']) >= turn:
                    turn_scores.append(result['turns'][turn-1]['score'])
            
            if turn_scores:
                stats['turn_wise'][f'turn_{turn}'] = {
                    'mean_score': np.mean(turn_scores),
                    'median_score': np.median(turn_scores),
                    'std_score': np.std(turn_scores),
                }
        
        # 得分分布
        score_ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        for low, high in score_ranges:
            count = sum(1 for score in total_scores if low <= score < high)
            stats['score_distribution'][f'{low}-{high}'] = {
                'count': count,
                'percentage': count / len(total_scores) * 100
            }
        
        return stats
    
    def save_report(self, report: Dict[str, Any], output_path: str = None):
        """保存评测报告"""
        if output_path is None:
            output_path = f"dotabench_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 评测报告已保存至: {output_path}")
    
    def print_summary(self, report: Dict[str, Any]):
        """打印评测摘要"""
        stats = report['statistics']
        
        print("\n" + "="*50)
        print("DotaBench 评测报告摘要")
        print("="*50)
        
        print(f"数据集: {report['dataset_info']['name']}")
        print(f"评测样本: {report['dataset_info']['evaluated_samples']}")
        print(f"评测时间: {report['dataset_info']['evaluation_time']}")
        
        print("\n整体表现:")
        overall = stats['overall']
        print(f"  平均得分: {overall['mean_score']:.3f}")
        print(f"  中位数得分: {overall['median_score']:.3f}")
        print(f"  标准差: {overall['std_score']:.3f}")
        print(f"  得分范围: {overall['min_score']:.3f} - {overall['max_score']:.3f}")
        
        print("\n各轮次表现:")
        for turn in range(1, 4):
            turn_key = f'turn_{turn}'
            if turn_key in stats['turn_wise']:
                turn_stats = stats['turn_wise'][turn_key]
                print(f"  第{turn}轮: {turn_stats['mean_score']:.3f} (±{turn_stats['std_score']:.3f})")
        
        print("\n得分分布:")
        for range_key, dist in stats['score_distribution'].items():
            print(f"  {range_key}: {dist['count']}个样本 ({dist['percentage']:.1f}%)")
        
        # 能力评级
        mean_score = overall['mean_score']
        if mean_score >= 0.8:
            level = "优秀"
        elif mean_score >= 0.6:
            level = "良好"
        elif mean_score >= 0.4:
            level = "一般"
        else:
            level = "待改进"
        
        print(f"\n整体评级: {level}")
        print("="*50)


def main():
    """主函数 - 使用示例"""
    # 创建评测器
    evaluator = DotaBenchEvaluator()
    
    # 评测前5个样本（测试用）
    print("开始DotaBench评测...")
    report = evaluator.evaluate_all(max_samples=5)
    
    # 打印摘要
    evaluator.print_summary(report)
    
    # 保存报告
    evaluator.save_report(report)
    
    print("\n评测完成！")


if __name__ == "__main__":
    main() 