# DotaBench 评测脚本使用指南

## 📋 概述

`dotabench_evaluator.py` 是一个独立的评测脚本，用于评价模型在医疗多轮对话诊断任务中的能力。该脚本基于 DotaBench 数据集，支持自动化评测和详细的结果分析。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install datasets pandas numpy scikit-learn jieba rouge-score bert-score
```

### 2. 基本使用

```python
# 导入评测器
from dotabench_evaluator import DotaBenchEvaluator

# 创建评测器实例
evaluator = DotaBenchEvaluator()

# 评测前5个样本（测试用）
report = evaluator.evaluate_all(max_samples=5)

# 打印摘要
evaluator.print_summary(report)

# 保存报告
evaluator.save_report(report)
```

## 🔧 核心功能

### 1. 模型接口集成

**最重要的步骤：修改 `call_model` 方法**

```python
def call_model(self, question: str, conversation_history: List[str] = None) -> str:
    """
    调用您的模型API进行推理
    
    Args:
        question: 当前问题
        conversation_history: 对话历史
    
    Returns:
        模型回答
    """
    # 示例1: OpenAI API
    # import openai
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": "你是一个医疗诊断助手"},
    #         {"role": "user", "content": question}
    #     ] + self._format_history(conversation_history)
    # )
    # return response.choices[0].message.content
    
    # 示例2: 本地模型API
    # response = requests.post(
    #     "http://localhost:8000/chat",
    #     json={
    #         "query": question,
    #         "history": conversation_history
    #     }
    # )
    # return response.json()["answer"]
    
    # 示例3: Hugging Face模型
    # from transformers import pipeline
    # generator = pipeline("text-generation", model="your-model-name")
    # response = generator(question, max_length=200)
    # return response[0]["generated_text"]
    
    # 临时示例（请替换为实际实现）
    return "这是模型的示例回答，请替换为实际的模型API调用"
```

### 2. 评测指标

脚本采用多维度评测指标：

- **ROUGE得分** - 文本相似度
- **关键词匹配** - Jaccard相似度
- **医疗术语准确性** - 专业术语匹配

### 3. 多轮对话处理

```python
# 三轮对话流程
for turn in range(1, 4):
    # 第1轮：初步病史分析
    # 第2轮：体格检查结果
    # 第3轮：辅助检查诊断
    
    # 每轮都会累积对话历史
    conversation_history.append(f"问题{turn}: {question}")
    conversation_history.append(f"回答{turn}: {model_answer}")
```

## 📊 评测结果解读

### 1. 整体得分

```json
{
  "overall": {
    "mean_score": 0.756,        // 平均得分
    "median_score": 0.720,      // 中位数得分
    "std_score": 0.134,         // 标准差
    "min_score": 0.450,         // 最低分
    "max_score": 0.950          // 最高分
  }
}
```

### 2. 轮次分析

```json
{
  "turn_wise": {
    "turn_1": {"mean_score": 0.680},  // 第1轮平均得分
    "turn_2": {"mean_score": 0.745},  // 第2轮平均得分
    "turn_3": {"mean_score": 0.812}   // 第3轮平均得分
  }
}
```

### 3. 得分分布

```json
{
  "score_distribution": {
    "0.8-1.0": {"count": 15, "percentage": 30.0},  // 优秀
    "0.6-0.8": {"count": 20, "percentage": 40.0},  // 良好
    "0.4-0.6": {"count": 12, "percentage": 24.0},  // 一般
    "0.2-0.4": {"count": 3, "percentage": 6.0},    // 待改进
    "0.0-0.2": {"count": 0, "percentage": 0.0}     // 不合格
  }
}
```

## 🎯 高级用法

### 1. 自定义评测策略

```python
class CustomDotaBenchEvaluator(DotaBenchEvaluator):
    def calculate_turn_score(self, model_answer: str, reference: str) -> float:
        """自定义评分策略"""
        # 添加您的评分逻辑
        custom_score = self.your_custom_metric(model_answer, reference)
        return custom_score
    
    def your_custom_metric(self, answer: str, reference: str) -> float:
        # 实现自定义评分算法
        pass
```

### 2. 批量评测

```python
# 评测所有样本
report = evaluator.evaluate_all()

# 评测指定数量样本
report = evaluator.evaluate_all(max_samples=50)

# 保存到指定路径
evaluator.save_report(report, "my_evaluation_results.json")
```

### 3. 结果分析

```python
# 分析具体案例
for result in report['results']:
    case_id = result['case_id']
    overall_score = result['overall_score']
    
    # 分析每轮表现
    for turn in result['turns']:
        turn_num = turn['turn']
        turn_score = turn['score']
        print(f"案例{case_id} 第{turn_num}轮得分: {turn_score}")
```

## 🔍 评测标准

### 得分等级划分

| 得分范围 | 等级 | 描述 |
|---------|------|------|
| 0.8-1.0 | 优秀 | 模型表现优异，接近专家水平 |
| 0.6-0.8 | 良好 | 模型表现良好，可实际应用 |
| 0.4-0.6 | 一般 | 模型表现一般，需要改进 |
| 0.2-0.4 | 待改进 | 模型表现较差，需要重点优化 |
| 0.0-0.2 | 不合格 | 模型表现极差，不建议使用 |

### 医疗特定评价维度

1. **诊断准确性** - 诊断结论的正确性
2. **推理逻辑** - 诊断过程的合理性
3. **专业术语** - 医学术语使用的准确性
4. **对话连贯性** - 多轮对话的逻辑一致性

## 📝 实际应用示例

```python
# 完整的评测流程
def run_evaluation():
    # 1. 创建评测器
    evaluator = DotaBenchEvaluator()
    
    # 2. 运行评测
    print("开始DotaBench评测...")
    report = evaluator.evaluate_all(max_samples=10)
    
    # 3. 分析结果
    evaluator.print_summary(report)
    
    # 4. 保存报告
    evaluator.save_report(report, "evaluation_report.json")
    
    # 5. 分析特定案例
    low_score_cases = [
        r for r in report['results'] 
        if r['overall_score'] < 0.5
    ]
    
    print(f"低分案例数量: {len(low_score_cases)}")
    for case in low_score_cases:
        print(f"案例 {case['case_id']} 得分: {case['overall_score']:.3f}")
    
    return report

# 运行评测
report = run_evaluation()
```

## 🚨 注意事项

1. **模型API集成** - 必须修改 `call_model` 方法对接您的模型
2. **评测时间** - 完整评测74个样本可能需要较长时间
3. **资源占用** - 确保有足够的计算资源和API调用额度
4. **结果解读** - 结合医疗专业知识理解评测结果

## 📚 扩展功能

### 1. 添加新的评测指标

```python
def calculate_clinical_accuracy(self, model_answer: str, reference: str) -> float:
    """计算临床准确性"""
    # 实现临床相关的评分逻辑
    pass
```

### 2. 支持不同模型比较

```python
def compare_models(self, model_configs: List[Dict]) -> Dict:
    """比较多个模型的性能"""
    results = {}
    for config in model_configs:
        # 切换模型配置
        self.model_config = config
        # 运行评测
        report = self.evaluate_all()
        results[config['name']] = report
    return results
```

这个评测脚本为您提供了完整的DotaBench评测解决方案，您只需要根据实际情况修改模型API调用部分即可开始评测。 