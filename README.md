<b> WindaEvalKit: 是一款为大模型评测集合工具包。该工具支持在多模态选择题、纯文本选择题、LLMJudge等不同类别的评测集上进行一键评估。 </b>

# 基本信息

## 版本

[2025-04-17] 初版发布。

## 支持的评测集

纯文本选择题评测集：MMLU、GPQA、CMB。
多模态选择题评测集：MMStar。
LLMJudge型评测集：MT-Bench。

## 支持的模型

gpt-3.5-turbo, gpt-4o, Qwen2-VL-7B-Instruct

# 快速开始

## 安装(SSH)

```bash
git clone git@github.com:FreedomIntelligence/WindaEvalKit.git
cd WindaEvalKit
pip install -e .
pip install -r requirements.txt
```

## 使用

### 选择题型评测集

```bash
python run.py --dataset MMLU --model gpt-3.5-turbo --workers 64 --evaluate_mode start_from_beginning
```
--dataset：评测集名称。你可以从dataset_info文件夹中找到评测集名称。

--model：模型名称。你可以从model_info文件夹中找到模型名称。

--workers：并行处理的工作进程数量。

--evaluate_mode：评估模式，分为两种：

1. start_from_beginning：从评测集的第一题开始，评估所有题目。会重置结果文件。

2. resume_from_checkpoint：检测结果文件，跳过已经完成评测的题目。

### LLMJudge型评测集

```bash
python run.py --dataset MT-Bench --model gpt-3.5-turbo --judgment_model gpt-4o --question_number 0 
```

--judgment_model：评判模型名称。

--question_number：评测集中问题的编号

## 结果生成

在使用选择题型评测集评估时，会生成两个结果文件。（1）选择题各单题的详细结果文件；（2）单次评测正确率结果文件。只有一个模型对于评测集中95%的题目都生成了答案，才会生成正确率结果文件。


