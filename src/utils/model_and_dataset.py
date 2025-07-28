"""
模型和数据集配置模块

该模块定义了MEvalKit支持的数据集和模型分类，用于在评测过程中
确定使用哪种评测方法和API接口。

作者: MEvalKit Team
版本: 1.0.0
"""

import os
import json

# 数据集分类定义

# 通用能力数据集（纯文本）
GENERAL_DATASETS = ["MMLU", "GPQA", "MT-Bench"]

# 通用多模态能力数据集（图像+文本）
GENERAL_MULTIMODAL_DATASETS = ["MMStar", "TestImageMCQ"]

# 医学知识数据集（纯文本）
MEDICAL_KNOWLEDGE_DATASETS = ["CMB", "CMMLUMed"]

# 医学伦理数据集（纯文本）
MEDICAL_ETHICS_DATASETS = ["MedEthicsMatrixMCQ", "MedEthicsMatrixCase"]

# 数据集分类映射表
DATASET_CATEGORIES = {
    "通用能力": GENERAL_DATASETS,
    "通用多模态能力": GENERAL_MULTIMODAL_DATASETS,
    "医学知识": MEDICAL_KNOWLEDGE_DATASETS,
    "医学伦理": MEDICAL_ETHICS_DATASETS
}

# 按评测类型分类的数据集

# 文本多选题数据集（使用TextMCQ_eval模块评测）
TEXT_DATASETS = ["MMLU", "GPQA", "CMB", "MedEthicsMatrixMCQ", "CMMLUMed"]

# 多模态数据集（使用ImageMCQ_eval模块评测）
MULTIMODAL_DATASETS = ["MMStar", "TestImageMCQ"]

# LLMJudge型数据集（使用LLMJudge_eval模块评测）
LLMJUDGE_DATASETS = ["MT-Bench", "MedEthicsMatrixCase"]

# 模型分类定义

# 纯文本模型（支持文本问答）
TEXT_MODELS = ["gpt-3.5-turbo", "gpt-4o", "doubao-1.5-pro-32k"]

# 多模态模型（支持图像+文本问答）
MULTIMODAL_MODELS = ["Qwen2-VL-7B-Instruct"]

# 评判模型（用于LLMJudge评测）
JUDGE_MODELS = ["gpt-4o"]


