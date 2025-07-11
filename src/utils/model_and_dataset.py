import os
import json



GENERAL_DATASETS = ["MMLU", "GPQA", "MT-Bench"]
GENERAL_MULTIMODAL_DATASETS = ["MMStar", "TestImageMCQ"]
MEDICAL_KNOWLEDGE_DATASETS = ["CMB", "CMMLUMed"]
MEDICAL_ETHICS_DATASETS = ["MedEthicsMatrixMCQ", "MedEthicsMatrixCase"]

DATASET_CATEGORIES = {
    "通用能力": GENERAL_DATASETS,
    "通用多模态能力": GENERAL_MULTIMODAL_DATASETS,
    "医学知识": MEDICAL_KNOWLEDGE_DATASETS,
    "医学伦理": MEDICAL_ETHICS_DATASETS
}

TEXT_DATASETS = ["MMLU", "GPQA", "CMB", "MedEthicsMatrixMCQ", "CMMLUMed"]
MULTIMODAL_DATASETS = ["MMStar", "TestImageMCQ"]
LLMJUDGE_DATASETS = ["MT-Bench", "MedEthicsMatrixCase"]

TEXT_MODELS = ["gpt-3.5-turbo", "gpt-4o", "doubao-1.5-pro-32k"]
MULTIMODAL_MODELS = ["Qwen2-VL-7B-Instruct"]
JUDGE_MODELS = ["gpt-4o"]


