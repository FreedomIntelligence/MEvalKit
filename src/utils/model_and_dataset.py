model_map = {
    "Qwen2-VL-7B-Instruct": "Pro/Qwen/Qwen2-VL-7B-Instruct",
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "gpt-4o": "gpt-4o"
}

GENERAL_DATASETS = ["MMLU", "GPQA", "MMStar", "MT-Bench"]
MEDICAL_KNOWLEDGE_DATASETS = ["CMB"]
MEDICAL_ETHICS_DATASETS = ["MedEthicMatrixMCQ"]

DATASET_CATEGORIES = {
    "通用能力": GENERAL_DATASETS,
    "医学知识": MEDICAL_KNOWLEDGE_DATASETS,
    "医学伦理": MEDICAL_ETHICS_DATASETS
}

TEXT_DATASETS = ["MMLU", "GPQA", "CMB", "MedEthicMatrixMCQ"]
IMAGE_DATASETS = ["MMStar"]
LLMJUDGE_DATASETS = ["MT-Bench"]

TEXT_MODELS = ["gpt-3.5-turbo", "gpt-4o"]
MULTIMODAL_MODELS = ["Qwen2-VL-7B-Instruct"]
JUDGE_MODELS = ["gpt-4o"]