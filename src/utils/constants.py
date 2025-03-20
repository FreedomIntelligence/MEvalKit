MCQ_TEMPLATE_ZH = """
你现在是一个能回答中文选择题的AI助手。请准确回答所提供给你的选择题, 并遵循以下要求:
1. 你每次的回答内容只能包含选项("A", "B", "C"，等等), 不允许出现其他内容。
"""

MCQ_TEMPLATE_EN = """
You are a helpful assistant that can answer multiple choice questions. Do not make any analysis or explanation, just answer with only the letter of the correct choice (A/B/C/D).
"""

SINGLE_MCQ_TEMPLATE_ZH = """
这道题只有一个正确答案。
"""

MULTIPLE_MCQ_TEMPLATE_ZH = """
这道题有多个正确答案。
"""

SINGLE_MCQ_TEMPLATE_EN = """
This question has only one correct answer.
"""

MULTIPLE_MCQ_TEMPLATE_EN = """
This question has multiple correct answers.
"""

MAX_CHOICE_MAP = {
    "CMB": 'E',
    "MMLU": 'D',
    "GPQA": 'D',
}

def get_choice_pattern(dataset_name: str) -> str:
    """根据数据集名称获取选项范围"""
    return f'[A-{MAX_CHOICE_MAP.get(dataset_name, "D")}]'

def build_patterns(dataset_name: str) -> list:
    """构建单选题的正则表达式模式"""
    choice = get_choice_pattern(dataset_name)
    max_letter = MAX_CHOICE_MAP.get(dataset_name, 'D')
    
    # 构建非选项字母的模式
    non_choice = f'[^A-{max_letter}]'
    
    return [
        # 直接匹配
        f'^\s*({choice})\s*$',                    # 单个字母
        f'{non_choice}*({choice}){non_choice}*$',  # 句子中的单个字母
        
        # 中文关键词
        f'答案[是为：:\s]*({choice})',             # "答案是A"，"答案为B"
        f'选[择项]?\s*({choice})',                 # "选A"，"选择B"，"选项C"
        f'({choice})\s*选[择项]',                  # "A选项"，"B选择"
        f'正确[的答案]?[是为：:\s]*({choice})',     # "正确答案是A"，"正确的是B"
        f'应[该当]?[是选择：:\s]*({choice})',       # "应该是A"，"应当选择B"
        f'我[的认为]?[选择认为]({choice})',         # "我选A"，"我认为B"
        
        # 英文关键词
        f'[Tt]he\s+answer\s+is\s+({choice})',     # "The answer is A"
        f'[Cc]hoose\s+({choice})',                # "Choose A"
        f'[Ss]elect\s+({choice})',                # "Select A"
        f'[Oo]ption\s+({choice})',                # "Option A"
        f'[Tt]he\s+correct\s+[answer\s+]?is\s+({choice})',  # "The correct answer is A"
        f'[Ii]\s+[choose\s+]?({choice})',         # "I choose A"
        
        # 带括号的格式
        f'\(({choice})\)',                        # "(A)"
        f'（({choice})）',                         # "（A）"
        f'【({choice})】',                         # "【A】"
        f'\[({choice})\]',                        # "[A]"
        
        # 特殊格式
        f'[选择项]\s*({choice})\s*[选择项]',        # "选项A选项"
        f'答案?对应\s*({choice})',                 # "答对应A"，"答案对应B"
        f'({choice})\s*[是为]正确[的答案]?',        # "A是正确答案"
        f'最终[选择答案]\s*[为是：:\s]*({choice})',  # "最终选择为A"
    ]

def build_patterns_multi(dataset_name: str) -> list:
    """构建多选题的正则表达式模式"""
    choice = get_choice_pattern(dataset_name)
    max_letter = MAX_CHOICE_MAP.get(dataset_name, 'D')
    
    # 构建非选项字母的模式
    non_choice = f'[^A-{max_letter}]'
    
    return [
        # 直接匹配多个选项
        f'({choice}[,，\s]*)+{choice}',           # "A,B,C" 或 "A B C"
        f'{non_choice}*({choice}[,，\s]*)+{choice}{non_choice}*$',  # 句子中的多个选项
        
        # 中文关键词
        f'答案[是为：:\s]*({choice}[,，\s]*)+{choice}',   # "答案是A,B,C"
        f'选[择项]?\s*({choice}[,，\s]*)+{choice}',      # "选A,B,C"
        
        # 英文关键词
        f'[Tt]he\s+answers?\s+(?:is|are)\s+({choice}[,，\s]*)+{choice}',  # "The answers are A,B,C"
        f'[Cc]hoose\s+({choice}[,，\s]*)+{choice}',      # "Choose A,B,C"
        
        # 带括号的格式
        f'\(({choice}[,，\s]*)+{choice}\)',              # "(A,B,C)"
        f'（({choice}[,，\s]*)+{choice}）',               # "（A,B,C）"
        
        # 列表格式
        f'[\[【\(（]({choice}[\]】\)）][,，\s]*)+{choice}[\]】\)）]'  # "[A][B][C]"
    ]

SINGLE_CHOICE_LIST = ["单项选择题", "single"]
MULTIPLE_CHOICE_LIST = ["多项选择题", "multiple"]

__all__ = [
    'MCQ_TEMPLATE_ZH',
    'MCQ_TEMPLATE_EN',
    'MAX_CHOICE_MAP',
    'get_choice_pattern',
    'build_patterns',
    'build_patterns_multi',
    'SINGLE_CHOICE_LIST',
    'MULTIPLE_CHOICE_LIST',
    'SINGLE_MCQ_TEMPLATE_ZH',
    'MULTIPLE_MCQ_TEMPLATE_ZH',
    'SINGLE_MCQ_TEMPLATE_EN',
    'MULTIPLE_MCQ_TEMPLATE_EN',
]

