import sys
import os
from pathlib import Path
from openai import BadRequestError, AuthenticationError

project_root = Path(__file__).resolve().parent.parent
print(project_root)
sys.path.append(str(project_root))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.Text.TextMCQ import *
from src.api.api import *
from src.utils.constants import *
from tqdm import tqdm
import concurrent.futures
from typing import List, Tuple



def process_single_prompt(args: Tuple[TestChat, dict, str, str]) -> str:
    """处理单个prompt的函数"""
    try:
        chatApi, prompt_dict, answer, dataset_name = args
        prompt = prompt_dict['prompt']
        question_type = prompt_dict['type']
        if question_type == 'single':
            response = chatApi.chat(prompt)
            #print("Initial response: ", response)
            response = extract_answer(response, dataset_name)
            if answer != '':
                print("Response: ", response, "Answer: ", answer)
                return "Correct" if response == answer else "Wrong"
            else:
                print("Response: ", response)
                return response
        elif question_type == 'multiple':
            response = chatApi.chat(prompt)
            #print("Initial response: ", response)
            response = extract_multi_answer(response, dataset_name)
            print("Response: ", response)
            return response

    except BadRequestError as e:
        return "Neglected"
    except AuthenticationError as e:
        return "Neglected"

def evaluate(chatApi: TestChat, dataset_name: str):
    dataset = TextMCQ(dataset_name)
    prompt_list = dataset.build_prompts()
    answers = [dataset.dataset[i]['answer'] for i in range(len(prompt_list))]
    length = len(prompt_list)
    correct_ans = 0
    
    # 创建参数列表
    args_list = [(chatApi, prompt, answer, dataset_name) 
                 for prompt, answer in zip(prompt_list, answers)]
    
    # 使用ThreadPoolExecutor进行并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        # 使用tqdm显示进度
        results = list(tqdm(
            executor.map(process_single_prompt, args_list),
            total=length,
            desc="Evaluating"
        ))
    
    valid_results = [result for result in results if result != "Neglected"]
    if len(answers) == 0:
        return valid_results
    else:
        correct_count = sum(1 for result in valid_results if result == "Correct")
        valid_count = len(valid_results)
        
        print(f"Accuracy: {correct_count / valid_count:.2%}")
        return correct_count / valid_count

def extract_answer(response: str, dataset_name: str):
    PATTERNS = build_patterns(dataset_name)
    for pattern in PATTERNS:
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    return None

def extract_multi_answer(response: str, dataset_name: str) -> List[str]:
    """提取多选题答案"""
    PATTERNS_MULTI = build_patterns_multi(dataset_name)
    # 预处理：移除多余空格，统一逗号格式
    response = response.strip().replace('，', ',')
    
    # 尝试所有模式匹配
    for pattern in PATTERNS_MULTI:
        matches = re.findall(pattern, response)
        if matches:
            # 提取所有选项并去重
            answers = []
            for match in matches:
                # 提取A-D的字母
                options = re.findall(r'[A-D]', match)
                answers.extend(options)
            
            # 去重并排序
            answers = sorted(list(set(answers)))
            return answers
    
    # 如果没有匹配到完整格式，尝试提取单个选项
    single_options = re.findall(r'[A-D]', response)
    if single_options:
        return sorted(list(set(single_options)))
    
    return None

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=100)
    chat = TestChat(model, language="en", random_seed=42)
    evaluate(chat, "MMLU")
    