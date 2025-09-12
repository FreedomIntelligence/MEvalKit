import sys
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import concurrent.futures

from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.dataset.QA import QA
from src.utils.utils_loading import load_dataset_info
from src.api.ConversationAPI import *

DEFAULT_GENERATE_SYSTEM_PROMPT_EN = """
You are a reliable assistant that can answer questions under the circumstance of the task.
You will be given a question, and you need to answer the question correctly, politely and in detail.
You may also be given the background of the task and the case of every single question for help.
"""

DEFAULT_GENERATE_SYSTEM_PROMPT_ZH = """
你是一个可靠的AI助手，可以在任务的特定情景下回答问题。
你将获取一个问题，并需要正确、礼貌且详细地回答问题。
作为帮助，你可能会获取任务的背景和每个问题的案例。
"""

def write_json_file(data, file_path):
    try:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"写入json文件出错：{str(e)}")
        return False

def read_json_file(file_path):
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"读取json文件出错：{str(e)}")
        return None


class QA_answer_generator:
    def __init__(self, 
                user_id: str,
                dataset_name: str,
                model_name: str,
                model_key: str,
                api_base: str,
                business_id: str,
                question_limitation: int,
                max_workers: int):
        dataset = QA(dataset_name)
        self.background = dataset.background
        self.language = dataset.language
        self.case = dataset.case
        self.question = dataset.question
        self.image = dataset.image
        self.answer = dataset.reference_answer
        self.reference = dataset.reference

        self.dataset_name = dataset_name
        self.user_id = user_id
        self.model_name = model_name
        self.model_key = model_key
        self.api_base = api_base
        self.business_id = business_id
        self.question_limitation = question_limitation
        self.max_workers = max_workers

    def generate_prompt(self, template, content):
        if isinstance(content, str):
            content = [content]

        placeholders = re.findall(r'\{(\w+)\}', template)
        format_dict = {}
        
        for i, placeholder in enumerate(placeholders):
            if i < len(content):
                # # 如果content[i]是字典，尝试获取对应的字段
                # if isinstance(content[i], dict):
                #     if placeholder in content[i]:
                #         format_dict[placeholder] = content[i][placeholder]
                #     elif 'content' in content[i]:  # HealthBench特殊情况
                #         format_dict[placeholder] = content[i]['content']
                #     else:
                #         format_dict[placeholder] = str(content[i])
                # else:
                format_dict[placeholder] = content[i]

        return template.format(**format_dict)

    def generate_business_id(self):
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        safe_model_name = re.sub(r'[\\/:*?"<>|]', '_', self.model_name).strip(' .') or 'unknown_model'
        return f"{self.dataset_name}_{safe_model_name}_{current_time}"


    
    def generate_single_response(self, args):
        i = args
        case_data_content = self.case['data'][i] if self.case['data'] is not None else None
        case_template = self.case['prompt_template']
        question_data_content = self.question['data'][i] if self.question['data'] is not None else None
        question_template = self.question['prompt_template']
        image = self.image['data'][i] if self.image['data'] is not None else None
        answer_data_content = self.answer['data'][i] if self.answer['data'] is not None else "Neglected"
        reference_data_content = self.reference['data'][i] if self.reference['data'] is not None else None
        
        system_prompt = ""
        if self.background is not None:
            system_prompt = self.background
        else:
            if self.language == "zh":
                system_prompt = DEFAULT_GENERATE_SYSTEM_PROMPT_ZH
            else:
                system_prompt = DEFAULT_GENERATE_SYSTEM_PROMPT_EN
        
        case_prompt = ""
        question_prompt = ""
        if case_data_content is not None:
            case_prompt = self.generate_prompt(case_template, case_data_content)
        temperature = 0 if self.answer['data'] is not None and self.answer['data'] is not None else 0.7
        # 检查是否为多轮对话（question_data_content是一个包含多个字符串的列表）
        if (question_data_content is not None and isinstance(question_data_content, list) and 
            len(question_data_content) > 1 and all(isinstance(item, str) for item in question_data_content)):
            # 多轮对话处理
            # 为多轮对话启用历史记录
            chat = ConversationAPI(
                model_name=self.model_name,
                system_prompt=system_prompt,
                user_prompt="",  # 初始user_prompt为空
                image_input=image,
                temperature=temperature,
                conversation_id=f"question_{i}",
                model_key=self.model_key,
                api_base=self.api_base,
                enable_history=True
            )
            
            # 如果有case信息，先发送case
            if case_prompt:
                chat.update_prompt(case_prompt, image)
                case_response = chat.generate_response()
            
            responses = []
            # 逐轮处理每个问题
            for turn_idx, turn_question in enumerate(question_data_content):
                if turn_question is not None:
                    question_prompt = self.generate_prompt(question_template, turn_question)
                    chat.update_prompt(question_prompt, image)
                    response = chat.generate_response()
                    # Get answer for this turn
                    turn_answer = "Neglected"
                    if answer_data_content != "Neglected" and isinstance(answer_data_content, list) and turn_idx < len(answer_data_content):
                        turn_answer = answer_data_content[turn_idx] if answer_data_content[turn_idx] is not None else "Neglected"
                    
                    responses.append({
                        "turn": turn_idx + 1,
                        "question": turn_question,
                        "answer": turn_answer,
                        "response": response,
                        "reference": reference_data_content
                    })
                else:
                    responses.append({
                        "turn": turn_idx + 1,
                        "question": None,
                        "answer": "Neglected",
                        "response": "No question provided for this turn",
                        "reference": reference_data_content
                    })
            
            return i, responses
        
        else:
            # 单轮对话处理（包括string类型的question_data_content）
            question_prompt = ""
            if question_data_content is not None:
                if isinstance(question_data_content, str):
                    # 直接使用字符串
                    question_prompt = self.generate_prompt(question_template, question_data_content)
                else:
                    # 其他类型也尝试用模板处理
                    question_prompt = self.generate_prompt(question_template, question_data_content)
            
            full_prompt = case_prompt + "\n" + question_prompt if case_prompt else question_prompt

            chat = ConversationAPI(
                model_name=self.model_name,
                system_prompt=system_prompt,
                user_prompt=full_prompt.strip(),
                image_input=image,
                temperature=temperature,
                conversation_id=None,
                model_key=self.model_key,
                api_base=self.api_base,
                enable_history=False  # 单轮对话不需要历史记录
            )
            
            response = chat.generate_response()
            
            # Create result structure for single-turn
            result = response
            
            return i, result

    def generate_responses(self):
        if self.question_limitation >= len(self.question['data']):
            self.question_limitation = len(self.question['data'])
        if self.business_id is None:
            self.business_id = self.generate_business_id()
            result_file = f"results/{self.user_id}/{self.business_id}_result.json"
            
            # Initialize result file with new structure
            initial_results = []
            for i in range(self.question_limitation):
                question_content = self.question['data'][i] if self.question['data'] is not None else None
                answer_content = self.answer['data'][i] if self.answer['data'] is not None else "Neglected"
                reference_content = self.reference['data'][i] if self.reference['data'] is not None else None
                
                # Check if it's multi-turn
                if (question_content is not None and isinstance(question_content, list) and 
                    len(question_content) > 1 and all(isinstance(item, str) for item in question_content)):
                    # Multi-turn format - initialize turns
                    turns = []
                    for turn_idx, turn_question in enumerate(question_content):
                        turn_answer = "Neglected"
                        if answer_content != "Neglected" and isinstance(answer_content, list) and turn_idx < len(answer_content):
                            turn_answer = answer_content[turn_idx] if answer_content[turn_idx] is not None else "Neglected"
                        
                        turns.append({
                            "turn": turn_idx + 1,
                            "question": turn_question,
                            "answer": turn_answer,
                            "response": "Neglected",
                            "reference": reference_content
                        })
                    initial_results.append({"id": i, "response": turns})
                else:
                    # Single-turn format
                    initial_results.append({
                        "id": i, 
                        "question": question_content,
                        "answer": answer_content,
                        "response": "Neglected",
                        "reference": reference_content
                    })
                    
            write_json_file(initial_results, result_file)
            
            
            args_list = list(range(self.question_limitation))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.generate_single_response, args): args for args in args_list}
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="评测中"):
                    idx, response = future.result()
                    initial_results[idx]['response'] = response
                    write_json_file(initial_results, result_file)
                    
            print(f"\n评测完成！结果已保存到: {result_file}")
            
        else:
            import glob
            result_pattern = f"results/{self.user_id}/*{self.business_id}_result.json"
            score_pattern = f"results/{self.user_id}/*{self.business_id}_score.json"
            matching_result_files = glob.glob(result_pattern)
            matching_score_files = glob.glob(score_pattern)
            if not matching_result_files or not matching_score_files:
                raise FileNotFoundError(f"找不到business_id为{self.business_id}的结果文件")
            # 如果找到多个匹配文件，使用第一个
            matching_result_file = matching_result_files[0]
            matching_score_file = matching_score_files[0]

            existing_results = read_json_file(matching_result_file)
            existing_score_results = read_json_file(matching_score_file)
            if not existing_results:
                # Initialize with new structure if file doesn't exist
                existing_results = []
                for i in range(self.question_limitation):
                    question_content = self.question['data'][i] if self.question['data'] is not None else None
                    answer_content = self.answer['data'][i] if self.answer['data'] is not None else "Neglected"
                    reference_content = self.reference['data'][i] if self.reference['data'] is not None else None
                    
                    # Check if it's multi-turn
                    if (question_content is not None and isinstance(question_content, list) and 
                        len(question_content) > 1 and all(isinstance(item, str) for item in question_content)):
                        # Multi-turn format - initialize turns
                        turns = []
                        for turn_idx, turn_question in enumerate(question_content):
                            turn_answer = "Neglected"
                            if answer_content != "Neglected" and isinstance(answer_content, list) and turn_idx < len(answer_content):
                                turn_answer = answer_content[turn_idx] if answer_content[turn_idx] is not None else "Neglected"
                            
                            turns.append({
                                "turn": turn_idx + 1,
                                "question": turn_question,
                                "answer": turn_answer,
                                "response": "Neglected",
                                "reference": reference_content
                            })
                        existing_results.append({"id": i, "response": turns})
                    else:
                        # Single-turn format
                        existing_results.append({
                            "id": i, 
                            "question": question_content,
                            "answer": answer_content,
                            "response": "Neglected",
                            "reference": reference_content
                        })
                        
                write_json_file(existing_results, matching_result_file)
            if not existing_score_results:
                existing_score_results = {
                    "valid_ratio": 0.0,
                    "score": 0.0
                }
                write_json_file(existing_score_results, matching_score_file)
            
            args_list = []

            for i in range(self.question_limitation):
                # Check if response needs to be generated based on new structure
                result_item = existing_results[i]
                
                # Handle multi-turn case
                if isinstance(result_item.get("response"), list):
                    # Multi-turn: check if any turn has "Neglected" response
                    needs_processing = any(turn.get("response") == "Neglected" for turn in result_item["response"])
                    if needs_processing:
                        args_list.append(i)
                # Handle single-turn case  
                elif result_item.get("response") == "Neglected":
                    args_list.append(i)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.generate_single_response, args): args for args in args_list}
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="评测中"):
                    idx, response = future.result()
                    existing_results[idx]['response'] = response
                    write_json_file(existing_results, matching_result_file)
                    
            print(f"\n评测完成！结果已保存到: {matching_result_file}")
        
        return self.business_id

    


if __name__ == "__main__":
    generator = QA_answer_generator(
        user_id="test",
        dataset_name="DotaBench",
        model_name="doubao-1.5-pro-32k",
        model_key="sk-fPz5uPZn2ubb9Qexx62yWcFl55Z46iRdBfdlvnjufQ6o0BVo",
        api_base="https://api.huatuogpt.cn/v1",
        business_id=None,
        question_limitation=10,
        max_workers=64
    )
    print(generator.generate_responses())
