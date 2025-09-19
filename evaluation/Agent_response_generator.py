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
from src.utils.config import config

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.dataset.Agent import Agent
from src.utils.utils_loading import load_dataset_info
from src.api.ConversationAPI import *

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


class Agent_response_generator:
    def __init__(self, 
                user_id: str,
                dataset_name: str,
                agent_1_model: str = "gpt-4o",  # Agent_1 固定为 gpt-4o
                agent_2_model: str = None,      # Agent_2 为待测模型
                response_agent_model: str = None,  # response_agent 为待测模型
                model_key: str = None,
                api_base: str = None,
                business_id: str = None,
                question_limitation: int = None,
                max_workers: int = 4):
        
        # 加载数据集
        dataset = Agent(dataset_name)
        
        # 获取所有agent信息
        self.agents = dataset.agents
        self.max_turn = dataset.max_turn if dataset.max_turn else 7
        self.language = dataset.language
        
        # 模型配置
        self.dataset_name = dataset_name
        self.user_id = user_id
        self.agent_1_model = agent_1_model
        self.agent_2_model = agent_2_model if agent_2_model else agent_1_model
        self.response_agent_model = response_agent_model if response_agent_model else agent_1_model
        self.model_key = model_key
        self.api_base = api_base
        self.business_id = business_id
        self.max_workers = max_workers
        
        # 获取数据长度
        if 'Agent_1' in self.agents and self.agents['Agent_1']['data']:
            self.data_length = len(self.agents['Agent_1']['data'])
        else:
            self.data_length = 0
            
        self.question_limitation = question_limitation if question_limitation else self.data_length

    def generate_business_id(self):
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        safe_model_name = re.sub(r'[\\/:*?"<>|]', '_', self.agent_2_model).strip(' .') or 'unknown_model'
        return f"{self.dataset_name}_Agent_{safe_model_name}_{current_time}"

    def create_conversation_api(self, model_name: str, system_prompt: str, conversation_id: str = None):
        """创建对话API实例"""
        return ConversationAPI(
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt="",
            image_input=None,
            temperature=0.7,
            conversation_id=conversation_id,
            model_key=self.model_key,
            api_base=self.api_base,
            enable_history=True
        )

    def format_agent_prompt(self, agent_key: str, data_index: int, **kwargs):
        """格式化agent的prompt模板"""
        if agent_key not in self.agents:
            return None
            
        agent = self.agents[agent_key]
        template = agent.get('prompt_template', None)
        data = agent['data'][data_index] if agent['data'] and data_index < len(agent['data']) else None
        
        if not template or data is None:
            return None
            
        # 准备格式化参数
        format_args = kwargs.copy()
        
        if isinstance(data, dict):
            format_args.update(data)
        elif isinstance(data, str):
            # 如果data是字符串，尝试用keys映射
            keys = agent.get('keys', [])
            if isinstance(keys, list) and len(keys) == 1:
                format_args[keys[0]] = data
            elif isinstance(keys, str):
                format_args[keys] = data
        
        try:
            return template.format(**format_args)
        except KeyError as e:
            print(f"Warning: Missing key {e} when formatting prompt for agent {agent_key}")
            return template

    def generate_single_agent_conversation(self, data_index: int):
        """为单个数据生成agent对话"""
        try:
            # 获取各agent的prompt
            agent_1_prompt = self.format_agent_prompt('Agent_1', data_index)
            agent_2_prompt = self.format_agent_prompt('Agent_2', data_index)
            response_agent_prompt = self.format_agent_prompt('response_agent', data_index)
            
            if not all([agent_1_prompt, agent_2_prompt, response_agent_prompt]):
                return data_index, {"error": "Failed to format agent prompts"}
            
            # 创建对话API实例
            agent_1_api = self.create_conversation_api(
                self.agent_1_model, 
                agent_1_prompt, 
                f"agent1_{data_index}"
            )
            agent_2_api = self.create_conversation_api(
                self.agent_2_model, 
                agent_2_prompt, 
                f"agent2_{data_index}"
            )
            response_agent_api = self.create_conversation_api(
                self.response_agent_model, 
                response_agent_prompt, 
                f"response_agent_{data_index}"
            )
            
            # 存储对话历史
            conversation_history = []
            
            # 开始对话 - Agent_2先提问
            agent_2_api.update_prompt("您好，请问有什么可以帮您？", None)
            initial_question = agent_2_api.generate_response()
            conversation_history.append({
                "turn": 0,
                "agent": "Agent_2",
                "message": initial_question
            })
            
            # 更新response_agent的对话历史
            response_agent_api.update_prompt(f"医生：{initial_question}", None)
            current_response = response_agent_api.generate_response()
            
            # 进行多轮对话
            for turn in range(1, self.max_turn + 1):
                # Agent_1 回应
                agent_1_api.update_prompt(initial_question if turn == 1 else agent_2_response, None)
                agent_1_response = agent_1_api.generate_response()
                conversation_history.append({
                    "turn": turn,
                    "agent": "Agent_1",
                    "message": agent_1_response
                })
                
                # 更新response_agent的对话历史
                response_agent_api.update_prompt(f"病人：{agent_1_response}", None)
                current_response = response_agent_api.generate_response()
                
                # 如果是最后一轮，结束对话
                if turn >= self.max_turn:
                    break
                
                # Agent_2 继续提问
                agent_2_api.update_prompt(agent_1_response, None)
                agent_2_response = agent_2_api.generate_response()
                conversation_history.append({
                    "turn": turn,
                    "agent": "Agent_2", 
                    "message": agent_2_response
                })
                
                # 更新response_agent的对话历史
                response_agent_api.update_prompt(f"医生：{agent_2_response}", None)
                current_response = response_agent_api.generate_response()
            
            # 获取参考答案
            reference_answer = None
            if 'answer' in self.agents and self.agents['answer']['data']:
                if data_index < len(self.agents['answer']['data']):
                    reference_answer = self.agents['answer']['data'][data_index]
            
            result = {
                "conversation_history": conversation_history,
                "final_response": current_response,
                "reference_answer": reference_answer,
                "max_turn_reached": len(conversation_history) >= self.max_turn * 2 - 1
            }
            
            return data_index, result
            
        except Exception as e:
            return data_index, {"error": f"Error in conversation generation: {str(e)}"}

    def generate_responses(self):
        """生成所有agent对话响应"""
        if self.question_limitation >= self.data_length:
            self.question_limitation = self.data_length
            
        if self.business_id is None:
            self.business_id = self.generate_business_id()
        
        result_file = f"results/{self.user_id}/{self.business_id}_result.json"
        
        # 初始化结果文件
        initial_results = []
        for i in range(self.question_limitation):
            # 获取参考答案
            reference_answer = None
            if 'answer' in self.agents and self.agents['answer']['data']:
                if i < len(self.agents['answer']['data']):
                    reference_answer = self.agents['answer']['data'][i]
            
            initial_results.append({
                "id": i,
                "response": "Neglected", 
                "answer": reference_answer
            })
        
        write_json_file(initial_results, result_file)
        
        # 并行生成响应
        args_list = list(range(self.question_limitation))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.generate_single_agent_conversation, args): args for args in args_list}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Agent对话评测中"):
                idx, response = future.result()
                if "error" not in response:
                    initial_results[idx]['response'] = response['final_response']
                else:
                    initial_results[idx]['response'] = f"Error: {response['error']}"
                
                write_json_file(initial_results, result_file)
        
        print(f"\nAgent对话评测完成！结果已保存到: {result_file}")
        return self.business_id


if __name__ == "__main__":
    # 使用配置模块获取API密钥，而不是硬编码
    try:
        api_key = config.get_api_key_safe()
        api_base = config.get_api_base_safe()
    except ValueError as e:
        print(f"配置错误: {e}")
        print("请设置环境变量OPENAI_API_KEY和OPENAI_API_BASE")
        exit(1)
    
    generator = Agent_response_generator(
        user_id="test",
        dataset_name="IOR-Dynamic",
        agent_1_model="gpt-4o",
        agent_2_model="doubao-1.5-pro-32k", 
        response_agent_model="doubao-1.5-pro-32k",
        model_key=api_key,
        api_base=api_base,
        business_id=None,
        question_limitation=50,
        max_workers=4
    )
    print(generator.generate_responses())