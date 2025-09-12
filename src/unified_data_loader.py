import os
import json
import pandas as pd
import random
import yaml
import re
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod


class UnifiedDataLoader:
    """统一的数据加载和prompt生成框架"""
    
    def __init__(self, config_path: str):
        """
        初始化统一数据加载器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.datasets = self.config['datasets']
        self.prompt_templates = self.config['prompt_templates']
        
    def load_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """
        通用数据加载方法
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            标准化的数据列表，每个元素包含question, choices, answer等字段
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found in config")
            
        dataset_config = self.datasets[dataset_name]
        return self._load_data_generic(dataset_config, dataset_name)
    
    def _load_data_generic(self, config: Dict[str, Any], dataset_name: str) -> List[Dict[str, Any]]:
        """通用数据加载实现"""
        dataset_path = config['dataset_path']
        column_mapping = config['column_mapping']
        preprocessing = config['preprocessing']
        
        # 判断数据源类型和加载方式
        if dataset_path.endswith('.csv'):
            # 单个CSV文件
            raw_data = self._load_csv_data(dataset_path)
        elif dataset_path.endswith('.json') or dataset_path.endswith('.jsonl'):
            # JSON/JSONL文件
            raw_data = self._load_json_data(dataset_path)
        elif dataset_path.endswith('.parquet'):
            # Parquet文件
            raw_data = self._load_parquet_data(dataset_path)
        elif os.path.isdir(dataset_path):
            # 目录包含多个文件
            raw_data = self._load_directory_data(config)
        else:
            raise ValueError(f"Unsupported data source: {dataset_path}")
        
        # 处理数据连接（如CMB的问题和答案分离）
        if 'join_config' in config:
            raw_data = self._join_data_sources(raw_data, config)
        
        # 标准化数据格式
        return self._standardize_data(raw_data, config, dataset_name)
    
    def _load_csv_data(self, file_path: str) -> List[Dict[str, Any]]:
        """加载CSV数据"""
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    
    def _load_parquet_data(self, file_path: str) -> List[Dict[str, Any]]:
        """加载Parquet数据"""
        df = pd.read_parquet(file_path)
        return df.to_dict('records')
    
    def _load_json_data(self, file_path: str) -> List[Dict[str, Any]]:
        """加载JSON/JSONL数据"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.jsonl'):
                # JSONL格式：每行一个JSON对象
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            else:
                # 标准JSON格式
                content = json.load(f)
                if isinstance(content, list):
                    data = content
                else:
                    data = [content]
        return data
    
    def _load_directory_data(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """加载目录中的多个文件"""
        dataset_path = config['dataset_path']
        all_data = []
        
        if 'subdatasets' in config:
            # 指定子数据集文件
            for subdataset in config['subdatasets']:
                file_path = os.path.join(dataset_path, subdataset)
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path, index_col=0 if 'index_col' in config.get('file_options', {}) else None)
                    data = df.to_dict('records')
                    # 为每条记录添加子数据集信息
                    for item in data:
                        item['subdataset'] = subdataset
                        item['subdataset_name'] = subdataset.replace('.csv', '')
                    all_data.extend(data)
                elif file_path.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
        else:
            # 自动发现所有文件
            for file_name in os.listdir(dataset_path):
                file_path = os.path.join(dataset_path, file_name)
                if file_name.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    all_data.extend(df.to_dict('records'))
                elif file_name.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
        
        return all_data
    
    def _join_data_sources(self, raw_data: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """连接多个数据源（如问题和答案文件）"""
        join_config = config['join_config']
        answer_file = config.get('answer_path')
        
        if answer_file:
            # 加载答案文件
            with open(answer_file, 'r', encoding='utf-8') as f:
                answers_data = json.load(f)
            
            # 创建答案映射
            join_key = join_config['join_key']
            answer_key = join_config['answer_file_key']
            answers_dict = {item[answer_key]: item for item in answers_data}
            
            # 连接数据
            for item in raw_data:
                if join_key in item and item[join_key] in answers_dict:
                    item.update(answers_dict[item[join_key]])
        
        return raw_data
    
    def _standardize_data(self, raw_data: List[Dict[str, Any]], config: Dict[str, Any], dataset_name: str) -> List[Dict[str, Any]]:
        """将原始数据标准化为统一格式"""
        column_mapping = config['column_mapping']
        preprocessing = config['preprocessing']
        
        standardized_data = []
        
        for item in raw_data:
            # 基础字段映射
            data_item = {'dataset_name': dataset_name}
            
            # 映射基础字段
            for std_field, source_field in column_mapping.items():
                if source_field in item:
                    data_item[std_field] = item[source_field]
            
            # 处理选项
            data_item = self._process_choices(data_item, item, config)
            
            # 处理特殊字段（如科目名称映射）
            data_item = self._process_special_fields(data_item, item, config)
            
            # 处理图像数据（如果是多模态数据集）
            if config.get('preprocessing', {}).get('multimodal', False):
                data_item = self._process_image_data(data_item, item, config)
            
            # 保留原始数据中的其他有用字段
            for key, value in item.items():
                if key not in data_item:
                    data_item[key] = value
            
            standardized_data.append(data_item)
        
        return standardized_data
    
    def _process_special_fields(self, data_item: Dict[str, Any], raw_item: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """处理特殊字段，如科目名称映射"""
        # 处理科目名称映射（CMMLU_Med类型）
        if 'subject_mapping' in config and 'subdataset_name' in raw_item:
            subject_mapping = config['subject_mapping']
            subdataset_name = raw_item['subdataset_name']
            data_item['subject_name'] = subject_mapping.get(subdataset_name, subdataset_name)
        
        return data_item
    
    def _process_choices(self, data_item: Dict[str, Any], raw_item: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """处理选项数据"""
        preprocessing = config['preprocessing']
        task_type = config.get('task_type', 'MCQ')
        
        # QA数据集不需要处理选项
        if task_type == 'QA':
            return data_item
        
        choice_source = preprocessing.get('choice_source', 'combination')
        
        if choice_source == 'combination':
            # GPQA类型：组合正确答案和错误答案
            choices = []
            correct_answer = None
            
            for choice_config in preprocessing['choice_combination']:
                column = choice_config['column']
                choice_type = choice_config['type']
                
                if column in data_item:
                    choices.append(data_item[column])
                    if choice_type == 'correct':
                        correct_answer = data_item[column]
            
            # 洗牌选项
            if preprocessing.get('shuffle_choices', False):
                random.shuffle(choices)
                correct_index = choices.index(correct_answer) if correct_answer else 0
                data_item['answer'] = chr(65 + correct_index)  # A, B, C, D
            else:
                data_item['answer'] = 'A'  # 默认正确答案在A
            
            # 设置标准化选项
            choice_labels = ['A', 'B', 'C', 'D', 'E']
            data_item['choices'] = {choice_labels[i]: choices[i] for i in range(len(choices))}
            data_item['shuffled_choices'] = choices
            
        elif choice_source == 'options_dict':
            # CMB类型：选项在options字段中
            options = raw_item.get('option', {})
            data_item['choices'] = options
            data_item['formatted_options'] = "\n".join([f"{k}. {v}" for k, v in options.items() if len(v) > 0 and v != " "])
            
        elif choice_source == 'separate_columns':
            # CMMLU类型：选项在单独列中
            choices = {}
            choice_mapping = {'choice_a': 'A', 'choice_b': 'B', 'choice_c': 'C', 'choice_d': 'D'}
            
            for std_field, choice_label in choice_mapping.items():
                if std_field in data_item:
                    choices[choice_label] = data_item[std_field]
            
            data_item['choices'] = choices
            
        elif choice_source == 'embedded_in_question':
            # MMStar类型：question字段已包含完整的问题和选项，直接使用
            # 不需要额外处理，在prompt模板中直接使用question字段
            pass
        
        return data_item
    
    def _process_image_data(self, data_item: Dict[str, Any], raw_item: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """处理图像数据"""
        image_config = config.get('preprocessing', {}).get('image_config', {})
        image_field = image_config.get('image_field', 'image')
        image_type = image_config.get('image_type', 'path')
        
        if image_field in raw_item:
            image_data = raw_item[image_field]
            
            if image_type == 'binary':
                # 图像数据是二进制格式，直接使用
                data_item['image'] = image_data
            elif image_type == 'path':
                # 图像数据是文件路径
                data_item['image'] = image_data
            else:
                # 其他类型，直接使用
                data_item['image'] = image_data
        
        return data_item
    
    def generate_prompt(self, data_item: Dict[str, Any], dataset_name: str, 
                       prompt_type: str = "zero_shot") -> str:
        """
        根据数据项和配置生成prompt
        
        Args:
            data_item: 数据项
            dataset_name: 数据集名称
            prompt_type: prompt类型 (zero_shot, chain_of_thought, five_shot)
            
        Returns:
            生成的prompt字符串
        """
        if prompt_type not in self.prompt_templates:
            raise ValueError(f"Prompt type {prompt_type} not found")
        
        if dataset_name not in self.prompt_templates[prompt_type]:
            raise ValueError(f"Dataset {dataset_name} not found in prompt templates")
        
        template_config = self.prompt_templates[prompt_type][dataset_name]
        template = template_config['template']
        placeholder_mapping = template_config['placeholder_mapping']
        
        # 准备填充参数
        fill_params = {}
        
        for placeholder, source in placeholder_mapping.items():
            if source.startswith('{') and source.endswith('}'):
                # 直接从数据项获取
                field_name = source[1:-1]  # 去掉花括号
                
                if '[' in field_name and ']' in field_name:
                    # 处理数组索引，如 shuffled_choices[0]
                    base_field = field_name.split('[')[0]
                    index = int(field_name.split('[')[1].split(']')[0])
                    fill_params[placeholder] = data_item[base_field][index]
                else:
                    fill_params[placeholder] = data_item.get(field_name, '')
            else:
                # 直接使用字符串值
                fill_params[placeholder] = source
        
        # 处理特殊的few-shot示例
        if prompt_type in ['chain_of_thought', 'five_shot'] and 'few_shot_examples' in fill_params:
            # 这里可以加载few-shot示例的逻辑
            fill_params['few_shot_examples'] = self._load_few_shot_examples(
                template_config.get('few_shot_source', ''), dataset_name)
        
        # 填充模板
        try:
            prompt = template.format(**fill_params)
        except KeyError as e:
            raise ValueError(f"Missing parameter for template filling: {e}")
        
        return prompt
    
    def _load_few_shot_examples(self, source_path: str, dataset_name: str) -> str:
        """加载few-shot示例"""
        if not source_path or not os.path.exists(source_path):
            return ""
        
        # 这里可以根据数据集类型加载不同的few-shot示例
        # 当前返回空字符串，实际使用时需要实现具体逻辑
        return ""
    
    def process_dataset(self, dataset_name: str, prompt_type: str = "zero_shot", 
                       max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        处理完整数据集，生成所有样本的prompt
        
        Args:
            dataset_name: 数据集名称
            prompt_type: prompt类型
            max_samples: 最大样本数量
            
        Returns:
            包含prompt和其他信息的数据列表
        """
        # 加载数据
        data = self.load_dataset(dataset_name)
        
        if max_samples:
            data = data[:max_samples]
        
        # 为每个样本生成prompt
        processed_data = []
        for item in data:
            prompt = self.generate_prompt(item, dataset_name, prompt_type)
            processed_item = {
                **item,
                'prompt': prompt,
                'prompt_type': prompt_type
            }
            processed_data.append(processed_item)
        
        return processed_data


# 使用示例
if __name__ == "__main__":
    # 初始化加载器
    loader = UnifiedDataLoader("dataset_info/unified_MCQ_config.yaml")
    
    # 测试GPQA数据集
    print("Testing GPQA dataset...")
    gpqa_data = loader.process_dataset("GPQA", "zero_shot", max_samples=2)
    for item in gpqa_data:
        print(f"Question: {item['question'][:50]}...")
        print(f"Choices: {item['choices']}")
        print(f"Answer: {item['answer']}")
        print(f"Prompt: {item['prompt'][:100]}...")
        print("-" * 50)
    
    # 测试CMB数据集
    print("Testing CMB dataset...")
    cmb_data = loader.process_dataset("CMB", "zero_shot", max_samples=2)
    for item in cmb_data:
        print(f"Question: {item['question'][:50]}...")
        print(f"Answer: {item.get('answer', 'N/A')}")
        print(f"Prompt: {item['prompt'][:100]}...")
        print("-" * 50)
    
    # 测试CMMLU_Med数据集
    print("Testing CMMLU_Med dataset...")
    cmmlu_data = loader.process_dataset("CMMLU_Med", "zero_shot", max_samples=2)
    for item in cmmlu_data:
        print(f"Question: {item['question'][:50]}...")
        print(f"Subject: {item.get('subject_name', 'N/A')}")
        print(f"Answer: {item.get('answer', 'N/A')}")
        print(f"Prompt: {item['prompt'][:100]}...")
        print("-" * 50)