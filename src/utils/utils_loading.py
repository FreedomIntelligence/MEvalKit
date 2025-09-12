import json
import yaml
import datasets
from datasets import load_dataset
import os
from pathlib import Path
from typing import Dict, Any, Union


def load_dataset_info(path: str) -> Dict[str, Any]:
    """
    加载数据集配置文件，支持JSON和YAML格式
    
    Args:
        path (str): 配置文件路径，支持.json、.yaml、.yml扩展名
        
    Returns:
        Dict[str, Any]: 解析后的配置字典
        
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 不支持的文件格式
        Exception: 解析错误
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    
    # 根据文件扩展名决定加载方式
    file_ext = Path(path).suffix.lower()
    
    try:
        if file_ext in ['.yaml', '.yml']:
            # 加载YAML文件
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        elif file_ext == '.json':
            # 加载JSON文件（保持原有功能）
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 尝试自动检测格式
            return _auto_detect_and_load(path)
    except yaml.YAMLError as e:
        raise ValueError(f"YAML格式错误: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON格式错误: {e}")
    except Exception as e:
        raise Exception(f"加载配置文件时出错: {e}")


def _auto_detect_and_load(path: str) -> Dict[str, Any]:
    """
    自动检测文件格式并加载
    
    Args:
        path (str): 文件路径
        
    Returns:
        Dict[str, Any]: 解析后的配置字典
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # 尝试作为YAML加载
        try:
            return yaml.safe_load(content)
        except yaml.YAMLError:
            pass
        
        # 尝试作为JSON加载
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # 如果都失败，抛出错误
        raise ValueError(f"无法识别文件格式: {path}")
        
    except Exception as e:
        raise Exception(f"自动检测文件格式失败: {e}")


def load_yaml_dataset_info(path: str) -> Dict[str, Any]:
    """
    专门加载YAML格式的数据集配置文件
    
    Args:
        path (str): YAML配置文件路径
        
    Returns:
        Dict[str, Any]: 解析后的配置字典
    """
    if not path.endswith(('.yaml', '.yml')):
        raise ValueError(f"文件路径必须以.yaml或.yml结尾: {path}")
    
    return load_dataset_info(path)


def load_json_dataset_info(path: str) -> Dict[str, Any]:
    """
    专门加载JSON格式的数据集配置文件（保持原有功能）
    
    Args:
        path (str): JSON配置文件路径
        
    Returns:
        Dict[str, Any]: 解析后的配置字典
    """
    if not path.endswith('.json'):
        raise ValueError(f"文件路径必须以.json结尾: {path}")
    
    return load_dataset_info(path)








# 保持原有的数据集加载函数不变
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True


def load_dataset_compile(dataset_info, loading_way):
    dataset_paths = dataset_info['path']
    
    # 修正loading_way，jsonl实际使用json格式加载
    actual_loading_way = "json" if loading_way == "jsonl" else loading_way
    
    if isinstance(dataset_paths, str):
        data_files = os.path.normpath(dataset_paths)
        dataset = load_dataset(
            actual_loading_way,
            data_files={'test': data_files}
        )['test']
    else:
        dataset_list = []
        for dataset_path in dataset_paths:
            data_files = os.path.normpath(dataset_path)
            dataset = load_dataset(
                actual_loading_way,
                data_files={'test': data_files}
            )['test']
            dataset_list.append(dataset)
        dataset = datasets.concatenate_datasets(dataset_list)
    return dataset
    

# # 将csv文件加载为dataset对象   
# def load_dataset_csv(dataset_info):
#     dataset_paths = dataset_info['path']
#     if isinstance(dataset_paths, str):
#         data_files = os.path.normpath(dataset_paths)
#         dataset = load_dataset(
#             'csv',
#             data_files={'test': data_files},
#             delimiter=','
#         )['test']
#     else:
#         dataset_list = []
#         for dataset_path in dataset_paths:
#             data_files = os.path.normpath(dataset_path)
#             dataset = load_dataset(
#                 'csv',
#                 data_files={'test': data_files},
#                 delimiter=','
#             )['test']
#             dataset_list.append(dataset)
#         dataset = datasets.concatenate_datasets(dataset_list)
#     return dataset

# # 将json文件加载为dataset对象
# def load_dataset_json(dataset_info):
#     dataset_paths = dataset_info['path']
#     if isinstance(dataset_paths, str):
#         data_files = os.path.normpath(dataset_paths)
#         dataset = load_dataset(
#             'json',
#             data_files= {'test': data_files}
#         )['test']
#     else:
#         dataset_list = []
#         for dataset_path in dataset_paths:
#             data_files = os.path.normpath(dataset_path)
#             dataset = load_dataset(
#                 'json',
#                 data_files= {'test': data_files}
#             )['test']
#             dataset_list.append(dataset)
#         dataset = datasets.concatenate_datasets(dataset_list)
#     return dataset

# # 将parquet文件加载为dataset对象
# def load_dataset_parquet(dataset_info):
#     dataset_paths = dataset_info['path']
#     if isinstance(dataset_paths, str):
#         data_files = os.path.normpath(dataset_paths)
#         dataset = load_dataset(
#             'parquet',
#             data_files={'test': data_files}
#         )['test']
#     else:
#         dataset_list = []
#         for dataset_path in dataset_paths:
#             data_files = os.path.normpath(dataset_path)
#             dataset = load_dataset(
#                 'parquet',
#                 data_files={'test': data_files}
#             )['test']
#             dataset_list.append(dataset)
#         dataset = datasets.concatenate_datasets(dataset_list)
#     return dataset

# loading_map = {
#     'csv': load_dataset_csv,
#     'json': load_dataset_json,
#     'jsonl': load_dataset_json,
#     'parquet': load_dataset_parquet
# }

if __name__ == '__main__':
    pass