import json
import datasets
from datasets import load_dataset
import os




def load_dataset_info(path):
    with open(path, 'r') as f:
        return json.load(f)

datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True
#将huggingface数据集加载为dataset对象
def load_dataset_huggingface(dataset_info):
    dataset_path = dataset_info['path']
    subset_name = dataset_info['subset_name']
    split_name = dataset_info['split_name']
    if isinstance(subset_name, str):
        dataset = load_dataset(path=dataset_path, name=subset_name, split=split_name)
    else:
        dataset_list = []
        for subset in subset_name:
            dataset = load_dataset(path=dataset_path, name=subset, split=split_name)
            dataset_list.append(dataset)
        dataset = datasets.concatenate_datasets(dataset_list)
    return dataset

# 将csv文件加载为dataset对象   
def load_dataset_csv(dataset_info):
    dataset_paths = dataset_info['path']
    if isinstance(dataset_paths, str):
        data_files = os.path.normpath(dataset_paths)
        dataset = load_dataset(
            'csv',
            data_files={'test': data_files},
            delimiter=','
        )['test']
    else:
        dataset_list = []
        for dataset_path in dataset_paths:
            data_files = os.path.normpath(dataset_path)
            dataset = load_dataset(
                'csv',
                data_files={'test': data_files},
                delimiter=','
            )['test']
            dataset_list.append(dataset)
        dataset = datasets.concatenate_datasets(dataset_list)
    return dataset

# 将json文件加载为dataset对象
def load_dataset_json(dataset_info):
    dataset_paths = dataset_info['path']
    if isinstance(dataset_paths, str):
        data_files = os.path.normpath(dataset_paths)
        dataset = load_dataset(
            'json',
            data_files= {'test': data_files}
        )['test']
    else:
        dataset_list = []
        for dataset_path in dataset_paths:
            data_files = os.path.normpath(dataset_path)
            dataset = load_dataset(
                'json',
                data_files= {'test': data_files}
            )['test']
            dataset_list.append(dataset)
        dataset = datasets.concatenate_datasets(dataset_list)
    return dataset



loading_map = {
    'huggingface': load_dataset_huggingface,
    'csv': load_dataset_csv,
    'json': load_dataset_json,
    'jsonl': load_dataset_json
}