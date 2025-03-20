import json
from datasets import load_dataset
import os




def load_dataset_info(path):
    with open(path, 'r') as f:
        return json.load(f)
    
def load_dataset_huggingface(dataset_info):
    dataset_path = dataset_info['dataset_path']
    subset_name = dataset_info['subset_name']
    split_name = dataset_info['split_name']
    dataset = load_dataset(path=dataset_path, name=subset_name, split=split_name, trust_remote_code=True)
    return dataset
    
def load_dataset_csv(dataset_info):
    dataset_path = dataset_info['dataset_path']
    data_files = os.path.normpath(dataset_path)
    dataset = load_dataset(
        'csv',
        data_files={'test': data_files},
        delimiter=','
    )['test']
    return dataset

def load_dataset_json(dataset_info):
    dataset_path = dataset_info['dataset_path']
    data_files = os.path.normpath(dataset_path)
    dataset = load_dataset(
        'json',
        data_files={'test': data_files}
    )['test']
    return dataset

loading_map = {
    'huggingface': load_dataset_huggingface,
    'csv': load_dataset_csv,
    'json': load_dataset_json
}