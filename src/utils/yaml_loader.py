import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        file_path (str): YAML文件路径
        
    Returns:
        Dict[str, Any]: 解析后的配置字典
        
    Raises:
        FileNotFoundError: 文件不存在
        yaml.YAMLError: YAML格式错误
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"配置文件不存在: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML格式错误: {e}")
    except Exception as e:
        raise Exception(f"加载配置文件时出错: {e}")


def load_yaml_config_with_validation(file_path: str, schema_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载YAML配置文件并进行验证
    
    Args:
        file_path (str): YAML文件路径
        schema_path (Optional[str]): JSON Schema文件路径（可选）
        
    Returns:
        Dict[str, Any]: 验证后的配置字典
    """
    config = load_yaml_config(file_path)
    
    if schema_path and os.path.exists(schema_path):
        try:
            import jsonschema
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = yaml.safe_load(f)
            jsonschema.validate(instance=config, schema=schema)
            print(f"✅ 配置文件验证通过: {file_path}")
        except ImportError:
            print("⚠️  jsonschema库未安装，跳过验证")
        except Exception as e:
            print(f"⚠️  配置文件验证失败: {e}")
    
    return config


def save_yaml_config(config: Dict[str, Any], file_path: str) -> bool:
    """
    保存配置到YAML文件
    
    Args:
        config (Dict[str, Any]): 要保存的配置字典
        file_path (str): 保存路径
        
    Returns:
        bool: 保存是否成功
    """
    try:
        # 确保目录存在
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
        return True
    except Exception as e:
        print(f"保存YAML文件时出错: {e}")
        return False


def convert_json_to_yaml(json_file_path: str, yaml_file_path: str) -> bool:
    """
    将JSON配置文件转换为YAML格式
    
    Args:
        json_file_path (str): JSON文件路径
        yaml_file_path (str): 输出的YAML文件路径
        
    Returns:
        bool: 转换是否成功
    """
    try:
        import json
        with open(json_file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return save_yaml_config(config, yaml_file_path)
    except Exception as e:
        print(f"转换文件时出错: {e}")
        return False


# 示例使用函数
def load_qa_config() -> Dict[str, Any]:
    """加载QA配置文件"""
    return load_yaml_config("dataset_info/QA_config.yaml")


def load_mcq_config() -> Dict[str, Any]:
    """加载MCQ配置文件"""
    return load_yaml_config("dataset_info/MCQ_config.yaml")


if __name__ == "__main__":
    # 测试函数
    try:
        # 测试加载QA配置
        qa_config = load_qa_config()
        print("QA配置加载成功")
        print(f"包含的数据集: {list(qa_config.keys())}")
        
        # 测试加载MCQ配置
        mcq_config = load_mcq_config()
        print("MCQ配置加载成功")
        print(f"包含的数据集: {list(mcq_config.keys())}")
        
    except Exception as e:
        print(f"测试失败: {e}") 