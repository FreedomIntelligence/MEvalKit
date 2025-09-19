"""
配置管理模块

该模块提供统一的环境变量管理和默认值处理，
确保API密钥等敏感信息的安全处理。

作者: MEvalKit Team
版本: 1.0.0
"""

import os
from typing import Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def get_api_key(key_name: str = "OPENAI_API_KEY", default: Optional[str] = None) -> str:
    """
    安全地获取API密钥
    
    Args:
        key_name: 环境变量名称
        default: 默认值（如果未设置环境变量）
        
    Returns:
        str: API密钥
        
    Raises:
        ValueError: 如果未找到API密钥且无默认值
    """
    api_key = os.getenv(key_name, default)
    if not api_key:
        raise ValueError(f"API密钥未设置: 请设置环境变量 {key_name} 或提供有效的默认值")
    return api_key

def get_api_base(base_name: str = "OPENAI_API_BASE", default: Optional[str] = None) -> str:
    """
    安全地获取API基础URL
    
    Args:
        base_name: 环境变量名称
        default: 默认值（如果未设置环境变量）
        
    Returns:
        str: API基础URL
        
    Raises:
        ValueError: 如果未找到API基础URL且无默认值
    """
    api_base = os.getenv(base_name, default)
    if not api_base:
        raise ValueError(f"API基础URL未设置: 请设置环境变量 {base_name} 或提供有效的默认值")
    return api_base

def get_default_model_config() -> dict:
    """
    获取默认的模型配置
    
    Returns:
        dict: 包含模型配置的字典
    """
    return {
        "model_key": get_api_key("OPENAI_API_KEY", ""),
        "api_base": get_api_base("OPENAI_API_BASE", "https://api.openai.com/v1"),
        "judge_model": os.getenv("JUDGE_MODEL", "gpt-4o"),
        "default_model": os.getenv("DEFAULT_MODEL", "gpt-4o")
    }

def validate_environment() -> bool:
    """
    验证必需的环境变量是否已设置
    
    Returns:
        bool: 如果所有必需的环境变量都已设置则返回True
    """
    required_vars = ["OPENAI_API_KEY", "OPENAI_API_BASE"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"警告: 以下环境变量未设置: {', '.join(missing_vars)}")
        print("请在.env文件中设置这些变量或通过其他方式提供")
        return False
    
    return True

# 全局配置对象
class Config:
    """全局配置类"""
    
    def __init__(self):
        # 尝试验证环境变量
        self.env_valid = validate_environment()
        
        # 获取配置值，提供默认值以避免程序崩溃
        self.default_api_key = os.getenv("OPENAI_API_KEY", "")
        self.default_api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.judge_model = os.getenv("JUDGE_MODEL", "gpt-4o")
        self.default_model = os.getenv("DEFAULT_MODEL", "gpt-4o")
        
        # 数据库配置
        self.mysql_host = os.getenv("MYSQL_HOST", "localhost")
        self.mysql_port = os.getenv("MYSQL_PORT", "3306")
        self.mysql_user = os.getenv("MYSQL_USER", "root")
        self.mysql_password = os.getenv("MYSQL_PASSWORD", "")
        self.mysql_database = os.getenv("MYSQL_DATABASE", "mevalkit")
    
    def get_api_key_safe(self, provided_key: Optional[str] = None) -> str:
        """
        安全地获取API密钥，优先使用提供的密钥
        
        Args:
            provided_key: 提供的API密钥
            
        Returns:
            str: API密钥
        """
        if provided_key and provided_key.strip():
            return provided_key
        if self.default_api_key:
            return self.default_api_key
        raise ValueError("未提供API密钥，且环境变量OPENAI_API_KEY未设置")
    
    def get_api_base_safe(self, provided_base: Optional[str] = None) -> str:
        """
        安全地获取API基础URL，优先使用提供的URL
        
        Args:
            provided_base: 提供的API基础URL
            
        Returns:
            str: API基础URL
        """
        if provided_base and provided_base.strip():
            return provided_base
        return self.default_api_base

# 全局配置实例
config = Config()