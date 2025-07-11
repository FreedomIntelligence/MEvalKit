#!/usr/bin/env python3
"""
测试business_id生成和文件名安全性
"""

import sys
import os
from pathlib import Path

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入app.py中的函数
from app import sanitize_filename, generate_business_id, get_original_model_name, model_name_mapping

def test_sanitize_filename():
    """测试文件名清理功能"""
    print("=== 测试文件名清理功能 ===")
    
    test_cases = [
        "Pro/Qwen/Qwen2-VL-7B-Instruct",
        "gpt-3.5-turbo",
        "doubao-1.5-pro-32k",
        "model/with/slashes",
        "model:with:colons",
        "model*with*asterisks",
        "model?with?question",
        "model\"with\"quotes",
        "model<with>brackets",
        "model|with|pipes",
        "model with spaces",
        "model.with.dots",
        "  model with leading/trailing spaces  ",
        "",  # 空字符串
        "   ",  # 只有空格
        "....",  # 只有点
    ]
    
    for original in test_cases:
        sanitized = sanitize_filename(original)
        restored = get_original_model_name(sanitized)
        print(f"原始: '{original}'")
        print(f"清理后: '{sanitized}'")
        print(f"恢复后: '{restored}'")
        is_safe = '/' not in sanitized and '\\' not in sanitized
        print(f"是否安全: {'✅' if is_safe else '❌'}")
        print(f"映射正确: {'✅' if restored == original else '❌'}")
        print("-" * 50)

def test_generate_business_id():
    """测试business_id生成功能"""
    print("\n=== 测试business_id生成功能 ===")
    
    test_cases = [
        ("MMLU", "Pro/Qwen/Qwen2-VL-7B-Instruct"),
        ("MMStar", "gpt-4o"),
        ("GPQA", "doubao-1.5-pro-32k"),
        ("CMB", "model/with/slashes"),
    ]
    
    for dataset, model in test_cases:
        business_id = generate_business_id(dataset, model)
        print(f"数据集: {dataset}")
        print(f"模型: {model}")
        print(f"business_id: {business_id}")
        is_safe = '/' not in business_id and '\\' not in business_id
        print(f"是否安全: {'✅' if is_safe else '❌'}")
        print("-" * 50)

def test_model_mapping():
    """测试模型名称映射"""
    print("\n=== 测试模型名称映射 ===")
    print("当前映射关系:")
    for safe_name, original_name in model_name_mapping.items():
        print(f"  '{safe_name}' -> '{original_name}'")

def test_file_path_safety():
    """测试文件路径安全性"""
    print("\n=== 测试文件路径安全性 ===")
    
    test_cases = [
        ("MMLU", "Pro/Qwen/Qwen2-VL-7B-Instruct"),
        ("MMStar", "model:with:colons"),
        ("GPQA", "model*with*asterisks"),
    ]
    
    for dataset, model in test_cases:
        business_id = generate_business_id(dataset, model)
        
        # 模拟文件路径
        result_file = f"results/test/{business_id}_result.json"
        score_file = f"results/test/{business_id}_score.json"
        
        print(f"数据集: {dataset}")
        print(f"模型: {model}")
        print(f"结果文件: {result_file}")
        print(f"分数文件: {score_file}")
        
        # 检查路径是否安全
        try:
            # 尝试创建路径对象
            path = Path(result_file)
            print(f"路径创建: ✅ 成功")
        except Exception as e:
            print(f"路径创建: ❌ 失败 - {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    print("开始测试business_id生成和文件名安全性...")
    
    test_sanitize_filename()
    test_generate_business_id()
    test_model_mapping()
    test_file_path_safety()
    
    print("\n=== 测试完成 ===") 