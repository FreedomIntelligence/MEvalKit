#!/usr/bin/env python3
"""
数据库迁移脚本
将现有的文件评测结果迁移到数据库
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime
import re

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.database.repository import evaluation_repo, task_repo
from src.database.models import db_manager

def extract_info_from_filename(filename):
    """从文件名中提取信息"""
    # 示例文件名: MMLU_gpt-4o_202507150700_score.json
    # 或者: MMLU_manual_testtest_manual_score.json
    
    # 移除_score.json后缀
    base_name = filename.replace("_score.json", "")
    
    # 检查是否为手动评测
    if "_manual_" in base_name:
        # 手动评测格式: dataset_manual_user_manual
        parts = base_name.split("_manual_")
        if len(parts) >= 2:
            dataset_name = parts[0]
            user_part = parts[1]
            # 移除最后的_manual
            if user_part.endswith("_manual"):
                user_part = user_part[:-7]
            user_id = user_part
            evaluation_mode = "manual"
            model_name = "manual"  # 手动评测没有具体模型名
        else:
            return None
    else:
        # 自动评测格式: dataset_model_timestamp
        # 找到最后一个下划线，前面是模型名，后面是时间戳
        last_underscore = base_name.rfind("_")
        if last_underscore > 0:
            prefix = base_name[:last_underscore]
            timestamp = base_name[last_underscore + 1:]
            
            # 从prefix中分离数据集和模型名
            # 数据集名通常不包含下划线，模型名可能包含下划线
            # 这里需要根据实际情况调整逻辑
            if "_" in prefix:
                # 假设第一个下划线分隔数据集和模型名
                first_underscore = prefix.find("_")
                dataset_name = prefix[:first_underscore]
                model_name = prefix[first_underscore + 1:]
            else:
                dataset_name = prefix
                model_name = "unknown"
            
            user_id = "test"  # 默认用户
            evaluation_mode = "automatic"
        else:
            return None
    
    return {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "user_id": user_id,
        "evaluation_mode": evaluation_mode,
        "business_id": base_name
    }

def migrate_file_to_database(score_file_path, result_file_path=None):
    """将单个文件迁移到数据库"""
    try:
        # 从文件名提取信息
        filename = os.path.basename(score_file_path)
        info = extract_info_from_filename(filename)
        
        if not info:
            print(f"无法解析文件名: {filename}")
            return False
        
        # 读取score.json文件
        with open(score_file_path, 'r', encoding='utf-8') as f:
            score_data = json.load(f)
        
        # 读取result.json文件（如果存在）
        result_data = None
        if result_file_path and os.path.exists(result_file_path):
            with open(result_file_path, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
        
        # 准备数据库记录
        db_record = {
            "business_id": info["business_id"],
            "user_id": info["user_id"],
            "dataset_name": info["dataset_name"],
            "model_name": info["model_name"],
            "evaluation_mode": info["evaluation_mode"],
            "eval_type": "unknown",  # 需要根据数据集推断
            "total_questions": score_data.get("total_questions", 0),
            "valid_questions": score_data.get("valid_questions", 0),
            "valid_ratio": score_data.get("completion_ratio", 0.0),
            "raw_score": score_data.get("raw_score", 0.0),
            "score": score_data.get("score", 0.0),
            "result_data": result_data,
            "response_data": None,  # 暂时设为None
            "is_completed": True
        }
        
        # 根据数据集名称推断评测类型
        dataset_name = info["dataset_name"].lower()
        if "mt-bench" in dataset_name or "llmjudge" in dataset_name:
            db_record["eval_type"] = "llmjudge"
        elif "mmstar" in dataset_name or "image" in dataset_name:
            db_record["eval_type"] = "imagemcq"
        else:
            db_record["eval_type"] = "textmcq"
        
        # 保存到数据库
        result = evaluation_repo.save_evaluation_result(db_record)
        
        if result:
            print(f"✓ 成功迁移: {filename}")
            return True
        else:
            print(f"✗ 迁移失败: {filename}")
            return False
            
    except Exception as e:
        print(f"✗ 迁移出错 {filename}: {str(e)}")
        return False

def migrate_all_files():
    """迁移所有文件到数据库"""
    results_dir = Path("results")
    
    if not results_dir.exists():
        print("results目录不存在")
        return
    
    # 统计信息
    total_files = 0
    success_count = 0
    failed_count = 0
    
    # 遍历所有用户目录
    for user_dir in results_dir.iterdir():
        if not user_dir.is_dir():
            continue
        
        user_id = user_dir.name
        print(f"\n处理用户: {user_id}")
        
        # 查找所有score.json文件
        score_files = list(user_dir.glob("*_score.json"))
        
        for score_file in score_files:
            total_files += 1
            
            # 构造对应的result.json文件路径
            business_id = score_file.stem.replace("_score", "")
            result_file = user_dir / f"{business_id}_result.json"
            
            # 迁移文件
            if migrate_file_to_database(str(score_file), str(result_file) if result_file.exists() else None):
                success_count += 1
            else:
                failed_count += 1
    
    print(f"\n迁移完成!")
    print(f"总文件数: {total_files}")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")

def main():
    """主函数"""
    print("开始数据库迁移...")
    print("=" * 50)
    
    # 确保数据库表已创建
    db_manager.create_tables()
    print("数据库表已创建")
    
    # 执行迁移
    migrate_all_files()
    
    print("=" * 50)
    print("迁移完成!")

if __name__ == "__main__":
    main() 