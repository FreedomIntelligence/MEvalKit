"""
数据库模块
提供评测结果的数据库存储功能
"""

from .models import db_manager, EvaluationResult, EvaluationTask, Base

__all__ = ['db_manager', 'EvaluationResult', 'EvaluationTask', 'Base'] 