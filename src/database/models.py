"""
数据库模型定义模块

该模块定义了MEvalKit项目使用的数据库表结构和数据模型。
主要包含评测结果表和评测任务表，用于存储和管理模型评测的相关数据。

主要功能：
- 定义数据库表结构
- 提供数据模型类
- 支持数据序列化和反序列化
- 管理数据库连接和会话

作者: MEvalKit Team
版本: 1.0.0
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json
from typing import Dict, Any, Optional

Base = declarative_base()

class EvaluationResult(Base):
    """
    评测结果表
    
    存储模型评测的详细结果数据，包括得分、统计信息和原始数据。
    支持多种评测类型：文本多选题、图像多选题、LLMJudge等。
    
    主要字段：
    - business_id: 业务标识符，用于区分不同的评测任务
    - user_id: 用户标识符
    - dataset_name: 数据集名称
    - model_name: 模型名称
    - evaluation_mode: 评测模式（automatic/manual）
    - eval_type: 评测类型（llmjudge/textmcq/imagemcq）
    - score: 最终得分
    - valid_ratio: 有效问题比例
    - result_data: 详细评测结果（JSON格式）
    - response_data: 模型响应数据（JSON格式）
    """
    __tablename__ = 'evaluation_results'
    
    # 主键和索引字段
    id = Column(Integer, primary_key=True, autoincrement=True)
    business_id = Column(String(255), nullable=False, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    dataset_name = Column(String(255), nullable=False)
    model_name = Column(String(255), nullable=False)
    evaluation_mode = Column(String(50), nullable=False, default='automatic')  # automatic/manual
    eval_type = Column(String(50), nullable=False)  # llmjudge/textmcq/imagemcq
    
    # 评测结果统计字段
    total_questions = Column(Integer, default=0)      # 总问题数
    valid_questions = Column(Integer, default=0)      # 有效问题数
    valid_ratio = Column(Float, default=0.0)         # 有效问题比例
    raw_score = Column(Float, default=0.0)           # 原始得分
    score = Column(Float, default=0.0)               # 最终得分
    
    # 详细结果数据（JSON格式存储）
    result_data = Column(JSON)  # 存储详细的评测结果
    response_data = Column(JSON)  # 存储模型响应数据
    
    # 元数据字段
    created_at = Column(DateTime, default=datetime.utcnow)      # 创建时间
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)  # 更新时间
    is_completed = Column(Boolean, default=False)               # 是否完成
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将模型实例转换为字典格式
        
        用于API响应和JSON序列化，便于数据传输和存储。
        
        返回:
            Dict[str, Any]: 包含所有字段的字典
        """
        return {
            'id': self.id,
            'business_id': self.business_id,
            'user_id': self.user_id,
            'dataset_name': self.dataset_name,
            'model_name': self.model_name,
            'evaluation_mode': self.evaluation_mode,
            'eval_type': self.eval_type,
            'total_questions': self.total_questions,
            'valid_questions': self.valid_questions,
            'valid_ratio': self.valid_ratio,
            'raw_score': self.raw_score,
            'score': self.score,
            'result_data': self.result_data,
            'response_data': self.response_data,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'is_completed': self.is_completed
        }

class EvaluationTask(Base):
    """
    评测任务表
    
    存储评测任务的执行状态和进度信息，用于任务管理和监控。
    支持任务状态跟踪、进度监控和错误处理。
    
    主要字段：
    - task_id: 任务唯一标识符
    - status: 任务状态（pending/running/completed/failed）
    - progress: 执行进度（0.0-1.0）
    - current_question: 当前处理的问题编号
    - total_questions: 总问题数
    - error_message: 错误信息（如果任务失败）
    """
    __tablename__ = 'evaluation_tasks'
    
    # 主键和索引字段
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(255), unique=True, nullable=False, index=True)
    business_id = Column(String(255), nullable=False, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    dataset_name = Column(String(255), nullable=False)
    model_name = Column(String(255), nullable=False)
    evaluation_mode = Column(String(50), nullable=False, default='automatic')
    eval_type = Column(String(50), nullable=False)
    
    # 任务状态字段
    status = Column(String(50), default='pending')  # pending/running/completed/failed
    progress = Column(Float, default=0.0)  # 进度百分比（0.0-1.0）
    current_question = Column(Integer, default=0)  # 当前处理的问题编号
    total_questions = Column(Integer, default=0)   # 总问题数
    
    # 任务配置字段
    question_limitation = Column(Integer, default=100)  # 问题数量限制
    max_workers = Column(Integer, default=1)           # 最大工作线程数
    
    # 错误信息字段
    error_message = Column(Text)  # 错误信息（如果任务失败）
    
    # 时间戳字段
    created_at = Column(DateTime, default=datetime.utcnow)  # 创建时间
    started_at = Column(DateTime)                           # 开始时间
    completed_at = Column(DateTime)                         # 完成时间
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将任务实例转换为字典格式
        
        用于API响应和JSON序列化，便于数据传输和存储。
        
        返回:
            Dict[str, Any]: 包含所有字段的字典
        """
        return {
            'id': self.id,
            'task_id': self.task_id,
            'business_id': self.business_id,
            'user_id': self.user_id,
            'dataset_name': self.dataset_name,
            'model_name': self.model_name,
            'evaluation_mode': self.evaluation_mode,
            'eval_type': self.eval_type,
            'status': self.status,
            'progress': self.progress,
            'current_question': self.current_question,
            'total_questions': self.total_questions,
            'question_limitation': self.question_limitation,
            'max_workers': self.max_workers,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }

# 数据库连接和会话管理
class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, db_url: str = "sqlite:///mevalkit.db"):
        self.engine = create_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """创建所有表"""
        Base.metadata.create_all(bind=self.engine)
        
    def get_session(self):
        """获取数据库会话"""
        return self.SessionLocal()
        
    def close_session(self, session):
        """关闭数据库会话"""
        session.close()

# 全局数据库管理器实例
db_manager = DatabaseManager() 