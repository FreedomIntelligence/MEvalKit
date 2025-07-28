"""
数据库操作仓库
提供评测结果的数据库操作接口
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from .models import db_manager, EvaluationResult, EvaluationTask

class EvaluationRepository:
    """评测结果数据库操作仓库"""
    
    def __init__(self):
        self.db_manager = db_manager
    
    def save_evaluation_result(self, result_data: Dict[str, Any]) -> Optional[EvaluationResult]:
        """保存评测结果"""
        session = self.db_manager.get_session()
        try:
            # 检查是否已存在相同business_id的结果
            existing_result = session.query(EvaluationResult).filter(
                EvaluationResult.business_id == result_data['business_id'],
                EvaluationResult.user_id == result_data['user_id']
            ).first()
            
            if existing_result:
                # 更新现有记录
                for key, value in result_data.items():
                    if hasattr(existing_result, key):
                        setattr(existing_result, key, value)
                existing_result.updated_at = datetime.utcnow()
                result = existing_result
            else:
                # 创建新记录
                result = EvaluationResult(**result_data)
                session.add(result)
            
            session.commit()
            return result
        except Exception as e:
            session.rollback()
            print(f"保存评测结果失败: {str(e)}")
            return None
        finally:
            self.db_manager.close_session(session)
    
    def get_evaluation_result(self, business_id: str, user_id: str) -> Optional[EvaluationResult]:
        """获取评测结果"""
        session = self.db_manager.get_session()
        try:
            result = session.query(EvaluationResult).filter(
                and_(
                    EvaluationResult.business_id == business_id,
                    EvaluationResult.user_id == user_id
                )
            ).first()
            return result
        except Exception as e:
            print(f"获取评测结果失败: {str(e)}")
            return None
        finally:
            self.db_manager.close_session(session)
    
    def get_user_evaluations(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户的所有评测记录"""
        session = self.db_manager.get_session()
        try:
            results = session.query(EvaluationResult).filter(
                EvaluationResult.user_id == user_id
            ).order_by(desc(EvaluationResult.created_at)).all()
            
            return [result.to_dict() for result in results]
        except Exception as e:
            print(f"获取用户评测记录失败: {str(e)}")
            return []
        finally:
            self.db_manager.close_session(session)
    
    def update_evaluation_progress(self, business_id: str, user_id: str, 
                                 current_question: int, total_questions: int,
                                 result_data: List[Dict[str, Any]] = None) -> bool:
        """更新评测进度"""
        session = self.db_manager.get_session()
        try:
            result = session.query(EvaluationResult).filter(
                and_(
                    EvaluationResult.business_id == business_id,
                    EvaluationResult.user_id == user_id
                )
            ).first()
            
            if result:
                result.current_question = current_question
                result.total_questions = total_questions
                if result_data:
                    result.result_data = result_data
                result.updated_at = datetime.utcnow()
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"更新评测进度失败: {str(e)}")
            return False
        finally:
            self.db_manager.close_session(session)
    
    def complete_evaluation(self, business_id: str, user_id: str, 
                          final_result: Dict[str, Any]) -> bool:
        """完成评测"""
        session = self.db_manager.get_session()
        try:
            result = session.query(EvaluationResult).filter(
                and_(
                    EvaluationResult.business_id == business_id,
                    EvaluationResult.user_id == user_id
                )
            ).first()
            
            if result:
                result.total_questions = final_result.get('total_questions', 0)
                result.valid_questions = final_result.get('valid_questions', 0)
                result.valid_ratio = final_result.get('valid_ratio', 0.0)
                result.raw_score = final_result.get('raw_score', 0.0)
                result.score = final_result.get('score', 0.0)
                result.result_data = final_result.get('result_data')
                result.response_data = final_result.get('response_data')
                result.is_completed = True
                result.updated_at = datetime.utcnow()
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"完成评测失败: {str(e)}")
            return False
        finally:
            self.db_manager.close_session(session)
    
    def get_leaderboard_data(self, dataset_name: str) -> List[Dict[str, Any]]:
        """获取排行榜数据"""
        session = self.db_manager.get_session()
        try:
            results = session.query(EvaluationResult).filter(
                and_(
                    EvaluationResult.dataset_name == dataset_name,
                    EvaluationResult.is_completed == True
                )
            ).order_by(desc(EvaluationResult.score)).all()
            
            leaderboard = []
            for result in results:
                leaderboard.append({
                    'model_name': result.model_name,
                    'score': result.score,
                    'raw_score': result.raw_score,
                    'timestamp': result.created_at.timestamp() if result.created_at else 0,
                    'date': result.created_at.strftime('%Y-%m-%d %H:%M:%S') if result.created_at else ''
                })
            
            return leaderboard
        except Exception as e:
            print(f"获取排行榜数据失败: {str(e)}")
            return []
        finally:
            self.db_manager.close_session(session)
    
    def delete_evaluation_result(self, business_id: str, user_id: str) -> bool:
        """删除评测结果"""
        session = self.db_manager.get_session()
        try:
            result = session.query(EvaluationResult).filter(
                and_(
                    EvaluationResult.business_id == business_id,
                    EvaluationResult.user_id == user_id
                )
            ).first()
            
            if result:
                session.delete(result)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"删除评测结果失败: {str(e)}")
            return False
        finally:
            self.db_manager.close_session(session)

class TaskRepository:
    """评测任务数据库操作仓库"""
    
    def __init__(self):
        self.db_manager = db_manager
    
    def create_task(self, task_data: Dict[str, Any]) -> Optional[EvaluationTask]:
        """创建评测任务"""
        session = self.db_manager.get_session()
        try:
            task = EvaluationTask(**task_data)
            session.add(task)
            session.commit()
            return task
        except Exception as e:
            session.rollback()
            print(f"创建任务失败: {str(e)}")
            return None
        finally:
            self.db_manager.close_session(session)
    
    def get_task(self, task_id: str) -> Optional[EvaluationTask]:
        """获取任务"""
        session = self.db_manager.get_session()
        try:
            task = session.query(EvaluationTask).filter(
                EvaluationTask.task_id == task_id
            ).first()
            return task
        except Exception as e:
            print(f"获取任务失败: {str(e)}")
            return None
        finally:
            self.db_manager.close_session(session)
    
    def update_task_status(self, task_id: str, status: str, 
                          progress: float = None, error_message: str = None) -> bool:
        """更新任务状态"""
        session = self.db_manager.get_session()
        try:
            task = session.query(EvaluationTask).filter(
                EvaluationTask.task_id == task_id
            ).first()
            
            if task:
                task.status = status
                if progress is not None:
                    task.progress = progress
                if error_message is not None:
                    task.error_message = error_message
                
                if status == 'running' and not task.started_at:
                    task.started_at = datetime.utcnow()
                elif status in ['completed', 'failed']:
                    task.completed_at = datetime.utcnow()
                
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"更新任务状态失败: {str(e)}")
            return False
        finally:
            self.db_manager.close_session(session)
    
    def update_task_progress(self, task_id: str, current_question: int, 
                           total_questions: int) -> bool:
        """更新任务进度"""
        session = self.db_manager.get_session()
        try:
            task = session.query(EvaluationTask).filter(
                EvaluationTask.task_id == task_id
            ).first()
            
            if task:
                task.current_question = current_question
                task.total_questions = total_questions
                task.progress = (current_question / total_questions * 100) if total_questions > 0 else 0
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"更新任务进度失败: {str(e)}")
            return False
        finally:
            self.db_manager.close_session(session)
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """获取活跃任务列表"""
        session = self.db_manager.get_session()
        try:
            tasks = session.query(EvaluationTask).filter(
                EvaluationTask.status.in_(['pending', 'running'])
            ).order_by(desc(EvaluationTask.created_at)).all()
            
            return [task.to_dict() for task in tasks]
        except Exception as e:
            print(f"获取活跃任务失败: {str(e)}")
            return []
        finally:
            self.db_manager.close_session(session)
    
    def cleanup_old_tasks(self, days: int = 7) -> int:
        """清理旧任务"""
        session = self.db_manager.get_session()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            deleted_count = session.query(EvaluationTask).filter(
                and_(
                    EvaluationTask.created_at < cutoff_date,
                    EvaluationTask.status.in_(['completed', 'failed'])
                )
            ).delete()
            session.commit()
            return deleted_count
        except Exception as e:
            session.rollback()
            print(f"清理旧任务失败: {str(e)}")
            return 0
        finally:
            self.db_manager.close_session(session)

# 全局仓库实例
evaluation_repo = EvaluationRepository()
task_repo = TaskRepository() 