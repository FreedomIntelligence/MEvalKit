"""
MEvalKit Web应用主文件

该文件提供了基于Flask的Web界面，用于管理模型评测任务。
主要功能包括：
- 任务创建和管理
- 实时进度监控
- 结果查看和排行榜
- API文档自动生成

作者: MEvalKit Team
版本: 2.0.0
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
from flasgger import Swagger, swag_from
import subprocess
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
import threading
import re
import glob
import socket

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入评测模块
from src.utils.model_and_dataset import *
from evaluation.MCQ_eval import evaluate_all_mcq_automatic
from evaluation.QA_eval import QA_evaluator
from evaluation.Agent_eval import Agent_evaluator

# 导入数据库模块 
try:
    from src.database.mysql_db import mysql_db_manager, save_evaluation_result, save_evaluation_score, load_evaluation_result, load_evaluation_score
    DATABASE_AVAILABLE = True
except ImportError:
    print("数据库模块不可用，将使用文件系统存储")
    DATABASE_AVAILABLE = False

# 创建Flask应用实例
app = Flask(__name__)

# Swagger配置
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/apispec.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs/"
}

# Swagger模板配置
swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "MEvalKit API",
        "description": "MEvalKit 模型评测系统 API 文档",
        "version": "2.0.0",
        "contact": {
            "name": "MEvalKit Team",
            "email": "support@mevalkit.com"
        }
    },
    "host": os.getenv('SWAGGER_HOST', 'localhost:1984'),
    "basePath": "/",
    "schemes": ["http", "https"],
    "tags": [
        {
            "name": "页面路由",
            "description": "Web界面页面路由"
        },
        {
            "name": "评测任务",
            "description": "评测任务管理相关接口"
        },
        {
            "name": "排行榜",
            "description": "排行榜数据接口"
        },
        {
            "name": "用户管理", 
            "description": "用户评测记录管理"
        }
    ]
}

# 初始化Swagger
swagger = Swagger(app, config=swagger_config, template=swagger_template)

# 初始化数据库
if DATABASE_AVAILABLE:
    try:
        mysql_db_manager.create_tables()
        print("数据库表已创建")
    except Exception as e:
        print(f"数据库初始化失败: {str(e)}")
        DATABASE_AVAILABLE = False

# 评估结果目录
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# 存储运行中的任务
active_tasks = {}

# 排行榜数据缓存
leaderboard_data = {}
last_leaderboard_update = 0

# 所有可用数据集
ALL_DATASETS = TEXT_DATASETS + MULTIMODAL_DATASETS + LLMJUDGE_DATASETS + ["IOR-Dynamic"]

# 默认用户
DEFAULT_USER = "test"

def get_user_evaluations(user_id):
    """获取用户的所有评测记录"""
    evaluations = []
    
    if DATABASE_AVAILABLE:
        try:
            # 从数据库获取评测记录的逻辑将在这里实现
            pass
        except Exception as e:
            print(f"从数据库获取用户评测记录失败: {str(e)}")
    
    # 文件系统兼容性处理
    user_results_dir = RESULTS_DIR / user_id
    
    if not user_results_dir.exists():
        return evaluations
    
    # 查找所有score.json文件
    score_files = list(user_results_dir.glob("*_score.json"))
    
    for score_file in score_files:
        try:
            # 从文件名提取business_id
            business_id = score_file.stem.replace("_score", "")
            
            # 读取score.json文件
            with open(score_file, 'r', encoding='utf-8') as f:
                score_data = json.load(f)
            
            # 检查对应的result.json文件是否存在
            result_file = user_results_dir / f"{business_id}_result.json"
            if result_file.exists():
                with open(result_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                
                # 计算valid_ratio
                total_questions = len(result_data)
                valid_questions = sum(1 for item in result_data if item.get("response") != "Neglected")
                valid_ratio = valid_questions / total_questions if total_questions > 0 else 0
                
                evaluation = {
                    "business_id": business_id,
                    "valid_ratio": valid_ratio,
                    "score": score_data.get("score", 0),
                    "raw_score": score_data.get("raw_score", 0),
                    "file_path": str(score_file),
                    "created_time": datetime.fromtimestamp(score_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                }
                evaluations.append(evaluation)
        except Exception as e:
            print(f"读取评测文件出错 {score_file}: {str(e)}")
            continue
    
    # 按创建时间降序排序
    evaluations.sort(key=lambda x: x["created_time"], reverse=True)
    return evaluations

def sanitize_filename(filename):
    """将文件名中的不安全字符替换为下划线"""
    return re.sub(r'[\\/:*?"<>|]', '_', filename).strip(' .') or 'unknown_model'

def generate_business_id(dataset, model_name):
    """生成business_id：{dataset}_{safe_model}_{当前时间}"""
    current_time = datetime.now().strftime("%Y%m%d%H%M")
    safe_model_name = sanitize_filename(model_name)
    return f"{dataset}_{safe_model_name}_{current_time}"

def update_leaderboard():
    """更新排行榜数据"""
    global leaderboard_data, last_leaderboard_update
    
    # 限制更新频率（5分钟更新一次）
    if time.time() - last_leaderboard_update < 300:
        return
    
    leaderboard_data = {dataset: {} for dataset in ALL_DATASETS}
    
    if DATABASE_AVAILABLE:
        try:
            # 从数据库更新排行榜数据的逻辑将在这里实现
            pass
        except Exception as e:
            print(f"从数据库更新排行榜失败: {str(e)}")
    
    # 文件系统兼容性处理
    for dataset in ALL_DATASETS:
        score_files = glob.glob(str(RESULTS_DIR / f"*/{dataset}_*_score.json"))
        for file in score_files:
            try:
                filename = os.path.basename(file)
                # 提取模型名称
                model_match = re.match(rf"{dataset}_(.+?)_\d{{12}}_score\.json", filename)
                if not model_match:
                    continue
                    
                model_name = model_match.group(1)
                
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    raw_score = data.get("raw_score", 0)
                    score = data.get("score", 0)

                    if model_name not in leaderboard_data[dataset] or score > leaderboard_data[dataset][model_name]["score"]:
                        leaderboard_data[dataset][model_name] = {
                            "raw_score": raw_score,
                            "score": score,
                            "timestamp": os.path.getmtime(file),
                            "date": datetime.fromtimestamp(os.path.getmtime(file)).strftime('%Y-%m-%d %H:%M:%S')
                        }
            except Exception as e:
                print(f"读取评测结果出错: {str(e)}")
    
    last_leaderboard_update = time.time()

def calculate_overall_rankings():
    """计算总榜排名数据"""
    update_leaderboard()
    
    overall_rankings = {}
    
    for category, datasets in DATASET_CATEGORIES.items():
        # 收集所有在此类别下至少有一个评测结果的模型
        models = set()
        for dataset in datasets:
            if dataset in leaderboard_data:
                models.update(leaderboard_data[dataset].keys())
        
        # 为每个模型计算在此类别下的平均分
        category_data = {}
        for model in models:
            model_scores = {}
            valid_scores = 0
            score_sum = 0
            
            # 收集此模型在该类别所有数据集上的分数
            for dataset in datasets:
                if dataset in leaderboard_data and model in leaderboard_data[dataset]:
                    score = leaderboard_data[dataset][model]["score"]
                    model_scores[dataset] = score
                    score_sum += score
                    valid_scores += 1
            
            # 只有至少有一个评测结果的模型才会出现在榜单上
            if valid_scores > 0:
                avg_score = score_sum / valid_scores
                category_data[model] = {
                    "scores": model_scores,
                    "average": avg_score,
                    "valid_datasets": valid_scores,
                    "total_datasets": len(datasets)
                }
        
        overall_rankings[category] = category_data
    
    return overall_rankings

# ==================== 页面路由 ====================

@app.route('/')
@swag_from({
    'tags': ['页面路由'],
    'summary': '总榜页面（主页）',
    'description': '显示所有分类的模型评测排行榜，提供模型在不同能力维度的排名对比',
    'responses': {
        200: {
            'description': '成功返回总榜页面',
            'content': {
                'text/html': {
                    'schema': {
                        'type': 'string'
                    }
                }
            }
        }
    }
})
def index():
    """总榜页面（主页）"""
    overall_rankings = calculate_overall_rankings()
    
    dataset_descriptions = {
        "MMLU": "多任务语言理解基准",
        "CMB": "中文医学知识基准",
        "GPQA": "通用物理问答基准", 
        "MMStar": "多模态评估基准",
        "MT-Bench": "多轮对话评估基准"
    }
    
    return render_template('overall_leaderboard.html', 
                          rankings=overall_rankings,
                          dataset_descriptions=dataset_descriptions,
                          categories=DATASET_CATEGORIES,
                          user_id=DEFAULT_USER,
                          last_update=datetime.fromtimestamp(last_leaderboard_update).strftime('%Y-%m-%d %H:%M:%S') if last_leaderboard_update > 0 else "从未更新")

@app.route('/specific-leaderboard')
@swag_from({
    'tags': ['页面路由'],
    'summary': '具体排行榜页面',
    'description': '显示各个数据集的详细排行榜，支持按数据集筛选和排序',
    'responses': {
        200: {
            'description': '成功返回具体排行榜页面'
        }
    }
})
def specific_leaderboard():
    """显示具体排行榜"""
    dataset_descriptions = {
        "MMLU": "多任务语言理解基准",
        "CMB": "中文医学知识基准", 
        "GPQA": "通用物理问答基准",
        "MMStar": "多模态评估基准",
        "MT-Bench": "多轮对话评估基准"
    }
    
    return render_template('specific_leaderboard.html',
                          datasets=ALL_DATASETS,
                          dataset_descriptions=dataset_descriptions,
                          leaderboard_data=leaderboard_data,
                          user_id=DEFAULT_USER,
                          last_update=datetime.fromtimestamp(last_leaderboard_update).strftime('%Y-%m-%d %H:%M:%S') if last_leaderboard_update > 0 else "从未更新")

@app.route('/new-evaluation')
@swag_from({
    'tags': ['页面路由'],
    'summary': '新建评测页面',
    'description': '显示创建新评测任务的表单页面，支持自动和手动两种评测模式',
    'responses': {
        200: {
            'description': '成功返回新建评测页面'
        }
    }
})
def new_evaluation():
    """显示开始新评测页面"""
    return render_template('new_evaluation.html', 
                          text_datasets=TEXT_DATASETS,
                          multimodal_datasets=MULTIMODAL_DATASETS,
                          llmjudge_datasets=LLMJUDGE_DATASETS,
                          user_id=DEFAULT_USER)

@app.route('/view-evaluations')
@swag_from({
    'tags': ['页面路由'],
    'summary': '查看评测记录页面',
    'description': '显示用户的所有评测记录，包括评测状态、分数等信息',
    'parameters': [
        {
            'name': 'user_id',
            'in': 'query',
            'type': 'string',
            'required': False,
            'default': 'test',
            'description': '用户ID'
        }
    ],
    'responses': {
        200: {
            'description': '成功返回评测记录页面'
        }
    }
})
def view_evaluations():
    """显示查看评测页面"""
    user_id = request.args.get('user_id', DEFAULT_USER)
    evaluations = get_user_evaluations(user_id)
    return render_template('view_evaluations.html', 
                          evaluations=evaluations,
                          user_id=user_id)

@app.route('/results')
@swag_from({
    'tags': ['页面路由'],
    'summary': '评测结果页面',
    'description': '显示所有评测任务的结果和状态，支持按数据集筛选',
    'parameters': [
        {
            'name': 'dataset',
            'in': 'query',
            'type': 'string',
            'required': False,
            'description': '筛选的数据集名称'
        }
    ],
    'responses': {
        200: {
            'description': '成功返回评测结果页面'
        }
    }
})
def results():
    """显示所有评估结果"""
    tasks = list(active_tasks.values())
    tasks.sort(key=lambda x: x.get('created_at', 0), reverse=True)
    
    available_datasets = list(set(task.get('dataset', '') for task in tasks if task.get('dataset')))
    available_datasets.sort()
    
    selected_dataset = request.args.get('dataset', '')
    
    if selected_dataset:
        tasks = [task for task in tasks if task.get('dataset') == selected_dataset]
    
    current_task_id = None
    for task in tasks:
        if task.get('status') in ['pending', 'running', 'evaluation_complete']:
            current_task_id = task.get('id')
            break
    
    return render_template('results.html', 
                          tasks=tasks,
                          active_tasks=active_tasks,
                          current_task_id=current_task_id,
                          available_datasets=available_datasets,
                          selected_dataset=selected_dataset)

@app.route('/task-detail/<task_id>')
@swag_from({
    'tags': ['评测任务'],
    'summary': '任务详情页面',
    'description': '显示指定评测任务的详细信息和实时状态',
    'parameters': [
        {
            'name': 'task_id',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': '任务ID'
        }
    ],
    'responses': {
        200: {
            'description': '成功返回任务详情页面'
        },
        302: {
            'description': '任务不存在时重定向到结果页面'
        }
    }
})
def task_detail(task_id):
    """显示任务详情"""
    if task_id not in active_tasks:
        return redirect(url_for("results"))
    
    task = active_tasks[task_id]
    user_id = task.get("user_id", DEFAULT_USER)
    return render_template('task_detail.html', task=task, user_id=user_id)

# ==================== API接口 ====================

@app.route('/api/overall-rankings')
@swag_from({
    'tags': ['排行榜'],
    'summary': '获取总榜排名数据',
    'description': '获取指定类别的模型排名数据，支持排序和筛选',
    'parameters': [
        {
            'name': 'category',
            'in': 'query',
            'type': 'string',
            'required': False,
            'default': '通用能力',
            'description': '排行榜类别',
            'enum': ['通用能力', '通用多模态能力', '医学知识', '医学伦理']
        },
        {
            'name': 'sort_by',
            'in': 'query',
            'type': 'string',
            'required': False,
            'default': 'average',
            'description': '排序字段'
        },
        {
            'name': 'order',
            'in': 'query',
            'type': 'string',
            'required': False,
            'default': 'desc',
            'description': '排序方式',
            'enum': ['asc', 'desc']
        }
    ],
    'responses': {
        200: {
            'description': '成功返回排名数据',
            'schema': {
                'type': 'object',
                'properties': {
                    'result': {'type': 'boolean'},
                    'msg': {'type': 'string'},
                    'data': {
                        'type': 'object',
                        'properties': {
                            'category': {'type': 'string'},
                            'sort_by': {'type': 'string'},
                            'order': {'type': 'string'},
                            'rankings': {
                                'type': 'array',
                                'items': {
                                    'type': 'object',
                                    'properties': {
                                        'model': {'type': 'string'},
                                        'average': {'type': 'number'},
                                        'valid_datasets': {'type': 'integer'},
                                        'total_datasets': {'type': 'integer'}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        400: {
            'description': '请求参数错误',
            'schema': {
                'type': 'object',
                'properties': {
                    'result': {'type': 'boolean'},
                    'msg': {'type': 'string'},
                    'data': {'type': 'null'}
                }
            }
        }
    }
})
def api_overall_rankings():
    """获取特定类别的排名数据"""
    category = request.args.get('category', '通用能力')
    sort_by = request.args.get('sort_by', 'average')
    order = request.args.get('order', 'desc')
    
    overall_rankings = calculate_overall_rankings()
    
    if category not in overall_rankings:
        return jsonify({
            "result": False,
            "msg": f"类别不存在: {category}",
            "data": None
        }), 400
    
    category_data = overall_rankings[category]
    
    models_list = []
    for model, data in category_data.items():
        model_data = {
            "model": model,
            "average": data["average"],
            "valid_datasets": data["valid_datasets"],
            "total_datasets": data["total_datasets"],
        }
        # 添加各个数据集的分数
        for dataset, score in data["scores"].items():
            model_data[dataset] = score
        
        models_list.append(model_data)
    
    # 排序
    reverse = (order == 'desc')
    if sort_by == 'average':
        models_list.sort(key=lambda x: x['average'], reverse=reverse)
    elif sort_by in DATASET_CATEGORIES.get(category, []):
        models_list.sort(key=lambda x: x.get(sort_by, 0), reverse=reverse)
    
    return jsonify({
        "result": True,
        "msg": "获取排名数据成功",
        "data": {
            "category": category,
            "sort_by": sort_by,
            "order": order,
            "rankings": models_list
        }
    })

@app.route('/run-evaluation', methods=['POST'])
@swag_from({
    'tags': ['评测任务'],
    'summary': '创建并运行评测任务',
    'description': '创建新的评测任务并在后台运行，支持自动和手动两种模式',
    'parameters': [
        {
            'name': 'evaluation_mode',
            'in': 'formData',
            'type': 'string',
            'required': True,
            'description': '评估模式',
            'enum': ['automatic', 'manual']
        },
        {
            'name': 'evaluation_type',
            'in': 'formData',
            'type': 'string',
            'required': False,
            'default': 'auto',
            'description': '评测类型：auto（自动检测）、MCQ（多选题）、QA（问答题）、Agent（对话）',
            'enum': ['auto', 'MCQ', 'QA', 'Agent']
        },
        {
            'name': 'dataset',
            'in': 'formData',
            'type': 'string',
            'required': True,
            'description': '数据集名称'
        },
        {
            'name': 'model_name',
            'in': 'formData',
            'type': 'string',
            'required': True,
            'description': '模型名称'
        },
        {
            'name': 'agent_1_model',
            'in': 'formData',
            'type': 'string',
            'required': False,
            'default': 'gpt-4o',
            'description': 'Agent_1模型（仅Agent评测，默认gpt-4o）'
        },
        {
            'name': 'agent_2_model',
            'in': 'formData',
            'type': 'string',
            'required': False,
            'description': 'Agent_2模型（仅Agent评测，默认使用model_name）'
        },
        {
            'name': 'response_agent_model',
            'in': 'formData',
            'type': 'string',
            'required': False,
            'description': 'Response Agent模型（仅Agent评测，默认使用model_name）'
        },
        {
            'name': 'api_base',
            'in': 'formData',
            'type': 'string',
            'required': False,
            'description': 'API接口地址（自动模式需要）'
        },
        {
            'name': 'model_key',
            'in': 'formData',
            'type': 'string',
            'required': False,
            'description': 'API密钥（自动模式需要）'
        },
        {
            'name': 'question_limitation',
            'in': 'formData',
            'type': 'integer',
            'required': False,
            'default': 100,
            'description': '评测的问题数量，0或不填表示评测全部题目'
        },
        {
            'name': 'max_workers',
            'in': 'formData',
            'type': 'integer',
            'required': False,
            'default': 32,
            'description': '并行工作线程数'
        },
        {
            'name': 'response_url',
            'in': 'formData',
            'type': 'string',
            'required': False,
            'description': '响应数据URL（手动模式需要）'
        },
        {
            'name': 'user_id',
            'in': 'formData',
            'type': 'string',
            'required': False,
            'default': 'test',
            'description': '用户ID'
        }
    ],
    'responses': {
        200: {
            'description': '任务创建成功',
            'schema': {
                'type': 'object',
                'properties': {
                    'result': {'type': 'boolean'},
                    'msg': {'type': 'string'},
                    'data': {
                        'type': 'object',
                        'properties': {
                            'task_id': {'type': 'string'},
                            'redirect_url': {'type': 'string'}
                        }
                    }
                }
            }
        },
        400: {
            'description': '参数错误',
            'schema': {
                'type': 'object',
                'properties': {
                    'result': {'type': 'boolean'},
                    'msg': {'type': 'string'},
                    'data': {'type': 'null'}
                }
            }
        }
    }
})
def run_evaluation():
    """创建并运行评测任务"""
    evaluation_mode = request.form.get('evaluation_mode')
    evaluation_type = request.form.get('evaluation_type', 'auto')
    dataset = request.form.get('dataset')
    model_name = request.form.get('model_name', '')
    agent_1_model = request.form.get('agent_1_model', 'gpt-4o')
    agent_2_model = request.form.get('agent_2_model', '')
    response_agent_model = request.form.get('response_agent_model', '')
    model_key = request.form.get('model_key', '')
    api_base = request.form.get('api_base', '')
    question_limitation = request.form.get('question_limitation', '100')
    max_workers = request.form.get('max_workers', '32')
    user_id = request.form.get('user_id', DEFAULT_USER)
    response_url = request.form.get('response_url', '')
    
    # 参数验证
    if not evaluation_mode or not dataset:
        return jsonify({
            "result": False,
            "msg": "缺少必要参数: evaluation_mode 和 dataset",
            "data": None
        }), 400
    
    if evaluation_mode == "automatic" and not model_name:
        return jsonify({
            "result": False,
            "msg": "自动模式需要提供模型名称",
            "data": None
        }), 400
    
    if evaluation_mode == "manual":
        if not model_name or not response_url:
            return jsonify({
                "result": False,
                "msg": "手动模式需要提供模型名称和响应数据URL",
                "data": None
            }), 400
    
    # 处理评测数量
    question_limit = None
    try:
        if question_limitation:
            question_limit = int(question_limitation)
            if question_limit <= 0:
                question_limit = None
    except (ValueError, TypeError):
        question_limit = None
    
    # 处理并行线程数
    workers = 32
    try:
        if max_workers:
            workers = int(max_workers)
            if workers <= 0:
                workers = 32
    except (ValueError, TypeError):
        workers = 32
    
    # 生成business_id
    if evaluation_mode == "automatic":
        business_id = generate_business_id(dataset, model_name)
    else:
        business_id = f"{dataset}_manual_{int(time.time())}"
    
    task_id = business_id
    
    # 自动检测或使用指定的评估类型
    if evaluation_type == "auto":
        if dataset in TEXT_DATASETS or dataset in MULTIMODAL_DATASETS:
            eval_type = "MCQ"
        elif dataset in LLMJUDGE_DATASETS:
            eval_type = "QA"
        elif dataset in ["IOR-Dynamic"]:
            eval_type = "Agent"
        else:
            return jsonify({
                "result": False,
                "msg": f"无法自动检测数据集类型: {dataset}，请手动指定evaluation_type参数",
                "data": None
            }), 400
    else:
        eval_type = evaluation_type
    
    # 验证评估类型
    if eval_type not in ["MCQ", "QA", "Agent"]:
        return jsonify({
            "result": False,
            "msg": f"不支持的评估类型: {eval_type}",
            "data": None
        }), 400
    
    # 初始化任务状态
    active_tasks[task_id] = {
        "id": task_id,
        "dataset": dataset,
        "model": model_name,
        "model_key": model_key,
        "api_base": api_base,
        "evaluation_mode": evaluation_mode,
        "evaluation_type": eval_type,
        "business_id": business_id,
        "eval_type": eval_type,  # 保持兼容性
        "status": "pending",
        "created_at": time.time(),
        "progress": 0,
        "total_questions": 0,
        "completed_questions": 0,
        "question_limit": question_limit,
        "max_workers": workers,
        "is_evaluation_complete": False,
        "error_message": None,
        "error_details": None,
        "user_id": user_id,
        "response_url": response_url,
        "agent_1_model": agent_1_model,
        "agent_2_model": agent_2_model or model_name,
        "response_agent_model": response_agent_model or model_name
    }

    def run_task():
        try:
            # 构建命令
            cmd = ["python", "run.py", 
                "--evaluation_mode", evaluation_mode,
                "--evaluation_type", eval_type,
                "--dataset", dataset, 
                "--business_id", business_id,
                "--user_id", user_id,
                "--question_limitation", str(question_limit) if question_limit else "100",
                "--max_workers", str(workers)]
            
            # 添加模式特定参数
            if evaluation_mode == "automatic":
                cmd.extend(["--model_name", model_name])
                
                # Agent评测特定参数
                if eval_type == "Agent":
                    cmd.extend(["--agent_1_model", agent_1_model])
                    cmd.extend(["--agent_2_model", agent_2_model or model_name])
                    cmd.extend(["--response_agent_model", response_agent_model or model_name])
                
                if api_base:
                    cmd.extend(["--api_base", api_base])
                if model_key:
                    cmd.extend(["--model_key", model_key])
            elif evaluation_mode == "manual":
                cmd.extend(["--model_name", model_name])
                cmd.extend(["--response_url", response_url])
                
            # 运行命令
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )
                
            # 更新任务状态
            active_tasks[task_id]["process"] = process
            active_tasks[task_id]["status"] = "running"
                
            # 实时获取输出
            output_lines = []
            stderr_lines = []
            
            def read_output(stream_type):
                stream = process.stdout if stream_type == "stdout" else process.stderr
                for line in iter(stream.readline, ''):
                    if stream_type == "stderr":
                        stderr_lines.append(line)
                    
                    # 解析进度信息
                    try:
                        # 匹配tqdm进度条格式
                        tqdm_match = re.search(r'(\d+)%\|.*?\| (\d+)/(\d+)', line)
                        if tqdm_match:
                            percent, current, total = map(int, tqdm_match.groups())
                            active_tasks[task_id]["progress"] = percent
                            active_tasks[task_id]["total_questions"] = total
                            active_tasks[task_id]["completed_questions"] = current
                    except Exception:
                        pass
                        
            # 启动输出读取线程
            stdout_thread = threading.Thread(target=read_output, args=("stdout",))
            stderr_thread = threading.Thread(target=read_output, args=("stderr",))
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
                
            # 等待进程完成
            process.wait()
                
            # 等待输出读取完成
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
                
            # 更新任务状态
            if process.returncode == 0:
                active_tasks[task_id]["is_evaluation_complete"] = True
                active_tasks[task_id]["status"] = "evaluation_complete"
                
                # 对于手动评测，立即设置进度为100%
                if evaluation_mode == "manual":
                    active_tasks[task_id]["progress"] = 100
                    active_tasks[task_id]["completed_questions"] = 1
                    active_tasks[task_id]["total_questions"] = 1
                    
                # 检查完成状态
                check_completion_status(task_id)
            else:
                active_tasks[task_id]["status"] = "failed"
                stderr_output = "".join(stderr_lines) if stderr_lines else ""
                active_tasks[task_id]["error_message"] = f"评测进程返回错误代码: {process.returncode}"
                active_tasks[task_id]["error_details"] = stderr_output if stderr_output else "无详细错误信息"
                    
        except Exception as e:
            active_tasks[task_id]["status"] = "failed"
            active_tasks[task_id]["error_message"] = f"评测任务执行异常: {str(e)}"
            active_tasks[task_id]["error_details"] = f"异常类型: {type(e).__name__}\n异常详情: {str(e)}"
        
    # 启动后台线程
    task_thread = threading.Thread(target=run_task)
    task_thread.daemon = True
    task_thread.start()
        
    # 返回任务创建成功的响应
    return jsonify({
        "result": True,
        "msg": "评估任务创建成功",
        "data": {
            "task_id": task_id,
            "redirect_url": url_for('task_detail', task_id=task_id)
        }
    })

@app.route('/task-status/<task_id>')
@swag_from({
    'tags': ['评测任务'],
    'summary': '获取任务状态',
    'description': '获取指定任务的实时状态信息，包括进度、完成情况等',
    'parameters': [
        {
            'name': 'task_id',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': '任务ID'
        }
    ],
    'responses': {
        200: {
            'description': '成功返回任务状态',
            'schema': {
                'type': 'object',
                'properties': {
                    'status': {'type': 'string', 'description': '任务状态'},
                    'progress': {'type': 'number', 'description': '任务进度百分比'},
                    'evaluation_complete': {'type': 'boolean', 'description': '评测是否完成'},
                    'total_questions': {'type': 'integer', 'description': '总问题数'},
                    'completed_questions': {'type': 'integer', 'description': '已完成问题数'},
                    'completion_rate': {'type': 'number', 'description': '完成率'},
                    'error_message': {'type': 'string', 'description': '错误信息'},
                    'error_details': {'type': 'string', 'description': '错误详情'}
                }
            }
        },
        404: {
            'description': '任务不存在',
            'schema': {
                'type': 'object',
                'properties': {
                    'result': {'type': 'boolean'},
                    'msg': {'type': 'string'},
                    'data': {'type': 'null'}
                }
            }
        }
    }
})
def task_status(task_id):
    """获取任务状态"""
    if task_id not in active_tasks:
        return jsonify({
            "result": False,
            "msg": "任务不存在",
            "data": None
        }), 404
    
    task = active_tasks[task_id]
    
    # 如果评测已完成但尚未检查完成状态，检查一次
    if task.get("status") == "evaluation_complete" and not task.get("status_checked", False):
        check_completion_status(task_id)
        task["status_checked"] = True
    
    # 构建响应数据
    response_data = {
        "status": task.get("status", "running"),
        "progress": task.get("progress", 0),
        "evaluation_complete": task.get("is_evaluation_complete", False),
        "total_questions": task.get("total_questions", 0),
        "completed_questions": task.get("completed_questions", 0),
        "completion_rate": task.get("completed_questions", 0) / task.get("total_questions", 1) 
                          if task.get("total_questions", 0) > 0 else 0,
        "error_message": task.get("error_message"),
        "error_details": task.get("error_details")
    }
    
    # 如果评测已完成，添加额外信息
    if task.get("status") in ["completed", "incomplete"]:
        response_data.update({
            "valid_questions": task.get("valid_questions", 0),
            "valid_rate": task.get("valid_rate", 0),
            "is_valid_evaluation": task.get("is_valid_evaluation", False),
            "score": task.get("score", 0),
            "raw_score": task.get("raw_score", 0)
        })
    
    return jsonify(response_data)

def check_completion_status(task_id):
    """检查任务完成状态并获取评测结果"""
    task = active_tasks[task_id]
    
    if not task.get("is_evaluation_complete", False):
        return False
        
    dataset = task["dataset"]
    model = task["model"]
    business_id = task.get("business_id", "")
    user_id = task.get("user_id", DEFAULT_USER)
    evaluation_mode = task.get("evaluation_mode", "automatic")
    
    try:
        # 尝试从数据库获取结果
        if DATABASE_AVAILABLE:
            try:
                result_data = load_evaluation_result(business_id)
                score_data = load_evaluation_score(business_id)
                
                if result_data and score_data:
                    # 从数据库获取的数据
                    total_questions = len(result_data) if isinstance(result_data, list) else 0
                    valid_questions = 0
                    
                    if isinstance(result_data, list):
                        for item in result_data:
                            if task["eval_type"] == "llmjudge":
                                if item.get("score", -1) >= 0 and item.get("score", -1) <= 10:
                                    valid_questions += 1
                            else:
                                if item.get("response") != "Neglected":
                                    valid_questions += 1
                    
                    valid_rate = valid_questions / total_questions if total_questions > 0 else 0
                    
                    task.update({
                        "total_questions": total_questions,
                        "valid_questions": valid_questions,
                        "valid_rate": valid_rate,
                        "is_valid_evaluation": valid_rate >= 0.95,
                        "score": score_data,
                        "raw_score": score_data,
                        "status": "completed" if valid_rate >= 0.95 else "incomplete"
                    })
                    
                    return True
            except Exception as e:
                print(f"从数据库获取结果失败: {str(e)}")
        
        # 文件系统兼容性处理
        user_results_dir = RESULTS_DIR / user_id
        
        # 根据评测模式确定文件名
        if evaluation_mode == "manual":
            result_path = user_results_dir / f"{business_id}_manual_result.json"
            score_result_path = user_results_dir / f"{business_id}_manual_score.json"
        else:
            result_path = user_results_dir / f"{business_id}_result.json"
            score_result_path = user_results_dir / f"{business_id}_score.json"
        
        # 等待文件生成
        if not result_path.exists():
            time.sleep(2)
            if not result_path.exists():
                print(f"结果文件不存在: {result_path}")
                return False
        
        if result_path.exists():
            with open(result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 分析响应数据
            total_questions = len(data)
            valid_questions = 0
            for d in data:
                if task["eval_type"] == "llmjudge":
                    if d.get("score", -1) >= 0 and d.get("score", -1) <= 10:
                        valid_questions += 1
                else:
                    if d.get("response") != "Neglected":
                        valid_questions += 1
            
            # 计算有效率
            valid_rate = valid_questions / total_questions if total_questions > 0 else 0
            task.update({
                "total_questions": total_questions,
                "valid_questions": valid_questions,
                "valid_rate": valid_rate,
                "is_valid_evaluation": valid_rate >= 0.95
            })
            
            # 尝试读取分数文件
            if score_result_path.exists():
                try:
                    with open(score_result_path, 'r', encoding='utf-8') as f:
                        score_data = json.load(f)
                        task["raw_score"] = score_data.get("raw_score", 0)
                        task["score"] = score_data.get("score", 0)
                except Exception as e:
                    print(f"读取分数文件出错: {str(e)}")
                    task["raw_score"] = 0
                    task["score"] = 0
            
            task["status"] = "completed" if valid_rate >= 0.95 else "incomplete"
            return True
            
    except Exception as e:
        print(f"读取评测结果出错: {str(e)}")
        task["status"] = "evaluation_complete"
        return False
    
    return False

# ==================== 辅助功能 ====================

@app.template_filter('timestamp_to_date')
def timestamp_to_date(timestamp):
    """将时间戳转换为可读日期"""
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

def start_leaderboard_update_thread():
    """启动排行榜更新后台线程"""
    def update():
        while True:
            try:
                update_leaderboard()
            except Exception as e:
                print(f"更新排行榜失败: {str(e)}")
            time.sleep(300)  # 5分钟更新一次
    
    threading.Thread(target=update, daemon=True).start()

@app.route('/test-swagger')
@swag_from({
    'tags': ['系统'],
    'summary': '测试Swagger',
    'description': '测试Swagger文档是否正常工作',
    'responses': {
        200: {
            'description': 'Swagger测试成功',
            'schema': {
                'type': 'object',
                'properties': {
                    'message': {'type': 'string'},
                    'swagger_ui_url': {'type': 'string'},
                    'api_spec_url': {'type': 'string'}
                }
            }
        }
    }
})
def test_swagger():
    """测试Swagger是否正常工作"""
    return jsonify({
        "message": "Swagger测试成功",
        "swagger_ui_url": "/apidocs/",
        "api_spec_url": "/apispec.json"
    })

if __name__ == '__main__':
    # 初始化排行榜
    update_leaderboard()
    # 启动排行榜更新后台线程
    start_leaderboard_update_thread()
    # 启动Flask应用
    app.run(host='0.0.0.0', debug=False, port=1984)