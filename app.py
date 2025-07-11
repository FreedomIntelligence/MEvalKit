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

from src.utils.model_and_dataset import *
from evaluation.TextMCQ_eval import *
from evaluation.ImageMCQ_eval import *
from evaluation.LLMJudge_eval import *

app = Flask(__name__)

# 简化的Swagger配置
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

# 动态获取host
def get_swagger_host():
    """动态获取Swagger host配置"""
    # 优先使用环境变量
    host = os.environ.get('SWAGGER_HOST')
    if host:
        return host
    
    # 如果没有环境变量，使用服务器IP
    return "localhost:5010"

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "MedUniBench API",
        "description": "MedUniBench API Documentation",
        "version": "1.0.0",
        "contact": {
            "name": "API Support"
        }
    },
    "host": "47.110.252.218:1984",
    #"host": "10.27.127.32:1984",
    "basePath": "/",
    "schemes": ["http"]
}

swagger = Swagger(app, config=swagger_config, template=swagger_template)

# Swagger文档配置
swagger_docs = {
    "index": {
        "tags": ["页面路由"],
        "summary": "总榜页面（主页）",
        "responses": {
            "200": {
                "description": "成功返回总榜页面"
            }
        }
    },
    "specific_leaderboard": {
        "tags": ["页面路由"],
        "summary": "显示具体排行榜",
        "responses": {
            "200": {
                "description": "成功返回具体排行榜页面"
            }
        }
    },
    "create_task": {
        "tags": ["页面路由"],
        "summary": "显示创建任务页面",
        "responses": {
            "200": {
                "description": "成功返回创建任务页面"
            }
        }
    },
    "new_evaluation": {
        "tags": ["页面路由"],
        "summary": "显示开始新评测页面",
        "responses": {
            "200": {
                "description": "成功返回开始新评测页面"
            }
        }
    },
    "view_evaluations": {
        "tags": ["页面路由"],
        "summary": "显示查看评测页面",
        "responses": {
            "200": {
                "description": "成功返回查看评测页面"
            }
        }
    },
    "run_evaluation": {
        "tags": ["评估任务"],
        "summary": "运行评估任务",
        "parameters": [
            {
                "name": "evaluation_mode",
                "in": "formData",
                "type": "string",
                "required": True,
                "description": "评估模式：automatic（自动模式）或manual（手动模式）"
            },
            {
                "name": "dataset",
                "in": "formData",
                "type": "string",
                "required": True,
                "description": "数据集名称，例如MMLU、GPQA、MMStar等"
            },
            {
                "name": "model_name",
                "in": "formData",
                "type": "string",
                "required": True,
                "description": "准备进行评测的模型名称，如gpt-4o、Qwen2-VL-7B-Instruct等"
            },
            {
                "name": "api_base",
                "in": "formData",
                "type": "string",
                "required": False,
                "description": "API接口路径，如http://localhost:8000/v1"
            },
            {
                "name": "model_key",
                "in": "formData",
                "type": "string",
                "required": False,
                "description": "API密钥或访问令牌"
            },
            {
                "name": "question_limitation",
                "in": "formData",
                "type": "string",
                "required": False,
                "default": "100",
                "description": "评测的问题数量，留空则评测全部题目"
            },
            {
                "name": "response_url",
                "in": "formData",
                "type": "string",
                "required": False,
                "description": "手动模式专用：响应数据URL，提供包含模型响应数据的JSON文件URL"
            },
            {
                "name": "user_id",
                "in": "formData",
                "type": "string",
                "required": False,
                "default": "test",
                "description": "用户ID，用于标识评测任务的创建者"
            }
        ],
        "responses": {
            "200": {
                "description": "任务创建成功",
                "schema": {
                    "type": "object",
                    "properties": {
                        "result": {"type": "boolean"},
                        "msg": {"type": "string"},
                        "data": {
                            "type": "object",
                            "properties": {
                                "task_id": {"type": "string"},
                                "redirect_url": {"type": "string"}
                            }
                        }
                    }
                }
            },
            "400": {
                "description": "参数错误",
                "schema": {
                    "type": "object",
                    "properties": {
                        "result": {"type": "boolean"},
                        "msg": {"type": "string"},
                        "data": {"type": "null"}
                    }
                }
            }
        }
    },
    "task_detail": {
        "tags": ["任务管理"],
        "summary": "显示任务详情",
        "parameters": [
            {
                "name": "task_id",
                "in": "path",
                "type": "string",
                "required": True,
                "description": "任务ID"
            }
        ],
        "responses": {
            "200": {
                "description": "成功返回任务详情页面"
            },
            "302": {
                "description": "任务不存在时重定向"
            }
        }
    },
    "results": {
        "tags": ["页面路由"],
        "summary": "显示所有评估结果",
        "responses": {
            "200": {
                "description": "成功返回结果页面"
            }
        }
    },
    "task_status": {
        "tags": ["任务管理"],
        "summary": "获取任务状态",
        "parameters": [
            {
                "name": "task_id",
                "in": "path",
                "type": "string",
                "required": True,
                "description": "任务ID"
            }
        ],
        "responses": {
            "200": {
                "description": "成功返回任务状态",
                "schema": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "progress": {"type": "number"},
                        "evaluation_complete": {"type": "boolean"},
                        "total_questions": {"type": "integer"},
                        "completed_questions": {"type": "integer"},
                        "completion_rate": {"type": "number"},
                        "error_message": {"type": "string"},
                        "error_details": {"type": "string"}
                    }
                }
            }
        }
    },
    "api_overall_rankings": {
        "tags": ["排行榜API"],
        "summary": "获取特定类别的排名数据",
        "parameters": [
            {
                "name": "category",
                "in": "query",
                "type": "string",
                "required": False,
                "default": "文本理解",
                "description": "排行榜类别"
            },
            {
                "name": "sort_by",
                "in": "query",
                "type": "string",
                "required": False,
                "default": "average",
                "description": "排序字段"
            },
            {
                "name": "order",
                "in": "query",
                "type": "string",
                "required": False,
                "default": "desc",
                "description": "排序方式(asc/desc)"
            }
        ],
        "responses": {
            "200": {
                "description": "成功返回排名数据",
                "schema": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "sort_by": {"type": "string"},
                        "order": {"type": "string"},
                        "rankings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "model": {"type": "string"},
                                    "average": {"type": "number"},
                                    "valid_datasets": {"type": "integer"},
                                    "total_datasets": {"type": "integer"}
                                }
                            }
                        }
                    }
                }
            },
            "400": {
                "description": "请求参数错误"
            }
        }
    }
}

# 评估结果目录
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# 存储运行中的任务
active_tasks = {}

leaderboard_data = {}
last_leaderboard_update = 0

LEADERBOARD_DATASETS = GENERAL_DATASETS + GENERAL_MULTIMODAL_DATASETS + MEDICAL_KNOWLEDGE_DATASETS + MEDICAL_ETHICS_DATASETS

# 默认用户
DEFAULT_USER = "test"

def get_user_evaluations(user_id):
    """获取用户的所有评测记录"""
    evaluations = []
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

# 存储模型名称映射关系，用于在文件名和显示名称之间转换
model_name_mapping = {}

def sanitize_filename(filename):
    """
    将文件名中的所有不安全字符替换为下划线
    """
    return re.sub(r'[\\/:*?"<>|]', '_', filename).strip(' .') or 'unknown_model'

def generate_business_id(dataset, model_name):
    """
    生成business_id：{dataset}_{safe_model}_{当前时间}
    """
    current_time = datetime.now().strftime("%Y%m%d%H%M")
    safe_model_name = sanitize_filename(model_name)
    return f"{dataset}_{safe_model_name}_{current_time}"

def init_leaderboard():
    global leaderboard_data
    leaderboard_data = {dataset: {} for dataset in LEADERBOARD_DATASETS}
    update_leaderboard()

def update_leaderboard():
    global leaderboard_data, last_leaderboard_update
    if time.time() - last_leaderboard_update < 300:
        return
    for dataset in LEADERBOARD_DATASETS:
        score_files = glob.glob(str(RESULTS_DIR / f"{dataset}_*_score.json"))
        for file in score_files:
            try:
                filename = os.path.basename(file)
                model_name = filename.replace(f"{dataset}_", "").replace("_score.json", "")
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
    # 确保排行榜数据最新
    update_leaderboard()
    
    # 按类别整理模型数据
    overall_rankings = {}
    
    for category, datasets in DATASET_CATEGORIES.items():
        # 收集所有在此类别下至少有一个评测结果的模型
        models = set()
        for dataset in datasets:
            models.update(leaderboard_data[dataset].keys())
        
        # 为每个模型计算在此类别下的平均分
        category_data = {}
        for model in models:
            model_scores = {}
            valid_scores = 0
            score_sum = 0
            
            # 收集此模型在该类别所有数据集上的分数
            for dataset in datasets:
                if model in leaderboard_data[dataset]:
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

@app.route('/')
@swag_from(swagger_docs["index"])
def index():
    """总榜页面（主页）"""
    # 计算总榜数据
    overall_rankings = calculate_overall_rankings()
    
    # 准备模板数据
    dataset_descriptions = {
        "MMLU": "多任务语言理解基准",
        "CMB": "中文医学知识基准",
        "GPQA": "通用物理问答基准",
        "MMStar": "多模态评估基准",
        "MT-Bench": "LLMJudge基准"
    }
    
    return render_template('overall_leaderboard.html', 
                          rankings=overall_rankings,
                          dataset_descriptions=dataset_descriptions,
                          categories=DATASET_CATEGORIES,
                          user_id=DEFAULT_USER,
                          last_update=datetime.fromtimestamp(last_leaderboard_update).strftime('%Y-%m-%d %H:%M:%S') if last_leaderboard_update > 0 else "从未更新")

# 添加API端点获取特定类别的排名
@app.route('/api/overall-rankings')
@swag_from(swagger_docs["api_overall_rankings"])
def api_overall_rankings():
    """返回特定类别的排名数据"""
    category = request.args.get('category', '文本理解')
    sort_by = request.args.get('sort_by', 'average')
    order = request.args.get('order', 'desc')
    
    overall_rankings = calculate_overall_rankings()
    
    if category not in overall_rankings:
        return jsonify({
            "result": False,
            "msg": "类别不存在",
            "data": None
        }), 400
    
    # 获取指定类别的数据
    category_data = overall_rankings[category]
    
    # 转换为列表便于排序
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
    elif sort_by in DATASET_CATEGORIES[category]:
        # 按特定数据集排序
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

def start_leaderboard_update_thread():
    def update():
        while True:
            try:
                update_leaderboard()
            except Exception as e:
                print(f"更新排行榜失败: {str(e)}")
            time.sleep(300)
    threading.Thread(target=update, daemon=True).start()

@app.route('/specific-leaderboard')
@swag_from(swagger_docs["specific_leaderboard"])
def specific_leaderboard():
    """显示具体排行榜"""
    # 准备模板数据
    datasets = ["MMLU", "CMB", "GPQA", "MMStar"]
    dataset_descriptions = {
        "MMLU": "多任务语言理解基准",
        "CMB": "中文医学知识基准", 
        "GPQA": "通用物理问答基准",
        "MMStar": "多模态评估基准"
    }
    
    return render_template('specific_leaderboard.html',
                          datasets=datasets,
                          dataset_descriptions=dataset_descriptions,
                          leaderboard_data=leaderboard_data,
                          user_id=DEFAULT_USER,
                          last_update=datetime.fromtimestamp(last_leaderboard_update).strftime('%Y-%m-%d %H:%M:%S') if last_leaderboard_update > 0 else "从未更新")

@app.route('/create-task')
@swag_from(swagger_docs["create_task"])
def create_task():
    """显示创建任务页面（重定向到新评测页面）"""
    return redirect(url_for('new_evaluation'))

@app.route('/new-evaluation')
@swag_from(swagger_docs["new_evaluation"])
def new_evaluation():
    """显示开始新评测页面"""
    return render_template('new_evaluation.html', 
                          text_datasets=TEXT_DATASETS,
                          multimodal_datasets=MULTIMODAL_DATASETS,
                          llmjudge_datasets=LLMJUDGE_DATASETS,
                          user_id=DEFAULT_USER)

@app.route('/view-evaluations')
@swag_from(swagger_docs["view_evaluations"])
def view_evaluations():
    """显示查看评测页面"""
    user_id = request.args.get('user_id', DEFAULT_USER)
    evaluations = get_user_evaluations(user_id)
    return render_template('view_evaluations.html', 
                          evaluations=evaluations,
                          user_id=user_id)

@app.route('/run-evaluation', methods=['POST'])
@swag_from(swagger_docs["run_evaluation"])
def run_evaluation():
    """运行评估任务"""
    evaluation_mode = request.form.get('evaluation_mode')
    dataset = request.form.get('dataset')
    model_name = request.form.get('model_name', '')
    model_key = request.form.get('model_key', '')
    api_base = request.form.get('api_base', '')
    question_limitation = request.form.get('question_limitation', '100')
    user_id = request.form.get('user_id', DEFAULT_USER)
    response_url = request.form.get('response_url', '')
    upload_file = request.files.get('upload_file', None)
    
    # 参数验证
    if not evaluation_mode or not dataset:
        response = jsonify({
            "result": False,
            "msg": "缺少必要参数",
            "data": None
        })
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response, 400
    
    # automatic模式需要模型名称
    if evaluation_mode == "automatic" and not model_name:
        response = jsonify({
            "result": False,
            "msg": "自动模式需要提供模型名称",
            "data": None
        })
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response, 400
    
    # manual模式需要模型名称和响应URL
    if evaluation_mode == "manual":
        if not model_name:
            response = jsonify({
                "result": False,
                "msg": "手动模式需要提供模型名称",
                "data": None
            })
            response.headers['Content-Type'] = 'application/json; charset=utf-8'
            return response, 400
        
        if not response_url:
            response = jsonify({
                "result": False,
                "msg": "手动模式需要提供响应数据URL",
                "data": None
            })
            response.headers['Content-Type'] = 'application/json; charset=utf-8'
            return response, 400
    
    # 处理评测数量
    question_limit = None
    try:
        if question_limitation:
            question_limit = int(question_limitation)
            if question_limit <= 0:
                question_limit = None
    except (ValueError, TypeError):
        question_limit = None
    
    # 生成business_id
    if evaluation_mode == "automatic":
        business_id = generate_business_id(dataset, model_name)
    else:
        business_id = f"{dataset}_manual_{int(time.time())}"
    
    # 创建任务ID
    task_id = business_id
    
    # 确定评估类型
    if dataset in TEXT_DATASETS:
        eval_type = "text"
    elif dataset in MULTIMODAL_DATASETS:
        eval_type = "multimodal"
    elif dataset in LLMJUDGE_DATASETS:
        eval_type = "llmjudge"
    else:
        response = jsonify({
            "result": False,
            "msg": "不支持的数据集类型",
            "data": None
        })
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response, 400
    
    # 初始化任务状态
    active_tasks[task_id] = {
        "id": task_id,
        "dataset": dataset,
        "model": model_name,
        "model_key": model_key,
        "api_base": api_base,
        "evaluation_mode": evaluation_mode,
        "business_id": business_id,
        "eval_type": eval_type,
        "status": "pending",
        "created_at": time.time(),
        "progress": 0,
        "total_questions": 0,
        "completed_questions": 0,
        "question_limit": question_limit,
        "is_evaluation_complete": False,
        "error_message": None,
        "error_details": None,
        "user_id": user_id
    }

    def run_task():
        try:
            # 构建命令，与run.py保持一致
            cmd = ["python", "run.py", 
                "--evaluation_mode", evaluation_mode,
                "--dataset", dataset, 
                "--business_id", business_id,
                "--user_id", user_id,
                "--question_limitation", str(question_limit) if question_limit else "100"]
            
            # 添加模型相关参数
            if evaluation_mode == "automatic":
                cmd.extend(["--model_name", model_name])
                cmd.extend(["--api_base", api_base])
                cmd.extend(["--model_key", model_key])
            elif evaluation_mode == "manual":
                cmd.extend(["--model_name", model_name])
                cmd.extend(["--api_base", api_base])
                cmd.extend(["--model_key", model_key])
                cmd.extend(["--response_url", response_url])
                
            # 运行命令
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONBUFFERED": "1"}
            )
                
            # 更新任务状态
            active_tasks[task_id]["process"] = process
            active_tasks[task_id]["status"] = "running"
                
            # 实时获取输出
            output_lines = []
            stderr_lines = []
            
            # 创建非阻塞读取函数
            def read_output(stream_type):
                stream = process.stdout if stream_type == "stdout" else process.stderr
                for line in iter(stream.readline, ''):
                    if stream_type == "stderr":
                        stderr_lines.append(line)
                    # 仅尝试从输出中提取tqdm进度信息
                    try:
                        # 尝试匹配不同格式的tqdm进度信息
                        # 匹配格式1: 处理文本问题:   0%|          | 11/14042 [00:01<20:06, 11.63it/s]
                        tqdm_match = re.search(r'处理文本问题:\s+(\d+)%\|.*?\| (\d+)/(\d+)', line)
                        if not tqdm_match:
                            # 匹配格式2: 任意文本: 45%|████▌     | 45/100 [00:05<00:06,  8.25it/s]
                            tqdm_match = re.search(r'.*?:\s+(\d+)%\|.*?\| (\d+)/(\d+)', line)
                        if not tqdm_match:
                            # 匹配格式3: 45%|████▌     | 45/100 [00:05<00:06,  8.25it/s]
                            tqdm_match = re.search(r'(\d+)%\|.*?\| (\d+)/(\d+)', line)
                                
                        if tqdm_match:
                            percent, current, total = map(int, tqdm_match.groups())
                            active_tasks[task_id]["progress"] = percent
                            active_tasks[task_id]["total_questions"] = total
                            active_tasks[task_id]["completed_questions"] = current
                    except Exception:
                        pass
                        
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
                    
                # 评测完成后检查完成状态
                check_completion_status(task_id)
            else:
                active_tasks[task_id]["status"] = "failed"
                # 获取错误输出
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
    response = jsonify({
        "result": True,
        "msg": "评估任务创建成功",
        "data": {
            "task_id": task_id,
            "redirect_url": url_for('task_detail', task_id=task_id)
        }
    })
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response

@app.route('/task-detail/<task_id>')
@swag_from(swagger_docs["task_detail"])
def task_detail(task_id):
    if task_id not in active_tasks:
        return redirect(url_for("results"))
    task = active_tasks[task_id]
    user_id = task.get("user_id", DEFAULT_USER)
    return render_template('task_detail.html', task=task, user_id=user_id)

@app.route('/results')
def results():
    """显示所有评估结果"""
    # 获取所有任务
    tasks = list(active_tasks.values())
    
    # 按创建时间降序排序
    tasks.sort(key=lambda x: x.get('created_at', 0), reverse=True)
    
    # 获取可用的数据集列表
    available_datasets = list(set(task.get('dataset', '') for task in tasks if task.get('dataset')))
    available_datasets.sort()
    
    # 获取筛选参数
    selected_dataset = request.args.get('dataset', '')
    
    # 如果有筛选条件，过滤任务
    if selected_dataset:
        tasks = [task for task in tasks if task.get('dataset') == selected_dataset]
    
    # 获取当前运行的任务ID（如果有的话）
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

@app.route('/task-status/<task_id>')
@swag_from(swagger_docs["task_status"])
def task_status(task_id):
    """获取任务状态"""
    if task_id not in active_tasks:
        return jsonify({
            "result": False,
            "msg": "任务不存在",
            "data": None
        })
    
    task = active_tasks[task_id]
    
    # 如果评测已完成但尚未检查完成状态，检查一次
    if task.get("status") == "evaluation_complete" and not task.get("status_checked", False):
        check_completion_status(task_id)
        task["status_checked"] = True
    
    # 构建基础响应数据
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
    """检查任务完成状态并从result.json获取评测结果"""
    task = active_tasks[task_id]
    
    # 评测完成后，读取result.json文件获取详细结果
    if task.get("is_evaluation_complete", False):
        dataset = task["dataset"]
        model = task["model"]
        business_id = task.get("business_id", "MMLUdoubao03")
        user_id = task.get("user_id", DEFAULT_USER)
        
        # 使用正确的文件路径格式（包含user_id和business_id）
        user_results_dir = RESULTS_DIR / user_id
        result_path = user_results_dir / f"{business_id}_result.json"
        score_result_path = user_results_dir / f"{business_id}_score.json"
        
        if result_path.exists():
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 分析响应数据
                total_questions = len(data)
                valid_questions = 0
                for d in data:
                    if task["eval_type"] == "llmjudge":
                        # 对于LLMJudge，直接检查分数是否在合理范围内
                        if d.get("score", -1) >= 0 and d.get("score", -1) <= 10:
                            valid_questions += 1
                    else:
                        if d.get("response") != "Neglected":
                            valid_questions += 1
                
                # 计算有效率和准确率
                valid_rate = valid_questions / total_questions if total_questions > 0 else 0
                task["total_questions"] = total_questions
                task["valid_questions"] = valid_questions
                task["valid_rate"] = valid_rate
                task["is_valid_evaluation"] = valid_rate >= 0.95  # 有效率>=95%为有效评测
                
                # 尝试读取分数文件
                if score_result_path.exists():
                    try:
                        with open(score_result_path, 'r', encoding='utf-8') as f:
                            score_data = json.load(f)
                            raw_score = score_data.get("raw_score", 0)
                            score = score_data.get("score", 0)
                            task["raw_score"] = raw_score
                            task["score"] = score
                    except Exception as e:
                        print(f"读取分数文件出错: {str(e)}")
                        task["raw_score"] = 0
                        task["score"] = 0
                
                if valid_rate >= 0.95:
                    task["status"] = "completed"
                else:
                    task["status"] = "incomplete"
                return True
            except Exception as e:
                print(f"读取评测结果出错: {str(e)}")
                # 出错时仍然标记为评测完成，但可能没有详细结果
                task["status"] = "evaluation_complete"
                return False
    
    return False

@app.template_filter('timestamp_to_date')
def timestamp_to_date(timestamp):
    """将时间戳转换为可读日期"""
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

@app.route('/test-swagger')
def test_swagger():
    """测试Swagger是否正常工作"""
    return jsonify({
        "message": "Swagger测试成功",
        "swagger_ui_url": "/apidocs/",
        "api_spec_url": "/apispec.json"
    })

@app.route('/debug-form')
def debug_form():
    """调试表单页面"""
    return send_file('debug_form.html')

@app.route('/continue-evaluation/<business_id>')
def continue_evaluation(business_id):
    """继续评测功能"""
    user_id = request.args.get('user_id', DEFAULT_USER)
    
    # 查找对应的评测文件
    user_results_dir = RESULTS_DIR / user_id
    result_file = user_results_dir / f"{business_id}_result.json"
    score_file = user_results_dir / f"{business_id}_score.json"
    
    if not result_file.exists():
        return jsonify({
            "result": False,
            "msg": "评测文件不存在",
            "data": None
        }), 404
    
    try:
        # 读取现有结果
        with open(result_file, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        
        # 计算valid_ratio
        total_questions = len(result_data)
        valid_questions = sum(1 for item in result_data if item.get("response") != "Neglected")
        valid_ratio = valid_questions / total_questions if total_questions > 0 else 0
        
        # 如果valid_ratio >= 0.95，说明评测已完成
        if valid_ratio >= 0.95:
            return jsonify({
                "result": False,
                "msg": "评测已完成，无需继续",
                "data": None
            }), 400
        
        # 从business_id中提取信息
        parts = business_id.split('_')
        if len(parts) < 2:
            return jsonify({
                "result": False,
                "msg": "无效的business_id格式",
                "data": None
            }), 400
        
        dataset = parts[0]
        model_name = '_'.join(parts[1:-1])  # 模型名可能包含下划线
        
        # 创建新的评测任务
        task_id = f"{dataset}_{model_name}_{int(time.time())}"
        
        # 初始化任务状态
        active_tasks[task_id] = {
            "id": task_id,
            "dataset": dataset,
            "model": model_name,
            "evaluation_mode": "automatic",
            "business_id": business_id,
            "eval_type": "text" if dataset in TEXT_DATASETS else "multimodal" if dataset in MULTIMODAL_DATASETS else "llmjudge",
            "status": "pending",
            "created_at": time.time(),
            "progress": 0,
            "total_questions": 0,
            "completed_questions": 0,
            "question_limit": None,
            "is_evaluation_complete": False,
            "error_message": None,
            "error_details": None,
            "user_id": user_id
        }
        
        def run_continue_task():
            try:
                # 构建命令
                cmd = ["python", "run.py", 
                    "--evaluation_mode", "automatic",
                    "--dataset", dataset, 
                    "--model_name", model_name,
                    "--business_id", business_id,
                    "--question_limitation", "100"]
                
                # 运行命令
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env={**os.environ, "PYTHONBUFFERED": "1"}
                )
                    
                # 更新任务状态
                active_tasks[task_id]["process"] = process
                active_tasks[task_id]["status"] = "running"
                    
                # 实时获取输出
                def read_output(stream_type):
                    stream = process.stdout if stream_type == "stdout" else process.stderr
                    for line in iter(stream.readline, ''):
                        try:
                            tqdm_match = re.search(r'(\d+)%\|.*?\| (\d+)/(\d+)', line)
                            if tqdm_match:
                                percent, current, total = map(int, tqdm_match.groups())
                                active_tasks[task_id]["progress"] = percent
                                active_tasks[task_id]["total_questions"] = total
                                active_tasks[task_id]["completed_questions"] = current
                        except Exception:
                            pass
                            
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
                    check_completion_status(task_id)
                else:
                    active_tasks[task_id]["status"] = "failed"
                    stderr_output = process.stderr.read() if process.stderr else ""
                    active_tasks[task_id]["error_message"] = f"评测进程返回错误代码: {process.returncode}"
                    active_tasks[task_id]["error_details"] = stderr_output if stderr_output else "无详细错误信息"
                        
            except Exception as e:
                active_tasks[task_id]["status"] = "failed"
                active_tasks[task_id]["error_message"] = f"评测任务执行异常: {str(e)}"
                active_tasks[task_id]["error_details"] = f"异常类型: {type(e).__name__}\n异常详情: {str(e)}"
            
        # 启动后台线程
        task_thread = threading.Thread(target=run_continue_task)
        task_thread.daemon = True
        task_thread.start()
        
        return jsonify({
            "result": True,
            "msg": "继续评测任务创建成功",
            "data": {
                "task_id": task_id,
                "redirect_url": url_for('task_detail', task_id=task_id)
            }
        })
        
    except Exception as e:
        return jsonify({
            "result": False,
            "msg": f"继续评测失败: {str(e)}",
            "data": None
        }), 500

if __name__ == '__main__':
    init_leaderboard()
    start_leaderboard_update_thread()
    #app.run(host='0.0.0.0', debug=False, port=1984)
    app.run(host='0.0.0.0', debug=False, port=1984)