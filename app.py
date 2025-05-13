from flask import Flask, render_template, request, jsonify, redirect, url_for
import subprocess
import os
import json
import time
from pathlib import Path
from datetime import datetime
import threading
import re
import glob
from src.utils.model_and_dataset import *
app = Flask(__name__)

# 评估结果目录
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# 存储运行中的任务
active_tasks = {}

leaderboard_data = {}
last_leaderboard_update = 0

LEADERBOARD_DATASETS = GENERAL_DATASETS + MEDICAL_KNOWLEDGE_DATASETS + MEDICAL_ETHICS_DATASETS

DATASET_CATEGORIES = {
    "文本理解": TEXT_DATASETS,
    "多模态": MULTIMODAL_DATASETS,
    "LLMJudge": LLMJUDGE_DATASETS
}

text_model_map = {
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "DeepSeek-R1-Distill-Qwen-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "huatuo-triage": "huatuo-triage",
    "Meta-Llama-3.1-8B-Instruct": "Meta-Llama/Meta-Llama-3.1-8B-Instruct",
}

multimodal_model_map = {
    "qwen2-vl": "Pro/Qwen/Qwen2-VL-7B-Instruct",
}

def init_leaderboard():
    global leaderboard_data
    leaderboard_data = {dataset: {} for dataset in LEADERBOARD_DATASETS}
    update_leaderboard()

def update_leaderboard():
    global leaderboard_data, last_leaderboard_update
    if time.time() - last_leaderboard_update < 300:
        return
    for dataset in LEADERBOARD_DATASETS:
        if dataset in LLMJUDGE_DATASETS:
            score_files = glob.glob(str(RESULTS_DIR / f"{dataset}_*_score.json"))
        else:
            score_files = glob.glob(str(RESULTS_DIR / f"{dataset}_*_result_accuracy.json"))
        for file in score_files:
            try:
                filename = os.path.basename(file)
                if dataset in LLMJUDGE_DATASETS:
                    model_name = filename.replace(f"{dataset}_", "").replace("_score.json", "")
                else:
                    model_name = filename.replace(f"{dataset}_", "").replace("_result_accuracy.json", "")
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if dataset in LLMJUDGE_DATASETS:
                        score = data.get("average_score", 0)
                    else:
                        score = data.get("accuracy", 0)

                    if model_name not in leaderboard_data[dataset] or score > leaderboard_data[dataset][model_name]["score"]:
                        leaderboard_data[dataset][model_name] = {
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
                          last_update=datetime.fromtimestamp(last_leaderboard_update).strftime('%Y-%m-%d %H:%M:%S') if last_leaderboard_update > 0 else "从未更新")

# 添加API端点获取特定类别的排名
@app.route('/api/overall-rankings')
def api_overall_rankings():
    """返回特定类别的排名数据"""
    category = request.args.get('category', '文本理解')
    sort_by = request.args.get('sort_by', 'average')
    order = request.args.get('order', 'desc')
    
    overall_rankings = calculate_overall_rankings()
    
    if category not in overall_rankings:
        return jsonify({"error": "类别不存在"}), 400
    
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
        "category": category,
        "sort_by": sort_by,
        "order": order,
        "rankings": models_list
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

@app.route('/specific_leaderboard')
def specific_leaderboard():
    """显示排行榜"""
    update_leaderboard()
    sorted_leaderboard = {}
    for dataset in LEADERBOARD_DATASETS:
        models_list = [{"model": model, "score": data["score"], "date": data["date"]} 
                     for model, data in leaderboard_data[dataset].items()]
        models_list.sort(key=lambda x: x["score"], reverse=True)  # 按分数降序排序
        sorted_leaderboard[dataset] = models_list
    
    dataset_descriptions = {
        "MMLU": "多任务语言理解基准",
        "CMB": "中文医学知识基准",
        "GPQA": "通用物理问答基准",
        "MMStar": "多模态评估基准",
        "MT-Bench": "LLMJudge基准"
    }

    dataset_categories = {
        "text": TEXT_DATASETS,
        "multimodal": MULTIMODAL_DATASETS,
        "llmjudge": LLMJUDGE_DATASETS
    }
    
    return render_template('specific_leaderboard.html', 
                           datasets=LEADERBOARD_DATASETS, 
                           dataset_descriptions=dataset_descriptions, 
                           dataset_categories=dataset_categories, 
                           leaderboard=sorted_leaderboard, 
                           last_update=datetime.fromtimestamp(last_leaderboard_update).strftime('%Y-%m-%d %H:%M:%S') if last_leaderboard_update > 0 else "从未更新")



@app.route('/create_task')
def create_task():
    """显示创建任务页面"""
    # 定义数据集和模型
    text_datasets = ["MMLU", "CMB", "GPQA"]
    multimodal_datasets = ["MMStar"]
    llmjudge_datasets = ["MT-Bench"]
    
    text_models = ["GPT-4o", "Claude-3.5-Sonnet", "Llama-3-70B", "Qwen-72B", "GLM-4"]
    multimodal_models = ["GPT-4o", "Claude-3.5-Sonnet", "Qwen-VL", "CogVLM"]
    judge_models = ["GPT-4o", "Claude-3.5-Sonnet", "GPT-4"]
    
    return render_template('create_task.html', 
                          text_datasets=text_datasets,
                          multimodal_datasets=multimodal_datasets,
                          llmjudge_datasets=llmjudge_datasets,
                          text_models=text_models,
                          multimodal_models=multimodal_models,
                          judge_models=judge_models)

@app.route('/run-evaluation', methods=['POST'])
def run_evaluation():
    """运行评估任务"""
    dataset = request.form.get('dataset')
    model_name = request.form.get('model')
    eval_mode = request.form.get('eval_mode')
    judgment_model_name = request.form.get('judgment_model')
    
    if not all([dataset, model_name, eval_mode]):
        return jsonify({"status": "error", "message": "缺少必要参数"}), 400
    
    model_map = {
        "gpt-3.5-turbo": "gpt-3.5-turbo",
        "gpt-4o": "gpt-4o",
        "Qwen2-VL-7B-Instruct": "Pro/Qwen/Qwen2-VL-7B-Instruct"
    }

    model = model_map.get(model_name)
    judgment_model = model_map.get(judgment_model_name)
    
    # 确定评估类型
    text_mcq_datasets = ["MMLU", "CMB", "GPQA"]
    image_mcq_datasets = ["MMStar"]
    llm_judge_datasets = ["MT-Bench"]
    
    if dataset in text_mcq_datasets:
        eval_type = "text"
    elif dataset in image_mcq_datasets:
        eval_type = "image"
    elif dataset in llm_judge_datasets:
        eval_type = "llmjudge"
        # 检查LLMJudge类型是否提供了判别模型
        if not judgment_model:
            return jsonify({"status": "error", "message": "LLMJudge评测需要提供判别模型"}), 400
    else:
        return jsonify({"status": "error", "message": "不支持的数据集"}), 400
    
    # 创建任务ID
    task_id = f"{dataset}_{model_name}_{int(time.time())}"

    # 初始化任务状态
    active_tasks[task_id] = {
        "id": task_id,
        "dataset": dataset,
        "model": model_name,
        "eval_mode": eval_mode,
        "judgment_model": judgment_model_name if eval_type == "llmjudge" else None,
        "eval_type": eval_type,
        "status": "pending",
        "created_at": time.time(),
        "progress": 0,
        "total_questions": 0,
        "completed_questions": 0,
        "is_evaluation_complete": False
    }
    
    # 启动评估任务
    def run_task():
        try:
            # 构建命令
            cmd = ["python", "run.py", 
                  "--dataset", dataset, 
                  "--model", model_name, 
                  "--evaluate_mode", eval_mode,
                  "--workers", "64",
                  "--question_limitation", "5"]
            
            # 如果是LLMJudge，添加judgment_model参数
            if eval_type == "llmjudge":
                cmd.extend(["--judgment_model", judgment_model_name])
            
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
            
            # 创建非阻塞读取函数
            def read_output(stream_type):
                stream = process.stdout if stream_type == "stdout" else process.stderr
                for line in iter(stream.readline, ''):
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
                
        except Exception as e:
            active_tasks[task_id]["status"] = "failed"
    
    # 启动后台线程
    task_thread = threading.Thread(target=run_task)
    task_thread.daemon = True
    task_thread.start()
    
    # 返回任务ID
    return redirect(url_for('task_detail', task_id=task_id))

@app.route('/task-detail/<task_id>')
def task_detail(task_id):
    if task_id not in active_tasks:
        return redirect(url_for("results"))
    return render_template('task_detail.html', task=active_tasks[task_id])

@app.route('/results')
def results():
    """显示所有评估结果"""
    # 获取所有任务
    tasks = list(active_tasks.values())
    
    # 按创建时间降序排序
    tasks.sort(key=lambda x: x.get('created_at', 0), reverse=True)
    
    return render_template('results.html', tasks=tasks)

@app.route('/task-status/<task_id>')
def task_status(task_id):
    """获取任务状态"""
    if task_id not in active_tasks:
        return jsonify({"status": "unknown"})
    
    task = active_tasks[task_id]
    
    # 如果评测已完成但尚未检查完成状态，检查一次
    if task.get("status") == "evaluation_complete" and not task.get("status_checked", False):
        check_completion_status(task_id)
        task["status_checked"] = True
    
    # 根据任务状态返回不同的信息
    response = {
        "status": task.get("status", "running"),
        "progress": task.get("progress", 0),
        "evaluation_complete": task.get("is_evaluation_complete", False),
        "total_questions": task.get("total_questions", 0),
        "completed_questions": task.get("completed_questions", 0),
        "completion_rate": task.get("completed_questions", 0) / task.get("total_questions", 1) 
                          if task.get("total_questions", 0) > 0 else 0
    }
    
    # 如果评测已完成，添加评测结果信息
    if task.get("status") in ["completed", "incomplete"]:
        response.update({
            "valid_questions": task.get("valid_questions", 0),
            "valid_rate": task.get("valid_rate", 0),
            "is_valid_evaluation": task.get("is_valid_evaluation", False),
            "score": task.get("score", 0)
        })
    
    #print(response)
    
    return jsonify(response)

def check_completion_status(task_id):
    """检查任务完成状态并从result.json获取评测结果"""
    task = active_tasks[task_id]
    
    # 评测完成后，读取result.json文件获取详细结果
    if task.get("is_evaluation_complete", False):
        dataset = task["dataset"]
        model = task["model"]
        if task["eval_type"] == "llmjudge":
            judgment_model = task["judgment_model"]
            result_path = RESULTS_DIR / f"{dataset}_{model}_{judgment_model}_result.json"
        else:
            result_path = RESULTS_DIR / f"{dataset}_{model}_result.json"
        print(result_path)
        if result_path.exists():
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 分析响应数据
                total_questions = len(data)
                valid_questions = 0
                for d in data:
                    valid_sub_questions = 0
                    if task["eval_type"] == "llmjudge":
                        results = d["results"]
                        for r in results:
                            print(r["score"])
                            if r["score"] >= 0 and r["score"] <= 10:
                                valid_sub_questions += 1
                        if valid_sub_questions == len(results):
                            valid_questions += 1
                    else:
                        if d["response"] != "Neglected":
                            valid_questions += 1
                
                # 计算有效率和准确率
                valid_rate = valid_questions / total_questions if total_questions > 0 else 0
                task["total_questions"] = total_questions
                task["valid_questions"] = valid_questions
                task["valid_rate"] = valid_rate
                task["is_valid_evaluation"] = valid_rate >= 0.8  # 有效率>=95%为有效评测
                if valid_rate >= 0.8:
                    task["status"] = "completed"
                    if task["eval_type"] == "llmjudge":
                        score_result_dir = RESULTS_DIR / f"{dataset}_{model}_{judgment_model}_score.json"
                    else:
                        score_result_dir = RESULTS_DIR / f"{dataset}_{model}_result_accuracy.json"
                    with open(score_result_dir, 'r', encoding='utf-8') as f:
                        score_data = json.load(f)
                        if task["eval_type"] == "llmjudge":
                            score = score_data["average_score"]
                        else:
                            score = score_data["accuracy"] * 100
                        # # 更新任务信息
                        # task["total_questions"] = total_questions
                        # task["valid_questions"] = valid_questions

                        # task["valid_rate"] = valid_rate
                        # task["is_valid_evaluation"] = valid_rate >= 0.95  # 有效率>=95%为有效评测
                        task["score"] = score
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

if __name__ == '__main__':
    init_leaderboard()
    start_leaderboard_update_thread()
    app.run(debug=True)