from flask import Flask, render_template, request, jsonify, redirect, url_for
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.model_and_dataset import *
from evaluation.TextMCQ_eval import *
from evaluation.ImageMCQ_eval import *
from evaluation.LLMJudge_eval import *

app = Flask(__name__)

# 评估结果目录
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# 存储运行中的任务
active_tasks = {}

leaderboard_data = {}
last_leaderboard_update = 0

LEADERBOARD_DATASETS = GENERAL_DATASETS + MEDICAL_KNOWLEDGE_DATASETS + MEDICAL_ETHICS_DATASETS



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
    eval_modes = ["start_from_beginning", "resume_from_checkpoint", "give_answers"]
    
    return render_template('create_task.html', 
                          text_datasets=TEXT_DATASETS,
                          multimodal_datasets=IMAGE_DATASETS,
                          llmjudge_datasets=LLMJUDGE_DATASETS,
                          text_models=TEXT_MODELS,
                          multimodal_models=MULTIMODAL_MODELS,
                          judge_models=JUDGE_MODELS,
                          eval_modes=eval_modes,
                    )

@app.route('/run-evaluation', methods=['POST'])
def run_evaluation():
    """运行评估任务"""
    dataset = request.form.get('dataset')
    model_name = request.form.get('model')
    eval_mode = request.form.get('eval_mode')
    judgment_model_name = request.form.get('judgment_model')
    
    # 处理评测数量
    question_limit = None
    try:
        limit_input = request.form.get('question_limit', '').strip()
        if limit_input:
            question_limit = int(limit_input)
            if question_limit <= 0:
                question_limit = None
    except (ValueError, TypeError):
        question_limit = None
    
    # 人工评估模式特殊处理
    if eval_mode == "give_answers":
        model_name = "人工评估"  # 设置一个标识性名称
    
    if not dataset or not eval_mode or (eval_mode != "give_answers" and not model_name):
        return jsonify({"status": "error", "message": "缺少必要参数"}), 400
    
    # 创建任务ID
    task_id = f"{dataset}_{model_name}_{int(time.time())}"
    
    # 确定评估类型
    if dataset in TEXT_DATASETS:
        eval_type = "text"
    elif dataset in MULTIMODAL_DATASETS:
        eval_type = "multimodal"
    elif dataset in LLMJUDGE_DATASETS:
        eval_type = "llmjudge"
    else:
        return jsonify({"status": "error", "message": "不支持的数据集类型"}), 400
    
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
        "question_limit": question_limit,  # 添加题目数量限制
        "is_evaluation_complete": False
    }
    
    # 处理人工评估模式
    if eval_mode == "give_answers":
        try:
            # 加载数据集
            if dataset in TEXT_DATASETS:
                data = TextMCQ(dataset)
            elif dataset in MULTIMODAL_DATASETS:
                data = ImageMCQ(dataset)
            elif dataset in LLMJUDGE_DATASETS:
                data = LLMJudge(dataset)
                
            # 获取问题和答案
            questions = data.questions
            answers = data.answers if hasattr(data, 'answers') else []
            choices = data.choices if hasattr(data, 'choices') else []
            
            # 应用题目数量限制
            if question_limit and question_limit < len(questions):
                questions = questions[:question_limit]
                if answers:
                    answers = answers[:question_limit]
                if choices:
                    choices = choices[:question_limit]
            
            # 更新任务信息
            active_tasks[task_id]["questions"] = questions
            active_tasks[task_id]["answers"] = answers
            active_tasks[task_id]["choices"] = choices
            active_tasks[task_id]["status"] = "human_evaluation"
            active_tasks[task_id]["total_questions"] = len(questions)
            
            return redirect(url_for('human_evaluation', task_id=task_id))
        except Exception as e:
            active_tasks[task_id]["status"] = "failed"
            active_tasks[task_id]["error"] = str(e)
            print(f"启动人工评估失败: {str(e)}")
            return redirect(url_for('task_detail', task_id=task_id))
            
    # 其他评估模式的处理逻辑
    # ... existing code ...

@app.route('/human-evaluation/<task_id>')
def human_evaluation(task_id):
    """显示人工评估页面"""
    if task_id not in active_tasks:
        return redirect(url_for('results'))
    
    task = active_tasks[task_id]
    
    # 检查是否是人工评估任务
    if task["eval_mode"] != "give_answers":
        return redirect(url_for('task_detail', task_id=task_id))
    
    # 获取当前问题索引（允许通过URL参数直接跳转）
    current_index = request.args.get('question_index', None)
    if current_index is not None:
        try:
            current_index = int(current_index)
            if current_index < 0 or current_index >= len(task["questions"]):
                current_index = task.get("current_question_index", 0)
        except ValueError:
            current_index = task.get("current_question_index", 0)
    else:
        current_index = task.get("current_question_index", 0)
    
    # 获取当前页码
    current_page = request.args.get('page', None)
    if current_page is not None:
        try:
            current_page = int(current_page)
        except ValueError:
            current_page = (current_index // 50) + 1
    else:
        current_page = (current_index // 50) + 1
    
    # 检查是否所有问题已回答
    if task.get("is_all_answered", False):
        # 所有问题已回答完毕
        calculate_human_evaluation_results(task_id)
        return redirect(url_for('task_detail', task_id=task_id))
    
    # 初始化已回答问题集合（如果不存在）
    if "answered_questions" not in task:
        task["answered_questions"] = set()
    
    # 初始化用户选择记录（如果不存在）
    if "user_selections" not in task:
        task["user_selections"] = [None] * len(task["questions"])
    
    # 获取当前问题
    current_question = task["questions"][current_index]
    
    # 渲染问题页面
    return render_template('human_evaluation.html', 
                          task=task,
                          question=current_question,
                          question_index=current_index,
                          total_questions=len(task["questions"]))

@app.route('/submit-answer/<task_id>', methods=['POST'])
def submit_answer(task_id):
    """接收用户提交的答案"""
    if task_id not in active_tasks:
        return redirect(url_for('results'))
    
    task = active_tasks[task_id]
    
    # 检查是否是人工评估任务
    if task["eval_mode"] != "give_answers":
        return redirect(url_for('task_detail', task_id=task_id))
    
    # 获取用户提交的答案
    answer = request.form.get('answer')
    question_index = int(request.form.get('question_index', 0))
    
    if not answer:
        # 如果未选择答案，重定向回原题目
        return redirect(url_for('human_evaluation', task_id=task_id, question_index=question_index))
    
    # 获取当前问题
    current_question = task["questions"][question_index]
    
    # 记录用户答案
    user_answer = {
        "question_index": question_index,
        "question": current_question,
        "user_answer": answer,
        "correct_answer": task["answers"][question_index],
        "is_correct": answer.upper() == task["answers"][question_index].upper()
    }
    
    # 标记题目为已回答
    if "answered_questions" not in task:
        task["answered_questions"] = set()
    task["answered_questions"].add(question_index)
    
    # 记录用户选择
    if "user_selections" not in task:
        task["user_selections"] = [None] * len(task["questions"])
    task["user_selections"][question_index] = answer
    
    # 如果之前没有记录过这个问题的答案，添加到answers列表
    found = False
    for i, ans in enumerate(task.get("user_answers", [])):
        if ans["question_index"] == question_index:
            task["user_answers"][i] = user_answer
            found = True
            break
    
    if not found:
        if "user_answers" not in task:
            task["user_answers"] = []
        task["user_answers"].append(user_answer)
    
    # 更新进度
    completed = len(task["answered_questions"])
    total = len(task["questions"])
    task["completed_questions"] = completed
    task["progress"] = int(completed / total * 100)
    
    # 检查是否完成所有问题
    if completed >= total:
        task["is_all_answered"] = True
        # 计算结果
        calculate_human_evaluation_results(task_id)
        return redirect(url_for('task_detail', task_id=task_id))
    
    # 移动到下一个问题（如果有）
    next_index = question_index + 1
    if next_index >= len(task["questions"]):
        # 如果达到最后一题，返回第一个未回答的题目
        for i in range(len(task["questions"])):
            if i not in task["answered_questions"]:
                next_index = i
                break
    
    task["current_question_index"] = next_index
    
    # 继续下一个问题
    return redirect(url_for('human_evaluation', task_id=task_id, question_index=next_index))

# 计算人工评估结果的函数
def calculate_human_evaluation_results(task_id):
    """计算人工评估的准确率和结果"""
    task = active_tasks[task_id]
    
    # 确保答案按题号排序
    if "user_answers" in task:
        task["user_answers"].sort(key=lambda x: x["question_index"])
    
    # 构建完整的答案列表（确保包含所有题目）
    complete_answers = []
    for i in range(len(task["questions"])):
        # 查找该题目的答案
        answer_found = False
        for ans in task.get("user_answers", []):
            if ans["question_index"] == i:
                complete_answers.append(ans)
                answer_found = True
                break
        
        # 如果没有找到答案，添加空答案
        if not answer_found:
            complete_answers.append({
                "question_index": i,
                "question": task["questions"][i],
                "user_answer": "未回答",
                "correct_answer": task["answers"][i] if i < len(task.get("answers", [])) else "未知",
                "is_correct": False
            })
    
    # 更新排序后的答案列表
    task["user_answers"] = complete_answers
    
    # 计算准确率（只考虑已回答的题目）
    answered_questions = [ans for ans in complete_answers if ans["user_answer"] != "未回答"]
    total_questions = len(answered_questions)
    correct_answers = sum(1 for answer in answered_questions if answer.get("is_correct", False))
    
    # 计算准确率
    accuracy = correct_answers / total_questions if total_questions > 0 else 0
    
    # 更新任务状态
    task["is_evaluation_complete"] = True
    task["status"] = "completed"
    task["valid_questions"] = total_questions
    task["valid_rate"] = total_questions / len(task["questions"]) if len(task["questions"]) > 0 else 0
    task["is_valid_evaluation"] = total_questions >= (len(task["questions"]) * 0.8)  # 至少回答80%的题目
    task["score"] = accuracy * 100  # 转换为百分比
    
    # 保存结果到文件
    save_human_evaluation_results(task_id)

# 保存人工评估结果
def save_human_evaluation_results(task_id):
    """将人工评估结果保存到文件"""
    task = active_tasks[task_id]
    
    # 创建结果目录
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # 再次确保答案是按题号排序的
    sorted_answers = sorted(task.get("user_answers", []), key=lambda x: x["question_index"])
    
    # 准备结果数据
    result_data = {
        "task_id": task_id,
        "dataset": task["dataset"],
        "model": task["model"],
        "eval_mode": task["eval_mode"],
        "completed_at": time.time(),
        "answers": sorted_answers,
        "total_questions": len(task["questions"]),
        "completed_questions": len([a for a in sorted_answers if a["user_answer"] != "未回答"]),
        "correct_answers": sum(1 for a in sorted_answers if a.get("is_correct", False)),
        "accuracy": task["score"] / 100  # 保存为小数形式
    }
    
    # 保存详细结果
    result_file = results_dir / f"{task['dataset']}_human_result.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    # 保存准确率结果
    accuracy_file = results_dir / f"{task['dataset']}_human_result_accuracy.json"
    with open(accuracy_file, 'w', encoding='utf-8') as f:
        json.dump({"accuracy": task["score"] / 100}, f, ensure_ascii=False, indent=2)
    

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