import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
print(project_root)
sys.path.append(str(project_root))

from src.dataset.LLMJudge.LLMJudgeBase import *
from src.api.text_api import *
from src.api.multiturn_text_api import *
from src.utils.LLMJudge_constants import *
from tqdm import tqdm
import concurrent.futures
from typing import List, Tuple
from openai import BadRequestError

def write_json_file(data, file_path):
    """将数据写入JSON文件"""
    try:
        # 确保目录存在
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"数据已成功写入: {file_path}")
        return True
    except Exception as e:
        print(f"写入JSON文件时出错: {str(e)}")
        return False

def read_json_file(file_path):
    """从JSON文件读取数据"""
    try:
        if not os.path.exists(file_path):
            return None
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"读取JSON文件时出错: {str(e)}")
        return None

def generate_question_list_response(question: list, model_name: str, temperature: float):
    results = []
    chat = TextAPI(model_name, GENERATE_SYSTEM_PROMPT, "", temperature)
    for q in question:
        try:
            print(q)
            response = chat.chat(q, conversation_id="test_conv")
            print(response)
            results.append(response)
        except BadRequestError as e:
            print("BadRequestError")
    return results

def generate_evaluate_response(questions: list, reference_answers: list, generate_model_name: str, evaluate_model_name: str, temperature: float):
    evaluate_responses = []
    for i, question in enumerate(questions):
        generate_chat = MultiturnTextAPI(generate_model_name, GENERATE_SYSTEM_PROMPT, question, temperature, "GenerateAgent")
        generate_response = generate_chat.generate_response()
        if reference_answers is not None:
            reference_answer = reference_answers[i]
            evaluate_prompt = f"Question: {question}\n\nGenerate Response: {generate_response}\n\nReference Answer: {reference_answer}"
            evaluate_chat = TextAPI(evaluate_model_name, JUDGE_SYSTEM_PROMPT_REASONING, evaluate_prompt, 0.7)
        else:
            evaluate_prompt = f"Question: {question}\n\nGenerate Response: {generate_response}"
            evaluate_chat = TextAPI(evaluate_model_name, JUDGE_SYSTEM_PROMPT, evaluate_prompt, 0.7)
        evaluate_response = evaluate_chat.generate_response()
        evaluate_responses.append({
                "question": question,
                "generate_response": generate_response,
                "reference_answer": reference_answers[i] if reference_answers is not None else None,
                "evaluate_response": evaluate_response
            })
    return evaluate_responses

    

def evaluate_llmjudge(dataset_name: str, generate_model_name: str, evaluate_model_name: str, question_number: int):
    dataset = LLMJudgeBase(dataset_name)
    questions = dataset.questions[question_number]
    answers = dataset.answers[question_number]
    result_dir = Path("results") / "LLMJudge"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{dataset_name}_{generate_model_name}_{evaluate_model_name}_result.json"
    if isinstance(questions, str):
        questions = [questions]
    if answers is not None:
        temperature = 0
    else:
        temperature = 0.7
    evaluate_responses = generate_evaluate_response(questions, answers, generate_model_name, evaluate_model_name, temperature)
    result_data = {
        "dataset_name": dataset_name,
        "generate_model_name": generate_model_name,
        "evaluate_model_name": evaluate_model_name,
        "question_number": question_number,
        "results": evaluate_responses
    }
    write_json_file(result_data, result_file)
    
    return evaluate_responses




    



if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    print(evaluate_llmjudge("MT-Bench", "gpt-3.5-turbo", "gpt-4o", 2))



