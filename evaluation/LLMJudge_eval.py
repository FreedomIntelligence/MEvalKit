import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
print(project_root)
sys.path.append(str(project_root))

from src.dataset.LLMJudge.LLMJudgeBase import *
from src.api.api import *
from src.utils.LLMJudge_constants import *
from tqdm import tqdm
import concurrent.futures
from typing import List, Tuple
from openai import BadRequestError

def generate_question_list_response(question: list, model_name: str, temperature: float):
    results = []
    model = ChatOpenAI(model_name=model_name, temperature=temperature)
    chat = TestChat(model, system_template=GENERATE_SYSTEM_PROMPT, agent_name="GenerateAgent")
    for q in question:
        try:
            print(q)
            response = chat.chat(q, conversation_id="test_conv")
            print(response)
            results.append(response)
        except BadRequestError as e:
            print("BadRequestError")
    return results

def generate_evaluate_response(questions: list, reference_answers: list, generate_model_name: str, evaluate_model_name: str, generate_temperature: float):
    generate_model = ChatOpenAI(model_name=generate_model_name, temperature=generate_temperature)
    evaluate_model = ChatOpenAI(model_name=evaluate_model_name, temperature=0)
    generate_chat = TestChat(generate_model, system_template=GENERATE_SYSTEM_PROMPT, agent_name="GenerateAgent")
    if reference_answers is not None:
        evaluate_chat = TestChat(evaluate_model, system_template=JUDGE_SYSTEM_PROMPT_REASONING, agent_name="JudgeAgent")
    else:
        evaluate_chat = TestChat(evaluate_model, system_template=JUDGE_SYSTEM_PROMPT, agent_name="JudgeAgent")
    evaluate_responses = []
    for i, question in enumerate(questions):
        generate_response = generate_chat.chat(question, conversation_id="generate_conv")
        if reference_answers is None:
            evaluate_prompt = f"Question: {question}\n\nGenerate Response: {generate_response}"
        else:
            reference_answer = reference_answers[i]
            evaluate_prompt = f"Question: {question}\n\nGenerate Response: {generate_response}\n\nReference Answer: {reference_answer}"
        evaluate_response = evaluate_chat.chat(evaluate_prompt, conversation_id="evaluate_conv")
        evaluate_responses.append(evaluate_response)
    return evaluate_responses


    

def evaluate(dataset_name: str, generate_model_name: str, evaluate_model_name: str, question_number: int):
    dataset = LLMJudgeBase(dataset_name)
    questions = dataset.questions[question_number]
    answers = dataset.answers[question_number]
    if isinstance(questions, str):
        questions = [questions]
    if answers is not None:
        temperature = 0
    else:
        temperature = 0.7
    evaluate_responses = generate_evaluate_response(questions, answers, generate_model_name, evaluate_model_name, temperature)
    return evaluate_responses




    



if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    print(evaluate("MT-Bench", "gpt-3.5-turbo", "gpt-4o", 2))



