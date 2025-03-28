from typing import Annotated, Literal, TypedDict, List
import io, os, sys, json, re
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from PIL import Image
from io import BytesIO
import base64
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils_stategraph import *
from utils.MCQ_constants import *
from utils.LLMJudge_constants import * 
import random




class TestChat():

    def __init__(self, model: ChatOpenAI, system_template: str, agent_name: str, random_seed: int = 42, language: Literal["en", "zh"] = "en"):
        self.model = model
        self.workflow = StateGraph(MessagesState)
        self.random_seed = random_seed
        self.language = language
        self.app = self.build(system_template, agent_name)
        self.conversation_history = {}  # 用于存储不同会话的历史记录

    def build(self, system_template: str, agent_name: str):
        workflow = self.workflow
        model = self.model
        @entrypoint(name=agent_name, workflow=workflow)
        @flow_node(name=agent_name, workflow=workflow)
        def agent_call(state: MessagesState):
            template = system_template
            messages = [SystemMessage(content=template)] + state['messages']
            response = model.invoke(messages)
            response.additional_kwargs['label'] = 'chat'
            response.additional_kwargs['node_name'] = agent_name

            return {
                "messages": [response]
            }

        checkpointer = MemorySaver()
        app = workflow.compile(checkpointer=checkpointer)
        return app
    


    def chat(self, question: str, conversation_id: str = "default"):
        # 如果是新的会话，初始化历史记录
        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []
        
        # 添加新的用户消息
        messages = [HumanMessage(content=question, additional_kwargs={'label': 'chat'})]
        
        # 如果有历史记录，将其添加到当前消息列表中
        if self.conversation_history[conversation_id]:
            messages = self.conversation_history[conversation_id] + messages
        
        # 调用模型获取回复
        final_state = self.app.invoke(
            {'messages': messages},
            config={"configurable": {"thread_id": conversation_id}}
        )
        
        # 获取最新回复
        final_response = final_state['messages'][-1].content
        
        # 更新会话历史记录
        self.conversation_history[conversation_id] = messages + [final_state['messages'][-1]]
        
        return final_response
    
    def clear_history(self, conversation_id: str = "default"):
        """清除指定会话的历史记录"""
        if conversation_id in self.conversation_history:
            self.conversation_history[conversation_id] = []
            return True
        return False





if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chat_trial = TestChat(model, system_template="You are a helpful assistant that can answer questions that have multiple choices. Do not make any analysis or explanation, just answer with only the letter of the correct choice (A/B/C/D).", agent_name="MCQ_Agent", language="en", random_seed=42)
    
    # 第一轮对话
    question1 = "Which of the cities are in Europe? A. Washington B. London C. Rome D. Madrid"
    answer1 = chat_trial.chat(question1, conversation_id="test_conv")
    print(f"问题1: {question1}")
    print(f"回答1: {answer1}")
    
    # 第二轮对话
    question2 = "在上述选项中，哪些城市是首都？"
    answer2 = chat_trial.chat(question2, conversation_id="test_conv")
    print(f"问题2: {question2}")
    print(f"回答2: {answer2}")
    
    # 清除历史记录
    chat_trial.clear_history("test_conv")

