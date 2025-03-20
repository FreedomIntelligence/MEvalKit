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
from utils.constants import *
import random




class TestChat():

    def __init__(self, model: ChatOpenAI, random_seed: int = 42, language: Literal["en", "zh"] = "en"):
        self.model = model
        self.workflow = StateGraph(MessagesState)
        self.random_seed = random_seed
        self.language = language
        self.app = self.build()
        self.history = []
        self.chat_history = []

    def build(self):
        workflow = self.workflow
        model = self.model
        system_template = MCQ_TEMPLATE_EN if self.language == "en" else MCQ_TEMPLATE_ZH
        @entrypoint(name="MCQ_Agent", workflow=workflow)
        @flow_node(name="MCQ_Agent", workflow=workflow)
        def MCQ_agent_call(state: MessagesState):
            template = system_template
            messages = [SystemMessage(content=template)] + state['messages']
            response = model.invoke(messages)
            response.additional_kwargs['label'] = 'chat'
            response.additional_kwargs['node_name'] = 'MCQ_Agent'

            return {
                "messages": [response]
            }

        checkpointer = MemorySaver()
        app = workflow.compile(checkpointer=checkpointer)
        return app
    


    def chat(self, question: str):
        messages = [HumanMessage(content=question, additional_kwargs={'label': 'chat'})]
        final_state = self.app.invoke(
            {'messages': messages},
            config={"configurable": {"thread_id": 42}}
        )
        final_response = final_state['messages'][-1].content
        self.chat_history.append([question, final_response])
        self.history = final_state['messages']
        return final_response





if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    chat_trial = TestChat(model, language="en", random_seed=42)
    prompt = "You are a helpful assistant that can answer questions that have multiple choices. Do not make any analysis or explanation, just answer with only the letter of the correct choice (A/B/C/D)."
    answer = chat_trial.chat("Which of the cities are in Europe? A. Washington B. London C. Rome D. Madrid")

