import json
from langgraph.graph import StateGraph

def flow_node(name, workflow: StateGraph):
    def decorator(func):
        workflow.add_node(name, func)
        return func
    return decorator

def conditional_edges(name, workflow: StateGraph):
    def decorator(func):
        workflow.add_conditional_edges(name, func)
        return func
    return decorator

def entrypoint(name, workflow: StateGraph):
    def decorator(func):
        workflow.set_entry_point(name)
        return func
    return decorator




def load_dataset_info(path):
    with open(path, 'r') as f:
        return json.load(f)