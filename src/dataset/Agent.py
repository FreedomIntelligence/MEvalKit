import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from datasets import load_dataset, Dataset
from src.utils.utils_loading import *

from typing import List, Tuple, Dict, Any

DATASET_INFO = load_dataset_info("dataset_info/Agent_config.yaml")

class Agent:

    def __init__(self, dataset_name: str):
        self.dataset_info = DATASET_INFO[dataset_name]
        
        # Load all agents defined in the data section
        self.agents = {}
        for key, value in self.dataset_info.items():
            if key not in ['language', 'max_score', 'scoring_criteria'] and isinstance(value, dict) and 'data' in value:
                self.agents[key] = self.load_agent_component(key)
        
        # Load metadata if available
        self.language = self.dataset_info.get('language', 'zh')
        self.max_score = self.dataset_info.get('max_score', 100)
        self.max_turn = self.dataset_info.get('max_turn')
        self.scoring_criteria = self.dataset_info.get('scoring_criteria', 'agent')

    def load_agent_component(self, agent_key: str):
        """
        Load an agent component including data and prompt template
        """
        agent_info = self.dataset_info[agent_key]
        agent_component = {
            'data': None,
            'prompt_template': None,
            'keys': None
        }
        
        data_info = agent_info.get('data', {})
        if not data_info:
            return agent_component
            
        loading_way = data_info.get('loading_way', '')
        key = data_info.get('key', [])
        sub_key = data_info.get('sub_key', None)
        
        if loading_way == "":
            return None
            
        # Load raw data using the existing loading infrastructure
        raw_data = load_dataset_compile(data_info, loading_way)
        data = []
        
        for d in raw_data:
            if isinstance(key, list):
                if len(key) == 1:
                    # Single key case
                    datum = d.get(key[0], None)
                    if sub_key and isinstance(sub_key, list):
                        # Handle sub_key extraction
                        if datum and isinstance(datum, list) and len(datum) > 0:
                            first_item = datum[0]
                            if isinstance(first_item, dict) and len(sub_key) == 1:
                                extracted_value = first_item.get(sub_key[0], "")
                                data.append(str(extracted_value))
                            else:
                                data.append(str(first_item))
                        else:
                            data.append("")
                    else:
                        # No sub_key, process datum directly
                        if isinstance(datum, list):
                            # If datum is a list, convert each item to string
                            agent_list = [str(item) for item in datum]
                            data.append(agent_list)
                        elif datum is None:
                            data.append(None)
                        else:
                            # Convert other types to string
                            data.append(str(datum))
                else:
                    # Multiple keys case - combine into a dictionary or formatted string
                    values = {}
                    for k in key:
                        val = d.get(k, "")
                        values[k] = str(val)
                    data.append(values)
            else:
                # key is not a list
                datum = d.get(key, None)
                if datum is None:
                    data.append("")
                else:
                    data.append(str(datum))
                    
        agent_component['data'] = data
        agent_component['prompt_template'] = agent_info.get("prompt_template", None)
        agent_component['keys'] = key
        return agent_component
    
    def get_agent_data(self, agent_key: str, index: int = None):
        """
        Get data for a specific agent
        """
        if agent_key not in self.agents:
            return None
            
        agent = self.agents[agent_key]
        if agent is None or agent['data'] is None:
            return None
            
        if index is not None:
            if 0 <= index < len(agent['data']):
                return agent['data'][index]
            else:
                return None
        else:
            return agent['data']
    
    def get_agent_prompt_template(self, agent_key: str):
        """
        Get prompt template for a specific agent
        """
        if agent_key not in self.agents:
            return None
            
        agent = self.agents[agent_key]
        if agent is None:
            return None
            
        return agent.get('prompt_template', None)
    
    def format_agent_prompt(self, agent_key: str, index: int, **kwargs):
        """
        Format agent prompt template with data at given index and additional kwargs
        """
        template = self.get_agent_prompt_template(agent_key)
        data = self.get_agent_data(agent_key, index)
        
        if template is None or data is None:
            return None
            
        # Prepare format arguments
        format_args = kwargs.copy()
        
        # Add data to format arguments
        if isinstance(data, dict):
            format_args.update(data)
        elif isinstance(data, str):
            # If data is a string and we have keys, try to map it
            agent = self.agents[agent_key]
            keys = agent.get('keys', [])
            if isinstance(keys, list) and len(keys) == 1:
                format_args[keys[0]] = data
        
        try:
            return template.format(**format_args)
        except KeyError as e:
            print(f"Warning: Missing key {e} when formatting prompt for agent {agent_key}")
            return template
    
    def get_available_agents(self):
        """
        Get list of available agent keys
        """
        return list(self.agents.keys())

if __name__ == "__main__":
    dataset = Agent("IOR-Dynamic")
    print("Available agents:", dataset.get_available_agents())
    
    # Test data loading for each agent
    for agent_key in dataset.get_available_agents():
        print(f"\n{agent_key} data sample:")
        sample_data = dataset.get_agent_data(agent_key, 0)
        print(sample_data)
        
        print(f"\n{agent_key} prompt template:")
        template = dataset.get_agent_prompt_template(agent_key)
        print(template)