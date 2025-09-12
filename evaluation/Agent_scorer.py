import yaml
import os
import json
import sys
import re
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.api.ConversationAPI import ConversationAPI
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print(f"Project root: {project_root}")
    raise

QA_SCORER_CONFIG_PATH = "dataset_info/QA_scorer_config.yaml"
with open(QA_SCORER_CONFIG_PATH, 'r', encoding='utf-8') as f:
    SCORER_CONFIG = yaml.safe_load(f)

def write_json_file(data, file_path):
    try:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"写入json文件出错：{str(e)}")
        return False

def read_json_file(file_path):
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"读取json文件出错：{str(e)}")
        return None
class Scorer:
    """Base scorer class that provides common functionality for all scorers."""
    
    def __init__(self, dataset_name, user_id, business_id):
        self.user_id = user_id
        self.dataset_name = dataset_name
        self.business_id = business_id
        self.result_file = None
        self.existing_results = None
        self.score_file = None
    
    def _prepare_scoring(self):
        """Common preparation workflow for all scorers."""
        import glob
        
        # Find result file
        result_file_pattern = f"results/{self.user_id}/*{self.business_id}_result.json"
        matching_result_files = glob.glob(result_file_pattern)
        if not matching_result_files:
            raise FileNotFoundError(f"找不到business_id为{self.business_id}的结果文件")
        
        self.result_file = matching_result_files[0]
        
        # Load existing results
        try:
            self.existing_results = read_json_file(self.result_file)
            if self.existing_results is None:
                raise ValueError("结果文件为空或格式错误")
        except Exception as e:
            print(f"结果读取失败，无法进行评分: {e}")
            raise
        
        # Set up score file path
        self.score_file = f"results/{self.user_id}/{self.business_id}_score.json"
        
        return self.existing_results
    
    def _save_score_results(self, score_results):
        """Save score results to file."""
        return write_json_file(score_results, self.score_file)
    
    def scoring(self):
        """Default scoring implementation - should be overridden by subclasses."""
        self._prepare_scoring()
        
        score_results = {
            "valid_ratio": 0.0,
            "score": 0.0
        }
        self._save_score_results(score_results)
        return score_results
    


class Accuracy_scorer(Scorer):
    def __init__(self, dataset_name, user_id, business_id):
        super().__init__(dataset_name, user_id, business_id)
    
    def scoring(self):
        # Use the common preparation workflow from base class
        existing_results = self._prepare_scoring()
        
        # TODO: Implement accuracy scoring logic here
        # For now, return basic results based on the data loaded
        total_questions = len(existing_results) if existing_results else 0
        valid_questions = 0
        correct_questions = 0
        score_results = {
            "valid_ratio": 0.0,
            "score": 0.0
        }
        self._save_score_results(score_results)
        
        for i in range(total_questions):
            response = existing_results[i]['response']
            if response != "Neglected":
                valid_questions += 1
                answer = existing_results[i]['answer']
                if response == answer:
                    correct_questions += 1
        
        valid_ratio = valid_questions / total_questions
        score = correct_questions / valid_questions * 100
        score_results = {
                "valid_ratio": valid_ratio,
                "score": score
            }
        self._save_score_results(score_results)
        return score_results




if __name__ == "__main__":
    scorer = Rubric_scorer("HealthBench", "test", "HealthBench_doubao-1.5-pro-32k_202509091443")
    scorer.scoring()



            
                
                

            
                







