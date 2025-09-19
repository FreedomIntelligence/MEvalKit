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
    from src.utils.config import config
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
    
class LLMJudge_scorer(Scorer):
    def __init__(self, dataset_name, user_id, business_id):
        super().__init__(dataset_name, user_id, business_id)
        
        # Use provided parameters or fall back to config defaults
        self.judge_model = SCORER_CONFIG[dataset_name]['judge_model']
        
        self.system_prompt = SCORER_CONFIG[dataset_name]['system_prompt']
        self.judge_prompt = SCORER_CONFIG[dataset_name]['judge_prompt']
        self.reference_answer_prompt = SCORER_CONFIG[dataset_name]['reference_answer_prompt']
        self.reference_prompt = SCORER_CONFIG[dataset_name]['reference_prompt']

        self.max_score = SCORER_CONFIG[dataset_name]['max_score']
    
    def extract_scores(self, response_text):
        """
        Extract scores from LLM judge response text.
        Supports multiple formats:
        1. Single score: "Rating: [[8]]" -> returns 8.0
        2. Multi-dimensional: "3,4,5,2" -> returns [3.0, 4.0, 5.0, 2.0]
        
        Returns:
            float or list of floats or None if no score found
        """
        if not response_text or not isinstance(response_text, str):
            return None
            
        # Try to extract single score format: Rating: [[8]]
        single_patterns = [
            r'Rating:\s*\[\[(\d+(?:\.\d+)?)\]\]',
            r'rating:\s*\[\[(\d+(?:\.\d+)?)\]\]', 
            r'\[\[(\d+(?:\.\d+)?)\]\]'
        ]
        
        for pattern in single_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        # Try to extract multi-dimensional scores: 3,4,5,2
        multi_patterns = [
            r'^(\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?)+)',  # Start of line
            r'(\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?)+)$',  # End of line
            r'(\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?){1,})'  # Anywhere in text
        ]
        
        for pattern in multi_patterns:
            matches = re.findall(pattern, response_text, re.MULTILINE)
            for match in matches:
                try:
                    scores = [float(s.strip()) for s in match.split(',')]
                    if len(scores) >= 2:  # At least 2 scores
                        return scores
                except ValueError:
                    continue
        
        return None
    
    def _generate_prompt(self, prompt_template, question=None, content=None, content_key=None):
        """
        Unified method to generate prompts for both response and answer cases.
        
        Args:
            prompt_template: The type of prompt template to use ('judge_prompt', 'reference_answer_prompt', 'reference_prompt')
            question: Either a string (simple case) or a list of questions (multi-turn) - not needed for reference_prompt
            content: Either a string (simple case) or a list of content (multi-turn) - only needed for reference_prompt when it's reference content
            content_key: The key to use for content formatting ('response' or 'answer') - not needed for reference_prompt
            
        Returns:
            str: The formatted judge prompt
        """
        prompt = ""
        if prompt_template == "judge_prompt":
            prompt = self.judge_prompt
        elif prompt_template == "reference_answer_prompt":
            prompt = self.reference_answer_prompt
        elif prompt_template == "reference_prompt":
            prompt = self.reference_prompt
        
        if prompt == "" or prompt is None:
            return ""
        
        # Special handling for reference_prompt - it doesn't use question/content_key pattern
        if prompt_template == "reference_prompt":
            # For reference_prompt, we only need to format with reference content if provided
            if content is not None:
                if isinstance(content, str):
                    return prompt.format(reference=content)
                elif isinstance(content, list):
                    # For multi-turn, format as reference content
                    format_dict = {'reference': '\n'.join(content) if content else ''}
                    return prompt.format(**format_dict)
            else:
                # If no content provided, return the prompt as-is (it might not need formatting)
                try:
                    return prompt.format()
                except:
                    return prompt
        
        # For judge_prompt and reference_answer_prompt, use the original logic
        if question is None or content is None or content_key is None:
            raise ValueError(f"question, content, and content_key are required for {prompt_template}")
        
        # Case 1: Simple string format
        if isinstance(question, str) and isinstance(content, str):
            return prompt.format(question=question, **{content_key: content})
        
        # Case 2: Multi-turn format (lists)
        elif isinstance(question, list) and isinstance(content, list):
            format_dict = {}
            
            # Create format dictionary with question_1, response_1/answer_1, question_2, response_2/answer_2, etc.
            for i, (q, c) in enumerate(zip(question, content), 1):
                format_dict[f'question_{i}'] = q
                format_dict[f'{content_key}_{i}'] = c
            
            return prompt.format(**format_dict)
        
        else:
            raise ValueError(f"question and {content_key} must both be strings or both be lists")

    def scoring(self):
        # Use the common preparation workflow from base class
        existing_results = self._prepare_scoring()
        
        # Initialize score results with default values
        score_results = {
            "valid_ratio": 0.0,
            "score": 0.0
        }
        self._save_score_results(score_results)

        total_questions = len(existing_results)
        valid_questions = 0
        scores = []

        for i in range(len(existing_results)):
            result = existing_results[i]['response']
            if isinstance(result, str):
                response = result
                question = existing_results[i]['question']
                answer = existing_results[i]['answer']
                response_prompt = self._generate_prompt("judge_prompt", question, response, "response")
                answer_prompt = self._generate_prompt("reference_answer_prompt", question, answer, "answer") if answer != "Neglected" else ""
                
                # Handle reference_prompt - get reference from existing_results
                reference_prompt = ""
                if self.reference_prompt and self.reference_prompt.strip():
                    reference_content = existing_results[i].get('reference', None)
                    if reference_content is not None:
                        reference_prompt = self._generate_prompt("reference_prompt", content=reference_content)
                    else:
                        reference_prompt = self._generate_prompt("reference_prompt")
                
                evaluate_prompt = response_prompt + '\n' + answer_prompt + '\n' + reference_prompt
            else:
                response = []
                question = []
                answer = []
                for j in range(len(result)):
                    response.append(result[j]['response'])
                    question.append(result[j]['question'])
                    answer.append(result[j]['answer'])
                response_prompt = self._generate_prompt("judge_prompt", question, response, "response")
                answer_prompt = self._generate_prompt("reference_answer_prompt", question, answer, "answer") if not all(x == "Neglected" for x in answer) else ""
                
                # Handle reference_prompt for multi-turn - get reference from existing_results
                reference_prompt = ""
                if self.reference_prompt and self.reference_prompt.strip():
                    # For multi-turn, reference is stored in the first turn or at the parent level
                    reference_content = None
                    if len(result) > 0 and 'reference' in result[0]:
                        reference_content = result[0]['reference']
                    elif 'reference' in existing_results[i]:
                        reference_content = existing_results[i]['reference']
                    
                    if reference_content is not None:
                        reference_prompt = self._generate_prompt("reference_prompt", content=reference_content)
                    else:
                        reference_prompt = self._generate_prompt("reference_prompt")
                
                evaluate_prompt = response_prompt + "\n" + answer_prompt + "\n" + reference_prompt
            
            # 使用配置模块安全获取API密钥
            try:
                api_key = config.get_api_key_safe()
                api_base = config.get_api_base_safe()
            except (ValueError, AttributeError):
                # 降级处理，使用默认值（但这通常会失败，因为没有硬编码密钥了）
                print("警告: 无法获取API配置，请检查环境变量设置")
                api_key = os.getenv('OPENAI_API_KEY', '')
                api_base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
                if not api_key:
                    raise ValueError("API密钥未设置，请设置环境变量OPENAI_API_KEY")
            
            scorer_api = ConversationAPI(
                model_name=self.judge_model,
                system_prompt=self.system_prompt,
                user_prompt=evaluate_prompt,
                image_input=None,
                temperature=0.7,
                conversation_id=None,
                model_key=api_key,
                api_base=api_base,
                enable_history=False
            )
            scorer_response = scorer_api.generate_response()
            #print(scorer_response)
            
            # Extract scores from the response
            extracted_scores = self.extract_scores(scorer_response)
            
            if extracted_scores is not None:
                print(f"Extracted scores: {extracted_scores}")
                valid_questions += 1
                
                # Calculate final score (average for multi-dimensional scores)
                if isinstance(extracted_scores, list):
                    final_score = sum(extracted_scores) / len(extracted_scores)
                    print(f"Final score (average): {final_score}")
                    scores.append(final_score)
                else:
                    print(f"Final score: {extracted_scores}")
                    scores.append(extracted_scores)
            else:
                print("No scores could be extracted from the response")

        # Calculate final statistics
        if valid_questions > 0:
            valid_ratio = valid_questions / total_questions
            average_score = sum(scores) / len(scores)
            final_score = average_score / self.max_score * 100
            score_results = {
                "valid_ratio": valid_ratio,
                "score": average_score,
                "final_score": final_score
            }
            
            self._save_score_results(score_results)
        else:
            score_results = {
                "valid_ratio": 0.0,
                "score": 0.0,
                "final_score": 0.0
            }
            self._save_score_results(score_results)

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


class Rubric_scorer(Scorer):
    def __init__(self, dataset_name, user_id, business_id):
        super().__init__(dataset_name, user_id, business_id)
        self.judge_model = SCORER_CONFIG[dataset_name]['judge_model']
        self.system_prompt = SCORER_CONFIG[dataset_name]['system_prompt']
        self.judge_prompt = SCORER_CONFIG[dataset_name]['judge_prompt']
        self.rubric_prompt = SCORER_CONFIG[dataset_name]['rubric_prompt']
        self.rubric_criterion = SCORER_CONFIG[dataset_name]['rubric_criterion']
        self.rubrics_data = None

    def load_rubrics(self):
        """
        根据配置加载rubrics数据
        Returns:
            list: 包含每个问题的rubrics列表，结构为[[question1_rubrics], [question2_rubrics], ...]
                 每个question_rubrics是包含{'criterion': str, 'points': int}的列表
        """
        try:
            # 从配置中获取参数
            data_path = self.rubric_criterion['path']
            rubrics_key = self.rubric_criterion['key']
            criterion_key = self.rubric_criterion['sub_keys']['criterion']
            points_key = self.rubric_criterion['sub_keys']['points']
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"找不到rubrics数据文件: {data_path}")
            
            all_questions_rubrics = []
            
            # 逐行读取JSONL文件
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            rubrics = data.get(rubrics_key, [])
                            
                            # 提取该问题的所有rubrics
                            question_rubrics = []
                            for rubric in rubrics:
                                criterion = rubric.get(criterion_key)
                                points = rubric.get(points_key)
                                if criterion is not None and points is not None:
                                    question_rubrics.append({
                                        'criterion': criterion,
                                        'points': points
                                    })
                            
                            # 即使该问题没有rubrics，也要添加空列表保持索引对应
                            all_questions_rubrics.append(question_rubrics)
                                    
                        except json.JSONDecodeError as e:
                            print(f"解析JSON行时出错: {e}")
                            # 添加空列表保持索引对应
                            all_questions_rubrics.append([])
                            continue
            
            self.rubrics_data = all_questions_rubrics
            total_rubrics = sum(len(q_rubrics) for q_rubrics in all_questions_rubrics)
            print(f"成功加载 {len(all_questions_rubrics)} 个问题的rubrics数据，总计 {total_rubrics} 个rubrics")
            return all_questions_rubrics
            
        except Exception as e:
            print(f"加载rubrics数据时出错: {e}")
            return None

    def _generate_prompt(self, question, response, rubric_items):
        """
        仿照LLMJudge_scorer生成包含rubrics的prompt
        
        Args:
            question: Either a string (simple case) or a list of questions (multi-turn)
            response: Either a string (simple case) or a list of responses (multi-turn)
            rubric_items: String containing formatted rubric criterion and points
            
        Returns:
            str: The formatted judge prompt
        """
        judge_prompt = self.judge_prompt
        
        # Case 1: Simple string format
        if isinstance(question, str) and isinstance(response, str):
            return judge_prompt.format(
                question_1=question, 
                response_1=response, 
                rubric_items=rubric_items
            )
        
        # Case 2: Multi-turn format (lists)
        elif isinstance(question, list) and isinstance(response, list):
            format_dict = {'rubric_items': rubric_items}
            
            # Create format dictionary with question_1, response_1, question_2, response_2, etc.
            for i, (q, r) in enumerate(zip(question, response), 1):
                format_dict[f'question_{i}'] = q
                format_dict[f'response_{i}'] = r
            
            return judge_prompt.format(**format_dict)
        
        else:
            raise ValueError("question and response must both be strings or both be lists")

    def _format_rubric_items(self, rubrics):
        """
        格式化rubrics为prompt字符串
        
        Args:
            rubrics: List of rubric dictionaries with 'criterion' and 'points' keys
            
        Returns:
            str: Formatted rubric items string
        """
        if not rubrics:
            return ""
        
        # 根据配置中的rubric_prompt模板格式化
        formatted_items = []
        for rubric in rubrics:
            criterion = rubric['criterion']
            points = rubric['points']
            formatted_items.append(f"Criterion: {criterion}\nPoints: {points}")
        
        return self.rubric_prompt.format(rubric_items="\n\n".join(formatted_items))

    
    def extract_rubric_score(self, response_text):
        """
        Extract score from rubric evaluation response.
        Expected format is JSON with "criteria_met" boolean field.
        
        Returns:
            float: 1.0 if criteria met, 0.0 if not met, None if parsing failed
        """
        if not response_text or not isinstance(response_text, str):
            return None
        
        try:
            # Try to find JSON block in markdown format
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON directly
                json_match = re.search(r'(\{.*?\})', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    return None
            
            # Parse JSON
            result = json.loads(json_str)
            criteria_met = result.get('criteria_met', False)
            
            return 1.0 if criteria_met else 0.0
            
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"解析JSON响应时出错: {e}")
            return None

    def scoring(self):
        # Use the common preparation workflow from base class
        existing_results = self._prepare_scoring()
        
        # Load rubrics data
        rubrics = self.load_rubrics()
        if rubrics is None:
            print("加载rubrics失败，无法进行评分")
            score_results = {
                "valid_ratio": 0.0,
                "score": 0.0
            }
            self._save_score_results(score_results)
            return score_results
        
        total_questions = len(existing_results)
        valid_questions = 0
        total_points = 0
        total_possible_points = 0
        
        print(f"开始对 {total_questions} 个问题进行rubric评分...")
        
        for i in range(total_questions):
            question_rubrics = rubrics[i] if i < len(rubrics) else []
            question = existing_results[i]['question']
            response = existing_results[i]['response']
            
            if isinstance(response, str) and response != "Neglected":
                valid_questions += 1
                question_score = 0
                question_max_score = 0
                
                print(f"Question {i+1}: 评估 {len(question_rubrics)} 个rubrics")
                
                # 对每个rubric进行单独评分
                for rubric_idx, rubric in enumerate(question_rubrics):
                    criterion = rubric['criterion']
                    points = rubric['points']
                    question_max_score += abs(points)  # 使用绝对值来计算最大可能得分
                    
                    # 格式化单个rubric项为prompt
                    single_rubric = [rubric]
                    rubric_items = self._format_rubric_items(single_rubric)
                    
                    # 生成完整的评估prompt
                    evaluate_prompt = self._generate_prompt(question, response, rubric_items)
                    
                    # 调用API进行评分
                    # 使用配置模块安全获取API密钥
                    try:
                        api_key = config.get_api_key_safe()
                        api_base = config.get_api_base_safe()
                    except (ValueError, AttributeError):
                        # 降级处理
                        print("警告: 无法获取API配置，请检查环境变量设置")
                        api_key = os.getenv('OPENAI_API_KEY', '')
                        api_base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
                        if not api_key:
                            raise ValueError("API密钥未设置，请设置环境变量OPENAI_API_KEY")
                    
                    scorer_api = ConversationAPI(
                        model_name=self.judge_model,
                        system_prompt=self.system_prompt,
                        user_prompt=evaluate_prompt,
                        image_input=None,
                        temperature=0.3,
                        conversation_id=None,
                        model_key=api_key,
                        api_base=api_base,
                        enable_history=False
                    )
                    
                    try:
                        scorer_response = scorer_api.generate_response()
                        print(f"Rubric {rubric_idx+1} response: {scorer_response}")
                        
                        # 提取评分结果
                        rubric_score = self.extract_rubric_score(scorer_response)
                        if rubric_score is not None:
                            # 根据criteria_met结果和points计算得分
                            if rubric_score == 1.0:  # criteria met
                                question_score += points
                            # 如果criteria不满足，得分为0（对于负分项也是如此）
                            
                            print(f"Rubric {rubric_idx+1} (points: {points}): criteria_met = {rubric_score == 1.0}, score = {points if rubric_score == 1.0 else 0}")
                        else:
                            print(f"Rubric {rubric_idx+1}: 无法解析评分结果")
                    
                    except Exception as e:
                        print(f"Rubric {rubric_idx+1} API调用失败: {e}")
                        continue
                
                total_points += question_score
                total_possible_points += question_max_score
                print(f"Question {i+1} 总分: {question_score}/{question_max_score}")
        
        # 计算最终统计结果
        if valid_questions > 0 and total_possible_points > 0:
            valid_ratio = valid_questions / total_questions
            # 将得分标准化为0-100分
            normalized_score = (total_points / total_possible_points) * 100
            
            score_results = {
                "valid_ratio": valid_ratio,
                "score": normalized_score,
                "total_points": total_points,
                "total_possible_points": total_possible_points
            }
            
            print(f"评分完成: {valid_questions}/{total_questions} 个有效问题")
            print(f"总分: {total_points}/{total_possible_points} ({normalized_score:.2f}%)")
            
            self._save_score_results(score_results)
        else:
            score_results = {
                "valid_ratio": 0.0,
                "score": 0.0,
                "total_points": 0,
                "total_possible_points": 0
            }
            self._save_score_results(score_results)
        
        return score_results

if __name__ == "__main__":
    scorer = LLMJudge_scorer("MedEthicsMatrixCase", "test", "MedEthicsMatrixCase_doubao-1.5-pro-32k_202509142032")
    scorer.scoring()



            
                
                

            
                







