DEFAULT_GENERATE_SYSTEM_PROMPT = """
You are a helpful assistant that can give helpful, detailed and polite responses to the instructions.
"""

DEFAULT_JUDGE_SYSTEM_PROMPT = """
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below.
Your evaluation should consider factors such as helpfulness, accuracy, depth, creativity and level of detail of the response.
Begin your evaluation by providing a short explanation. Be as objective as possible.
After providing your explanation, you must rate the response on a scale of 1 to 10, by strictly following the format "Rating: X/10".
"""

DEFAULT_JUDGE_SYSTEM_PROMPT_REASONING = """
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below.
Your evaluation should consider correctness and helpfulness of the response.
You will be given a reference answer and the assistant's response.
Begin your evaluation by comparing both of the answers. Identify and correct any mistakes. Be as objective as possible.
After providing your explanation, you must rate the response on a scale of 1 to 10, by strictly following the format "Rating: X/10".
"""

MCQ_TEMPLATE_SINGLE_ZH = """
你现在是一个回答中文选择题的AI助手。该选择题只有一个正确选项。
你的回答中只能有一个选项，且只能包含正确选项的字母。
"""

MCQ_TEMPLATE_SINGLE_EN = """
You are a helpful assistant that can answer multiple choice questions. This question has only one correct answer.
Your answer must only contain the letter of the correct answer.
"""

MCQ_TEMPLATE_MULTIPLE_ZH = """
你现在是一个回答中文选择题的AI助手。该选择题有多个正确选项。
你的回答中必须包含多个选项，且只能包含正确选项的字母。
"""

MCQ_TEMPLATE_MULTIPLE_EN = """
You are a helpful assistant that can answer multiple choice questions. This question has multiple correct answers.
Your answer must only contain the letters of the correct answers.
"""