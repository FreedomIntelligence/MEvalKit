DEFAULT_GENERATE_SYSTEM_PROMPT_EN = """
You are a reliable assistant that can answer questions under the circumstance of the task.
You will be given a question, and you need to answer the question correctly, politely and in detail.
You may also be given the background of the task and the case of every single question for help.
"""

DEFAULT_GENERATE_SYSTEM_PROMPT_ZH = """
你是一个可靠的AI助手，可以在任务的特定情景下回答问题。
你将获取一个问题，并需要正确、礼貌且详细地回答问题。
作为帮助，你可能会获取任务的背景和每个问题的案例。
"""

DEFAULT_JUDGE_SYSTEM_PROMPT_WITH_GIVEN_ZH = """
请根据以下评分标准对模型回答进行评分：
"""

DEFAULT_JUDGE_SYSTEM_PROMPT_WITH_GIVEN_EN = """
Please evaluate the following model response based on the scoring criteria provided.
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