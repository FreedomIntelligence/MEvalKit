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

# MCQ_JSON_TEMPLATE_SINGLE_ZH = """
# 你是一个回答中文选择题的AI助手。该选择题只有一个正确选项。
# 请仔细阅读题目，分析各选项，然后以JSON格式回答。

# 你的回答必须是一个JSON对象，包含以下一个字段：
# - "answer": 你选择的正确选项字母（只能是A、B、C、D中的一个）


# 示例：
# 问题：中国的首都是哪里？
# A. 上海
# B. 北京  
# C. 广州
# D. 深圳

# 回答：
# ```json
# {
#   "answer": "B"
# }
# ```
# """.strip()

# MCQ_JSON_TEMPLATE_SINGLE_EN = """
# You are a helpful assistant that can answer multiple choice questions. This question has only one correct answer.
# Please read the question carefully, analyze each option, and respond in JSON format.

# Your answer must be a JSON object containing the following one field:
# - "answer": The letter of the correct option you choose (must be one of A, B, C, D)  

# Example:
# Question: What is the capital of France?
# A. London
# B. Berlin
# C. Paris
# D. Madrid

# Answer:
# ```json
# {
#   "answer": "C"
# }
# ```
# """.strip()

# MCQ_JSON_TEMPLATE_MULTIPLE_ZH = """
# 你是一个回答中文选择题的AI助手。该选择题有多个正确选项。
# 请仔细阅读题目，分析各选项，然后以JSON格式回答。

# 你的回答必须是一个JSON对象，包含以下一个字段：
# - "answer": 你选择的正确选项字母列表（如["A", "B"]或["A", "C", "D"]）

# 示例：
# 问题：以下哪些是中国的直辖市？
# A. 北京
# B. 上海
# C. 广州
# D. 天津
# E. 重庆

# 回答：
# ```json
# {
#   "answer": ["A", "B", "D", "E"]
# }
# ```
# """.strip()

# MCQ_JSON_TEMPLATE_MULTIPLE_EN = """
# You are a helpful assistant that can answer multiple choice questions. This question has multiple correct answers.
# Please read the question carefully, analyze each option, and respond in JSON format.

# Your answer must be a JSON object containing the following one field:
# - "answer": A list of letters for the correct options you choose (e.g., ["A", "B"] or ["A", "C", "D"])

# Example:
# Question: Which of the following are programming languages?
# A. Python
# B. JavaScript
# C. HTML
# D. Java
# E. CSS

# Answer:
# ```json
# {
#   "answer": ["A", "B", "D"]
# }
# ```
# """.strip()



MCQ_JSON_TEMPLATE_SINGLE_ZH = """
你是一个回答中文选择题的AI助手。该选择题只有一个正确选项。
请仔细阅读题目，分析各选项，然后以JSON格式回答。

你的回答必须是一个JSON对象，包含以下两个字段：
- "answer": 你选择的正确选项字母（只能是A、B、C、D中的一个）
- "reasoning": 你选择这个答案的详细理由和分析过程

示例：
问题：中国的首都是哪里？
A. 上海
B. 北京  
C. 广州
D. 深圳

回答：
```json
{
  "answer": "B",
  "reasoning": "中国的首都是北京。北京是中华人民共和国的政治中心，也是全国的首都城市。上海是经济中心，广州和深圳是重要的经济特区，但都不是首都。"
}
```
""".strip()

MCQ_JSON_TEMPLATE_SINGLE_EN = """
You are a helpful assistant that can answer multiple choice questions. This question has only one correct answer.
Please read the question carefully, analyze each option, and respond in JSON format.

Your answer must be a JSON object containing the following two fields:
- "answer": The letter of the correct option you choose (must be one of A, B, C, D)  
- "reasoning": Your detailed reasoning and analysis process for choosing this answer

Example:
Question: What is the capital of France?
A. London
B. Berlin
C. Paris
D. Madrid

Answer:
```json
{
  "answer": "C",
  "reasoning": "The capital of France is Paris. Paris has been the capital and largest city of France since the 12th century. London is the capital of the United Kingdom, Berlin is the capital of Germany, and Madrid is the capital of Spain."
}
```
""".strip()

MCQ_JSON_TEMPLATE_MULTIPLE_ZH = """
你是一个回答中文选择题的AI助手。该选择题有多个正确选项。
请仔细阅读题目，分析各选项，然后以JSON格式回答。

你的回答必须是一个JSON对象，包含以下两个字段：
- "answer": 你选择的正确选项字母列表（如["A", "B"]或["A", "C", "D"]）
- "reasoning": 你选择这些答案的详细理由和分析过程

示例：
问题：以下哪些是中国的直辖市？
A. 北京
B. 上海
C. 广州
D. 天津
E. 重庆

回答：
```json
{
  "answer": ["A", "B", "D", "E"],
  "reasoning": "中国目前有四个直辖市：北京、上海、天津和重庆。北京是首都和政治中心，上海是经济金融中心，天津是重要的港口城市，重庆是西南地区的重要城市。广州虽然是重要的经济城市，但它是广东省的省会城市，不是直辖市。"
}
```
""".strip()

MCQ_JSON_TEMPLATE_MULTIPLE_EN = """
You are a helpful assistant that can answer multiple choice questions. This question has multiple correct answers.
Please read the question carefully, analyze each option, and respond in JSON format.

Your answer must be a JSON object containing the following two fields:
- "answer": A list of letters for the correct options you choose (e.g., ["A", "B"] or ["A", "C", "D"])
- "reasoning": Your detailed reasoning and analysis process for choosing these answers

Example:
Question: Which of the following are programming languages?
A. Python
B. JavaScript
C. HTML
D. Java
E. CSS

Answer:
```json
{
  "answer": ["A", "B", "D"],
  "reasoning": "Python, JavaScript, and Java are all programming languages. Python is a high-level programming language used for various applications. JavaScript is primarily used for web development. Java is an object-oriented programming language. HTML and CSS, while essential for web development, are markup and styling languages respectively, not programming languages."
}
```
""".strip()

RUBRIC_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()