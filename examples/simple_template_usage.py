"""
简单的模板处理示例
展示如何在MCQ_eval.py中统一处理不同占位符
"""

def render_template(template, data):
    """简单的模板渲染，支持不同占位符"""
    if not template:
        return ""
    
    # 数据标准化处理
    normalized_data = {}
    
    # 处理choices字段的不同格式
    if 'choices' in data and data['choices']:
        choices = data['choices']
        if isinstance(choices, list):
            normalized_data['choices'] = choices
            normalized_data['choice_sentence'] = ', '.join(choices)
        elif isinstance(choices, str):
            if ',' in choices:
                normalized_data['choices'] = [c.strip() for c in choices.split(',')]
                normalized_data['choice_sentence'] = choices
            else:
                normalized_data['choices'] = [choices]
                normalized_data['choice_sentence'] = choices
    
    # 处理其他字段
    for key, value in data.items():
        if key not in normalized_data:
            normalized_data[key] = value
    
    # 占位符映射
    placeholder_mappings = {
        'choice_sentence': 'choice_sentence',
        'choices': 'choices',
        'question': 'question',
        'case': 'case',
        'hint': 'hint',
        'background': 'background',
        'reference': 'reference_answer',
        'reference_answer': 'reference_answer',
        'image_path': 'image',
        'image': 'image'
    }
    
    # 渲染模板
    result = template
    for old_placeholder, new_placeholder in placeholder_mappings.items():
        if f"{{{{ {old_placeholder} }}}}" in template:
            value = normalized_data.get(new_placeholder, '')
            if isinstance(value, list):
                # 如果是列表，转换为格式化字符串
                if old_placeholder == 'choices':
                    formatted_choices = []
                    for i, choice in enumerate(value):
                        formatted_choices.append(f"{chr(65 + i)}. {choice}")
                    value = '\n'.join(formatted_choices)
                else:
                    value = ', '.join(str(v) for v in value)
            result = result.replace(f"{{{{ {old_placeholder} }}}}", str(value))
    
    return result

def example_different_templates():
    """展示如何处理不同的模板"""
    print("=== 不同模板处理示例 ===")
    
    # 不同的模板配置
    templates = [
        # MCQ类型 - 使用choice_sentence
        {
            'name': 'MCQ with choice_sentence',
            'template': "Question: {{ question }}\nThe choices are: {{ choice_sentence }}",
            'data': {
                'question': 'What is the capital of France?',
                'choices': ['Paris', 'London', 'Berlin']
            }
        },
        # MCQ类型 - 使用choices列表
        {
            'name': 'MCQ with choices list',
            'template': "Question: {{ question }}\nChoices: {{ choices }}",
            'data': {
                'question': 'What is the capital of France?',
                'choices': ['Paris', 'London', 'Berlin']
            }
        },
        # MCQ类型 - 使用字符串choices
        {
            'name': 'MCQ with string choices',
            'template': "Question: {{ question }}\nChoices: {{ choice_sentence }}",
            'data': {
                'question': 'What is the capital of France?',
                'choices': 'Paris, London, Berlin'
            }
        },
        # 包含case和hint的模板
        {
            'name': 'MCQ with case and hint',
            'template': "Case: {{ case }}\nQuestion: {{ question }}\nChoices: {{ choice_sentence }}\nHint: {{ hint }}",
            'data': {
                'case': 'Geography quiz',
                'question': 'What is the capital of France?',
                'choices': ['Paris', 'London', 'Berlin'],
                'hint': 'Think about the Eiffel Tower'
            }
        },
        # QA类型 - 使用reference
        {
            'name': 'QA with reference',
            'template': "Question: {{ question }}\nReference: {{ reference }}",
            'data': {
                'question': 'Explain machine learning.',
                'reference': 'Machine learning is a subset of AI...'
            }
        },
        # QA类型 - 使用reference_answer
        {
            'name': 'QA with reference_answer',
            'template': "Question: {{ question }}\nReference Answer: {{ reference_answer }}",
            'data': {
                'question': 'Explain machine learning.',
                'reference_answer': 'Machine learning is a subset of AI...'
            }
        }
    ]
    
    for example in templates:
        print(f"\n--- {example['name']} ---")
        print(f"模板: {example['template']}")
        print(f"数据: {example['data']}")
        
        # 渲染模板
        result = render_template(example['template'], example['data'])
        print(f"渲染结果:\n{result}")

def example_data_normalization():
    """展示数据标准化过程"""
    print("\n=== 数据标准化过程示例 ===")
    
    # 不同格式的原始数据
    raw_data_examples = [
        {
            'name': 'List choices',
            'data': {
                'question': 'What is the capital of France?',
                'choices': ['Paris', 'London', 'Berlin']
            }
        },
        {
            'name': 'String choices',
            'data': {
                'question': 'What is the capital of France?',
                'choices': 'Paris, London, Berlin'
            }
        },
        {
            'name': 'Single choice',
            'data': {
                'question': 'What is the capital of France?',
                'choices': 'Paris'
            }
        }
    ]
    
    for example in raw_data_examples:
        print(f"\n--- {example['name']} ---")
        print(f"原始数据: {example['data']}")
        
        # 模拟标准化过程
        normalized_data = {}
        choices = example['data']['choices']
        
        if isinstance(choices, list):
            normalized_data['choices'] = choices
            normalized_data['choice_sentence'] = ', '.join(choices)
        elif isinstance(choices, str):
            if ',' in choices:
                normalized_data['choices'] = [c.strip() for c in choices.split(',')]
                normalized_data['choice_sentence'] = choices
            else:
                normalized_data['choices'] = [choices]
                normalized_data['choice_sentence'] = choices
        
        print(f"标准化后: {normalized_data}")

def example_placeholder_mapping():
    """展示占位符映射"""
    print("\n=== 占位符映射示例 ===")
    
    # 占位符映射表
    mappings = {
        'choice_sentence': 'choice_sentence',
        'choices': 'choices',
        'question': 'question',
        'case': 'case',
        'hint': 'hint',
        'background': 'background',
        'reference': 'reference_answer',
        'reference_answer': 'reference_answer',
        'image_path': 'image',
        'image': 'image'
    }
    
    print("支持的占位符映射:")
    for old_placeholder, new_placeholder in mappings.items():
        print(f"  {old_placeholder} -> {new_placeholder}")
    
    # 测试不同占位符的模板
    test_templates = [
        "Question: {{ question }}",
        "Choices: {{ choice_sentence }}",
        "Choices: {{ choices }}",
        "Case: {{ case }}",
        "Hint: {{ hint }}",
        "Reference: {{ reference }}",
        "Reference Answer: {{ reference_answer }}"
    ]
    
    test_data = {
        'question': 'What is the capital of France?',
        'choices': ['Paris', 'London', 'Berlin'],
        'case': 'Geography quiz',
        'hint': 'Think about the Eiffel Tower',
        'reference': 'Paris is the capital of France.',
        'reference_answer': 'Paris is the capital of France.'
    }
    
    print(f"\n测试数据: {test_data}")
    
    for template in test_templates:
        result = render_template(template, test_data)
        print(f"\n模板: {template}")
        print(f"结果: {result}")

if __name__ == "__main__":
    example_different_templates()
    example_data_normalization()
    example_placeholder_mapping() 