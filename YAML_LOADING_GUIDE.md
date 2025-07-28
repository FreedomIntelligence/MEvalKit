# YAML配置文件加载指南

## 概述

本指南介绍如何在不改变现有代码的情况下使用YAML配置文件。项目已经提供了完整的YAML加载工具，可以无缝集成到现有系统中。

## 快速开始

### 1. 基本使用

```python
from src.utils.yaml_loader import load_yaml_config

# 加载YAML配置文件
config = load_yaml_config("dataset_info/QA_config.yaml")
print(f"配置包含 {len(config)} 个数据集")
```

### 2. 使用预定义的加载函数

```python
from src.utils.yaml_loader import load_qa_config, load_mcq_config

# 加载QA配置
qa_config = load_qa_config()

# 加载MCQ配置
mcq_config = load_mcq_config()
```

### 3. 访问配置数据

```python
# 获取特定数据集配置
mt_bench_config = qa_config.get('MT-Bench', {})

# 访问配置属性
language = mt_bench_config.get('language')
max_score = mt_bench_config.get('max_score')
question_config = mt_bench_config.get('question', {})
```

## 与现有代码集成

### 方法1：直接替换JSON加载

```python
# 原来的JSON加载方式
from src.utils.utils_loading import load_dataset_info
json_config = load_dataset_info("dataset_info/text_dataset.json")

# 新的YAML加载方式
from src.utils.yaml_loader import load_yaml_config
yaml_config = load_yaml_config("dataset_info/text_dataset.yaml")

# 两种方式返回的数据结构完全相同，可以直接替换
```

### 方法2：渐进式迁移

```python
import os
from src.utils.utils_loading import load_dataset_info
from src.utils.yaml_loader import load_yaml_config

def load_config_smart(file_path):
    """智能加载配置文件，支持JSON和YAML"""
    if file_path.endswith('.yaml') or file_path.endswith('.yml'):
        return load_yaml_config(file_path)
    else:
        return load_dataset_info(file_path)

# 使用示例
config = load_config_smart("dataset_info/QA_config.yaml")
```

## 高级功能

### 1. 配置验证

```python
from src.utils.yaml_loader import load_yaml_config_with_validation

# 加载并验证配置
config = load_yaml_config_with_validation(
    "dataset_info/QA_config.yaml",
    "dataset_info/QA_config_schema.yaml"
)
```

### 2. 保存配置

```python
from src.utils.yaml_loader import save_yaml_config

# 保存配置到YAML文件
config = {"dataset": {"language": "zh", "max_score": 100}}
success = save_yaml_config(config, "output_config.yaml")
```

### 3. JSON转YAML

```python
from src.utils.yaml_loader import convert_json_to_yaml

# 将JSON配置文件转换为YAML
success = convert_json_to_yaml(
    "dataset_info/text_dataset.json",
    "dataset_info/text_dataset.yaml"
)
```

## 实际应用示例

### 示例1：在现有评估代码中使用YAML

```python
# 在 evaluation/TextMCQ_eval.py 中
from src.utils.yaml_loader import load_yaml_config

def evaluate_mcq_automatic(user_id, dataset_name, model_name, ...):
    # 加载YAML配置而不是JSON
    dataset_info = load_yaml_config("dataset_info/MCQ_config.yaml")
    
    # 其余代码保持不变
    dataset_config = dataset_info.get(dataset_name, {})
    # ... 继续处理
```

### 示例2：在Web应用中加载YAML配置

```python
# 在 app.py 中
from src.utils.yaml_loader import load_yaml_config

@app.route('/api/datasets')
def get_datasets():
    # 加载YAML配置
    qa_config = load_yaml_config("dataset_info/QA_config.yaml")
    mcq_config = load_yaml_config("dataset_info/MCQ_config.yaml")
    
    # 返回数据集列表
    datasets = {
        'qa': list(qa_config.keys()),
        'mcq': list(mcq_config.keys())
    }
    return jsonify(datasets)
```

## 配置文件格式

### YAML格式优势

1. **可读性更好**：YAML格式更接近自然语言
2. **支持注释**：可以在配置文件中添加说明
3. **多行文本**：支持复杂的多行文本内容
4. **层次结构清晰**：缩进表示层次关系

### 示例配置

```yaml
MT-Bench:
  language: en
  max_score: 10
  background: null
  case: null
  
  question:
    loading_way: json
    path:
      - data/mt-bench/question.jsonl
    key: turns
    prompt_template: "Question: {question}"
  
  reference_answer:
    loading_way: json
    path:
      - data/mt-bench/question.jsonl
    key: reference
    prompt_template: "Reference Answer: {reference_answer}"
```

## 错误处理

### 常见错误及解决方案

1. **文件不存在**
```python
try:
    config = load_yaml_config("nonexistent.yaml")
except FileNotFoundError as e:
    print(f"配置文件不存在: {e}")
```

2. **YAML格式错误**
```python
try:
    config = load_yaml_config("malformed.yaml")
except yaml.YAMLError as e:
    print(f"YAML格式错误: {e}")
```

3. **编码问题**
```python
# 确保使用UTF-8编码
with open("config.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
```

## 性能考虑

### 缓存配置

```python
import functools

@functools.lru_cache(maxsize=10)
def load_cached_config(file_path):
    """缓存配置加载结果"""
    return load_yaml_config(file_path)

# 使用缓存加载
config = load_cached_config("dataset_info/QA_config.yaml")
```

### 异步加载

```python
import asyncio
import aiofiles
import yaml

async def load_yaml_config_async(file_path):
    """异步加载YAML配置"""
    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
        content = await f.read()
        return yaml.safe_load(content)

# 使用异步加载
config = await load_yaml_config_async("dataset_info/QA_config.yaml")
```

## 最佳实践

1. **使用相对路径**：配置文件路径使用相对路径，便于部署
2. **添加注释**：在YAML文件中添加有意义的注释
3. **验证配置**：使用schema验证配置文件格式
4. **错误处理**：始终包含适当的错误处理代码
5. **版本控制**：将配置文件纳入版本控制

## 总结

通过使用提供的YAML加载工具，您可以：

- ✅ 在不改变现有代码的情况下使用YAML配置文件
- ✅ 保持与JSON配置的完全兼容性
- ✅ 享受YAML格式的可读性和灵活性
- ✅ 逐步迁移现有配置而不影响系统运行

开始使用YAML配置文件，让您的配置管理更加清晰和高效！ 