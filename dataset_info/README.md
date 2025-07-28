# 大模型评测集Schema体系

## 概述

本schema体系参考OpenAPI Specification的设计理念，为大模型评测数据集提供标准化的配置规范。该体系具有以下特点：

- **结构化描述**：使用JSON Schema定义清晰的数据结构
- **类型安全**：提供完整的类型定义和约束条件
- **可扩展性**：支持自定义扩展和引用
- **工具友好**：支持验证、文档生成和代码生成

## 核心组件

### 1. LoadingConfig（数据加载配置）
定义如何从不同格式的数据文件中加载数据：

```json
{
  "loading_way": "json|csv|parquet|jsonl",
  "path": "文件路径或路径列表",
  "key": "字段键名或键名列表",
  "sub_key": "子键名（可选）"
}
```

### 2. QuestionConfig（问题配置）
继承LoadingConfig，增加问题类型字段：

```json
{
  "question_type_key": "问题类型字段名"
}
```

### 3. AnswerConfig（答案配置）
继承LoadingConfig，增加答案类型：

```json
{
  "answer_type": "choice|content|code|multichoice"
}
```

### 4. DatasetConfig（数据集配置）
完整的数据集配置结构：

```json
{
  "language": "zh|en|multilingual",
  "max_score": 100,
  "background": "背景信息配置",
  "case": "案例信息配置",
  "question": "问题配置",
  "answer": "答案配置",
  "model_response": "模型响应配置",
  "choices": "选项配置",
  "hint": "提示信息配置",
  "metadata": "元数据信息"
}
```

## 新增特性

相比原始配置，新schema体系增加了以下特性：

### 1. 元数据管理
- **版本控制**：数据集版本信息
- **作者信息**：数据集创建者
- **许可证**：使用许可条款
- **标签系统**：便于分类和搜索
- **难度等级**：问题难度标识
- **学科领域**：专业分类

### 2. 增强的数据加载配置
- **多文件支持**：支持从多个文件加载数据
- **灵活的子键配置**：支持复杂的嵌套数据结构
- **统一的加载接口**：标准化不同格式的数据加载

### 3. 模型响应配置
- **输出格式**：定义模型响应的格式要求
- **长度限制**：设置最大token数
- **质量要求**：定义响应质量标准

### 4. 背景和案例信息
- **背景描述**：提供数据集背景信息
- **案例场景**：支持复杂的案例描述
- **上下文信息**：增强问题理解

## 使用示例

### 基本使用

#### JSON格式
```json
{
  "$schema": "./evaluation_schema.json",
  "DatasetName": {
    "language": "zh",
    "max_score": 100,
    "question": {
      "loading_way": "json",
      "path": ["data/dataset.json"],
      "key": "question"
    },
    "answer": {
      "loading_way": "json", 
      "path": ["data/dataset.json"],
      "key": "answer",
      "answer_type": "choice"
    }
  }
}
```

#### YAML格式
```yaml
$schema: ./evaluation_schema.yaml

DatasetName:
  language: zh
  max_score: 100
  question:
    loading_way: json
    path: [data/dataset.json]
    key: question
  answer:
    loading_way: json
    path: [data/dataset.json]
    key: answer
    answer_type: choice
```

### 高级配置

#### JSON格式
```json
{
  "$schema": "./evaluation_schema.json",
  "AdvancedDataset": {
    "language": "en",
    "max_score": 100,
    "background": {
      "description": "Advanced dataset description",
      "domain": "academic_research"
    },
    "question": {
      "loading_way": "csv",
      "path": ["data/questions.csv"],
      "key": "Question",
      "question_type_key": "Category"
    },
    "answer": {
      "loading_way": "csv",
      "path": ["data/answers.csv"],
      "key": "Correct_Answer",
      "answer_type": "content"
    },
    "model_response": {
      "format": "text",
      "max_tokens": 1500
    },
    "metadata": {
      "version": "2.0",
      "description": "Advanced evaluation dataset",
      "author": "Research Team",
      "license": "MIT",
      "tags": ["advanced", "research", "comprehensive"],
      "difficulty_levels": ["intermediate", "advanced"],
      "subject_areas": ["computer_science", "mathematics"]
    }
  }
}
```

#### YAML格式
```yaml
$schema: ./evaluation_schema.yaml

AdvancedDataset:
  language: en
  max_score: 100
  background:
    description: Advanced dataset description
    domain: academic_research
  question:
    loading_way: csv
    path: [data/questions.csv]
    key: Question
    question_type_key: Category
  answer:
    loading_way: csv
    path: [data/answers.csv]
    key: Correct_Answer
    answer_type: content
  model_response:
    format: text
    max_tokens: 1500
  metadata:
    version: "2.0"
    description: Advanced evaluation dataset
    author: Research Team
    license: MIT
    tags: [advanced, research, comprehensive]
    difficulty_levels: [intermediate, advanced]
    subject_areas: [computer_science, mathematics]
```

## 验证和工具

### Schema验证
使用JSON Schema验证器验证配置文件：

```bash
# 验证JSON格式配置文件
jsonschema -i text_dataset_schema_example.json evaluation_schema.json

# 验证YAML格式配置文件
jsonschema -i text_dataset_schema_example.yaml evaluation_schema.yaml

# 使用Python验证
python -c "
import yaml
import json
from jsonschema import validate

# 加载YAML文件
with open('text_dataset_schema_example.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 加载JSON Schema
with open('evaluation_schema.json', 'r') as f:
    schema = json.load(f)

# 验证
validate(instance=config, schema=schema)
print('验证通过！')
"
```

### 代码生成
基于schema可以生成：
- 数据加载器代码
- 验证函数
- 文档模板
- 测试用例

### IDE支持
在支持JSON Schema的IDE中可以获得：
- 自动补全
- 类型检查
- 错误提示
- 文档提示

## 扩展指南

### 添加新的数据类型
1. 在`LoadingConfig`的`loading_way`枚举中添加新格式
2. 实现对应的数据加载器
3. 更新文档和示例

### 添加新的答案类型
1. 在`AnswerConfig`的`answer_type`枚举中添加新类型
2. 实现对应的评估逻辑
3. 更新验证规则

### 自定义元数据字段
在`metadata`对象中添加自定义字段，确保不影响核心功能。

## 最佳实践

1. **版本管理**：为每个数据集配置明确的版本号
2. **文档完整**：提供详细的数据集描述和使用说明
3. **标签规范**：使用统一的标签体系进行分类
4. **测试覆盖**：为配置创建完整的测试用例
5. **向后兼容**：保持配置格式的向后兼容性

## 文件结构

```
dataset_info/
├── evaluation_schema.json          # JSON格式Schema定义文件
├── evaluation_schema.yaml          # YAML格式Schema定义文件
├── text_dataset.json               # 原始配置文件
├── text_dataset_schema_example.json # JSON格式新schema示例
├── text_dataset_schema_example.yaml # YAML格式新schema示例
└── README.md                       # 本文档
```

## 格式选择

本schema体系同时支持JSON和YAML两种格式：

### JSON格式优势
- 广泛支持，几乎所有编程语言都有原生支持
- 严格的语法，减少歧义
- 适合机器处理和API交互

### YAML格式优势
- 更简洁易读，减少冗余字符
- 支持注释，便于文档化
- 层次结构更清晰
- 适合配置文件场景

### 使用建议
- **开发阶段**：推荐使用YAML格式，便于快速编辑和调试
- **生产环境**：可以选择JSON格式，确保最大兼容性
- **工具集成**：根据具体工具的支持情况选择合适格式 