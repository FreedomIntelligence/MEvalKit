**使用一句命令即可完成对已有的或自定义的评测集的评测！**

## 目录

- [项目特色](#项目特色)
- [版本](#版本)
- [支持的评测集](#支持的评测集)
- [评测方法](#评测方法)
    - [结果生成](#结果生成)
    - [从头评估](#从头评估)
    - [断点评估](#断点评估)
- [使用](#使用)
    - [安装](#安装)
    - [命令](#命令)
- [评测集集成](#评测集集成)

## 项目特色

- **运行简便**：仅用一句命令即可完成评测。
- **多种模型**：通过标准OpenAI接口即可支持GPT系列、Qwen系列等多种纯文本及多模态模型。
- **多种评测集**：支持多种评测集，如纯文本评测集MMLU、多模态评测集MMStar、LLMJudge型评测集
MT-Bench等。

## 版本

- **[2025-04-23]** v1.0初始版本，支持MMLU、MMStar、MT-Bench、CMB等评测集

## 支持的评测集
| 评测集名称       | 评测集类型         | 评测集路径           |
| --------------- | ----------------- | ----------------- |
| MMLU            | 纯文本评测集       | https://huggingface.co/datasets/cais/mmlu |
| GPQA            | 纯文本评测集       | https://github.com/idavidrein/gpqa                                         |
| CMB             | 纯文本评测集       | https://github.com/FreedomIntelligence/CMB                                         |
| MMStar          | 多模态评测集       | https://huggingface.co/datasets/Lin-Chen/MMStar |
| MT-Bench        | LLMJudge型评测集   | https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge                                         |

## 评测模式与结果生成

分为两种模式：[从头评估](#从头评估)和[断点评估](#断点评估)

### 结果生成

选择“从头评估”以后，会生成一个“评测集名称_模型名称_result.json”文件，记录每一个问题的模型response。

在一轮评估中，如果95%的题目都已经生成了结果，则会生成一个“评测集名称_模型名称_score_result.json”文件，记录本轮评估最后的分数。

### 从头评估

从头开始一轮评测。会重置评测的结果文件，将所有题目的模型response都设为null。

### 断点评估

进行一次断点评估时，会读取result.json文件中每一个问题是否已经生成response。若生成，则跳过该问题的评估。

## 使用

### 安装

```bash
git clone git@github.com:FreedomIntelligence/MEvalKit.git
cd MEvalKit
pip install -e .
pip install -r requirements.txt
```

### 命令

```bash
python run.py --dataset MMLU --model gpt-3.5-turbo --judgment_model None --workers 64 --evaluate_mode start
```
--dataset：在dataset_info.json中已经配置好的评测集名称。

--model：标准OpenAI接口支持的模型。

--workers：并行处理的工作线程数量。

--evaluate_mode：评测模式，可选值为start（从头开始）和resume（断点继续）。

--judgment_model：LLMJudge评测集中用于评判的模型。如果不是LLMJudge评测集，则指定为None。

## 评测集集成

所有评测集的配置文件都在dataset_info文件夹中。如果您要集成您自定义的评测集，请在不同类型评测集的json文件中进行配置（纯文本评测集放在text_dataset.json中，多模态评测集放在image_dataset.json中，LLMJudge型评测集放在llmjudge_dataset.json中）。

以下的各部分（choices, answer, hint, image等），如果不被您的评测集所包含的话，请用{}进行配置。

如果各部分的子内容无需指定的话，用""表示。

```json
"数据集名称": {
        "language": "评测集使用的语言",
        "question": {
            "loading_way": "评测集问题部分的加载方式，目前支持huggingface、csv、json",
            "path": "评测集问题部分的加载路径",
            "subset_name": "如果是huggingface加载方式，还需指定子集",
            "split_name": "如果是huggingface加载方式，还需指定划分",
            "key": "数据集选项部分的表头名称",
            "question_type_key": "如果是混合单选、多选的选择题型的评测集，还需指定评测集问题类型的表头名称"
        },
        "choices": {
            "loading_way": "评测集选项部分的加载方式，目前支持huggingface、csv、json",
            "path": "评测集选项部分的加载路径",
            "subset_name": "如果是huggingface加载方式，还需指定子集",
            "split_name": "如果是huggingface加载方式，还需指定划分",
            "key": "评测集选项部分的表头名称。如果你的选项分散在多列中，请用list的形式指定，如['Option1', 'Option2', 'Option3', 'Option4']",
            "sub_key": "如果选项是以'选项名称'：'选项内容'的形式呈现的，请用list的形式指定所有'选项名称'"
        },
        "answer": {
            "loading_way": "评测集答案部分的加载方式，目前支持huggingface、csv、json",
            "path": "评测集答案部分的加载路径",
            "subset_name": "如果是huggingface加载方式，还需指定子集",
            "split_name": "如果是huggingface加载方式，还需指定划分",
            "key": "评测集答案部分的表头名称",
            "answer_type": "评测集答案的类型，'choice'表示答案是以选项（即ABCD）的形式呈现的，'content'表示答案是以选项内容的形式呈现的"
        },
        "hint": {
            "loading_way": "评测集提示部分的加载方式，目前支持huggingface、csv、json",
            "path": "评测集提示部分的加载路径",
            "subset_name": "如果是huggingface加载方式，还需指定子集",
            "split_name": "如果是huggingface加载方式，还需指定划分",
            "key": "评测集提示部分的表头名称"
        },
        "image": {
            "loading_way": "评测集图片部分的加载方式，目前支持huggingface、csv、json",
            "path": "评测集图片部分的加载路径",
            "subset_name": "如果是huggingface加载方式，还需指定子集",
            "split_name": "如果是huggingface加载方式，还需指定划分",
            "key": "评测集图片部分的表头名称"
        }
    }
```

## 协议

本仓库的代码依照[Apache License 2.0](LICENSE)协议开源。

## 致谢

本项目受益于[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)与[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory),感谢以上作者的付出！