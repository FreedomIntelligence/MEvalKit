# MEvalKit - 多模态大语言模型评测平台

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

**MEvalKit** 是一个功能强大的多模态大语言模型评测平台，支持纯文本、多模态和LLMJudge型评测集，提供Web界面和命令行两种使用方式，让模型评测变得简单高效。

## 🌟 项目特色

- **🎯 一键评测**：仅用一句命令即可完成对已有或自定义评测集的评测
- **🔧 多种模型支持**：通过标准OpenAI接口支持GPT系列、Qwen系列等多种纯文本及多模态模型
- **📊 丰富评测集**：支持MMLU、MMStar、MT-Bench、CMB等多种评测集
- **🌐 Web界面**：提供直观的Web管理界面，支持任务创建、进度监控、结果查看
- **🐳 Docker部署**：提供完整的Docker部署方案，快速搭建评测环境
- **🔄 断点续评**：支持评测任务的中断和恢复，避免重复计算
- **📈 排行榜系统**：自动生成模型性能排行榜，便于比较分析

## 📋 目录

- [快速开始](#快速开始)
- [支持的评测集](#支持的评测集)
- [使用方式](#使用方式)
  - [命令行使用](#命令行使用)
  - [Web界面使用](#web界面使用)
- [部署指南](#部署指南)
  - [Docker快速部署](#docker快速部署)
  - [手动安装部署](#手动安装部署)
- [评测模式](#评测模式)
- [API文档](#api文档)
- [自定义评测集](#自定义评测集)
- [项目结构](#项目结构)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## 🚀 快速开始

### 使用Docker（推荐）

```bash
# 克隆项目
git clone https://github.com/FreedomIntelligence/MEvalKit.git
cd MEvalKit

# 一键部署
./deploy.sh

# 访问Web界面
# http://localhost:5000
```

### 手动安装

```bash
# 克隆项目
git clone https://github.com/FreedomIntelligence/MEvalKit.git
cd MEvalKit

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp env.example .env
# 编辑.env文件，设置API密钥等

# 启动Web服务
python app.py

# 或使用命令行评测
python run.py --dataset MMLU --model_name gpt-4o --evaluation_mode automatic
```

## 📊 支持的评测集

| 评测集名称 | 类型 | 描述 | 数据源 |
|-----------|------|------|--------|
| **MMLU** | 纯文本 | 大规模多任务语言理解评测集 | [HuggingFace](https://huggingface.co/datasets/cais/mmlu) |
| **GPQA** | 纯文本 | 研究生水平物理问答评测集 | [GitHub](https://github.com/idavidrein/gpqa) |
| **CMB** | 纯文本 | 中文医学基准评测集 | [GitHub](https://github.com/FreedomIntelligence/CMB) |
| **MMStar** | 多模态 | 多模态科学问答评测集 | [HuggingFace](https://huggingface.co/datasets/Lin-Chen/MMStar) |
| **MT-Bench** | LLMJudge | 多轮对话评测集 | [GitHub](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) |

## 💻 使用方式

### 命令行使用

#### 自动模式（推荐）

```bash
# 基本用法
python run.py --dataset MMLU --model_name gpt-4o --evaluation_mode automatic

# 完整参数示例
python run.py \
  --evaluation_mode automatic \
  --dataset MMLU \
  --model_name gpt-4o \
  --api_base "https://api.openai.com/v1" \
  --model_key "your-api-key" \
  --question_limitation 100 \
  --user_id "test_user"
```

#### 手动模式

```bash
# 使用预生成的响应文件进行评测
python run.py \
  --evaluation_mode manual \
  --dataset MMLU \
  --model_name gpt-4o \
  --response_url "https://example.com/responses.json" \
  --question_limitation 100
```

### Web界面使用

1. **访问主页**：打开浏览器访问 `http://localhost:5000`
2. **创建评测任务**：点击"开始新评测"，填写评测参数
3. **监控进度**：在任务详情页面查看实时进度
4. **查看结果**：评测完成后查看详细结果和排行榜

## 🐳 部署指南

### Docker快速部署

#### 一键部署（推荐）

```bash
# 首次部署（构建镜像并启动服务）
./deploy.sh

# 只构建镜像，不启动服务
./deploy.sh --build-only

# 只启动服务，不构建镜像
./deploy.sh --start-only

# 重新构建镜像并启动服务
./deploy.sh --rebuild
```

#### 手动Docker部署

```bash
# 构建镜像
docker build -t mevalkit:latest .

# 配置环境变量
cp env.example .env
# 编辑.env文件

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f
```

### 手动安装部署

```bash
# 1. 安装Python依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp env.example .env
# 编辑.env文件，设置必要的API密钥

# 3. 启动Web服务
python app.py

# 4. 访问应用
# Web界面: http://localhost:5000
# API文档: http://localhost:5000/apidocs/
```

## 🔄 评测模式

### 自动模式（Automatic）

- **特点**：实时调用模型API进行评测
- **适用场景**：有模型API访问权限的情况
- **优势**：实时性好，支持大规模评测
- **参数**：需要提供API密钥和接口地址

### 手动模式（Manual）

- **特点**：使用预生成的模型响应文件进行评测
- **适用场景**：模型响应已预先生成或API访问受限
- **优势**：成本低，可重复使用响应数据
- **参数**：需要提供响应数据URL

## 📚 API文档

### 主要接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/` | GET | 主页（总排行榜） |
| `/new-evaluation` | GET | 创建评测任务页面 |
| `/run-evaluation` | POST | 运行评测任务 |
| `/task-status/<task_id>` | GET | 获取任务状态 |
| `/task-detail/<task_id>` | GET | 查看任务详情 |
| `/results` | GET | 查看所有结果 |

### Swagger文档

访问 `http://localhost:5000/apidocs/` 查看完整的API文档。

## 🔧 自定义评测集

### 配置文件结构

评测集配置位于 `dataset_info/` 目录下：

- `text_dataset.json` - 纯文本评测集配置
- `image_dataset.json` - 多模态评测集配置  
- `LLMJudge_dataset.json` - LLMJudge型评测集配置

### 配置示例

```json
{
  "MyDataset": {
    "language": "zh",
    "question": {
      "loading_way": "huggingface",
      "path": "your-dataset-path",
      "subset_name": "default",
      "split_name": "test",
      "key": "question"
    },
    "choices": {
      "loading_way": "huggingface",
      "path": "your-dataset-path",
      "key": ["A", "B", "C", "D"]
    },
    "answer": {
      "loading_way": "huggingface",
      "path": "your-dataset-path",
      "key": "answer",
      "answer_type": "choice"
    }
  }
}
```

## 📁 项目结构

```
MEvalKit/
├── app.py                 # Flask Web应用主文件
├── run.py                 # 命令行评测入口
├── requirements.txt       # Python依赖
├── Dockerfile            # Docker镜像配置
├── docker-compose.yml    # Docker Compose配置
├── deploy.sh             # 一键部署脚本
├── evaluation/           # 评测核心模块
│   ├── TextMCQ_eval.py   # 文本多选题评测
│   ├── ImageMCQ_eval.py  # 图像多选题评测
│   └── LLMJudge_eval.py  # LLMJudge评测
├── src/                  # 工具模块
│   └── utils/           # 工具函数
├── dataset_info/        # 数据集配置
│   ├── text_dataset.json
│   ├── image_dataset.json
│   └── LLMJudge_dataset.json
├── templates/           # Web模板
├── results/            # 评测结果存储
├── data/              # 数据集缓存
└── logs/              # 日志文件
```

## 🤝 贡献指南

我们欢迎社区贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 贡献类型

- 🐛 Bug修复
- ✨ 新功能开发
- 📚 文档改进
- 🧪 测试用例
- 🔧 性能优化

## 📄 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可证。

## 🙏 致谢

本项目受益于以下开源项目：

- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) - 多模态评测框架
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - 大语言模型训练框架

## 📞 联系我们

- 项目主页：https://github.com/FreedomIntelligence/MEvalKit
- 问题反馈：https://github.com/FreedomIntelligence/MEvalKit/issues
- 邮箱：1481345518@qq.com

---

**MEvalKit** - 让模型评测变得简单高效！ 🚀