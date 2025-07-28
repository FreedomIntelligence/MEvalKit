# MEvalKit 项目结构说明

本文档详细介绍了MEvalKit项目的目录结构和各个文件的作用，帮助开发者快速理解项目架构。

## 📁 项目根目录

```
MEvalKit/
├── app.py                 # Flask Web应用主文件
├── run.py                 # 命令行评测入口
├── requirements.txt       # Python依赖包列表
├── Dockerfile            # Docker镜像构建文件
├── docker-compose.yml    # Docker Compose配置文件
├── deploy.sh             # 一键部署脚本
├── README.md             # 项目说明文档
└── PROJECT_STRUCTURE.md  # 项目结构说明（本文档）
```

## 🔧 核心应用文件

### `app.py` - Web应用主文件
- **功能**: Flask Web应用的核心文件
- **主要特性**:
  - 提供Web界面管理评测任务
  - 实时进度监控
  - 结果查看和排行榜
  - 自动生成API文档
- **关键路由**:
  - `/`: 主页（总排行榜）
  - `/new-evaluation`: 创建评测任务
  - `/task-status/<task_id>`: 任务状态监控
  - `/results`: 查看评测结果

### `run.py` - 命令行评测入口
- **功能**: 提供命令行界面进行模型评测
- **支持模式**:
  - `automatic`: 自动模式（实时调用API）
  - `manual`: 手动模式（使用预生成响应）
- **主要参数**:
  - `--dataset`: 数据集名称
  - `--model_name`: 模型名称
  - `--evaluation_mode`: 评测模式
  - `--question_limitation`: 问题数量限制

## 📊 评测模块 (`evaluation/`)

```
evaluation/
├── TextMCQ_eval.py       # 文本多选题评测
├── ImageMCQ_eval.py      # 图像多选题评测
└── LLMJudge_eval.py      # LLMJudge型评测
```

### `TextMCQ_eval.py` - 文本多选题评测
- **功能**: 处理纯文本多选题数据集
- **支持数据集**: MMLU, GPQA, CMB等
- **主要函数**:
  - `evaluate_mcq_automatic()`: 自动模式评测
  - `evaluate_mcq_manual()`: 手动模式评测
  - `extract_answer()`: 提取单选题答案
  - `extract_multi_answer()`: 提取多选题答案

### `ImageMCQ_eval.py` - 图像多选题评测
- **功能**: 处理多模态（图像+文本）数据集
- **支持数据集**: MMStar, TestImageMCQ等
- **主要特性**:
  - 图像预处理和编码
  - 多模态提示构建
  - 图像+文本联合评测

### `LLMJudge_eval.py` - LLMJudge型评测
- **功能**: 处理对话型评测数据集
- **支持数据集**: MT-Bench, MedEthicsMatrixCase等
- **主要特性**:
  - 多轮对话评测
  - 自动评判生成
  - 对话质量评估

## 🛠️ 工具模块 (`src/`)

```
src/
├── utils/                # 工具函数
│   ├── model_and_dataset.py  # 模型和数据集配置
│   ├── utils_loading.py      # 数据加载工具
│   ├── yaml_loader.py        # YAML配置加载器
│   ├── MCQ_constants.py      # 多选题常量定义
│   └── default_prompts.py    # 默认提示模板
├── database/            # 数据库模块
│   ├── models.py        # 数据模型定义
│   └── repository.py    # 数据访问层
├── api/                 # API接口模块
│   ├── text_api.py      # 文本API接口
│   └── image_api.py     # 图像API接口
└── dataset/             # 数据集处理模块
    ├── Text/            # 文本数据集处理
    └── Image/           # 图像数据集处理
```

### `src/utils/` - 工具函数
- **model_and_dataset.py**: 定义支持的数据集和模型分类
- **utils_loading.py**: 提供数据加载的通用工具函数
- **yaml_loader.py**: YAML配置文件的加载和解析
- **MCQ_constants.py**: 多选题相关的常量定义
- **default_prompts.py**: 默认的提示模板

### `src/database/` - 数据库模块
- **models.py**: 定义数据库表结构和模型
- **repository.py**: 提供数据访问的抽象层

### `src/api/` - API接口模块
- **text_api.py**: 文本模型的API调用接口
- **image_api.py**: 多模态模型的API调用接口

## 📋 配置文件 (`dataset_info/`)

```
dataset_info/
├── text_dataset.json         # 文本数据集配置
├── image_dataset.json        # 图像数据集配置
├── LLMJudge_dataset.json     # LLMJudge数据集配置
├── MCQ_config.yaml          # 多选题配置
├── QA_config.yaml           # 问答配置
└── README.md                # 配置说明文档
```

### 配置文件说明
- **text_dataset.json**: 定义文本数据集的加载方式和字段映射
- **image_dataset.json**: 定义多模态数据集的配置
- **LLMJudge_dataset.json**: 定义对话型数据集的配置
- **MCQ_config.yaml**: 多选题的详细配置参数
- **QA_config.yaml**: 问答任务的配置参数

## 🐳 Docker相关文件

```
├── Dockerfile              # Docker镜像构建文件
├── docker-compose.yml      # Docker Compose配置
├── docker-compose.prod.yml # 生产环境配置
├── Dockerfile.prod         # 生产环境镜像
├── .dockerignore           # Docker忽略文件
└── build_image.sh          # 镜像构建脚本
```

### Docker文件说明
- **Dockerfile**: 开发环境的镜像构建配置
- **docker-compose.yml**: 本地开发环境的容器编排
- **Dockerfile.prod**: 生产环境的镜像构建配置
- **docker-compose.prod.yml**: 生产环境的容器编排

## 📁 数据目录

```
├── data/                   # 数据集缓存目录
├── results/               # 评测结果存储
├── logs/                  # 日志文件
└── stress_test_results/   # 压力测试结果
```

### 数据目录说明
- **data/**: 存储下载和缓存的数据集文件
- **results/**: 存储评测任务的详细结果
- **logs/**: 存储应用运行日志
- **stress_test_results/**: 存储压力测试的结果数据

## 🌐 Web模板 (`templates/`)

```
templates/
├── index.html             # 主页模板
├── new_evaluation.html    # 创建评测任务页面
├── task_detail.html       # 任务详情页面
├── results.html           # 结果查看页面
└── leaderboard.html       # 排行榜页面
```

## 📄 文档文件

```
├── README.md              # 项目主要说明文档
├── PROJECT_STRUCTURE.md   # 项目结构说明（本文档）
├── README_DOCKER.md       # Docker部署说明
├── DOCKER_DEPLOYMENT.md   # Docker部署指南
├── DATABASE_MIGRATION.md  # 数据库迁移指南
├── DATABASE_SECURITY_GUIDE.md # 数据库安全指南
├── YAML_LOADING_GUIDE.md  # YAML配置加载指南
└── dotabench_usage_guide.md # DotaBench使用指南
```

## 🔧 脚本文件

```
├── deploy.sh              # 一键部署脚本
├── build_image.sh         # 镜像构建脚本
├── migrate_to_database.py # 数据库迁移脚本
├── secure_database.py     # 数据库安全模块
└── debug_api_response.py  # API响应调试工具
```

## 📊 数据库文件

```
├── mevalkit.db           # 主数据库文件
├── mevalkit_secure.db    # 加密数据库文件
├── demo_secure.db        # 演示数据库文件
└── db_config.json        # 数据库配置文件
```

## 🎯 使用建议

1. **首次使用**: 从 `README.md` 开始，了解项目基本功能
2. **快速部署**: 使用 `deploy.sh` 脚本进行一键部署
3. **自定义评测**: 参考 `dataset_info/` 目录下的配置文件
4. **问题排查**: 查看 `logs/` 目录下的日志文件
5. **API使用**: 访问 `http://localhost:5000/apidocs/` 查看API文档

## 🔄 开发流程

1. **环境准备**: 安装Python 3.8+和Docker
2. **依赖安装**: 运行 `pip install -r requirements.txt`
3. **配置设置**: 复制 `env.example` 为 `.env` 并配置
4. **启动服务**: 运行 `python app.py` 或使用Docker
5. **开始评测**: 通过Web界面或命令行进行评测

---

**注意**: 本文档会随着项目的发展而更新，请定期查看最新版本。 