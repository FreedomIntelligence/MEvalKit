# MEvalKit Docker镜像构建文件
# 
# 该文件定义了MEvalKit的Docker镜像构建过程，包括：
# - 基础镜像选择
# - 系统依赖安装
# - Python环境配置
# - 应用代码部署
# - 健康检查配置
#
# 作者: MEvalKit Team
# 版本: 1.0.0

# 使用Python 3.11作为基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
# 禁用Python输出缓冲
ENV PYTHONUNBUFFERED=1
# 不生成.pyc文件
ENV PYTHONDONTWRITEBYTECODE=1
# 指定Flask应用入口
ENV FLASK_APP=app.py
# 设置Flask环境为生产模式
ENV FLASK_ENV=production

# 使用国内镜像源并安装系统依赖
# 安装编译工具、图像处理库等必要的系统依赖
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件并安装Python依赖
# 使用清华大学镜像源加速下载
COPY requirements.txt .
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn --no-cache-dir -r requirements.txt

# 复制应用代码到容器
COPY . .

# 创建必要的目录结构
# 这些目录用于存储评测结果、日志、压力测试结果和数据缓存
RUN mkdir -p /app/results /app/logs /app/stress_test_results /app/data

# 设置run.py为可执行文件
RUN chmod +x run.py

# 暴露Web服务端口
EXPOSE 1984

# 健康检查配置
# 每30秒检查一次，超时10秒，启动后等待5秒开始检查，失败3次后认为不健康
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:1984/ || exit 1

# 启动命令：运行Flask应用
CMD ["python", "app.py"] 