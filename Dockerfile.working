# MEvalKit Docker镜像构建文件
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# 安装系统依赖包，OpenCV需要这些库
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件并安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码到容器
COPY . .

# 创建必要的目录结构
RUN mkdir -p /app/results /app/logs /app/stress_test_results /app/data

# 设置run.py为可执行文件
RUN chmod +x run.py

# 暴露Web服务端口
EXPOSE 1984

# 启动命令：运行Flask应用
CMD ["python", "app.py"]