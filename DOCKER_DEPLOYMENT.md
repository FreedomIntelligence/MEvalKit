# MEvalKit Docker 部署指南

本文档详细介绍了如何在 rootless Docker 环境中部署 MEvalKit 应用。

## 目录

- [环境要求](#环境要求)
- [快速部署](#快速部署)
- [手动部署](#手动部署)
- [配置说明](#配置说明)
- [数据持久化](#数据持久化)
- [故障排除](#故障排除)
- [开发模式](#开发模式)

## 环境要求

### 系统要求
- Linux 系统（推荐 Ubuntu 20.04+ 或 CentOS 8+）
- Docker 20.10+ 
- Docker Compose 2.0+（可选，用于编排）

### Rootless Docker 环境
本项目已针对 rootless Docker 环境进行了优化。确认您的 Docker 环境：

```bash
# 检查 Docker 版本
docker --version

# 检查是否为 rootless 模式
docker info | grep "Docker Root Dir"
# 输出应该类似：Docker Root Dir: /home/username/.local/share/docker
```

## 快速部署

### 一键部署（推荐）

我们提供了便捷的一键部署脚本：

```bash
# 首次部署（构建镜像并启动服务）
./deploy.sh

# 只构建镜像，不启动服务
./deploy.sh --build-only

# 只启动服务，不构建镜像（需要镜像已存在）
./deploy.sh --start-only

# 重新构建镜像并启动服务
./deploy.sh --rebuild

# 停止服务
./deploy.sh --stop

# 重启服务
./deploy.sh --restart

# 查看服务日志
./deploy.sh --logs

# 查看服务状态
./deploy.sh --status

# 清理资源
./deploy.sh --clean
```

### 使用 Makefile

项目也支持使用 Makefile 进行管理：

```bash
# 构建镜像
make build

# 启动服务
make up

# 停止服务
make down

# 查看日志
make logs

# 进入容器
make shell

# 查看状态
make status
```

## 手动部署

如果您更喜欢手动控制部署过程：

### 1. 准备环境文件

```bash
# 复制环境配置文件
cp env.example .env

# 编辑配置文件，设置您的 API 密钥
nano .env
```

### 2. 构建镜像

```bash
# 使用构建脚本
cd docker && ./build_image.sh

# 或直接使用 Docker 命令
docker build -t mevalkit:latest .
```

### 3. 启动服务

```bash
# 使用 Docker Compose 启动
docker-compose up -d

# 或直接使用 Docker 命令
docker run -d \
    --name mevalkit \
    -p 5000:5000 \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/stress_test_results:/app/stress_test_results \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/dataset_info:/app/dataset_info \
    -v $(pwd)/custom_models:/app/custom_models \
    -v $(pwd)/datasets_and_models:/app/datasets_and_models \
    --env-file .env \
    --restart unless-stopped \
    mevalkit:latest
```

## 配置说明

### 环境变量配置

编辑 `.env` 文件配置以下重要参数：

```bash
# OpenAI API 配置
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1

# 其他模型API配置
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# 应用配置
FLASK_ENV=production
SECRET_KEY=your_secret_key_here_change_this_in_production

# 评估配置
DEFAULT_WORKERS=64
DEFAULT_QUESTION_LIMITATION=100
```

### 端口配置

默认端口为 5000，如需修改：

```bash
# 在 docker-compose.yml 中修改
ports:
  - "8080:5000"  # 将主机端口改为 8080

# 或在 docker run 命令中修改
-p 8080:5000
```

## 数据持久化

Docker 配置已包含数据卷映射，以下目录的数据将被持久化：

- `./results` - 评估结果文件
- `./logs` - 应用日志
- `./stress_test_results` - 压力测试结果
- `./data` - 数据集文件
- `./dataset_info` - 数据集配置
- `./custom_models` - 自定义模型
- `./datasets_and_models` - 数据集和模型

### 备份数据

```bash
# 备份重要数据
tar -czf mevalkit_backup_$(date +%Y%m%d).tar.gz \
    results/ logs/ stress_test_results/ data/ \
    dataset_info/ custom_models/ datasets_and_models/
```

## 访问应用

部署完成后，您可以通过以下地址访问：

- **Web 界面**: http://localhost:5000
- **API 文档**: http://localhost:5000/apidocs/
- **健康检查**: http://localhost:5000/

## 故障排除

### 常见问题

1. **端口被占用**
   ```bash
   # 检查端口占用
   netstat -tlnp | grep :5000
   
   # 修改端口
   # 在 docker-compose.yml 中修改端口映射
   ```

2. **权限问题**
   ```bash
   # 确保脚本有执行权限
   chmod +x deploy.sh docker/build_image.sh
   
   # 确保目录有写权限
   chmod 755 results/ logs/ data/
   ```

3. **镜像构建失败**
   ```bash
   # 清理缓存重新构建
   docker system prune -f
   ./deploy.sh --rebuild
   ```

4. **服务无法启动**
   ```bash
   # 查看详细日志
   ./deploy.sh --logs
   
   # 检查环境文件
   cat .env
   ```

### 日志查看

```bash
# 查看实时日志
./deploy.sh --logs

# 查看容器日志
docker logs -f mevalkit

# 查看 Docker Compose 日志
docker-compose logs -f
```

### 进入容器调试

```bash
# 进入运行中的容器
docker exec -it mevalkit bash

# 或使用 Makefile
make shell
```

## 开发模式

### 开发环境部署

```bash
# 使用开发模式启动（源代码同步）
cd docker && docker-compose -f docker-compose.dev.yml up -d

# 或使用 Makefile
make dev-sync
```

### 开发模式特点

- 源代码实时同步到容器
- 支持热重载
- 调试模式启用
- 详细的错误信息

### 开发工具

```bash
# 查看开发模式日志
make dev-sync-logs

# 停止开发模式
make dev-sync-down
```

## 性能优化

### 镜像优化

- 使用多阶段构建减少镜像大小
- 生产环境使用非 root 用户运行
- 优化依赖安装顺序

### 运行时优化

- 设置合适的内存限制
- 配置 CPU 限制
- 使用健康检查监控服务状态

### 示例配置

```yaml
# docker-compose.yml 中添加资源限制
services:
  mevalkit:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
```

## 安全考虑

### 生产环境安全

1. **修改默认密钥**
   ```bash
   # 在 .env 文件中设置强密钥
   SECRET_KEY=your_very_strong_secret_key_here
   ```

2. **限制网络访问**
   ```bash
   # 只允许本地访问
   ports:
     - "127.0.0.1:5000:5000"
   ```

3. **使用 HTTPS**
   ```bash
   # 配置反向代理（如 Nginx）
   # 或使用 Traefik 等容器化代理
   ```

### 数据安全

- 定期备份重要数据
- 使用加密存储敏感信息
- 限制容器权限

## 监控和维护

### 健康检查

应用内置健康检查，可通过以下方式监控：

```bash
# 检查容器健康状态
docker ps --filter name=mevalkit

# 查看健康检查日志
docker inspect mevalkit | grep -A 10 Health
```

### 日志管理

```bash
# 配置日志轮转
# 在 docker-compose.yml 中添加日志配置
services:
  mevalkit:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### 定期维护

```bash
# 清理未使用的镜像和容器
docker system prune -f

# 更新基础镜像
docker pull python:3.11-slim

# 重新构建应用镜像
./deploy.sh --rebuild
```

## 支持

如果您在部署过程中遇到问题，请：

1. 查看本文档的故障排除部分
2. 检查应用日志：`./deploy.sh --logs`
3. 确认环境配置是否正确
4. 提交 Issue 到项目仓库

---

**注意**: 本文档针对 rootless Docker 环境进行了优化，如果您使用的是传统 Docker 安装，大部分配置仍然适用，但可能需要调整权限设置。 