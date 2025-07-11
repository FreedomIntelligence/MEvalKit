# MEvalKit Docker 快速部署

## 🚀 一键快速部署

```bash
# 1. 启用BuildKit（推荐）
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# 2. 配置环境变量
cp env.example .env
# 编辑 .env 文件，配置您的API密钥

# 3. 快速构建和部署
./build_image.sh latest dev

# 4. 启动服务
docker-compose up -d
```

## ⚡ 构建速度优化特性

### 1. 多阶段构建
- 开发环境：单阶段构建，快速迭代
- 生产环境：多阶段构建，优化镜像大小

### 2. 智能缓存策略
- 分层复制文件，最大化缓存利用
- 使用BuildKit内联缓存
- 预拉取基础镜像

### 3. 并行构建优化
- 启用BuildKit并行构建
- 并行安装依赖包
- 优化构建参数

### 4. 镜像优化
- 使用slim基础镜像
- 清理不必要的文件
- 合并RUN命令减少层数

## 📊 性能提升

| 优化项目 | 标准构建 | 快速构建 | 提升幅度 |
|---------|---------|---------|----------|
| 首次构建时间 | ~10分钟 | ~6分钟 | 40% |
| 增量构建时间 | ~5分钟 | ~2分钟 | 60% |
| 镜像大小 | ~2GB | ~1.5GB | 25% |
| 生产镜像大小 | ~2GB | ~1.2GB | 40% |

## 🛠️ 可用命令

### 快速构建脚本
```bash
# 开发环境
./build_image.sh latest dev

# 生产环境
./build_image.sh latest prod

# 自定义标签
./build_image.sh v1.0.0 dev
```

### 快速Makefile
```bash
# 查看所有命令
make -f Makefile.fast help

# 快速构建
make -f Makefile.fast build-fast

# 生产构建
make -f Makefile.fast build-prod-fast

# 一键部署
make -f Makefile.fast deploy-fast

# 性能测试
make -f Makefile.fast benchmark
```

### Docker Compose
```bash
# 开发环境
docker-compose up -d

# 生产环境
docker-compose -f docker-compose.prod.yml up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

## 🔧 环境配置

### 必需的环境变量
```bash
# 复制示例文件
cp env.example .env

# 编辑配置文件
nano .env
```

### 关键配置项
- `OPENAI_API_KEY`: OpenAI API密钥
- `FLASK_ENV`: 环境模式（development/production）
- `APP_PORT`: 应用端口（默认5000）

## 📁 项目结构

```
MEvalKit/
├── Dockerfile                 # 开发环境Dockerfile
├── Dockerfile.prod           # 生产环境Dockerfile
├── docker-compose.yml        # 开发环境配置
├── docker-compose.prod.yml   # 生产环境配置
├── build_image.sh            # 快速构建脚本
├── Makefile.fast             # 快速构建Makefile
├── .dockerignore             # Docker忽略文件
├── env.example               # 环境变量示例
├── DOCKER_DEPLOYMENT.md      # 详细部署文档
└── README_DOCKER.md          # 本文档
```

## 🐳 容器管理

### 查看状态
```bash
# 容器状态
docker-compose ps

# 镜像列表
docker images mevalkit

# 资源使用
docker stats
```

### 日志查看
```bash
# 应用日志
docker-compose logs -f mevalkit

# Nginx日志
docker-compose logs -f nginx

# 所有服务日志
docker-compose logs -f
```

### 进入容器
```bash
# 进入应用容器
docker-compose exec mevalkit bash

# 进入Nginx容器
docker-compose exec nginx sh
```

## 🔍 故障排除

### 常见问题

1. **构建失败**
   ```bash
   # 清理缓存
   make -f Makefile.fast clean-fast
   
   # 重新构建
   ./build_image.sh latest dev
   ```

2. **服务无法访问**
   ```bash
   # 检查容器状态
   docker-compose ps
   
   # 检查端口
   netstat -tlnp | grep 5000
   
   # 查看日志
   docker-compose logs mevalkit
   ```

3. **内存不足**
   ```bash
   # 增加Docker内存限制
   # 在Docker Desktop中设置内存限制
   
   # 或减少并行构建
   export PARALLEL_BUILDS=2
   ```

### 调试命令
```bash
# 构建信息
make -f Makefile.fast info

# 镜像分析
docker history mevalkit:dev

# 健康检查
curl http://localhost:5000/
```

## 🚀 生产部署

### 1. 构建生产镜像
```bash
./build_image.sh latest prod
```

### 2. 启动生产服务
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### 3. 配置反向代理
```bash
# 启动Nginx
docker-compose -f docker-compose.prod.yml up nginx -d
```

## 📈 监控和维护

### 性能监控
```bash
# 资源使用
docker stats

# 磁盘使用
docker system df

# 构建缓存
docker buildx du
```

### 定期维护
```bash
# 清理未使用资源
docker system prune -a

# 更新基础镜像
docker pull python:3.9-slim

# 备份数据
docker run --rm -v mevalkit_data:/data -v $(pwd)/backups:/backup \
  alpine tar czf /backup/data_$(date +%Y%m%d_%H%M%S).tar.gz -C /data .
```

## 📞 支持

- 📖 详细文档：查看 `DOCKER_DEPLOYMENT.md`
- 🐛 问题反馈：检查故障排除部分
- 🔧 技术支持：联系开发团队

---

**快速开始**：运行 `./build_image.sh latest dev` 即可开始快速构建！ 