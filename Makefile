# MEvalKit Docker 管理 Makefile

# 变量定义
IMAGE_NAME := mevalkit
TAG := latest
CONTAINER_NAME := mevalkit-app
PORT := 5000

# 默认目标
.PHONY: help
help:
	@echo "MEvalKit Docker 管理命令:"
	@echo ""
	@echo "构建相关:"
	@echo "  build          - 构建开发版镜像"
	@echo "  build-prod     - 构建生产版镜像"
	@echo "  rebuild        - 强制重新构建镜像"
	@echo ""
	@echo "运行相关:"
	@echo "  up             - 使用 docker-compose 启动服务"
	@echo "  down           - 停止 docker-compose 服务"
	@echo "  run            - 使用 docker 命令运行容器"
	@echo "  stop           - 停止容器"
	@echo "  restart        - 重启容器"
	@echo ""
	@echo "管理相关:"
	@echo "  logs           - 查看容器日志"
	@echo "  shell          - 进入容器 shell"
	@echo "  status         - 查看容器状态"
	@echo "  clean          - 清理容器和镜像"
	@echo "  clean-all      - 清理所有相关资源"
	@echo ""
	@echo "工具相关:"
	@echo "  env            - 创建环境配置文件"
	@echo "  test           - 测试容器健康状态"
	@echo ""
	@echo "压力测试相关:"
	@echo "  stress-test        - 运行交互式压力测试菜单"
	@echo "  stress-test-quick  - 快速验证测试 (1-2分钟)"
	@echo "  stress-test-fast   - 快速压力测试 (3-5分钟)"  
	@echo "  stress-test-full   - 完整压力测试 (20-30分钟)"
	@echo ""
	@echo "文件查看相关:"
	@echo "  inspect        - 查看镜像详细信息"
	@echo "  files          - 查看容器内文件结构"
	@echo "  config         - 查看容器配置信息"
	@echo ""
	@echo "开发模式相关:"
	@echo "  dev            - 启动标准开发环境"
	@echo "  dev-sync       - 启动源代码同步开发模式"
	@echo "  dev-sync-logs  - 查看开发模式日志"
	@echo "  dev-sync-down  - 停止开发模式"

# 构建相关
.PHONY: build
build:
	cd docker && ./build_image.sh $(TAG)

.PHONY: build-prod
build-prod:
	cd docker && docker build -f Dockerfile.prod -t $(IMAGE_NAME):prod ..

.PHONY: rebuild
rebuild:
	cd docker && docker build --no-cache -f Dockerfile -t $(IMAGE_NAME):$(TAG) ..

# 运行相关
.PHONY: up
up:
	cd docker && docker-compose up -d

.PHONY: down
down:
	cd docker && docker-compose down

.PHONY: run
run: build
	docker run -d \
		--name $(CONTAINER_NAME) \
		-p $(PORT):5000 \
		-v $$(pwd)/results:/app/results \
		-v $$(pwd)/logs:/app/logs \
		-v $$(pwd)/stress_test_results:/app/stress_test_results \
		-v $$(pwd)/data:/app/data \
		--env-file .env \
		$(IMAGE_NAME):$(TAG)

.PHONY: stop
stop:
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)

.PHONY: restart
restart: stop run

# 管理相关
.PHONY: logs
logs:
	@if cd docker && docker-compose ps | grep -q mevalkit; then \
		cd docker && docker-compose logs -f; \
	else \
		docker logs -f $(CONTAINER_NAME); \
	fi

.PHONY: shell
shell:
	@if cd docker && docker-compose ps | grep -q mevalkit; then \
		cd docker && docker-compose exec mevalkit bash; \
	else \
		docker exec -it $(CONTAINER_NAME) bash; \
	fi

.PHONY: status
status:
	@echo "=== Docker Compose 状态 ==="
	@cd docker && docker-compose ps 2>/dev/null || echo "Docker Compose 未运行"
	@echo ""
	@echo "=== Docker 容器状态 ==="
	@docker ps -a --filter name=$(CONTAINER_NAME) 2>/dev/null || echo "未找到相关容器"
	@echo ""
	@echo "=== Docker 镜像 ==="
	@docker images $(IMAGE_NAME) 2>/dev/null || echo "未找到相关镜像"

.PHONY: clean
clean: down stop
	-docker rmi $(IMAGE_NAME):$(TAG)
	-docker rmi $(IMAGE_NAME):prod

.PHONY: clean-all
clean-all: clean
	-docker system prune -f
	-docker volume prune -f

# 工具相关
.PHONY: env
env:
	@if [ ! -f .env ]; then \
		cp env.example .env; \
		echo "已创建 .env 文件，请编辑配置您的 API 密钥"; \
	else \
		echo ".env 文件已存在"; \
	fi

.PHONY: test
test:
	@echo "测试容器健康状态..."
	@if curl -f http://localhost:$(PORT)/ >/dev/null 2>&1; then \
		echo "✅ 服务运行正常"; \
	else \
		echo "❌ 服务无法访问"; \
		exit 1; \
	fi

# 压力测试相关
.PHONY: stress-test
stress-test:
	@echo "运行交互式压力测试..."
	@if cd docker && docker-compose ps | grep -q mevalkit; then \
		cd docker && docker-compose exec mevalkit ./stress_test/run_stress_test.sh; \
	else \
		docker exec -it $(CONTAINER_NAME) ./stress_test/run_stress_test.sh; \
	fi

.PHONY: stress-test-quick  
stress-test-quick:
	@echo "运行快速验证测试..."
	@if cd docker && docker-compose ps | grep -q mevalkit; then \
		cd docker && docker-compose exec mevalkit python stress_test/quick_stress_test.py; \
	else \
		docker exec -it $(CONTAINER_NAME) python stress_test/quick_stress_test.py; \
	fi

.PHONY: stress-test-full
stress-test-full:
	@echo "运行完整压力测试..."
	@if cd docker && docker-compose ps | grep -q mevalkit; then \
		cd docker && docker-compose exec mevalkit python stress_test/evaluation_stress_test.py; \
	else \
		docker exec -it $(CONTAINER_NAME) python stress_test/evaluation_stress_test.py; \
	fi

.PHONY: stress-test-fast
stress-test-fast:
	@echo "运行快速压力测试模式..."
	@if cd docker && docker-compose ps | grep -q mevalkit; then \
		cd docker && docker-compose exec mevalkit python stress_test/evaluation_stress_test.py --quick-test; \
	else \
		docker exec -it $(CONTAINER_NAME) python stress_test/evaluation_stress_test.py --quick-test; \
	fi

# 文件查看相关
.PHONY: inspect
inspect:
	@echo "=== 镜像详细信息 ==="
	@docker image inspect $(IMAGE_NAME):$(TAG) --format='{{json .Config}}' | jq
	@echo ""
	@echo "=== 镜像构建历史 ==="
	@docker history $(IMAGE_NAME):$(TAG)

.PHONY: files
files:
	@echo "=== 容器文件结构 ==="
	@if cd docker && docker-compose ps | grep -q mevalkit; then \
		echo "🗂️  项目根目录:"; \
		cd docker && docker-compose exec mevalkit ls -la /app/; \
		echo ""; \
		echo "🐍 Python 文件:"; \
		cd docker && docker-compose exec mevalkit find /app -name "*.py" -type f | grep -E "(app|stress|run)" | head -10; \
		echo ""; \
		echo "📁 重要目录:"; \
		cd docker && docker-compose exec mevalkit ls -la /app/src/ /app/templates/ /app/data/ 2>/dev/null || true; \
	else \
		echo "🗂️  项目根目录:"; \
		docker exec $(CONTAINER_NAME) ls -la /app/; \
		echo ""; \
		echo "🐍 Python 文件:"; \
		docker exec $(CONTAINER_NAME) find /app -name "*.py" -type f | grep -E "(app|stress|run)" | head -10; \
		echo ""; \
		echo "📁 重要目录:"; \
		docker exec $(CONTAINER_NAME) ls -la /app/src/ /app/templates/ /app/data/ 2>/dev/null || true; \
	fi

.PHONY: config
config:
	@echo "=== 容器配置信息 ==="
	@if cd docker && docker-compose ps | grep -q mevalkit; then \
		echo "🔧 环境变量:"; \
		cd docker && docker-compose exec mevalkit env | grep -E "(FLASK|OPENAI|PYTHON)" | sort; \
		echo ""; \
		echo "📦 Python 版本和路径:"; \
		cd docker && docker-compose exec mevalkit python --version; \
		cd docker && docker-compose exec mevalkit which python; \
		echo ""; \
		echo "📋 主要依赖包:"; \
		cd docker && docker-compose exec mevalkit pip list | grep -E "(Flask|requests|openai|pandas|numpy)" | head -10; \
	else \
		echo "🔧 环境变量:"; \
		docker exec $(CONTAINER_NAME) env | grep -E "(FLASK|OPENAI|PYTHON)" | sort; \
		echo ""; \
		echo "📦 Python 版本和路径:"; \
		docker exec $(CONTAINER_NAME) python --version; \
		docker exec $(CONTAINER_NAME) which python; \
		echo ""; \
		echo "📋 主要依赖包:"; \
		docker exec $(CONTAINER_NAME) pip list | grep -E "(Flask|requests|openai|pandas|numpy)" | head -10; \
	fi

# 开发相关快捷命令
.PHONY: dev
dev: env build up logs

.PHONY: dev-sync
dev-sync: env
	@echo "启动开发模式（源代码同步）..."
	cd docker && docker-compose -f docker-compose.dev.yml up -d

.PHONY: dev-sync-logs
dev-sync-logs:
	@echo "查看开发模式日志..."
	cd docker && docker-compose -f docker-compose.dev.yml logs -f

.PHONY: dev-sync-down
dev-sync-down:
	@echo "停止开发模式..."
	cd docker && docker-compose -f docker-compose.dev.yml down

.PHONY: prod
prod: env build-prod
	docker run -d \
		--name $(CONTAINER_NAME)-prod \
		-p $(PORT):5000 \
		-v $$(pwd)/results:/app/results \
		-v $$(pwd)/logs:/app/logs \
		-v $$(pwd)/stress_test_results:/app/stress_test_results \
		-v $$(pwd)/data:/app/data \
		--env-file .env \
		$(IMAGE_NAME):prod 