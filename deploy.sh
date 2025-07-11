#!/bin/bash

# MEvalKit 一键部署脚本
# 支持 rootless Docker 环境

set -e

# 默认配置
IMAGE_NAME="mevalkit"
TAG="latest"
PORT="1984"
CONTAINER_NAME="mevalkit"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  MEvalKit 一键部署脚本${NC}"
    echo -e "${BLUE}================================${NC}"
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --build-only           只构建镜像，不启动服务"
    echo "  --start-only           只启动服务，不构建镜像"
    echo "  --rebuild              重新构建镜像并启动服务"
    echo "  --stop                 停止服务"
    echo "  --restart              重启服务"
    echo "  --logs                 查看服务日志"
    echo "  --status               查看服务状态"
    echo "  --clean                清理镜像和容器"
    echo "  -h, --help             显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                     构建镜像并启动服务"
    echo "  $0 --build-only        只构建镜像"
    echo "  $0 --start-only        只启动服务"
    echo "  $0 --rebuild           重新构建并启动"
}

# 检查Docker环境
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker 未安装或不在 PATH 中"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker 守护进程未运行"
        exit 1
    fi
    
    print_message "Docker 环境检查通过"
}

# 检查环境文件
check_env_file() {
    if [[ ! -f ".env" ]]; then
        print_warning ".env 文件不存在，正在创建..."
        if [[ -f "env.example" ]]; then
            cp env.example .env
            print_message "已从 env.example 创建 .env 文件"
            print_warning "请编辑 .env 文件配置您的 API 密钥"
        else
            print_error "env.example 文件不存在"
            exit 1
        fi
    else
        print_message ".env 文件已存在"
    fi
}

# 构建镜像
build_image() {
    print_message "开始构建镜像..."
    if [[ -d "docker" ]]; then
        cd docker && ./build_image.sh && cd ..
    else
        docker build -t "$IMAGE_NAME:$TAG" .
    fi
    print_message "镜像构建完成"
}

# 启动服务
start_service() {
    print_message "启动 MEvalKit 服务..."
    
    # 检查镜像是否存在
    if ! docker images "$IMAGE_NAME:$TAG" | grep -q "$IMAGE_NAME"; then
        print_error "镜像 $IMAGE_NAME:$TAG 不存在，请先构建镜像"
        exit 1
    fi
    
    # 使用 docker-compose 启动服务
    if [[ -f "docker-compose.yml" ]]; then
        docker compose up -d
    elif [[ -f "docker/docker-compose.yml" ]]; then
        cd docker && docker compose up -d && cd ..
    else
        # 直接使用 docker run
        docker run -d \
            --name "$CONTAINER_NAME" \
            -p "$PORT:1984" \
            -v "$(pwd)/results:/app/results" \
            -v "$(pwd)/logs:/app/logs" \
            -v "$(pwd)/stress_test_results:/app/stress_test_results" \
            -v "$(pwd)/data:/app/data" \
            -v "$(pwd)/dataset_info:/app/dataset_info" \
            -v "$(pwd)/custom_models:/app/custom_models" \
            -v "$(pwd)/datasets_and_models:/app/datasets_and_models" \
            --env-file .env \
            --restart unless-stopped \
            "$IMAGE_NAME:$TAG"
    fi
    
    print_message "服务启动完成"
    print_message "访问地址: http://localhost:$PORT"
    print_message "API文档: http://localhost:$PORT/apidocs/"
}

# 停止服务
stop_service() {
    print_message "停止 MEvalKit 服务..."
    
    if [[ -f "docker-compose.yml" ]]; then
        docker-compose down
    elif [[ -f "docker/docker-compose.yml" ]]; then
        cd docker && docker-compose down && cd ..
    else
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
    fi
    
    print_message "服务已停止"
}

# 查看日志
show_logs() {
    print_message "显示服务日志..."
    
    if [[ -f "docker-compose.yml" ]]; then
        docker-compose logs -f
    elif [[ -f "docker/docker-compose.yml" ]]; then
        cd docker && docker-compose logs -f && cd ..
    else
        docker logs -f "$CONTAINER_NAME"
    fi
}

# 查看状态
show_status() {
    print_message "服务状态:"
    
    if [[ -f "docker-compose.yml" ]]; then
        docker-compose ps
    elif [[ -f "docker/docker-compose.yml" ]]; then
        cd docker && docker-compose ps && cd ..
    else
        docker ps -a --filter name="$CONTAINER_NAME"
    fi
    
    echo ""
    print_message "镜像信息:"
    docker images "$IMAGE_NAME"
}

# 清理资源
clean_resources() {
    print_message "清理 Docker 资源..."
    
    # 停止并删除容器
    stop_service
    
    # 删除镜像
    docker rmi "$IMAGE_NAME:$TAG" 2>/dev/null || true
    
    print_message "清理完成"
}

# 主函数
main() {
    print_header
    
    # 检查Docker环境
    check_docker
    
    # 检查环境文件
    check_env_file
    
    # 解析命令行参数
    case "${1:-}" in
        --build-only)
            build_image
            ;;
        --start-only)
            start_service
            ;;
        --rebuild)
            stop_service
            build_image
            start_service
            ;;
        --stop)
            stop_service
            ;;
        --restart)
            stop_service
            start_service
            ;;
        --logs)
            show_logs
            ;;
        --status)
            show_status
            ;;
        --clean)
            clean_resources
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        "")
            # 默认行为：构建并启动
            build_image
            start_service
            ;;
        *)
            print_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@" 