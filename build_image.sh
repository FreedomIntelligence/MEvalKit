#!/bin/bash

# MEvalKit 快速镜像构建脚本
# 使用多种优化技术加快构建速度

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置变量
IMAGE_NAME="mevalkit"
TAG=${1:-"latest"}
BUILD_TYPE=${2:-"dev"}
BUILDKIT_ENABLED=${BUILDKIT_ENABLED:-1}
PARALLEL_BUILDS=${PARALLEL_BUILDS:-4}

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Docker是否运行
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker未运行或无法访问"
        exit 1
    fi
}

# 清理旧的构建缓存
clean_build_cache() {
    log_info "清理构建缓存..."
    docker builder prune -f
}

# 预拉取基础镜像
prepull_base_images() {
    log_info "预拉取基础镜像..."
    docker pull python:3.9-slim &
    docker pull nginx:alpine &
    wait
}

# 构建镜像
build_image() {
    local dockerfile="Dockerfile"
    local target_image="${IMAGE_NAME}:${TAG}"
    
    if [ "$BUILD_TYPE" = "prod" ]; then
        dockerfile="Dockerfile.prod"
        target_image="${IMAGE_NAME}:prod"
    fi
    
    log_info "开始构建镜像: ${target_image}"
    log_info "使用Dockerfile: ${dockerfile}"
    log_info "构建类型: ${BUILD_TYPE}"
    
    # 设置构建参数
    local build_args=""
    if [ "$BUILD_TYPE" = "prod" ]; then
        build_args="--target production"
    fi
    
    # 使用BuildKit和并行构建优化
    DOCKER_BUILDKIT=${BUILDKIT_ENABLED} docker build \
        --file ${dockerfile} \
        --tag ${target_image} \
        --cache-from ${IMAGE_NAME}:latest \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --build-arg BUILDKIT_STEP_LOG_MAX_SIZE=10485760 \
        --build-arg BUILDKIT_STEP_LOG_MAX_SPEED=10485760 \
        --progress=plain \
        ${build_args} \
        .
    
    log_success "镜像构建完成: ${target_image}"
}

# 优化构建性能
optimize_build_performance() {
    log_info "优化构建性能..."
    
    # 设置Docker构建参数
    export DOCKER_BUILDKIT=${BUILDKIT_ENABLED}
    export COMPOSE_DOCKER_CLI_BUILD=${BUILDKIT_ENABLED}
    
    # 设置构建缓存目录
    export BUILDX_CACHE_DIR="/tmp/docker-buildx-cache"
    mkdir -p ${BUILDX_CACHE_DIR}
    
    # 设置并行构建
    export DOCKER_BUILDKIT_STEP_LOG_MAX_SIZE=10485760
    export DOCKER_BUILDKIT_STEP_LOG_MAX_SPEED=10485760
}

# 验证镜像
validate_image() {
    local target_image="${IMAGE_NAME}:${TAG}"
    if [ "$BUILD_TYPE" = "prod" ]; then
        target_image="${IMAGE_NAME}:prod"
    fi
    
    log_info "验证镜像: ${target_image}"
    
    # 检查镜像是否存在
    if ! docker image inspect ${target_image} > /dev/null 2>&1; then
        log_error "镜像构建失败: ${target_image}"
        exit 1
    fi
    
    # 检查镜像大小
    local image_size=$(docker image inspect ${target_image} --format='{{.Size}}')
    local size_mb=$((image_size / 1024 / 1024))
    log_info "镜像大小: ${size_mb}MB"
    
    # 运行健康检查
    log_info "运行健康检查..."
    docker run --rm ${target_image} python -c "import flask; print('Flask导入成功')" || {
        log_error "健康检查失败"
        exit 1
    }
    
    log_success "镜像验证通过"
}

# 显示构建信息
show_build_info() {
    log_info "构建信息:"
    echo "  镜像名称: ${IMAGE_NAME}"
    echo "  标签: ${TAG}"
    echo "  构建类型: ${BUILD_TYPE}"
    echo "  BuildKit: ${BUILDKIT_ENABLED}"
    echo "  并行构建: ${PARALLEL_BUILDS}"
    echo "  Docker版本: $(docker --version)"
    echo "  Docker Compose版本: $(docker-compose --version)"
}

# 主函数
main() {
    log_info "开始MEvalKit镜像构建..."
    
    show_build_info
    check_docker
    optimize_build_performance
    clean_build_cache
    prepull_base_images
    build_image
    validate_image
    
    log_success "MEvalKit镜像构建完成！"
    log_info "使用方法:"
    echo "  开发环境: docker-compose up -d"
    echo "  生产环境: docker-compose -f docker-compose.prod.yml up -d"
    echo "  运行容器: docker run -p 5000:5000 ${IMAGE_NAME}:${TAG}"
}

# 脚本入口
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 