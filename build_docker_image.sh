#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   ./build_docker_image.sh [镜像名:标签] [--online]
#
# 默认是离线构建模式：
# - 依赖 wheel 来自本地 wheelhouse/
# - 若 wheelhouse 不存在，会提示先执行 prepare_offline_wheels.sh
#
# 示例：
#   ./build_docker_image.sh qwentrain:offline
#   ./build_docker_image.sh qwentrain:online --online

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_TAG="${1:-qwentrain:offline}"
MODE="${2:-}"
USE_OFFLINE_WHEELS=1

if [[ "${MODE}" == "--online" ]]; then
  USE_OFFLINE_WHEELS=0
fi

if [[ "${USE_OFFLINE_WHEELS}" = "1" ]]; then
  if [[ ! -d "${ROOT_DIR}/wheelhouse" || ! -s "${ROOT_DIR}/requirements.offline.lock.txt" ]]; then
    echo "[1/3] 未检测到离线依赖，请先执行："
    echo "  ./prepare_offline_wheels.sh"
    echo "若要为 Linux x86_64 服务器准备："
    echo "  ./prepare_offline_wheels.sh --docker-platform linux/amd64"
    exit 1
  else
    echo "[1/3] 已检测到离线依赖，跳过下载。"
  fi
else
  echo "[1/3] 在线构建模式，跳过离线 wheel 检查。"
fi

echo "[2/3] 构建 Docker 镜像: ${IMAGE_TAG}"
docker build \
  --build-arg USE_OFFLINE_WHEELS="${USE_OFFLINE_WHEELS}" \
  -t "${IMAGE_TAG}" \
  -f "${ROOT_DIR}/Dockerfile" \
  "${ROOT_DIR}"

echo "[3/3] 完成。"
echo "运行示例："
echo "docker run --rm -it ${IMAGE_TAG}"
echo
echo "若需挂载本地模型目录："
echo "docker run --rm -it -v ${ROOT_DIR}/model:/opt/qwentrain/model ${IMAGE_TAG}"
