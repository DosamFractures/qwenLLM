#!/usr/bin/env bash
set -euo pipefail

# 生成离线依赖：
# 1) requirements.offline.lock.txt（Linux 容器内 freeze）
# 2) wheelhouse/（离线 whl 包）
#
# 默认使用 Docker 容器生成 Linux x86_64 依赖，适合离线服务器部署。
#
# 用法：
#   ./prepare_offline_wheels.sh
#   ./prepare_offline_wheels.sh --docker-platform linux/arm64
#   ./prepare_offline_wheels.sh --python-image python:3.11-slim
#   ./prepare_offline_wheels.sh --use-host-pip

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK_FILE="${ROOT_DIR}/requirements.offline.lock.txt"
WHEEL_DIR="${ROOT_DIR}/wheelhouse"

PY_IMAGE="python:3.11-slim"
DOCKER_PLATFORM="linux/amd64"
USE_HOST_PIP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-image)
      PY_IMAGE="$2"
      shift 2
      ;;
    --docker-platform)
      DOCKER_PLATFORM="$2"
      shift 2
      ;;
    --use-host-pip)
      USE_HOST_PIP=1
      shift
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

if [[ "${USE_HOST_PIP}" = "1" ]]; then
  PIP_BIN="${ROOT_DIR}/.venv/bin/pip"
  if [[ ! -x "${PIP_BIN}" ]]; then
    echo "未找到 ${PIP_BIN}，请先在项目根目录创建并安装 .venv。"
    exit 1
  fi

  echo "[Host 1/5] 升级打包工具（pip/setuptools/wheel）..."
  "${PIP_BIN}" install --upgrade pip setuptools wheel
  echo "[Host 2/5] 导出离线锁定依赖: ${LOCK_FILE}"
  "${PIP_BIN}" freeze > "${LOCK_FILE}"
  echo "[Host 3/5] 重建 wheel 目录: ${WHEEL_DIR}"
  rm -rf "${WHEEL_DIR}"
  mkdir -p "${WHEEL_DIR}"
  echo "[Host 4/5] 下载业务依赖 wheel..."
  "${PIP_BIN}" download --dest "${WHEEL_DIR}" --requirement "${LOCK_FILE}" --only-binary=:all:
  echo "[Host 5/5] 下载基础安装工具 wheel..."
  "${PIP_BIN}" download --dest "${WHEEL_DIR}" --only-binary=:all: pip setuptools wheel
else
  if ! command -v docker >/dev/null 2>&1; then
    echo "未检测到 docker。可改用：./prepare_offline_wheels.sh --use-host-pip"
    exit 1
  fi

  echo "[Docker 1/4] 使用容器生成离线依赖"
  echo "  image=${PY_IMAGE}"
  echo "  platform=${DOCKER_PLATFORM}"

  docker run --rm \
    --platform "${DOCKER_PLATFORM}" \
    -v "${ROOT_DIR}:/work" \
    -w /work \
    "${PY_IMAGE}" \
    /bin/bash -lc "
      set -euo pipefail
      python -m pip install --upgrade pip setuptools wheel
      python -m pip install -r requirements.txt
      python -m pip freeze > requirements.offline.lock.txt
      rm -rf wheelhouse
      mkdir -p wheelhouse
      python -m pip download --dest wheelhouse --requirement requirements.offline.lock.txt --only-binary=:all:
      python -m pip download --dest wheelhouse --only-binary=:all: pip setuptools wheel
    "
fi

COUNT="$(find "${WHEEL_DIR}" -type f | wc -l | tr -d ' ')"
echo "[Done] 离线 wheel 准备完成，共 ${COUNT} 个文件。"
echo "锁定依赖文件：${LOCK_FILE}"
