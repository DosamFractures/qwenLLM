FROM python:3.11-slim

# 基础运行环境配置
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/opt/qwentrain \
    VENV_PATH=/opt/qwentrain/.venv

WORKDIR ${APP_HOME}

# 把项目源码、wheelhouse、离线锁定文件一起打进镜像
COPY . ${APP_HOME}

# 1 表示优先离线安装；0 表示在线安装（回退模式）
ARG USE_OFFLINE_WHEELS=1

RUN python -m venv "${VENV_PATH}" \
    && if [ "${USE_OFFLINE_WHEELS}" = "1" ]; then \
         test -d "${APP_HOME}/wheelhouse" || (echo "wheelhouse not found. Please run prepare_offline_wheels.sh first." && exit 1); \
         test -s "${APP_HOME}/requirements.offline.lock.txt" || (echo "requirements.offline.lock.txt not found. Please run prepare_offline_wheels.sh first." && exit 1); \
         "${VENV_PATH}/bin/pip" install --no-index --find-links "${APP_HOME}/wheelhouse" --upgrade pip setuptools wheel; \
         "${VENV_PATH}/bin/pip" install --no-index --find-links "${APP_HOME}/wheelhouse" -r "${APP_HOME}/requirements.offline.lock.txt"; \
       else \
         "${VENV_PATH}/bin/pip" install --upgrade pip; \
         "${VENV_PATH}/bin/pip" install -r "${APP_HOME}/requirements.txt"; \
       fi

ENV PATH="${VENV_PATH}/bin:${PATH}"

# 默认启动聊天脚本（可在 docker run 时覆盖）
CMD ["python", "-m", "qwentrain.chat_cli", "-model", "qwen3.5-2b", "--local-files-only"]
