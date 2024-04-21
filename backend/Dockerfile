ARG ARCHITECTURE=arm64v8

FROM ${ARCHITECTURE}/python:3.11-slim
ARG UID=1000
ARG GID=1000
ARG USERNAME=user

LABEL maintainer="tnakagami"
LABEL description="build the environment of Django application"

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    DEBIAN_FRONTEND=noninteractive \
    SRC_ROOT_PATH=/opt/app

RUN    apt-get update \
    && apt-get install -y libmagic1 gcc libc-dev python3-dev libpq-dev \
    && addgroup --gid ${GID} ${USERNAME} \
    && adduser --home /opt/home --shell /bin/bash --uid ${UID} --gid ${GID} --disabled-password ${USERNAME} \
    && pip install poetry \
    && mkdir -p ${SRC_ROOT_PATH} \
    && chown -R ${USERNAME}:${USERNAME} /opt/home \
    && rm -rf /var/lib/apt/lists/*

COPY ./pyproject.toml /opt/pyproject.toml

RUN    cd /opt \
    && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

WORKDIR ${SRC_ROOT_PATH}
USER ${USERNAME}

ENTRYPOINT [ "uvicorn", "config.asgi:application", "--lifespan", "on", "--host", "0.0.0.0", "--port", "8000" ]