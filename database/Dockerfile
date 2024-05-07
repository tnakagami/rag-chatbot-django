ARG ARCHITECTURE=arm64v8
FROM ${ARCHITECTURE}/postgres:16
ARG PGVECTOR_VERSION=0.7.0

LABEL maintainer="tnakagami"
LABEL description="build postgres environment"

ENV POSTGRES_USER= \
    POSTGRES_PASSWORD= \
    POSTGRES_DB= \
    POSTGRES_PORT=5432 \
    PGDATA=/var/lib/postgresql/data/pgdata

RUN    apt-get update \
    && apt-get install -y git make gcc postgresql-server-dev-16 \
    && cd /tmp \
    && git clone --branch v${PGVECTOR_VERSION} https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make \
    && make install \
    && cd / \
    && apt-get remove -y git make gcc \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /tmp/* \
    && rm -rf /var/lib/apt/lists/*

VOLUME ["/var/lib/postgresql/data"]

EXPOSE 5432

HEALTHCHECK --start-interval=1s --start-period=5s --interval=5s --retries=5 CMD pg_isready -U ${POSTGRES_USER}
