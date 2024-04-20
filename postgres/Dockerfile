ARG ARCHITECTURE=arm64v8
FROM ${ARCHITECTURE}/postgres:16-alpine3.18

LABEL maintainer="tnakagami"
LABEL description="build postgres environment"

ENV POSTGRES_USER= \
    POSTGRES_PASSWORD= \
    POSTGRES_DB= \
    POSTGRES_PORT=5432 \
    PGDATA=/var/lib/postgresql/data

VOLUME ["/var/lib/postgresql/data"]

EXPOSE 5432

HEALTHCHECK --start-interval=1s --start-period=5s --interval=5s --retries=5 CMD pg_isready -U ${POSTGRES_USER}