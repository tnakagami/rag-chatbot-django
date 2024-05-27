ARG ARCHITECTURE=arm64v8
FROM ${ARCHITECTURE}/redis:7-alpine

LABEL maintainer="tnakagami"
LABEL description="build redis environment"

# To store redis's config file
RUN    mkdir -p /usr/local/etc/redis \
    && mkdir -p /var/log/redis

EXPOSE 6379