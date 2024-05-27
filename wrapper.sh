#!/bin/bash

readonly base_dir=$(cd $(dirname $0) && pwd)

function Usage() {
    echo $0 "build|start|down|stop|ps|logs"
    echo "  Options:"
    echo "    * build"
    echo "            build docker image"
    echo "    * start"
    echo "            create containers and start them"
    echo "    * down"
    echo "            stop and destroy containers"
    echo "    * stop"
    echo "            stop containers"
    echo "    * ps"
    echo "            show process status for each containers"
    echo "    * logs [service name]"
    echo "            show log for target service or each container"
}

while [ -n "$1" ]; do
  case "$1" in
    help | -h )
      Usage
      exit 0
      ;;

    build )
      # build
      docker-compose build --no-cache --progress=plain
      # remove old images
      docker images | grep none | awk '{print $3;}' | xargs -I{} docker rmi {}
      shift
      ;;

    start )
      docker-compose up -d
      shift
      ;;

    down | stop | ps )
      docker-compose "$1"
      shift
      ;;

    logs )
      target_service="$2"

      if [ -n "${target_service}" ]; then
        docker-compose logs ${target_service}
        shift
      else
        docker-compose logs database
        docker-compose logs backend
        docker-compose logs celery
        docker-compose logs redis
      fi
      shift
      ;;

    *)
      shift
      ;;
  esac
done
