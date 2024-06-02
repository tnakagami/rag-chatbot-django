#!/bin/bash

is_running=1

# Setup handler
handler(){
  echo sigterm accepted

  is_running=0
}
trap handler 1 2 3 15

# Create pid directory and log directory
readonly WORKDIR=/opt/app
readonly CELERY_ROOT_DIR=/opt/home/celery
readonly PID_DIR=${CELERY_ROOT_DIR}/run
readonly LOG_DIR=${CELERY_ROOT_DIR}/log
mkdir -p ${PID_DIR}
mkdir -p ${LOG_DIR}

# =============
# = Main loop =
# =============
celery multi start \
       --app=config --workdir=${WORKDIR} \
       worker -l INFO \
       --pidfile="${PID_DIR}/celeryd-%n.pid" \
       --logfile="${LOG_DIR}/celeryd-%n%I.log"

while [ ${is_running} -eq 1 ]; do
  sleep 1
done

# Finalize
celery multi stop worker \
       --pidfile="${PID_DIR}/celeryd-%n.pid" \
       --logfile="${LOG_DIR}/celeryd-%n%I.log"