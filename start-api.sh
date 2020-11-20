#!/usr/bin/env bash

model_type=${1:-"classification"}

exec gunicorn --log-level INFO --access-logfile - --workers 2 \
    --worker-class sync --timeout 60 --bind 0.0.0.0:9876 "telesto.app:get_app('${model_type}')"
