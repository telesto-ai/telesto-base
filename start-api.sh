#!/usr/bin/env bash

exec gunicorn --log-level INFO --access-logfile - --workers 2 \
    --worker-class sync --timeout 60 --bind 0.0.0.0:9876 'telesto.app:app'
