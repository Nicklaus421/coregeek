#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
# PYTHONUNBUFFERED：stdout 被重定向到文件/管道时默认是块缓冲，
# 进程被 kill 会丢掉整块日志，这里强制实时输出。
exec env PYTHONUNBUFFERED=1 python3 main3.py "$1"
