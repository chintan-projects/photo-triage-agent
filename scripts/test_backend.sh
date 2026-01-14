#!/bin/bash
cd "$(dirname "$0")/../backend"
uv run pytest "${@:---v}"
