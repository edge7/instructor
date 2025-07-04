#!/usr/bin/env bash
set -euo pipefail

# Ensure uv is installed
if ! command -v uv >/dev/null 2>&1; then
    pip install uv
fi

# Install dependencies using uv
uv pip install --system -r requirements.txt
uv pip install --system -r requirements-doc.txt

# Ensure newly installed executables are available
pyenv rehash

# Build the documentation
mkdocs build
