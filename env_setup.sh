#!/usr/bin/env bash
set -euo pipefail

# 1. Install uv if not already installed
if ! command -v uv &> /dev/null; then
  echo "🔽 Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$PATH"
fi

# 2. Create virtual environment with Python 3.11 using uv
echo "🐍 Creating virtual environment with Python 3.11..."
uv venv .venv --python 3.11

# 3. Activate venv
source .venv/bin/activate

# 4. Install dependencies (like poetry install)
echo "📦 Installing dependencies with uv sync..."
uv sync

echo "✅ Setup complete! Virtual environment is at: $(pwd)/.venv"
echo "👉 To activate it later, run: source .venv/bin/activate"
