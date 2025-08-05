#!/usr/bin/env bash
set -euo pipefail

# Initialize conda in this shell (works for bash/zsh)
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "Cannot find conda initialization script; ensure conda is installed." >&2
  exit 1
fi

# Create or update environment
if conda env list | grep -q "^ttct_full[[:space:]]"; then
  echo "Updating existing ttct_full environment..."
  conda env update -f environment.yml
else
  echo "Creating ttct_full environment..."
  conda env create -f environment.yml
fi

# Activate it
conda activate ttct_full

# (Optional) Ensure backend requirements are satisfied if you keep a separate requirements.txt
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
fi

echo ""
echo "✅ Environment ready."
echo "To run the backend:"
echo "  conda activate ttct_full"
echo "  cd ttct_backend"
echo "  python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"
