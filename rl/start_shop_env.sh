#!/usr/bin/env bash
set -euo pipefail

GRPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${GRPO_DIR}/.." && pwd)
SHOP_SIMULATOR_ROOT=${SHOP_SIMULATOR_ROOT:-${PROJECT_ROOT}/shopSimulator}
SHOP_PYTHON=${SHOP_PYTHON:-/root/miniconda3/envs/shopsim/bin/python}

if curl -fsS --max-time 2 -X POST http://127.0.0.1:5000/api/shop_agent \
  -H 'Content-Type: application/json' -d '{"action":"release_all"}' >/dev/null; then
  echo "ShopSimulator is already available at http://127.0.0.1:5000 (20 slots reset)."
  exit 0
fi

cd "${SHOP_SIMULATOR_ROOT}/shop_env/shop_env"
export PYTHONPATH="${SHOP_SIMULATOR_ROOT}/shop_env${PYTHONPATH:+:${PYTHONPATH}}"
export SHOPSIM_ENV_MAX_NUM=20
exec "${SHOP_PYTHON}" pack_api.py
