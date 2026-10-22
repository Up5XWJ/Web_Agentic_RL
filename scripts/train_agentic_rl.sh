#!/usr/bin/env bash
set -euo pipefail

AGENTIC_RL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
AGENTIC_RL_PYTHON="${OPENAPPS_RL_PYTHON:-${AGENTIC_RL_ROOT}/.venv-rl/bin/python}"
if [[ ! -x "${AGENTIC_RL_PYTHON}" ]]; then
  echo "Set OPENAPPS_RL_PYTHON to the Python executable in your separate GPU environment." >&2
  exit 1
fi
export PYTHONPATH="${AGENTIC_RL_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
cd -- "${AGENTIC_RL_ROOT}"
exec "${AGENTIC_RL_PYTHON}" -m open_apps.agentic_rl.cli train "$@"
