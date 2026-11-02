#!/usr/bin/env bash
# AgentFlight Kaggle session bootstrap (build-prompt section 9).
# Runs at the start of EVERY Kaggle session: sessions start from a clean
# image, so this installs pinned deps (~5-10 min) and records the exact
# versions into results/ENVIRONMENT.json.
set -euo pipefail

PY=${PYTHON:-python}
echo "== AgentFlight bootstrap: $(date -Is) =="

# 1. System sanity (day-0 checklist head)
nvidia-smi || echo "WARNING: nvidia-smi failed — not a GPU session?"
$PY -c "import torch; print('torch', torch.__version__, 'capability', torch.cuda.get_device_capability(0))"

# 2. Pinned deps. vllm/lmcache versions get pinned at first successful E0 and
#    then frozen here — until then this installs known-T4-compatible releases.
$PY -m pip install -q "vllm==0.11.0" "lmcache" "huggingface_hub" || \
  echo "WARNING: pinned vllm install failed — record the working version in ENVIRONMENT.json and update this line"

# 3. AgentFlight itself (from the cloned repo)
cd /kaggle/working/agentflight
$PY -m pip install -q -e ".[gpu,report]"

# 4. Clone/pull latest results from the HF dataset repo (resume state)
if [ -n "${HF_TOKEN:-}" ]; then
  $PY -c "
from huggingface_hub import snapshot_download
import os
try:
    snapshot_download('agentflight/agentflight-results', repo_type='dataset', local_dir='/kaggle/working/results_prev', token=os.environ['HF_TOKEN'])
    print('previous results restored')
except Exception as e:
    print('no previous results (first session?):', e)
"
fi

echo "== bootstrap done =="
