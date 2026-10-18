#!/usr/bin/env bash
# Runs ON the Nebius GPU VM. Starts every DC-Ops training job in parallel on the one GPU,
# mirrors logs to the Hub every 10 minutes, and powers the VM off when all jobs end
# (hard cap: 10 hours, so credits cannot run away).
#
# Expects: ~/dcops/jobs/*.py and ~/.hf_token (copied by launch_nebius.sh).
set -uo pipefail
cd ~/dcops
export HF_TOKEN="$(cat ~/.hf_token)"
export PATH="$HOME/.local/bin:$PATH"
LOGS=~/dcops/logs; mkdir -p "$LOGS"

sudo shutdown -h +600 "dcops: 10 hour cap" || true

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
nvidia-smi | tee "$LOGS/nvidia-smi.txt"

start() {  # name script args...
    local name="$1" script="$2"; shift 2
    nohup uv run "jobs/$script" "$@" > "$LOGS/$name.log" 2>&1 &
    echo $! > "$LOGS/$name.pid"
    echo "$(date +%T) started $name (pid $!)"
}

start teacher        teacher.py
start s_r160_kd      student.py --name r160_kd     --res 160 --kd-alpha 0.7
start s_r128_kd      student.py --name r128_kd     --res 128 --kd-alpha 0.7
start s_r192_kd      student.py --name r192_kd     --res 192 --kd-alpha 0.7
start s_r160_nokd    student.py --name r160_nokd   --res 160 --kd-alpha 0
start s_r160_noaug   student.py --name r160_noaug  --res 160 --kd-alpha 0.7 --domain-aug 0
start s_r160_kd_e120 student.py --name r160_kd_e120 --res 160 --kd-alpha 0.7 --epochs 120
start kws_s_kd       kws.py --name dscnn_s_kd --ch 64  --kd-alpha 0.5 --data /tmp/sc_s
start kws_m_kd       kws.py --name dscnn_m_kd --ch 128 --kd-alpha 0.5 --data /tmp/sc_m

upload_logs() {
    uv run --with huggingface_hub python - <<'PY' || true
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo("abhijitbetigeri/dcops-e8-student", private=True, exist_ok=True)
api.upload_folder(folder_path=os.path.expanduser("~/dcops/logs"), path_in_repo="logs",
                  repo_id="abhijitbetigeri/dcops-e8-student", commit_message="logs")
PY
}

while true; do
    sleep 600
    { date; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; } > "$LOGS/status.txt"
    running=0
    for f in "$LOGS"/*.pid; do kill -0 "$(cat "$f")" 2>/dev/null && running=$((running+1)); done
    echo "running jobs: $running" >> "$LOGS/status.txt"
    upload_logs
    if [ "$running" -eq 0 ]; then
        echo "$(date +%T) all jobs finished" >> "$LOGS/status.txt"; upload_logs
        sudo shutdown -h now
    fi
done
