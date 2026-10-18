#!/usr/bin/env bash
# Launch the whole DC-Ops overnight run on HF Jobs GPUs, all in parallel.
#   NAMESPACE=iteratehack ./launch_overnight.sh     (org credits; needs a write/contributor role in the org)
#   NAMESPACE=abhijitbetigeri ./launch_overnight.sh (personal credits)
# The teacher builds crops + fine-tunes ConvNeXt; the 6 students wait for its crops on the Hub;
# the 2 keyword spotters are independent. Every job has a hard timeout. ~$15-25 total on A10G.
set -euo pipefail
cd "$(dirname "$0")"
HF="${HF:-/Users/abhijitbetigeri/OneRobotLoop/.venv/bin/hf}"
NS="${NAMESPACE:-iteratehack}"
.venv-train/bin/python -c "from huggingface_hub import HfApi; a=HfApi(); a.create_repo('abhijitbetigeri/dcops-e8-jobs', private=True, exist_ok=True); a.upload_folder(folder_path='jobs', repo_id='abhijitbetigeri/dcops-e8-jobs', allow_patterns=['*.py'])"
run() {  # label script timeout args...
    local label="$1" script="$2" timeout="$3"; shift 3
    "$HF" jobs run --namespace "$NS" --flavor "${FLAVOR:-a10g-small}" --timeout "$timeout" \
        --secrets HF_TOKEN -d -l "run=$label" ghcr.io/astral-sh/uv:python3.12-bookworm \
        bash -c "set -e; uv run --with huggingface_hub python -c \"from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('abhijitbetigeri/dcops-e8-jobs', '$script', force_download=True), '/tmp/$script')\" && uv run /tmp/$script $*" \
        | tee -a launched_jobs.log
}
run teacher        teacher.py 3h
run s_r160_kd      student.py 5h --name r160_kd     --res 160 --kd-alpha 0.7
run s_r128_kd      student.py 5h --name r128_kd     --res 128 --kd-alpha 0.7
run s_r192_kd      student.py 5h --name r192_kd     --res 192 --kd-alpha 0.7
run s_r160_nokd    student.py 5h --name r160_nokd   --res 160 --kd-alpha 0      # ablation: no distillation
run s_r160_noaug   student.py 5h --name r160_noaug  --res 160 --kd-alpha 0.7 --domain-aug 0  # ablation: no board-camera aug
run s_r160_kd_long student.py 6h --name r160_kd_e120 --res 160 --kd-alpha 0.7 --epochs 120
run kws_s_kd       kws.py     5h --name dscnn_s_kd   --ch 64  --kd-alpha 0.5
run kws_m_kd       kws.py     6h --name dscnn_m_kd   --ch 128 --kd-alpha 0.5
echo "launched; see launched_jobs.log and: $HF jobs ps --namespace $NS"
