#!/bin/bash
set -euo pipefail

# KathaVaani Fast GPU Runner
# Executes commands directly inside the persistent H200 GPU worker pod
POD_NAME="kathavaani-worker"
NAMESPACE="dgx-s-bmu-soet-230737-restricted"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <command to run inside GPU container>"
    echo "Example: $0 python3 kathavaani/tts/bakeoff.py"
    exit 1
fi

# Ensure worker pod is running
STATUS=$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "NotFound")
if [ "$STATUS" != "Running" ]; then
    echo "Worker pod $POD_NAME is not running (status: $STATUS). Starting it..."
    kubectl apply -f /home/dgx-s-bmu-soet-230737/kathavaani/k8s/worker-pod.yaml
    kubectl wait --for=condition=Ready pod/"$POD_NAME" -n "$NAMESPACE" --timeout=60s
fi

# Ensure GPU environment is initialized inside the pod
if ! kubectl exec "$POD_NAME" -n "$NAMESPACE" -- test -f /tmp/venv_gpu/bin/python 2>/dev/null; then
    echo "Initializing GPU environment in $POD_NAME..."
    kubectl exec "$POD_NAME" -n "$NAMESPACE" -- env HOME=/tmp /workspace/kathavaani/bin/uv venv --python /opt/conda/bin/python3 --system-site-packages /tmp/venv_gpu
    kubectl exec "$POD_NAME" -n "$NAMESPACE" -- env HOME=/tmp UV_CACHE_DIR=/tmp/uv_cache /workspace/kathavaani/bin/uv pip install --python /tmp/venv_gpu/bin/python soundfile openai-whisper TTS "transformers>=4.38.0,<4.40.0" torchvision accelerate speechbrain faiss-cpu pyloudnorm
fi

# Execute command inside worker pod with exact argument preservation
kubectl exec -i "$POD_NAME" -n "$NAMESPACE" -- env HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib COQUI_TOS_AGREED=1 PATH=/tmp/venv_gpu/bin:/workspace/kathavaani/bin:$PATH PYTHONPATH=/workspace/kathavaani HF_HOME=/tmp/hf_cache TORCH_HOME=/tmp/torch_cache "$@"
