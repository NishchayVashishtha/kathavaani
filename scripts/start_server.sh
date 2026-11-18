#!/usr/bin/env bash
# KathaVaani Web Studio Server Launch Script
set -e

NAMESPACE="dgx-s-bmu-soet-230737-restricted"
POD_NAME="kathavaani-worker"
PORT=8000

echo "=== Starting KathaVaani Studio Web Server ==="

# Check if pod is running
STATUS=$(kubectl get pod "${POD_NAME}" -n "${NAMESPACE}" -o jsonpath='{.status.phase}' 2>/dev/null || echo "NotFound")
if [ "${STATUS}" != "Running" ]; then
    echo "Worker pod ${POD_NAME} is not running. Starting it..."
    kubectl apply -f /home/dgx-s-bmu-soet-230737/kathavaani/k8s/worker-pod.yaml
    kubectl wait --for=condition=Ready pod/${POD_NAME} -n ${NAMESPACE} --timeout=60s
fi

echo "Starting FastAPI server inside worker pod..."
kubectl exec "${POD_NAME}" -n "${NAMESPACE}" -- bash -c "
pkill -f uvicorn || true
cd /workspace/kathavaani && \
nohup /tmp/venv_gpu/bin/uvicorn server.app:app --host 0.0.0.0 --port ${PORT} > logs/server.log 2>&1 &
"

sleep 2
echo "Establishing port-forwarding to localhost:${PORT}..."
# Kill any existing port-forward on port 8000
pkill -f "port-forward.*${PORT}" || true

kubectl port-forward pod/"${POD_NAME}" "${PORT}:${PORT}" -n "${NAMESPACE}" > logs/port_forward.log 2>&1 &
PF_PID=$!

echo "======================================================================"
echo " KathaVaani Studio is LIVE!"
echo " Web UI & REST API: http://localhost:${PORT}"
echo " Health Endpoint:   http://localhost:${PORT}/api/health"
echo " Voice Bank API:    http://localhost:${PORT}/api/voicebank"
echo " Port-Forward PID:  ${PF_PID}"
echo "======================================================================"
