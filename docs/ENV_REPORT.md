# KathaVaani — Phase 0 Environment & Hardware Audit Report

**Audit Timestamp**: 2026-11-19 01:00 IST  
**Cluster Architecture**: Bright Computing / Base Command Manager (BCM) 11 on Ubuntu 24.04.1 LTS  
**Master Node**: `bmu-headnode` (Linux 6.8.0-51-generic x86_64)  
**Compute Worker Node**: `bmu-worker` (Linux 6.8.0-87-generic x86_64, 224 vCPUs, 2.1 TB RAM)  

---

## 1. Hardware & GPU Specifications

| Parameter | Host Specification | User Allocated Quota |
| :--- | :--- | :--- |
| **GPU Model** | NVIDIA H200 Tensor Core GPU | **1x `nvidia.com/mig-1g.18gb`** |
| **VRAM Slice** | 141 GB HBM3e (Full Physical GPU) | **18 GB VRAM** (`mig-1g.18gb` profile) |
| **Driver / CUDA** | Driver `580.105.08` / CUDA `13.0` | CUDA 12.4 / 13.0 compatible |
| **MIG Configuration** | Active (`GI: 9, CI: 0, Dev: 0`) | `16384 MiB` allocated addressable memory |
| **Compute / Memory Quota** | 224 CPUs / 2 TB RAM total | **16 vCPUs / 32 GiB RAM** limit |
| **Namespace** | Cluster-level | `dgx-s-bmu-soet-230737-restricted` |

### GPU Verification Output
```
Wed Nov 18 19:24:21 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA H200                    On  |   00000000:43:00.0 Off |                   On |
|                                         |                        |              Enabled |
+-----------------------------------------+------------------------+----------------------+
| MIG devices:                                                                            |
| GPU  GI  CI  MIG |              Shared Memory-Usage | SM     Unc| CE ENC  DEC  OFA  JPG |
|  0    9   0   0  |              15MiB / 16384MiB    | 16      0 |  1   0    1    0    1 |
+------------------+----------------------------------+-----------+-----------------------+
```

### PyTorch Runtime Verification
- Container Image: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime` (pre-pulled on compute node)
- Output: `PyTorch: 2.4.0 CUDA available: True Device: NVIDIA H200 MIG 1g.18gb`

---

## 2. Cluster Scheduler & Workload Orchestration

- **Primary Orchestrator**: Kubernetes `v1.32.11` via `kubectl`.
- **Admission Webhook**: Kyverno `validate.kyverno.svc-fail` policy `require-gpu`.
  - Every job in namespace `dgx-s-bmu-soet-230737-restricted` must specify GPU limits (`limits: { nvidia.com/mig-1g.18gb: 1 }`). Non-GPU pods are rejected by admission control.
- **Storage Mounting**: Direct `hostPath` mount `/home/dgx-s-bmu-soet-230737` into `/workspace` inside GPU pods is fully operational and shared between headnode and compute worker.
- **Batch & Interactive Execution**: Supported via `scripts/run_gpu.sh` which launches transient or long-running Kubernetes pods/jobs mounting workspace files and streaming logs to standard output.
- **SLURM Compatibility**: Slurm client binaries (`/cm/local/apps/slurm/25.05/bin/sinfo`) and templates (`slurm/*.sbatch`) are documented and provided for cross-cluster portability.

---

## 3. Storage Hierarchy & Cache Strategy

| Mount Point | Total Size | Available | Status / Strategy |
| :--- | :--- | :--- | :--- |
| `/dev/sdb6` (`/` & `/home`) | 5.3 TB | **7.4 GB** (100% full) | **Constrained**: No large model weights or datasets stored here. |
| `/dev/sdb5` (`/var`) | 180 GB | **165 GB** (9% used) | **Optimal Large Storage**: Allocated `/var/tmp/dgx-s-bmu-soet-230737`. |
| `/dev/sdb4` (`/tmp`) | 20 GB | **19 GB** | Used for transient batch scratch files. |
| `tmpfs` (`/dev/shm`) | 63 GB | **63 GB** | Available for PyTorch multi-worker DataLoader shared memory. |

### Configured Cache Links
- `kathavaani/.cache/huggingface` $\to$ `/var/tmp/dgx-s-bmu-soet-230737/hf_cache`
- `kathavaani/.cache/torch` $\to$ `/var/tmp/dgx-s-bmu-soet-230737/torch_cache`
- `kathavaani/models_cache` $\to$ `/var/tmp/dgx-s-bmu-soet-230737/models`
- `kathavaani/checkpoints` $\to$ `/var/tmp/dgx-s-bmu-soet-230737/checkpoints`
- `kathavaani/data_storage` $\to$ `/var/tmp/dgx-s-bmu-soet-230737/datasets`
- `kathavaani/.venv` $\to$ `/var/tmp/dgx-s-bmu-soet-230737/venv`

---

## 4. Software Toolchain

- **Package Manager**: `uv 0.13.9` installed in `/var/tmp/dgx-s-bmu-soet-230737/bin`.
- **Python**: Python 3.10.22 isolated venv in `/var/tmp/dgx-s-bmu-soet-230737/venv`.
- **FFmpeg**: Static release `ffmpeg` and `ffprobe` in `/var/tmp/dgx-s-bmu-soet-230737/bin`.
- **Version Control**: Git 2.43.0 initialized.
- **Network**: Direct outbound HTTPS connectivity verified (Hugging Face, GitHub, PyPI HTTP 200).
