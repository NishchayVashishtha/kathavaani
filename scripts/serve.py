#!/usr/bin/env python3
"""
KathaVaani Persistent Studio Server Launcher
Spawns uvicorn in an independent process group (start_new_session=True)
so it survives shell session termination.
"""

import subprocess
import sys
import os
import time

def start():
    log_dir = "/home/dgx-s-bmu-soet-230737/kathavaani/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "server.log")
    pid_file = os.path.join(log_dir, "server.pid")

    # Stop any existing server
    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r") as f:
                old_pid = int(f.read().strip())
            os.kill(old_pid, 9)
            print(f"Stopped old server PID {old_pid}")
        except Exception:
            pass

    out_fd = open(log_file, "a")
    proc = subprocess.Popen(
        ["/var/tmp/dgx-s-bmu-soet-230737/venv/bin/uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8888"],
        cwd="/home/dgx-s-bmu-soet-230737/kathavaani",
        stdout=out_fd,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    with open(pid_file, "w") as f:
        f.write(str(proc.pid))

    print(f"KathaVaani Studio Server started with PID {proc.pid} on port 8888.")
    time.sleep(2)

if __name__ == "__main__":
    start()
