import os
import sys
import time
from pathlib import Path
from box import Box
from typing import Optional

class TeeLogger(object):
    """
    Hijacks stdout/stderr to output to both terminal and file simultaneously.
    """
    def __init__(self, filename, stream):
        self.terminal = stream
        self.log_file = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

def setup_run(config_path: str, output_dir_suffix: Optional[str] = None):
    """
    Initialize a temporal experiment run directory structure.

    Args:
        config_path (str): Path to the config file (e.g., .../Exp00_Baseline/config.py)
        output_dir_suffix (Optional[str]): Suffix appended to the timestamped output directory name,
                            e.g., '20251210_222910__grid_loss_10e5'.
                            If None, only the timestamp is used for the directory name.
    Returns:
        Box: Object containing three key paths {log, ckpt, viz}
    """
    # 1. Resolve base path
    config_file = Path(config_path).resolve()
    exp_dir = config_file.parent  # e.g., .../Exp00_Baseline

    # 2. Generate timestamp ID
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # 3. Create the main run directory
    # e.g., .../Exp00_Baseline/output/20231027_123045
    run_root = exp_dir / "output" / f"{timestamp}{f'__{output_dir_suffix}' if output_dir_suffix else ''}"

    # 4. Create three sub-directories (singular naming)
    log_dir = run_root / "log"
    ckpt_dir = run_root / "ckpt"
    viz_dir = run_root / "viz"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    print(f"✅ [Logger] Experiment Initialized: {timestamp}")
    print(f"   - Log : {log_dir}")
    print(f"   - Ckpt: {ckpt_dir}")
    print(f"   - Viz : {viz_dir}")

    # 5. Activate log hijacking (Console Log)
    log_file_path = log_dir / "console.log"
    sys.stdout = TeeLogger(str(log_file_path), sys.stdout)
    sys.stderr = TeeLogger(str(log_file_path), sys.stderr)

    # 6. Return path package (converted to strings for convenient subsequent use)
    return Box({
        "root": str(run_root),
        "log": str(log_dir),
        "ckpt": str(ckpt_dir),
        "viz": str(viz_dir)
    })