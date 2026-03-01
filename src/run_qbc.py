import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def run_one(seed: int, label_idx: int, workdir: Path):
    cmd = [
        sys.executable,
        "qbc_Active_learning.py",
        "--random_state",
        str(seed),
        "--label_idx",
        str(label_idx),
    ]
    proc = subprocess.run(cmd, cwd=str(workdir), capture_output=True, text=True)
    return seed, label_idx, proc.returncode, proc.stderr[-300:]


def main():
    workdir = Path(__file__).resolve().parent
    tasks = [(seed, label_idx) for seed in range(40, 50) for label_idx in range(3)]
    max_workers = 1

    print(f"Start {len(tasks)} tasks with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_one, seed, label_idx, workdir) for seed, label_idx in tasks]
        for future in as_completed(futures):
            seed, label_idx, rc, err_tail = future.result()
            if rc == 0:
                print(f"[OK] seed={seed}, label_idx={label_idx}")
            else:
                print(f"[FAIL] seed={seed}, label_idx={label_idx}, rc={rc}")
                if err_tail:
                    print(err_tail)

    print("All tasks finished.")


if __name__ == "__main__":
    main()
