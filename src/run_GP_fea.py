import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def run_one(seed: int, workdir: Path):
    cmd = [
        sys.executable,
        "GP_Active_learning_fea.py",
        "--random_state",
        str(seed)
    ]
    proc = subprocess.run(cmd, cwd=str(workdir), capture_output=True, text=True)
    return seed, proc.returncode, proc.stderr[-300:]


def main():
    workdir = Path(__file__).resolve().parent
    tasks = [seed for seed in range(30, 50)]
    max_workers = 1

    print(f"Start {len(tasks)} tasks with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_one, seed, workdir) for seed in tasks]
        for future in as_completed(futures):
            seed, rc, err_tail = future.result()
            if rc == 0:
                print(f"[OK] seed={seed}")
            else:
                print(f"[FAIL] seed={seed}, rc={rc}")
                if err_tail:
                    print(err_tail)

    print("All tasks finished.")


if __name__ == "__main__":
    main()
