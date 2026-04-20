"""
Generate combined replay GIFs for all runs in the results folder.

Skips runs that fail (e.g. missing data, old format) and prints a summary.

Usage:
    python experiments/commonroad/generate_all_gifs.py
    python experiments/commonroad/generate_all_gifs.py --fps 30
    python experiments/commonroad/generate_all_gifs.py --force  # overwrite existing
"""

import os
import sys
import argparse
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
REPLAY_SCRIPT = os.path.join(SCRIPT_DIR, "replay_combined.py")
REPO_ROOT = os.path.join(SCRIPT_DIR, "..", "..")


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate combined replay GIFs for all runs")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing GIFs")
    return p.parse_args()


def main():
    args = parse_args()

    runs = []
    for name in sorted(os.listdir(RESULTS_DIR)):
        d = os.path.join(RESULTS_DIR, name)
        if os.path.isfile(os.path.join(d, "episode.json")):
            runs.append((name, d))

    if not runs:
        print("No runs found in results/")
        return

    print(f"Found {len(runs)} runs\n")

    succeeded, skipped, failed = [], [], []

    for name, run_dir in runs:
        gif_path = os.path.join(run_dir, "combined_replay.gif")

        if os.path.exists(gif_path) and not args.force:
            skipped.append(name)
            print(f"  SKIP  {name}")
            continue

        print(f"  GEN   {name} ...", end="", flush=True)
        try:
            result = subprocess.run(
                [sys.executable, REPLAY_SCRIPT, run_dir,
                 "-o", gif_path,
                 "--fps", str(args.fps),
                 "--dpi", str(args.dpi)],
                capture_output=True, text=True,
                cwd=REPO_ROOT,
                timeout=120,
            )
            if result.returncode == 0:
                succeeded.append(name)
                print("  OK")
            else:
                failed.append((name, result.stderr[-300:]))
                print("  FAIL")
                print(f"        {result.stderr[-200:]}")
        except subprocess.TimeoutExpired:
            failed.append((name, "timeout"))
            print("  TIMEOUT")
        except Exception as e:
            failed.append((name, str(e)))
            print(f"  ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"  Generated: {len(succeeded)}")
    print(f"  Skipped:   {len(skipped)}  (use --force to overwrite)")
    print(f"  Failed:    {len(failed)}")
    if failed:
        print(f"\n  Failed runs:")
        for name, err in failed:
            print(f"    - {name}: {err[:100]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
