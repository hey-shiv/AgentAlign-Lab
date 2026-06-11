import argparse
from pathlib import Path

from agentalign.data.trajectories import load_trajectories, save_trajectory
from agentalign.verifier.score import calculate_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="data/trajectories/raw")
    parser.add_argument("--out-dir", type=str, default="data/trajectories/scored")
    parser.add_argument("--clear", action="store_true", help="Remove existing scored JSONL files first.")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.clear:
        for path in out_dir.glob("*.jsonl"):
            path.unlink()

    if not runs_dir.exists():
        print(f"Directory {runs_dir} does not exist.")
        return

    for path in sorted(runs_dir.glob("*.jsonl")):
        print(f"Scoring trajectories in {path.name}...")
        trajectories = load_trajectories(path)
        out_path = out_dir / path.name

        # Clear out file
        if out_path.exists():
            out_path.unlink()

        for traj in trajectories:
            task_path = Path(f"data/tasks/{traj.task_id}.json")
            if not task_path.exists():
                print(f"Task {traj.task_id} not found, skipping trajectory {traj.run_id}.")
                continue

            # Since we can't easily rebuild the workspace state at the end of the trajectory
            # without replaying, and we just want to update scores based on logic,
            # we just recalculate the composite score from the existing verifier pass/fail.
            # If we wanted to re-verify, we'd have to replay the actions.

            # Recalculate score from existing verifier outcome
            calculate_score(traj)
            save_trajectory(traj, out_path)

    print("Scoring complete.")

if __name__ == "__main__":
    main()
