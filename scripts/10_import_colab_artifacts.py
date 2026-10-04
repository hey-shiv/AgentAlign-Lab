#!/usr/bin/env python3
"""Script 10: Import GPU artifacts back from Kaggle/Colab.

Usage:
    python scripts/10_import_colab_artifacts.py --zip-path /path/to/agentalign_results.zip
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Import GPU artifacts")
    parser.add_argument("--zip-path", required=True, help="Path to downloaded zip")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    if not zip_path.exists():
        print(f"ERROR: {zip_path} not found")
        return

    # Extract to temp dir
    extract_dir = Path("_import_tmp")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # Copy artifacts to project
    artifacts = [
        "outputs/adapters/qwen_dpo_final",
        "outputs/adapters/qwen_sft_final",
        "outputs/evals",
        "data/preferences/dpo_llm_train.jsonl",
        "data/trajectories/scored_llm_train",
        "data/trajectories/scored_llm_test",
        "runs/llm_train",
        "runs/llm_test",
    ]

    imported = 0
    for artifact in artifacts:
        src = extract_dir / artifact
        dst = Path(artifact)

        if not src.exists():
            print(f"  SKIP (not in zip): {artifact}")
            continue

        if dst.exists() and not args.force:
            print(f"  SKIP (exists, use --force): {artifact}")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

        imported += 1
        print(f"  {artifact}")

    # Cleanup
    shutil.rmtree(extract_dir)

    print(f"\nImported {imported} artifacts from {zip_path}")

    # Show summary if eval results exist
    eval_path = Path("outputs/evals/eval_results.json")
    if eval_path.exists():
        results = json.loads(eval_path.read_text())
        print("\n--- Evaluation Summary ---")
        for agent in ["qwen_base", "qwen_sft", "qwen_dpo"]:
            if agent in results:
                m = results[agent]
                print(f"  {agent}: {m['pass_rate']*100:.1f}% pass rate "
                      f"({m['passed']}/{m['total']})")
        for key in ["dpo_vs_base", "sft_vs_base", "dpo_vs_sft"]:
            if key in results:
                r = results[key]
                print(f"  {key}: {r['diff']:+.4f} "
                      f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]")


if __name__ == "__main__":
    main()
