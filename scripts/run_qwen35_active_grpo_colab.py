"""Colab launcher for the Qwen3.5-9B active-bank GRPO experiment."""

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


PROMPT_SUFFIX = (
    "Return only the final answer in \\boxed{...}. "
    "No explanation. No text after the box. /no_think"
)
CHAT_KWARGS = '{"enable_thinking": false}'


def run(cmd, cwd=None):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    ret = proc.wait()
    if ret:
        raise subprocess.CalledProcessError(ret, cmd)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def compact_one(summary_path):
    d = load_json(summary_path)
    variant_name, variant = next(iter(d["variants"].items()))
    return {
        "summary_exists": summary_path.exists(),
        "bank_exists": (summary_path.parent / "active_bank.json").exists(),
        "target_modules_exists": (summary_path.parent / "target_modules.json").exists(),
        "map_params_exists": (summary_path.parent / "map_params" / "Map-G2048_active_o.json").exists(),
        "bank_summary": d["bank_summary"],
        "baseline_eval": d["baseline_eval"],
        "config": d["config"],
        "variant_name": variant_name,
        "variant": {
            "updates": variant["updates"],
            "skipped": variant["skipped"],
            "pass": variant["pass"],
            "trainable_params": variant["trainable_params"],
            "checkpoint_bytes_est": variant["checkpoint_bytes_est"],
            "mean_step_s": variant["mean_step_s"],
            "tokens_per_s": variant["tokens_per_s"],
            "peak_alloc_gb": variant["peak_alloc_gb"],
            "elapsed_train_s": variant["elapsed_train_s"],
            "eval": variant["eval"],
            "best_correct_mean": variant["best_correct_mean"],
            "final_correct_mean": variant["final_correct_mean"],
        },
    }


def write_multiseed(out_root, seeds):
    payload = {"model": "Qwen/Qwen3.5-9B", "task": "MATH-500 level1-3 answer_only", "seeds": seeds}
    for seed in seeds:
        map_dir = out_root / f"seed{seed}-map"
        lora_dir = out_root / f"seed{seed}-lora"
        payload[str(seed)] = {
            "map": compact_one(map_dir / "active_train_summary.json"),
            "lora": compact_one(lora_dir / "active_train_summary.json"),
        }
    (out_root / "qwen35-multiseed-summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="/content/mapping-networks-src.tar.gz")
    ap.add_argument("--workdir", default="/content/mapping-networks")
    ap.add_argument("--out-root", default="results/9b-math500/qwen35-active-grpo-20260707")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--eval-baseline-seed", type=int, default=0)
    ap.add_argument("--target-updates", default="30")
    ap.add_argument("--max-attempts", default="120")
    ap.add_argument("--candidate-n", default="50")
    ap.add_argument("--probe-k", default="8")
    ap.add_argument("--max-new", default="64")
    ap.add_argument("--max-new-eval", default="128")
    ap.add_argument("--eval-n", default="200")
    ap.add_argument("--train-batch", default="1")
    ap.add_argument("--beta-kl", default="0.05")
    ap.add_argument("--stdout-artifact", action="store_true")
    ap.add_argument("--stdout-artifact-max-mb", type=float, default=256.0)
    args, _ = ap.parse_known_args()

    archive = Path(args.archive)
    workdir = Path(args.workdir)
    if not archive.exists():
        raise FileNotFoundError(f"missing repo archive: {archive}")
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(workdir)

    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--upgrade",
        "transformers",
        "datasets",
        "accelerate",
        "matplotlib",
        "sentencepiece",
    ], cwd=workdir)
    run([
        sys.executable,
        "-c",
        (
            "import torch\n"
            "print('torch', torch.__version__)\n"
            "print('cuda', torch.cuda.is_available())\n"
            "print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)\n"
            "print('mem_gb', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2) if torch.cuda.is_available() else None)\n"
        ),
    ], cwd=workdir)

    out_root = workdir / args.out_root
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    base_cmd = [
        sys.executable,
        "experiments/math500_active_grpo_9b.py",
        "--model",
        "Qwen/Qwen3.5-9B",
        "--dtype",
        "bf16",
        "--min-level",
        "1",
        "--max-level",
        "3",
        "--candidate-n",
        args.candidate_n,
        "--probe-k",
        args.probe_k,
        "--K",
        "8",
        "--max-new",
        args.max_new,
        "--max-new-eval",
        args.max_new_eval,
        "--eval-n",
        args.eval_n,
        "--eval-batch",
        "1",
        "--target-updates",
        args.target_updates,
        "--max-attempts",
        args.max_attempts,
        "--train-batch",
        args.train_batch,
        "--beta-kl",
        args.beta_kl,
        "--prompt-suffix",
        PROMPT_SUFFIX,
        "--chat-template-kwargs",
        CHAT_KWARGS,
        "--eval-after-train",
    ]
    for seed in seeds:
        map_cmd = base_cmd + [
            "--seed",
            str(seed),
            "--variants",
            "map",
            "--out-dir",
            str(out_root / f"seed{seed}-map"),
        ]
        if seed != args.eval_baseline_seed:
            map_cmd.append("--skip-baseline-eval")
        run(map_cmd, cwd=workdir)
        bank = out_root / f"seed{seed}-map" / "active_bank.json"
        lora_cmd = base_cmd + [
            "--seed",
            str(seed),
            "--variants",
            "lora",
            "--skip-baseline-eval",
            "--active-bank-json",
            str(bank),
            "--out-dir",
            str(out_root / f"seed{seed}-lora"),
        ]
        run(lora_cmd, cwd=workdir)

    write_multiseed(out_root, seeds)
    artifacts = Path("/content/qwen35-active-grpo-artifacts.tar.gz")
    with tarfile.open(artifacts, "w:gz") as tf:
        tf.add(out_root, arcname=args.out_root)
    print(f"\nARTIFACTS={artifacts}", flush=True)
    print(f"ARTIFACT_BYTES={artifacts.stat().st_size}", flush=True)
    if args.stdout_artifact and artifacts.stat().st_size <= int(args.stdout_artifact_max_mb * 1024 * 1024):
        b64 = base64.b64encode(artifacts.read_bytes()).decode()
        print("ARTIFACT_TAR_GZ_B64_BEGIN", flush=True)
        for i in range(0, len(b64), 4096):
            print(b64[i:i + 4096], flush=True)
        print("ARTIFACT_TAR_GZ_B64_END", flush=True)


if __name__ == "__main__":
    main()
