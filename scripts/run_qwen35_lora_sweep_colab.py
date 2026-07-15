"""Colab launcher: Qwen3.5-9B Map + LoRA lr/rank sweep on one shared active bank.

Run with: colab run --gpu G4 --timeout 86400 scripts/run_qwen35_lora_sweep_colab.py

GPU-occupancy strategy:
  * One Colab VM builds the active bank once (Map variant) and reuses it for
    every LoRA variant via --active-bank-json. No bank rebuild per variant.
  * All variants run sequentially in the same VM -> single GPU rental.
  * --stdout-artifact base64-encodes the result tarball to stdout so the run is
    recoverable even if the Colab VM is released before manual download.
  * --time-budget-s bounds each variant so the sweep cannot blow past a cap.
"""

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path


PROMPT_SUFFIX = (
    "Return only the final answer in \\boxed{...}. "
    "No explanation. No text after the box. /no_think"
)
CHAT_KWARGS = '{"enable_thinking": false}'

# (label, lora_r, lora_lr)
DEFAULT_LORA_VARIANTS = [
    ("r8-lr3e-5", 8, 3e-5),
    ("r8-lr1e-4", 8, 1e-4),
    ("r8-lr3e-4", 8, 3e-4),
    ("r8-lr1e-3", 8, 1e-3),
    ("r4-lr1e-4", 4, 1e-4),
    ("r16-lr1e-4", 16, 1e-4),
]


def run(cmd, cwd=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
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


def compact(summary_path):
    d = load_json(summary_path)
    variant_name, variant = next(iter(d["variants"].items()))
    return {
        "variant_name": variant_name,
        "bank_summary": d.get("bank_summary"),
        "baseline_eval": d.get("baseline_eval"),
        "config": d.get("config"),
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


def heartbeat(msg, stop_event, interval=30):
    """Print a heartbeat every `interval`s until stop_event is set."""
    start = time.time()
    while not stop_event.wait(interval):
        print(f"[heartbeat] {msg} ({time.time()-start:.0f}s elapsed)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-url", default="https://github.com/cklxx/mapping-networks.git")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--workdir", default="/content/mapping-networks")
    ap.add_argument("--out-root", default="results/9b-math500/qwen35-lora-sweep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--candidate-n", default="100")
    ap.add_argument("--probe-k", default="8")
    ap.add_argument("--max-new", default="64")
    ap.add_argument("--max-new-eval", default="128")
    ap.add_argument("--eval-n", default="200")
    ap.add_argument("--train-batch", default="2")
    ap.add_argument("--micro-batch", default="2")
    ap.add_argument("--beta-kl", default="0.05")
    ap.add_argument("--target-updates", default="50")
    ap.add_argument("--max-attempts", default="200")
    ap.add_argument("--time-budget-s", default="1200")  # 20min per variant
    ap.add_argument("--lora-variants", default="")  # override: "r8-lr1e-4,r8-lr3e-4"
    ap.add_argument("--stdout-artifact", action="store_true")
    ap.add_argument("--stdout-artifact-max-mb", type=float, default=256.0)
    ap.add_argument("--hf-token", default="", help="HF token (colab run does not forward env vars)")
    args, _ = ap.parse_known_args()

    workdir = Path(args.workdir)
    if workdir.exists():
        shutil.rmtree(workdir)

    # Clone repo from GitHub (public repo, HTTPS).
    print(f"[setup] cloning {args.repo_url} ({args.branch}) -> {workdir}", flush=True)
    run(["git", "clone", "--depth", "1", "--branch", args.branch, args.repo_url, str(workdir)])

    # HF_XET_HIGH_PERFORMANCE enables high-performance transfer via Xet
    # (replaces deprecated HF_HUB_ENABLE_HF_TRANSFER). HF_TOKEN lifts the
    # anonymous rate limit. Set both before any download.
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    # Disable tqdm progress bars (they use \r and get lost in line-buffered
    # pipes). A heartbeat thread provides visible progress instead.
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    # HF_TOKEN must be passed via --hf-token (colab run does not forward
    # local env vars to the remote VM). Set it before any download so the
    # subprocess inherits it.
    _hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    if _hf_token:
        os.environ["HF_TOKEN"] = _hf_token

    run([
        sys.executable, "-m", "pip", "install", "-q", "--upgrade",
        "transformers", "datasets", "accelerate", "matplotlib", "sentencepiece",
        "huggingface_hub",
    ], cwd=workdir)

    # Authenticate so downloads use the higher, authenticated rate limit.
    if _hf_token:
        run([
            sys.executable, "-c",
            f"from huggingface_hub import login; login(token='{_hf_token}')",
        ], cwd=workdir)

    # Pre-download the model with a heartbeat so progress is visible and the VM
    # stays alive (no idle reclaim) during the long download.
    stop_evt = threading.Event()
    hb = threading.Thread(target=heartbeat, args=("downloading model", stop_evt), daemon=True)
    hb.start()
    t0 = time.time()
    print("[download] starting snapshot_download ...", flush=True)
    run([
        sys.executable, "-c",
        (
            "from huggingface_hub import snapshot_download\n"
            "import os, time\n"
            "t0 = time.time()\n"
            "path = snapshot_download('Qwen/Qwen3.5-9B')\n"
            "total = sum(os.path.getsize(os.path.join(dp,f)) "
            "for dp,dn,fn in os.walk(path) for f in fn)\n"
            "print(f'[download] done: {total/1024**3:.2f} GB in {time.time()-t0:.0f}s at {path}', flush=True)\n"
        ),
    ], cwd=workdir)
    stop_evt.set()
    print(f"[download] total wall-clock: {time.time()-t0:.0f}s", flush=True)

    run([
        sys.executable, "-c",
        (
            "import torch\n"
            "print('torch', torch.__version__)\n"
            "print('cuda', torch.cuda.is_available())\n"
            "print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)\n"
            "print('mem_gb', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2) if torch.cuda.is_available() else None)\n"
        ),
    ], cwd=workdir)

    out_root = workdir / args.out_root

    base_cmd = [
        sys.executable, "experiments/math500_active_grpo_9b.py",
        "--model", "Qwen/Qwen3.5-9B",
        "--dtype", "bf16",
        "--min-level", "1", "--max-level", "3",
        "--candidate-n", args.candidate_n,
        "--probe-k", args.probe_k,
        "--K", "8",
        "--max-new", args.max_new,
        "--max-new-eval", args.max_new_eval,
        "--eval-n", args.eval_n,
        "--eval-batch", "1",
        "--target-updates", args.target_updates,
        "--max-attempts", args.max_attempts,
        "--train-batch", args.train_batch,
        "--micro-batch", args.micro_batch,
        "--beta-kl", args.beta_kl,
        "--time-budget-s", args.time_budget_s,
        "--prompt-suffix", PROMPT_SUFFIX,
        "--chat-template-kwargs", CHAT_KWARGS,
        "--eval-after-train",
        "--seed", str(args.seed),
    ]

    # 1. Map: builds the active bank + baseline eval.
    map_dir = out_root / "map"
    run(base_cmd + ["--variants", "map", "--out-dir", str(map_dir)], cwd=workdir)
    bank = map_dir / "active_bank.json"

    # 2. LoRA variants: reuse the bank, skip baseline eval.
    if args.lora_variants:
        labels = [s.strip() for s in args.lora_variants.split(",") if s.strip()]
        lora_variants = [v for v in DEFAULT_LORA_VARIANTS if v[0] in labels]
    else:
        lora_variants = DEFAULT_LORA_VARIANTS

    payload = {
        "model": "Qwen/Qwen3.5-9B",
        "task": "MATH-500 level1-3 answer_only",
        "seed": args.seed,
        "map": compact(map_dir / "active_train_summary.json"),
        "lora": {},
    }
    for label, lora_r, lora_lr in lora_variants:
        lora_dir = out_root / f"lora-{label}"
        cmd = base_cmd + [
            "--variants", "lora",
            "--skip-baseline-eval",
            "--active-bank-json", str(bank),
            "--lora-r", str(lora_r),
            "--lora-lr", f"{lora_lr:g}",
            "--out-dir", str(lora_dir),
        ]
        try:
            run(cmd, cwd=workdir)
            payload["lora"][label] = compact(lora_dir / "active_train_summary.json")
        except subprocess.CalledProcessError as e:
            payload["lora"][label] = {"error": f"exit {e.returncode}"}
        # Persist incrementally so a mid-sweep crash still yields a summary.
        (out_root / "sweep-summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2)
        )

    artifacts = Path("/content/qwen35-lora-sweep-artifacts.tar.gz")
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
