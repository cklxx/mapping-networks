"""Colab-side launcher for the 9B MATH-500 experiment.

The local agent uploads a repo snapshot to /content/mapping-networks-src.tar.gz and
then executes this script with `colab exec -f`. The script is intentionally plain
Python so it can run inside a fresh Colab kernel without depending on local shell state.
"""

import argparse
import base64
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


def run(cmd, cwd=None, env=None):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="/content/mapping-networks-src.tar.gz")
    ap.add_argument("--workdir", default="/content/mapping-networks")
    ap.add_argument("--model", default="01-ai/Yi-1.5-9B-Chat")
    ap.add_argument("--hardware-label", default="Colab L4")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--attn-impl", default="")
    ap.add_argument("--max-steps", default="350")
    ap.add_argument("--time-budget-s", default=str(6 * 3600))
    ap.add_argument("--n-eval", default="200")
    ap.add_argument("--max-new", default="256")
    ap.add_argument("--max-new-eval", default="512")
    ap.add_argument("--eval-batch", default="1")
    ap.add_argument("--B", default="1")
    ap.add_argument("--K", default="3")
    ap.add_argument("--g-sweep", default="256,2048")
    ap.add_argument("--lora-lr-sweep", default="1e-4,1e-3")
    ap.add_argument("--baseline-json", default="")
    ap.add_argument("--print-every", default="20")
    ap.add_argument("--save-every", default="20")
    ap.add_argument("--min-train-level", default="3")
    ap.add_argument("--max-train-level", default="5")
    ap.add_argument("--train-selection", default="stride", choices=["head", "stride"])
    ap.add_argument("--skip-plot", action="store_true")
    ap.add_argument("--stdout-artifact", action="store_true")
    ap.add_argument("--stdout-artifact-max-mb", type=float, default=32.0)
    ap.add_argument("--baseline-only", action="store_true")
    args = ap.parse_args()

    archive = Path(args.archive)
    workdir = Path(args.workdir)
    if not archive.exists():
        raise FileNotFoundError(f"missing repo archive: {archive}")

    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(workdir)

    # Do not install the repo's broad requirements.txt on Colab: it can upgrade
    # torch to a CUDA build that no longer matches preinstalled torchaudio. This
    # text-only experiment only needs the HF/data/plotting stack; keep Colab's
    # native torch build intact.
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

    probe = (
        "import torch\n"
        "print('torch', torch.__version__)\n"
        "print('cuda', torch.cuda.is_available())\n"
        "print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)\n"
        "print('mem_gb', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2) if torch.cuda.is_available() else None)\n"
    )
    run([sys.executable, "-c", probe], cwd=workdir)

    out = "results/9b-math500/results.txt"
    cost_out = "results/9b-math500/cost-table.md"
    cmd = [
        sys.executable,
        "experiments/math500_rl_9b.py",
        "--model",
        args.model,
        "--hardware-label",
        args.hardware_label,
        "--dtype",
        args.dtype,
        "--max-steps",
        args.max_steps,
        "--time-budget-s",
        args.time_budget_s,
        "--n-eval",
        args.n_eval,
        "--max-new",
        args.max_new,
        "--max-new-eval",
        args.max_new_eval,
        "--eval-batch",
        args.eval_batch,
        "--B",
        args.B,
        "--K",
        args.K,
        "--g-sweep",
        args.g_sweep,
        "--lora-lr-sweep",
        args.lora_lr_sweep,
        "--print-every",
        args.print_every,
        "--save-every",
        args.save_every,
        "--min-train-level",
        args.min_train_level,
        "--max-train-level",
        args.max_train_level,
        "--train-selection",
        args.train_selection,
        "--out",
        out,
        "--cost-out",
        cost_out,
    ]
    if args.attn_impl:
        cmd.extend(["--attn-impl", args.attn_impl])
    if args.baseline_json:
        cmd.extend(["--baseline-json", args.baseline_json])
    if args.baseline_only:
        cmd.append("--baseline-only")
    artifacts = Path("/content/mapping-networks-9b-artifacts.tar.gz")
    try:
        run(cmd, cwd=workdir)
        if not args.baseline_only and not args.skip_plot:
            run([sys.executable, "plot_curves.py"], cwd=workdir / "results/9b-math500")
    finally:
        with tarfile.open(artifacts, "w:gz") as tf:
            result_dir = workdir / "results/9b-math500"
            if result_dir.exists():
                tf.add(result_dir, arcname="results/9b-math500")
        print(f"\nARTIFACTS={artifacts}", flush=True)
        if args.stdout_artifact and artifacts.exists():
            max_bytes = int(args.stdout_artifact_max_mb * 1024 * 1024)
            size = artifacts.stat().st_size
            print(f"ARTIFACT_BYTES={size}", flush=True)
            if size <= max_bytes:
                b64 = base64.b64encode(artifacts.read_bytes()).decode()
                print("ARTIFACT_TAR_GZ_B64_BEGIN", flush=True)
                for i in range(0, len(b64), 4096):
                    print(b64[i:i + 4096], flush=True)
                print("ARTIFACT_TAR_GZ_B64_END", flush=True)
            else:
                print(f"ARTIFACT_TAR_GZ_B64_SKIPPED size>{max_bytes}", flush=True)


if __name__ == "__main__":
    main()
