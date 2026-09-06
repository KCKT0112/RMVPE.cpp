# SPDX-License-Identifier: MPL-2.0
"""Compare two GGUF models on identical native WAV frontends in strict arithmetic."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from benchmark import errors


def infer(cli, model, backend, wav, stem, threads):
    probabilities = stem.with_suffix(".prob.f32")
    mel = stem.with_suffix(".mel.f32")
    csv = stem.with_suffix(".csv")
    command = [str(cli), "--model", str(model), "--backend", backend,
               "--wav", str(wav), "--probabilities", str(probabilities),
               "--dump-mel", str(mel), "--output", str(csv),
               "--threads", str(threads), "--warmup", "0", "--runs", "1"]
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("RMVPE_VK_", "GGML_VK_DISABLE_")):
            env.pop(key)
    run = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    output = np.fromfile(probabilities, dtype=np.float32).reshape(-1, 360)
    frontend = np.fromfile(mel, dtype=np.float32).reshape(-1, 128)
    frames = len(np.loadtxt(csv, delimiter=",", skiprows=1, ndmin=2))
    if not 0 < frames <= len(output) or not np.isfinite(output).all() or not np.isfinite(frontend).all():
        raise ValueError(f"Invalid native outputs for {wav}")
    return output[:frames], frontend, json.loads(run.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-cli", type=Path, default=Path("build-vulkan/bin/rmvpe-cli.exe"))
    parser.add_argument("--reference-backend", default="Vulkan0")
    parser.add_argument("--cli", type=Path, default=Path("build-cpu/bin/rmvpe-cli.exe"))
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--fixtures", type=Path, help="Directory of WAV fixtures produced by validate.py")
    parser.add_argument("--wav", type=Path, action="append", default=[])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--prob-atol", type=float, default=2e-4)
    parser.add_argument("--pitch-atol", type=float, default=.1)
    parser.add_argument("--work", type=Path, default=Path("work/compare-models"))
    parser.add_argument("--output", type=Path, default=Path("work/compare-models.json"))
    args = parser.parse_args()
    wavs = sorted(args.fixtures.glob("*.wav")) if args.fixtures else []
    wavs += args.wav
    if not wavs or len({p.stem for p in wavs}) != len(wavs):
        parser.error("Provide WAVs with unique stems")
    if args.threads < 1 or not all(np.isfinite(v) and v >= 0 for v in (args.prob_atol, args.pitch_atol)):
        parser.error("Positive threads and finite nonnegative tolerances are required")
    args.work.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "Same WAV and native frontend; strict arithmetic; agreement, not ground-truth pitch accuracy",
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "reference_backend": args.reference_backend, "backend": args.backend,
        "threads": args.threads, "probability_tolerance": args.prob_atol,
        "pitch_tolerance_cents": args.pitch_atol, "cases": [],
    }
    for wav in wavs:
        reference, ref_mel, ref_execution = infer(args.reference_cli, args.reference, args.reference_backend,
                                                  wav, args.work / (wav.stem + "-reference"), args.threads)
        output, mel, execution = infer(args.cli, args.model, args.backend,
                                      wav, args.work / (wav.stem + "-candidate"), args.threads)
        if reference.shape != output.shape:
            raise ValueError(f"Frame count differs for {wav}")
        metrics = errors(reference, output)
        same_frontend = ref_mel.shape == mel.shape and np.array_equal(ref_mel, mel)
        passed = (same_frontend and metrics["uv_disagreements"] == 0 and
                  metrics["prob_max_abs"] <= args.prob_atol and metrics["pitch_max_cents"] <= args.pitch_atol)
        report["cases"].append({"case": wav.stem, "wav_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
                                "frames": len(output), "identical_frontend": same_frontend,
                                "passed": bool(passed), **metrics,
                                "reference_execution": ref_execution, "execution": execution})
        report["passed"] = all(case["passed"] for case in report["cases"])
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(wav.stem, "PASS" if passed else "FAIL", metrics, flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
