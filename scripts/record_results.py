# SPDX-License-Identifier: MPL-2.0
"""Archive completed local runs and generate transparent comparison tables."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("docs/results/windows-2026-09-06"))
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    names=["benchmark-final","benchmark-initial","tuning","validation-cpu-f32","validation-vulkan-f32",
           "validation-cpu-f16","validation-vulkan-f16","validation-vulkan-fast","diagnostics"]
    for name in names:
        shutil.copyfile(Path("work")/(name+".json"),args.output/(name+".json"))
    final=json.loads((args.output/"benchmark-final.json").read_text())
    rows=["# Full network timings","","Batch 1, host F32 Mel to host F32 probabilities; 3 warmups, 10 timed runs. CPU threads: 4.","",
          "| Engine | Audio seconds | Padded frames | Median ms | P95 ms | RTF | Probability max abs | CPU/GPU compute nodes |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in final["results"]:
        rows.append(f'| {r.get("variant",r["backend"])} | {r["seconds"]:g} | {r["frames"]} | {r["median_ms"]:.3f} | {np.percentile(r["samples_ms"],95):.3f} | {r["rtf"]:.6f} | {r["prob_max_abs"]:.8g} | {r.get("cpu_compute_nodes","n/a")}/{r.get("accelerator_compute_nodes","n/a")} |')
    (args.output/"all-results.md").write_text("\n".join(rows)+"\n",encoding="utf-8")
    hashes={}
    for pattern in ["src/**/*","include/**/*","cmake/**/*","tools/*","scripts/*.py","tests/*"]:
        for path in Path(".").glob(pattern):
            if path.is_file() and "__pycache__" not in str(path): hashes[path.as_posix()]=sha(path)
    binaries={}
    for folder in ["build-cpu/bin","build-vulkan/bin"]:
        for path in Path(folder).glob("*"):
            if path.suffix.lower() in [".exe",".dll"]: binaries[path.as_posix()]=sha(path)
    inputs={path.as_posix():sha(path) for path in Path("work/benchmark").glob("*.mel.f32")}
    inputs.update({path.as_posix():sha(path) for path in Path("work/benchmark").glob("*.wav")})
    provenance={
        "hardware":{"cpu":"AMD Ryzen 7 5700X, 8 cores / 16 threads","gpu":"NVIDIA GeForce RTX 4060, 8 GB",
                    "driver":"620.02","vulkan_sdk":"1.4.341.1","compiler":"MSVC 19.51.36256","cpu_build_isa":"AVX2/FMA"},
        "source_hashes":hashes,"final_binary_hashes":binaries,"input_hashes":inputs,
        "model_manifests":{str(p):json.loads(p.read_text()) for p in Path("models").glob("*.gguf.json")},
        "note":"Final hashes include subsequent CLI-only changes (single ordinary inference, optional end-to-end timing) and tests/docs; network algorithms are those of benchmark-final.json. Initial/tuning runs are exploratory and retain their original metadata. Exact raw samples are never rewritten.",
    }
    (args.output/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(args.output)

if __name__=="__main__": main()
