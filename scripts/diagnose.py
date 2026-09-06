# SPDX-License-Identifier: MPL-2.0
"""Record native audio-to-F0 timing and separate ONNX CUDA node placement."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import Counter

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("work/diagnostics.json"))
    p.add_argument("--onnx",type=Path,default=Path("work/benchmark/rmvpe.onnx"))
    p.add_argument("--cuda-dir",type=Path,default=Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"))
    p.add_argument("--ort-path",type=Path)
    args=p.parse_args()
    results={"audio_to_f0":[]}
    for backend,binary in [("cpu","build-cpu/bin/rmvpe-cli.exe"),("Vulkan0","build-vulkan/bin/rmvpe-cli.exe")]:
        for seconds in [1,3,10]:
            command=[binary,"--model","models/rmvpe-f32.gguf","--wav",f"work/benchmark/{seconds}s.wav",
                     "--backend",backend,"--threads","4","--benchmark-audio","--warmup","3","--runs","10"]
            run=subprocess.run(command,capture_output=True,text=True,check=True)
            result=json.loads(run.stdout); result["seconds"]=seconds; result["command"]=command
            results["audio_to_f0"].append(result)
    if args.ort_path:
        sys.path.insert(0,str(args.ort_path))
    import numpy as np
    import torch
    import onnxruntime as ort
    if "CUDAExecutionProvider" in ort.get_available_providers():
        handles=[]
        if os.name=="nt":
            torch_lib=Path(torch.__file__).parent/"lib"
            for directory in (args.cuda_dir/"bin",torch_lib): handles.append(os.add_dll_directory(str(directory)))
            ort.preload_dlls(cuda=True,cudnn=False,directory=str(args.cuda_dir/"bin"))
            ort.preload_dlls(cuda=False,cudnn=True,directory=str(torch_lib))
        options=ort.SessionOptions(); options.intra_op_num_threads=4; options.inter_op_num_threads=1
        options.enable_profiling=True; options.profile_file_prefix="work/ort-profile"
        session=ort.InferenceSession(str(args.onnx),sess_options=options,
            providers=[("CUDAExecutionProvider",{"use_tf32":"0","cudnn_conv_algo_search":"HEURISTIC"}),"CPUExecutionProvider"])
        if session.get_providers()[0]!="CUDAExecutionProvider": raise RuntimeError("CUDA fallback")
        x=np.fromfile("work/benchmark/1s.mel.f32",dtype=np.float32).reshape(128,128).T.copy()[None]
        session.run(None,{"mel":x})
        path=session.end_profiling(); events=json.loads(Path(path).read_text())
        kernels=[e for e in events if e.get("cat")=="Node" and e.get("name","").endswith("_kernel_time")]
        results["ort_placement"]={"providers":session.get_providers(),"version":ort.__version__,
            "counts":dict(Counter(e["args"].get("provider","unknown") for e in kernels)),
            "ops_by_provider":{provider:dict(Counter(e["args"].get("op_name","unknown") for e in kernels if e["args"].get("provider")==provider))
                               for provider in ["CUDAExecutionProvider","CPUExecutionProvider"]},
            "note":"Separate instrumented placement run; its timings are excluded from benchmarks."}
    args.output.write_text(json.dumps(results,indent=2)+"\n")
    print(args.output)

if __name__=="__main__": main()
