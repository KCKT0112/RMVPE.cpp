# SPDX-License-Identifier: MPL-2.0
"""Sequential, synchronized network benchmarks on identical padded log-Mel inputs."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import statistics
import subprocess
import time
import numpy as np
import torch
if os.environ.get("RMVPE_ORT_PATH"):
    sys.path.insert(0, os.environ["RMVPE_ORT_PATH"])
import onnxruntime as ort
import soundfile as sf
from reference import load_model

def signal(seconds, sr=16000):
    t = np.arange(round(seconds*sr), dtype=np.float64)/sr
    freq = 170 + 35*np.sin(2*np.pi*0.7*t) + 12*np.sin(2*np.pi*3.1*t)
    phase = 2*np.pi*np.cumsum(freq)/sr
    envelope = np.sin(np.pi*np.minimum(t/0.04,1))**2 if seconds < .04 else np.minimum(t/.04,1)
    envelope *= (np.sin(2*np.pi*1.4*t) > -.65)
    y = sum(np.sin(k*phase + k*.13)/k for k in range(1,13))*.18*envelope
    y += np.random.default_rng(20260906).normal(0,.0005,len(t))
    return y.astype(np.float32)

def pitch(p):
    center = p.argmax(-1)
    idx = np.arange(360)[None,:]
    weight = p*((idx >= center[:,None]-4)&(idx <= center[:,None]+4))
    cents = (weight*(1997.3794084376191+20*idx)).sum(-1)/np.maximum(weight.sum(-1),1e-20)
    return np.where(p.max(-1)>.03,10*2**(cents/1200),0)

def errors(reference, output):
    a,b = pitch(reference),pitch(output)
    both = (a>0)&(b>0)
    cents = np.abs(1200*np.log2(a[both]/b[both]))
    return {"prob_max_abs":float(np.abs(reference-output).max()),
            "prob_mean_abs":float(np.abs(reference-output).mean()),
            "uv_disagreements":int(((a>0)!=(b>0)).sum()), "voiced_frames":int(both.sum()),
            "pitch_max_cents":float(cents.max()) if cents.size else 0.,
            "pitch_mean_cents":float(cents.mean()) if cents.size else 0.,
            "pitch_p50_cents":float(np.percentile(cents,50)) if cents.size else 0.,
            "pitch_p99_cents":float(np.percentile(cents,99)) if cents.size else 0.,
            "frames_above_5_cents":int((cents>5).sum()),
            "frames_above_20_cents":int((cents>20).sum())}

def timed(fn, warmup, runs, sync=lambda: None):
    for _ in range(warmup): fn(); sync()
    values=[]
    for _ in range(runs):
        sync(); start=time.perf_counter(); output=fn(); sync(); values.append((time.perf_counter()-start)*1000)
    return output, {"median_ms":statistics.median(values),"samples_ms":values}

def native(args, variant, mel, prob, wav=None, dump=None):
    cpu = variant.startswith("cpu")
    binary = args.cpu_cli if cpu else args.gpu_cli
    command=[str(binary),"--model",str(args.model),"--backend","cpu" if cpu else args.gpu_backend,
             "--threads",str(args.threads),"--warmup",str(args.warmup),"--runs",str(args.runs),"--probabilities",str(prob)]
    command += ["--wav",str(wav)] if wav else ["--mel",str(mel)]
    if dump: command += ["--dump-mel",str(dump)]
    if "unroll" in variant: command += ["--unroll-gru"]
    if "fast" in variant: command += ["--fast"]
    if "direct" in variant: command += ["--direct-conv"]
    if "im2col" in variant: command += ["--im2col"]
    env=os.environ.copy()
    for key in list(env):
        if key.startswith(("RMVPE_VK_","GGML_VK_DISABLE_")): env.pop(key)
    if "fast" in variant: env["RMVPE_VK_FAST"]="1"
    if "cpu-gru" in variant: env["RMVPE_VK_CPU_GRU"]="1"
    if "host-visible" in variant: env["RMVPE_VK_HOST_VISIBLE"]="1"
    if "row-major" in variant: env["RMVPE_VK_GRU_ROW_MAJOR"]="1"
    result=subprocess.run(command,env=env,text=True,capture_output=True,check=True)
    stats=json.loads(result.stdout)
    stats["command"]=command
    stats["environment"]={k:v for k,v in env.items() if k.startswith(("RMVPE_VK_","GGML_VK_"))}
    stats["backend_log"]=result.stderr
    return np.fromfile(prob,dtype=np.float32).reshape(-1,360),stats

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,default=Path("models/original/model.pt"))
    p.add_argument("--model",type=Path,default=Path("models/rmvpe-f32.gguf"))
    p.add_argument("--upstream",default="reference/RMVPE")
    p.add_argument("--cpu-cli",type=Path,default=Path("build-cpu/bin/rmvpe-cli.exe"))
    p.add_argument("--gpu-cli",type=Path,default=Path("build-vulkan/bin/rmvpe-cli.exe"))
    p.add_argument("--gpu-backend",default="Vulkan0")
    p.add_argument("--seconds",type=float,nargs="+",default=[1,3,10])
    p.add_argument("--variants",nargs="+",default=["cpu","vulkan","vulkan-fast"])
    p.add_argument("--cuda-dir",type=Path,default=Path(os.environ.get("CUDA_PATH","C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")))
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--warmup",type=int,default=3)
    p.add_argument("--runs",type=int,default=7)
    p.add_argument("--output",type=Path,default=Path("work/benchmark.json"))
    p.add_argument("--work",type=Path,default=Path("work/benchmark"))
    p.add_argument("--skip-reference-timing",action="store_true")
    args=p.parse_args()
    if min(args.seconds)<=0 or args.warmup<0 or args.runs<1: p.error("Positive durations/runs and nonnegative warmup required")
    args.work.mkdir(parents=True,exist_ok=True); args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    model=load_model(args.checkpoint,args.upstream)
    spec=importlib.import_module("rmvpe_upstream.spec").MelSpectrogram(128,16000,1024,160,None,30,8000)
    report={"date":time.strftime("%Y-%m-%d"),"platform":platform.platform(),"python":platform.python_version(),
        "torch":torch.__version__,"onnxruntime":ort.__version__,"ort_providers":ort.get_available_providers(),
        "cuda_gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "threads":args.threads,"warmup":args.warmup,"runs":args.runs,
        "model_sha256":hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "scope":"batch 1 network only, same padded log-Mel, host F32 input to host F32 output including GPU transfers and synchronization; loading/building/compilation excluded",
        "inputs":"deterministic harmonic glide with voiced/unvoiced intervals and seeded noise, not an accuracy dataset",
        "results":[],"frontend":[]}
    cases=[]
    with torch.inference_mode():
        for seconds in args.seconds:
            name=f"{seconds:g}s"; a=signal(seconds); frames=((len(a)//160+1+31)//32)*32
            tensor=torch.from_numpy(np.pad(a,(0,frames*160-160-len(a))))[None]
            mel=spec(tensor); reference=model(mel).numpy()[0]
            mel_path=args.work/f"{name}.mel.f32"; mel[0].T.contiguous().numpy().tofile(mel_path)
            wav=args.work/f"{name}.wav"; sf.write(wav,a,16000,subtype="FLOAT")
            cases.append((seconds,frames,mel,reference,mel_path,wav))
    if not args.skip_reference_timing:
        onnx_path=args.work/"rmvpe.onnx"
        print("Exporting original E2E0 to ONNX...",flush=True)
        torch.onnx.export(model,cases[0][2].clone(),str(onnx_path),input_names=["mel"],output_names=["probabilities"],
            dynamic_axes={"mel":{2:"frames"},"probabilities":{1:"frames"}},opset_version=17,dynamo=False)
        import onnx
        onnx.checker.check_model(str(onnx_path))
        for backend in ["torch-cpu"]+(["torch-cuda"] if torch.cuda.is_available() else []):
            device="cuda" if backend.endswith("cuda") else "cpu"; model.to(device)
            for seconds,frames,mel,reference,_,_ in cases:
                x=mel
                with torch.inference_mode(): y,stats=timed(lambda:model(x.to(device)).cpu().numpy(),args.warmup,args.runs,torch.cuda.synchronize if device=="cuda" else lambda:None)
                stats.update(errors(reference,y[0])); stats.update(backend=backend,seconds=seconds,frames=frames,rtf=stats["median_ms"]/(seconds*1000))
                report["results"].append(stats); print(backend,seconds,stats["median_ms"],flush=True)
            model.cpu()
            if device=="cuda": del x,y; torch.cuda.empty_cache()
        options=ort.SessionOptions(); options.intra_op_num_threads=args.threads; options.inter_op_num_threads=1
        handles=[]
        providers=["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            if os.name=="nt":
                torch_lib=Path(torch.__file__).parent/"lib"
                for directory in (args.cuda_dir/"bin",torch_lib):
                    handles.append(os.add_dll_directory(str(directory)))
                ort.preload_dlls(cuda=True,cudnn=False,directory=str(args.cuda_dir/"bin"))
                ort.preload_dlls(cuda=False,cudnn=True,directory=str(torch_lib))
            providers.append("CUDAExecutionProvider")
        for provider in providers:
            configuration=[(provider,{"use_tf32":"0","cudnn_conv_algo_search":"HEURISTIC"}),"CPUExecutionProvider"] if provider.startswith("CUDA") else [provider]
            session=ort.InferenceSession(str(onnx_path),sess_options=options,providers=configuration)
            if session.get_providers()[0]!=provider: raise RuntimeError("Requested ONNX provider did not initialize")
            for seconds,frames,mel,reference,_,_ in cases:
                x=mel.numpy(); y,stats=timed(lambda:session.run(None,{"mel":x})[0],args.warmup,args.runs)
                backend="onnxruntime-cuda" if provider.startswith("CUDA") else "onnxruntime-cpu"
                stats.update(errors(reference,y[0])); stats.update(backend=backend,seconds=seconds,frames=frames,rtf=stats["median_ms"]/(seconds*1000),providers=session.get_providers(),provider_options=session.get_provider_options())
                report["results"].append(stats); print(backend,seconds,stats["median_ms"],flush=True)
            del session
    for variant in args.variants:
        for seconds,frames,mel,reference,mel_path,wav in cases:
            print("Running",variant,seconds,flush=True)
            y,stats=native(args,variant,mel_path,args.work/"output.f32")
            stats.update(errors(reference,y)); stats.update(variant=variant,seconds=seconds,rtf=stats["median_ms"]/(seconds*1000))
            report["results"].append(stats)
            print(variant,seconds,stats["median_ms"],stats["prob_max_abs"],flush=True)
            args.output.write_text(json.dumps(report,indent=2)+"\n")
    # Independent C++ waveform frontend compared with the original torch/librosa frontend.
    for seconds,frames,mel,reference,mel_path,wav in cases:
        y,stats=native(args,"cpu",mel_path,args.work/"audio-prob.f32",wav=wav,dump=args.work/"audio-mel.f32")
        native_mel=np.fromfile(args.work/"audio-mel.f32",dtype=np.float32).reshape(-1,128)
        error=np.abs(native_mel-mel[0].T.numpy())
        report["frontend"].append({"seconds":seconds,"mel_max_abs":float(error.max()),"mel_mean_abs":float(error.mean()),"frontend_ms":stats["frontend_ms"],**errors(reference,y)})
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(args.output,flush=True)

if __name__=="__main__": main()
