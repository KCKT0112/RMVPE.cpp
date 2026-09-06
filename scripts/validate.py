# SPDX-License-Identifier: MPL-2.0
"""End-to-end parity against the original checkpoint, including short/edge inputs."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import soundfile as sf
import torch
from torchaudio.transforms import Resample
from benchmark import errors, signal
from reference import load_model

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cli",type=Path,default=Path("build-cpu/bin/rmvpe-cli.exe"))
    p.add_argument("--model",type=Path,default=Path("models/rmvpe-f32.gguf"))
    p.add_argument("--checkpoint",type=Path,default=Path("models/original/model.pt"))
    p.add_argument("--backend",default="cpu")
    p.add_argument("--upstream",default="reference/RMVPE")
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--fast",action="store_true")
    p.add_argument("--unroll-gru",action="store_true")
    p.add_argument("--direct-conv",action="store_true")
    p.add_argument("--prob-atol",type=float,default=2e-4)
    p.add_argument("--pitch-atol",type=float,default=.1)
    p.add_argument("--wav",type=Path,action="append",default=[])
    p.add_argument("--output",type=Path,default=Path("work/validation.json"))
    p.add_argument("--work",type=Path,default=Path("work/validation"))
    args=p.parse_args()
    torch.set_num_threads(args.threads)
    args.work.mkdir(parents=True,exist_ok=True); args.output.parent.mkdir(parents=True,exist_ok=True)
    model=load_model(args.checkpoint,args.upstream)
    spec=importlib.import_module("rmvpe_upstream.spec").MelSpectrogram(128,16000,1024,160,None,30,8000)
    cases=[("silence",np.zeros(16000,np.float32),16000),
           ("noise",np.random.default_rng(71).normal(0,.1,16000).astype(np.float32),16000)]
    for n in [1,159,160,511,512,4959,4960,5119,5120,5121]:
        cases.append((f"length_{n}",signal(n/16000),16000))
    for sr in [8000,16000,22050,44100,48000]:
        cases.append((f"harmonic_{sr}",signal(1.2,sr),sr))
    mono=signal(1.0); cases.append(("stereo",np.stack((mono,mono*.7),axis=-1),16000))
    for path in args.wav:
        audio,sr=sf.read(path,dtype="float32"); cases.append((path.stem,audio,sr))
    report={"model_sha256":hashlib.sha256(args.model.read_bytes()).hexdigest(),"backend":args.backend,
            "options":{"fast":args.fast,"unroll_gru":args.unroll_gru,"direct_conv":args.direct_conv},
            "probability_tolerance":args.prob_atol,"pitch_tolerance_cents":args.pitch_atol,"cases":[]}
    failed=False
    for name,audio,sr in cases:
        print(name,flush=True)
        mono=audio.mean(axis=-1) if audio.ndim==2 else audio
        x=torch.from_numpy(mono.copy())[None]
        if sr!=16000: x=Resample(sr,16000,lowpass_filter_width=128)(x)
        valid=x.shape[-1]//160+1; padded=((valid+31)//32)*32
        x=torch.nn.functional.pad(x,(0,padded*160-160-x.shape[-1]))
        with torch.inference_mode(): mel=spec(x); reference=model(mel).numpy()[0]
        wav=args.work/(name+".wav"); sf.write(wav,audio,sr,subtype="FLOAT")
        probabilities=args.work/(name+".prob.f32"); mel_path=args.work/(name+".mel.f32")
        csv=args.work/(name+".csv")
        command=[str(args.cli),"--model",str(args.model),"--backend",args.backend,"--wav",str(wav),
            "--probabilities",str(probabilities),"--dump-mel",str(mel_path),"--output",str(csv),
            "--threads",str(args.threads),"--warmup","0","--runs","1"]
        if args.fast: command.append("--fast")
        if args.unroll_gru: command.append("--unroll-gru")
        if args.direct_conv: command.append("--direct-conv")
        env=os.environ.copy()
        if args.fast: env["RMVPE_VK_FAST"]="1"
        else: env.pop("RMVPE_VK_FAST",None)
        run=subprocess.run(command,env=env,capture_output=True,text=True,check=True)
        output=np.fromfile(probabilities,dtype=np.float32).reshape(-1,360)
        native_mel=np.fromfile(mel_path,dtype=np.float32).reshape(-1,128)
        mel_error=np.abs(native_mel-mel[0].T.numpy())
        metrics=errors(reference[:valid],output[:valid])
        pitch_csv=np.loadtxt(csv,delimiter=",",skiprows=1,ndmin=2)
        ok=(metrics["prob_max_abs"]<=args.prob_atol and metrics["uv_disagreements"]==0 and metrics["pitch_max_cents"]<=args.pitch_atol
            and native_mel.shape[0]==padded and pitch_csv.shape[0]==valid and float(mel_error.max())<.005)
        failed|=not ok
        report["cases"].append({"name":name,"sample_rate":sr,"samples":len(audio),"frames":valid,"padded_frames":padded,
            "wav_sha256":hashlib.sha256(wav.read_bytes()).hexdigest(),"mel_max_abs":float(mel_error.max()),
            "mel_mean_abs":float(mel_error.mean()),"passed":bool(ok),**metrics,"execution":json.loads(run.stdout)})
        print("PASS" if ok else "FAIL",metrics,flush=True)
        args.output.write_text(json.dumps(report,indent=2)+"\n")
    report["passed"]=not failed; args.output.write_text(json.dumps(report,indent=2)+"\n")
    raise SystemExit(1 if failed else 0)

if __name__=="__main__": main()
