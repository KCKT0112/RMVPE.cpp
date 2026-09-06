# SPDX-License-Identifier: MPL-2.0
"""Convert the 230917 E2E0 checkpoint to GGUF with evaluation BatchNorm folded."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

STORAGE_TYPES = ("f32", "f16", "f16-intermediate")


def use_half_storage(name, ndim, dtype):
    if dtype not in STORAGE_TYPES:
        raise ValueError(f"Unknown storage type: {dtype}")
    eligible = ndim >= 2 and not name.startswith(("mel.", "gru."))
    return eligible and (dtype == "f16" or
                         (dtype == "f16-intermediate" and name.startswith("unet.intermediate.")))

def convert(checkpoint, output, dtype="f32", upstream="reference/RMVPE"):
    import gguf
    import torch
    from librosa.filters import mel
    from reference import load_model

    if dtype not in STORAGE_TYPES:
        raise ValueError(f"Unknown storage type: {dtype}")
    torch.set_num_threads(4)
    print("Loading upstream checkpoint...", flush=True)
    model = load_model(checkpoint, upstream)
    print("Folding evaluation BatchNorm and writing tensors...", flush=True)
    tensors = {}
    bn = model.unet.encoder.bn
    scale = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
    tensors["input.scale"] = scale.numpy()
    tensors["input.bias"] = (bn.bias.detach() - bn.running_mean * scale).numpy()
    modules = dict(model.named_modules())
    for name, layer in modules.items():
        if name.startswith("unet.tf."):
            continue  # E2E0 registers this unused timbre filter but never executes it.
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            w = layer.weight.detach().clone()
            b = torch.zeros(layer.out_channels) if layer.bias is None else layer.bias.detach().clone()
            parent, _, index = name.rpartition(".")
            following = modules.get(parent + "." + str(int(index) + 1)) if index.isdigit() else None
            if isinstance(following, torch.nn.BatchNorm2d):
                s = following.weight.detach() / torch.sqrt(following.running_var + following.eps)
                w *= s.reshape(1, -1, 1, 1) if isinstance(layer, torch.nn.ConvTranspose2d) else s.reshape(-1, 1, 1, 1)
                b = (b - following.running_mean) * s + following.bias.detach()
            tensors[name + ".weight"] = w.numpy()
            tensors[name + ".bias"] = b.numpy()
    state = model.state_dict()
    for name in ("fc.1.weight", "fc.1.bias"):
        tensors[name] = state[name].numpy()
    for direction, suffix in enumerate(("", "_reverse")):
        for part in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            tensors[f"gru.{direction}.{part}"] = state[f"fc.0.gru.{part}_l0{suffix}"].numpy()
    tensors["mel.basis"] = mel(sr=16000, n_fft=1024, n_mels=128, fmin=30, fmax=8000, htk=True)
    tensors["mel.window"] = torch.hann_window(1024).numpy()
    provenance = {
        "source": "https://github.com/yxlllc/RMVPE", "release": "230917",
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "storage": dtype, "batchnorm": "folded, epsilon=1e-5", "unused_timbre_filter": "omitted",
        "model_license": "No explicit license found in inspected upstream source or archive",
    }
    if dtype == "f16-intermediate":
        provenance["storage_policy"] = "Only unet.intermediate convolution weights F16; all other tensors F32"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(output), "rmvpe")
    writer.add_name("RMVPE 230917 E2E0")
    writer.add_uint32("rmvpe.version", 1)
    for key, value in provenance.items():
        writer.add_string("rmvpe." + key, value)
    for name, array in tensors.items():
        # Recurrent weights and all affine terms stay F32; convolutions/linears may be F16.
        half = use_half_storage(name, array.ndim, dtype)
        array = np.ascontiguousarray(array, dtype=np.float16 if half else np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite tensor: {name}")
        writer.add_tensor(name, array)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    provenance["gguf_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    provenance["tensors"] = len(tensors)
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dtype", choices=STORAGE_TYPES, default="f32")
    parser.add_argument("--upstream", default="reference/RMVPE")
    args = parser.parse_args()
    convert(args.checkpoint, args.output, args.dtype, args.upstream)
