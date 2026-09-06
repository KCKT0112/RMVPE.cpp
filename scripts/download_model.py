# SPDX-License-Identifier: MPL-2.0
"""Download the original, hash-pinned RMVPE release. Weights are not MPL code."""
import argparse
import hashlib
from pathlib import Path
import urllib.request
import zipfile

URL = "https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip"
SHA256 = "54ae40d9c066d998b94574f6ef0deea19ed1565bd655b3f0d9b1ad612fb5309c"

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("models"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "rmvpe.zip"
    if not archive.exists():
        temp = archive.with_suffix(".download")
        urllib.request.urlretrieve(URL, temp)
        temp.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Archive SHA256 mismatch; remove it and download again")
    target = args.output / "original" / "model.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        # Extract only the expected member, never arbitrary archive paths.
        target.write_bytes(z.read("model.pt"))
    print(f"{target}: {hashlib.sha256(target.read_bytes()).hexdigest()}")

if __name__ == "__main__":
    main()
