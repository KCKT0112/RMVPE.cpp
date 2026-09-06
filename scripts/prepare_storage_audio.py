# SPDX-License-Identifier: MPL-2.0
"""Prepare the five held-out storage fixtures from hash-checked public librosa examples.

Supply --download to retrieve missing source audio, and optionally --jfk for
the speech file used in the original validation. Audio assets retain their
original terms and are not included in this repository.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen

import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("work/storage-audio"))
    parser.add_argument("--download", action="store_true", help="Fetch missing OGG sources from librosa.org")
    parser.add_argument("--jfk", type=Path, help="Original JFK WAV (copied without re-encoding)")
    args = parser.parse_args()
    manifest = Path(__file__).resolve().parents[1] / "docs/results/windows-2026-09-06-storage/audio-sources.json"
    sources = json.loads(manifest.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    fixtures = []
    for row in sources:
        path = args.output / row["source"].rsplit("/", 1)[1]
        if not path.exists():
            if not args.download:
                parser.error(f"Missing {path}; supply --download or place the source there")
            with tempfile.TemporaryDirectory(dir=args.output) as temporary:
                fetched = Path(temporary) / "source.ogg"
                with urlopen(row["source"], timeout=60) as src, fetched.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                if hashlib.sha256(fetched.read_bytes()).hexdigest() != row["sha256"]:
                    raise ValueError(f"Downloaded source hash mismatch: {row['source']}")
                shutil.copyfile(fetched, path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Source hash mismatch: {path}")
        audio, sr = sf.read(path, dtype="float32")
        if sr != row["sample_rate"]:
            raise ValueError(f"Unexpected sample rate: {path}")
        start = round(row.get("start", 0) * sr)
        end = start + round(row["duration"] * sr)
        if end > len(audio):
            raise ValueError(f"Source is too short: {path}")
        output = args.output / (row["name"] + ".wav")
        samples = audio[start:end]
        sf.write(output, samples, sr, subtype="FLOAT")
        fixtures.append({"name": row["name"], "sample_rate": sr, "samples": len(samples),
                         "pcm_f32le_sha256": hashlib.sha256(samples.astype("<f4").tobytes()).hexdigest()})
        print("--wav", output)
    if args.jfk:
        output = args.output / "recording_0_jfk.wav"
        if args.jfk.resolve() != output.resolve():
            shutil.copyfile(args.jfk, output)
        print("--wav", output)
    # libsndfile writes a timestamp in the WAV PEAK chunk; sample hashes are
    # stable even when the complete generated WAV bytes differ.
    (args.output / "fixtures.json").write_text(json.dumps(fixtures, indent=2) + "\n", encoding="utf-8")
    (args.output / "sources.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
