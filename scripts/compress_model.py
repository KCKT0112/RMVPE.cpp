# SPDX-License-Identifier: MPL-2.0
"""Pack/unpack GGUF bytes using reversible byte shuffle and checksummed Zstandard.

RMVPEBS1 is a distribution container; unpack it before native inference.
Requires numpy and zstandard (see requirements-compression.txt).
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import time

import numpy as np
import zstandard as zstd

MAGIC = b"RMVPEBS1"
HEADER = struct.Struct("<8sQ32s")
BLOCK_HEADER = struct.Struct("<QQ")
BLOCK_BYTES = 8 * 1024 * 1024
MAX_PACKED_BYTES = 16 * 1024 * 1024


@contextmanager
def new_output(path):
    """Publish a complete file without replacing an existing destination."""
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError(path)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".part", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w+b") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        # Windows rename refuses an existing destination. POSIX rename would
        # replace it, so use link there for an atomic create-if-absent operation.
        if os.name == "nt":
            os.rename(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("Truncated archive")
    return data


def pack(source, output, level=9):
    source, output = Path(source), Path(output)
    start = time.perf_counter()
    digest = hashlib.sha256()
    total = 0
    compressor = zstd.ZstdCompressor(level=level, write_checksum=True)
    with source.open("rb") as src, new_output(output) as dst:
        dst.write(HEADER.pack(MAGIC, 0, bytes(32)))
        while block := src.read(BLOCK_BYTES):
            if len(block) % 4:
                raise ValueError("Input size must be a multiple of four bytes")
            digest.update(block)
            total += len(block)
            shuffled = np.frombuffer(block, np.uint8).reshape(-1, 4).T.copy().tobytes()
            packed = compressor.compress(shuffled)
            dst.write(BLOCK_HEADER.pack(len(block), len(packed)))
            dst.write(packed)
        if not total:
            raise ValueError("Input is empty")
        dst.seek(0)
        dst.write(HEADER.pack(MAGIC, total, digest.digest()))
    return {"operation": "pack", "original_bytes": total,
            "compressed_bytes": output.stat().st_size, "model_sha256": digest.hexdigest(),
            "level": level, "zstandard_version": zstd.__version__,
            "seconds": time.perf_counter() - start}


def unpack(source, output):
    source, output = Path(source), Path(output)
    start = time.perf_counter()
    digest = hashlib.sha256()
    with source.open("rb") as src, new_output(output) as dst:
        magic, expected_size, expected_digest = HEADER.unpack(read_exact(src, HEADER.size))
        if magic != MAGIC:
            raise ValueError("Unknown archive format")
        if not expected_size or expected_size % 4:
            raise ValueError("Invalid original size")
        remaining = expected_size
        decoder = zstd.ZstdDecompressor(max_window_size=BLOCK_BYTES // 1024)
        while remaining:
            length, packed_size = BLOCK_HEADER.unpack(read_exact(src, BLOCK_HEADER.size))
            if not 0 < length <= min(BLOCK_BYTES, remaining) or length % 4:
                raise ValueError("Invalid block length")
            if not 0 < packed_size <= MAX_PACKED_BYTES:
                raise ValueError("Invalid compressed block length")
            packed = read_exact(src, packed_size)
            params = zstd.get_frame_parameters(packed)
            # max_output_size alone does not bound frames declaring their own
            # content size. Validate it and the decoder window before allocation.
            if params.content_size != length or params.window_size > BLOCK_BYTES or not params.has_checksum:
                raise ValueError("Invalid Zstandard frame parameters")
            shuffled = decoder.decompress(packed, max_output_size=length, allow_extra_data=False)
            if len(shuffled) != length:
                raise ValueError("Invalid decoded block length")
            block = np.frombuffer(shuffled, np.uint8).reshape(4, -1).T.copy().tobytes()
            dst.write(block)
            digest.update(block)
            remaining -= length
        if src.read(1):
            raise ValueError("Trailing archive data")
        if digest.digest() != expected_digest:
            raise ValueError("Archive SHA256 mismatch")
    return {"operation": "unpack", "original_bytes": expected_size,
            "compressed_bytes": source.stat().st_size, "model_sha256": digest.hexdigest(),
            "sha256_verified": True, "seconds": time.perf_counter() - start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    for name in ("pack", "unpack"):
        sub = commands.add_parser(name)
        sub.add_argument("source", type=Path)
        sub.add_argument("output", type=Path, help="New file; existing files are never overwritten")
        if name == "pack":
            sub.add_argument("--level", type=int, choices=range(1, 20), default=9, metavar="1..19")
    args = parser.parse_args()
    try:
        result = pack(args.source, args.output, args.level) if args.operation == "pack" else unpack(args.source, args.output)
    except (OSError, ValueError, zstd.ZstdError) as error:
        parser.exit(1, f"compress_model: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
