# SPDX-License-Identifier: MPL-2.0
"""Regression checks for precision-sensitive layers and archive integrity."""
import hashlib
from pathlib import Path
import struct
import sys

import numpy as np
import pytest
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compress_model import BLOCK_BYTES, BLOCK_HEADER, HEADER, MAGIC, new_output, pack, unpack
from convert_rmvpe import use_half_storage


@pytest.mark.parametrize("name,ndim", [
    ("unet.encoder.layers.0.conv.0.weight", 4),
    ("unet.encoder.layers.3.conv.0.weight", 4),
    ("unet.decoder.layers.4.conv.0.weight", 4),
    ("cnn.weight", 4), ("fc.1.weight", 2),
    ("gru.0.weight_ih", 2), ("mel.basis", 2),
    ("unet.intermediate.layers.0.conv.0.bias", 1),
])
def test_selected_profile_preserves_sensitive_tensors(name, ndim):
    assert not use_half_storage(name, ndim, "f16-intermediate")


def test_storage_profiles():
    name = "unet.intermediate.layers.0.conv.0.weight"
    assert use_half_storage(name, 4, "f16-intermediate")
    assert use_half_storage(name, 4, "f16")
    assert not use_half_storage(name, 4, "f32")
    assert use_half_storage("cnn.weight", 4, "f16")
    assert not use_half_storage("gru.0.weight_ih", 2, "f16")
    with pytest.raises(ValueError):
        use_half_storage(name, 4, "unknown")


@pytest.mark.parametrize("size", [4, BLOCK_BYTES, BLOCK_BYTES + 12])
def test_round_trip(tmp_path, size):
    data = np.random.default_rng(71).integers(0, 256, size, dtype=np.uint8).tobytes()
    source, archive, restored = [tmp_path / n for n in ("source", "model.bsz", "restored")]
    source.write_bytes(data)
    result = pack(source, archive)
    assert result["original_bytes"] == size
    result = unpack(archive, restored)
    assert result["model_sha256"] == hashlib.sha256(data).hexdigest()
    assert restored.read_bytes() == data


@pytest.mark.parametrize("damage", [
    "magic", "short_header", "short_block_header", "short_payload",
    "hash", "checksum", "trailing", "zero_block", "huge_block", "huge_packed",
    "wrong_content_size", "extra_frame", "no_checksum", "wrong_total",
])
def test_reject_damaged_archive(tmp_path, damage):
    source, archive, output = [tmp_path / n for n in ("source", "model.bsz", "output")]
    source.write_bytes(bytes(range(256)) * 4)
    pack(source, archive)
    data = bytearray(archive.read_bytes())
    offset = HEADER.size + BLOCK_HEADER.size
    if damage == "magic":
        data[0] ^= 1
    elif damage == "short_header":
        data = data[:10]
    elif damage == "short_block_header":
        data = data[:HEADER.size + 7]
    elif damage == "short_payload":
        data = data[:-1]
    elif damage == "hash":
        data[16] ^= 1
    elif damage == "checksum":
        data[-1] ^= 1
    elif damage == "trailing":
        data += b"extra"
    elif damage == "zero_block":
        struct.pack_into("<Q", data, HEADER.size, 0)
    elif damage == "huge_block":
        struct.pack_into("<Q", data, HEADER.size, BLOCK_BYTES + 4)
    elif damage == "huge_packed":
        struct.pack_into("<Q", data, HEADER.size + 8, 2**63)
    elif damage == "wrong_total":
        struct.pack_into("<Q", data, 8, 4)
    else:
        if damage == "extra_frame":
            payload = bytes(data[offset:]) + zstd.ZstdCompressor().compress(b"extra")
        else:
            payload = zstd.ZstdCompressor(write_checksum=damage != "no_checksum").compress(
                bytes(1028 if damage == "wrong_content_size" else 1024))
        data = data[:offset] + payload
        struct.pack_into("<Q", data, HEADER.size + 8, len(payload))
    archive.write_bytes(data)
    unrelated = tmp_path / "output.part"
    unrelated.write_bytes(b"keep")
    with pytest.raises((ValueError, zstd.ZstdError)):
        unpack(archive, output)
    assert not output.exists()
    assert unrelated.read_bytes() == b"keep"
    assert not list(tmp_path.glob(".output.*.part"))


def test_preserve_existing_output_and_input(tmp_path):
    source, archive, existing = [tmp_path / n for n in ("source", "model.bsz", "existing")]
    source.write_bytes(b"GGUF" * 20)
    existing.write_bytes(b"keep")
    pack(source, archive)
    with pytest.raises(FileExistsError):
        unpack(archive, existing)
    with pytest.raises(FileExistsError):
        pack(source, source)
    assert existing.read_bytes() == b"keep"
    assert source.read_bytes() == b"GGUF" * 20


def test_destination_created_during_write_is_preserved(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(FileExistsError):
        with new_output(output) as stream:
            stream.write(b"temporary")
            output.write_bytes(b"someone else's file")
    assert output.read_bytes() == b"someone else's file"
    assert not list(tmp_path.glob(".output.*.part"))


@pytest.mark.parametrize("data", [b"", b"abc"])
def test_invalid_input_does_not_publish(tmp_path, data):
    source, archive = tmp_path / "source", tmp_path / "model.bsz"
    source.write_bytes(data)
    with pytest.raises(ValueError):
        pack(source, archive)
    assert not archive.exists()


def test_existing_experimental_format(tmp_path):
    # Independently construct the format used by the recorded quality experiment.
    original = b"GGUF" + bytes(range(256)) * 2
    shuffled = b"".join(original[i::4] for i in range(4))
    frame = zstd.ZstdCompressor(level=9, write_checksum=True).compress(shuffled)
    archive, restored = tmp_path / "legacy.bsz", tmp_path / "restored"
    archive.write_bytes(MAGIC + len(original).to_bytes(8, "little") +
                        hashlib.sha256(original).digest() +
                        len(original).to_bytes(8, "little") + len(frame).to_bytes(8, "little") + frame)
    unpack(archive, restored)
    assert restored.read_bytes() == original
