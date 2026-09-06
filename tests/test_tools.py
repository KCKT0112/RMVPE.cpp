# SPDX-License-Identifier: MPL-2.0
"""Model-independent CLI rejection tests; RMVPE_CLI selects the built executable."""
import os
from pathlib import Path
import subprocess
import pytest

CLI = Path(os.environ.get("RMVPE_CLI", "build-cpu/bin/rmvpe-cli.exe"))
@pytest.mark.parametrize("args", [
    [], ["--unknown"], ["--threads"], ["--threads", "4bad"],
    ["--model", "missing.gguf", "--wav", "a", "--mel", "b"],
    ["--model", "missing.gguf", "--mel", "a", "--runs", "0"],
    ["--model", "missing.gguf", "--mel", "a", "--threshold", "nan"],
])
def test_invalid_cli(args):
    result = subprocess.run([str(CLI), *args], capture_output=True, text=True)
    assert result.returncode != 0
    assert "rmvpe:" in result.stderr

def test_help_and_backends():
    assert subprocess.run([str(CLI), "--help"], capture_output=True).returncode == 0
    result = subprocess.run([str(CLI), "--list-backends"], capture_output=True, text=True, check=True)
    assert "CPU" in result.stdout

def test_invalid_model(tmp_path):
    model = tmp_path / "bad.gguf"
    model.write_bytes(b"not a gguf")
    result = subprocess.run([str(CLI), "--model", str(model), "--mel", "unused"], capture_output=True)
    assert result.returncode != 0
