"""Check the Sage stub contracts, including expected type errors."""

import subprocess
import sys
from pathlib import Path


def test_sage_typing_contracts():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyright",
            "--pythonpath",
            sys.executable,
            "tests/typing/contracts.py",
            "typings",
        ],
        check=False,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
