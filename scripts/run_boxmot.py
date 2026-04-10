#!/usr/bin/env python3
"""
Launch BoxMOT with SSL that trusts the Windows certificate store (managed-network CAs).

Use this instead of `boxmot.exe` when you see SSL errors after IT has installed
the proxy/root CA in Windows (Group Policy), even if REQUESTS_CA_BUNDLE alone did not help.

Requires: pip install truststore uv
  (BoxMOT installs optional deps with `uv pip install`; both must be on PATH — this script
   prepends the venv Scripts folder so `uv` is found even if you did not run Activate.ps1.)
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def main() -> None:
    # Same directory as this interpreter: ...\.venv\Scripts — contains uv.exe after `pip install uv`
    scripts_dir = Path(sys.executable).resolve().parent
    os.environ["PATH"] = str(scripts_dir) + os.pathsep + os.environ.get("PATH", "")

    if not shutil.which("uv"):
        print(
            "The `uv` executable was not found next to this Python.\n"
            "  .venv\\Scripts\\pip install uv\n"
            "BoxMOT uses `uv pip install` to pull YOLOX and other detector extras.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        import truststore
    except ImportError as e:
        print(
            "Missing package: truststore\n"
            "  .venv\\Scripts\\pip install truststore\n"
            "Or use the merged CA bundle from scripts/Configure-PythonSSL.ps1",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    truststore.inject_into_ssl()

    sys.argv[0] = "boxmot"
    from boxmot.engine.cli import main as boxmot_main

    boxmot_main()


if __name__ == "__main__":
    main()
