#!/usr/bin/env python3
"""Merge certifi's CA bundle with one or more corporate / proxy root certificates (PEM)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    # UTF-8 with BOM or ASCII PEM
    for enc in ("utf-8-sig", "utf-8", "ascii", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "corporate_cas",
        nargs="+",
        type=Path,
        help="PEM file(s) with your org root/intermediate CA(s). "
        "If you only have a .cer (DER), convert: "
        "openssl x509 -inform DER -in corp.cer -out corp.pem",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "ssl" / "cacert-combined.pem",
        help="Where to write the merged bundle (default: ./ssl/cacert-combined.pem)",
    )
    args = p.parse_args()

    try:
        import certifi
    except ImportError as e:
        print("Install certifi in the same environment: pip install certifi", file=sys.stderr)
        raise SystemExit(1) from e

    base_path = Path(certifi.where())
    parts: list[str] = [_read_text(base_path).rstrip()]

    for ca in args.corporate_cas:
        if not ca.is_file():
            print(f"Not found: {ca}", file=sys.stderr)
            return 1
        text = _read_text(ca).strip()
        if "BEGIN CERTIFICATE" not in text:
            print(
                f"Warning: {ca} does not look like PEM (expected -----BEGIN CERTIFICATE-----). "
                "Convert DER .cer to PEM or export Base-64 from certmgr.",
                file=sys.stderr,
            )
        parts.append(text)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged = "\n\n".join(parts) + "\n"
    args.output.write_text(merged, encoding="utf-8")
    print(f"Wrote {args.output} ({len(merged)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
