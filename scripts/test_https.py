#!/usr/bin/env python3
"""Quick HTTPS probe: default ssl vs truststore vs optional REQUESTS_CA_BUNDLE."""

from __future__ import annotations

import os
import ssl
import sys
import urllib.error
import urllib.request


def _get(url: str, ctx: ssl.SSLContext | None) -> tuple[int, str]:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=15, context=ctx) as r:
            return r.status, str(r.geturl())
    except urllib.error.HTTPError as e:
        return e.code, str(e.reason)
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"


def main() -> int:
    urls = (
        "https://pypi.org",
        "https://github.com",
        "https://drive.google.com",
    )

    print("--- Default ssl (certifi / system defaults as loaded by urllib) ---")
    for u in urls:
        code, msg = _get(u, None)
        print(f"  {u} -> {code} {msg}")

    try:
        import truststore
    except ImportError:
        print("\ntruststore not installed; skip OS store test. pip install truststore")
        return 0

    print("\n--- After truststore.inject_into_ssl() (Windows cert store) ---")
    truststore.inject_into_ssl()
    for u in urls:
        code, msg = _get(u, None)
        print(f"  {u} -> {code} {msg}")

    bundle = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if bundle and os.path.isfile(bundle):
        print(f"\n--- Explicit bundle file: {bundle} ---")
        ctx = ssl.create_default_context(cafile=bundle)
        for u in urls:
            code, msg = _get(u, ctx)
            print(f"  {u} -> {code} {msg}")
    elif bundle:
        print(f"\nSSL_CERT_FILE / REQUESTS_CA_BUNDLE set but not a file: {bundle}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
