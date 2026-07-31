"""Minimal HTTPS server for the iOS audio test (T2.1 item 5).

getUserMedia requires a secure context. Over plain HTTP on a LAN address iOS
Safari blocks the API outright, so the test cannot even start -- and a blocked
API would look like a failure of (a) for entirely the wrong reason.

Tailscale would give a real certificate, but it is not set up yet. This serves
the same page over HTTPS with a self-signed certificate; iOS lets you tap through
the warning, and once accepted the origin IS a secure context, so getUserMedia
behaves exactly as it will in production.

ponytail: stdlib http.server + ssl, no framework. This runs a handful of times
and then gets deleted.
"""

from __future__ import annotations

import functools
import http.server
import socket
import ssl
import sys
from pathlib import Path

CERT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/tls")
PORT = 8443


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no packet is sent; this just picks the route
        return str(s.getsockname()[0])
    finally:
        s.close()


def main() -> None:
    # directory is a CONSTRUCTOR argument, not a class attribute. Setting it on
    # the class is silently ignored -- __init__ does `self.directory = directory
    # or os.getcwd()` -- so the server quietly served the repo root and every
    # request 404'd.
    root = Path(__file__).parent
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    httpd = http.server.HTTPServer(("0.0.0.0", PORT), handler)
    assert (root / "ios-audio-test.html").exists(), f"page missing under {root}"

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(CERT_DIR / "cert.pem", CERT_DIR / "key.pem")
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

    print(f"\n  Open this on the iPhone:\n\n    https://{lan_ip()}:{PORT}/ios-audio-test.html\n")
    print("  Safari will warn about the certificate. Tap Show Details -> visit this website.")
    print("  Ctrl-C to stop.\n")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
