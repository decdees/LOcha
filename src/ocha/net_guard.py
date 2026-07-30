"""Outbound-network guard (NFR-3, T1.10).

Ocha has no cloud path. This makes that testable rather than aspirational, by
intercepting socket connections and refusing anything outside loopback and the
Tailscale CGNAT range.

It is a TEST instrument, not a runtime firewall -- a determined code path could
bypass it. Its job is to fail the build when someone adds an outbound call.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

# Tailscale hands out 100.64.0.0/10 (CGNAT). Loopback is the local model server,
# VOICEVOX on :50021, and the app itself.
ALLOWED_NETWORKS = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("100.64.0.0/10"),
)


class OutboundNetworkError(AssertionError):
    """Raised when a code path tries to reach the internet."""


def is_allowed(host: str) -> bool:
    if host in ("localhost", ""):
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        # A hostname that is not an IP literal means DNS, which means a name we
        # do not control -- treat as outbound.
        return False
    return any(addr in net for net in ALLOWED_NETWORKS)


@contextmanager
def no_outbound_network() -> Iterator[None]:
    """Fail loudly on any connection outside loopback/Tailscale."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def guard(self: Any, address: Any) -> Any:
        if isinstance(address, tuple) and address:
            host = str(address[0])
            if not is_allowed(host):
                raise OutboundNetworkError(
                    f"outbound connection to {host} -- Ocha must make no cloud calls (NFR-3)"
                )
        return real_connect(self, address)

    def guard_ex(self: Any, address: Any) -> Any:
        if isinstance(address, tuple) and address:
            host = str(address[0])
            if not is_allowed(host):
                raise OutboundNetworkError(
                    f"outbound connection to {host} -- Ocha must make no cloud calls (NFR-3)"
                )
        return real_connect_ex(self, address)

    # Patching a C-level method needs the Any-typed shims above; mypy cannot
    # express "same signature as socket.connect" for a wrapper.
    socket.socket.connect = guard  # type: ignore[method-assign]
    socket.socket.connect_ex = guard_ex  # type: ignore[method-assign]
    try:
        yield
    finally:
        socket.socket.connect = real_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = real_connect_ex  # type: ignore[method-assign]
