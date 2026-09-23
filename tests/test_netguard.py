"""Unit tests for netguard: the address classifier and the PublicUrl brand."""

from __future__ import annotations

import ipaddress
import re
import socket

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pplx_agent_tools.errors import BlockedUrlError, NetworkError
from pplx_agent_tools.netguard import (
    BlockedAddress,
    PublicAddress,
    PublicUrl,
    check_url,
    classify,
)
from tests.test_fetch_refusals import ALLOWED, BLOCKED, IANA_V4, IANA_V6


@pytest.mark.parametrize(("label", "addr"), BLOCKED, ids=[f"{a} {b}" for b, a in BLOCKED])
def test_classify_blocks(label: str, addr: str) -> None:
    verdict = classify(ipaddress.ip_address(addr))
    assert isinstance(verdict, BlockedAddress), label
    assert verdict.reason


@pytest.mark.parametrize("addr", ALLOWED)
def test_classify_allows(addr: str) -> None:
    assert classify(ipaddress.ip_address(addr)) == PublicAddress(ipaddress.ip_address(addr))


@pytest.mark.parametrize("addr", [a for _, a in BLOCKED])
def test_public_address_cannot_wrap_a_blocked_address(addr: str) -> None:
    with pytest.raises(ValueError, match="not public"):
        PublicAddress(ipaddress.ip_address(addr))


def test_public_url_needs_an_address() -> None:
    with pytest.raises(ValueError):
        PublicUrl(url="http://x.test/", host="x.test", addresses=())


_SPECIAL = [ipaddress.ip_network(c) for c, _ in IANA_V4 + IANA_V6]
_EMBEDDING = [ipaddress.ip_network(c) for c in ("::ffff:0:0/96", "64:ff9b::/96", "2002::/16")]


@given(st.ip_addresses())
def test_nothing_inside_a_registry_entry_is_public(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> None:
    verdict = classify(ip)
    if isinstance(verdict, PublicAddress):
        assert not ip.is_multicast
        # The only registry entries that can hold a public address are the
        # three that embed a public IPv4 address.
        inside = [n for n in _SPECIAL if n.version == ip.version and ip in n]
        assert all(n in _EMBEDDING for n in inside), (ip, inside)
        if not inside:
            assert ip.is_global


def test_check_url_blocks_hostname_resolving_internal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *_a, **_k: [(socket.AF_INET, 1, 6, "", ("100.64.0.1", 0))]
    )
    with pytest.raises(BlockedUrlError, match=re.escape("100.64.0.0/10 Shared Address Space")):
        check_url("https://cgnat.test/")


def test_check_url_dns_failure_is_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(*_a: object, **_k: object) -> object:
        raise socket.gaierror(socket.EAI_AGAIN, "temporary failure")

    monkeypatch.setattr(socket, "getaddrinfo", _fail)
    with pytest.raises(NetworkError):
        check_url("https://flaky.test/")


def test_check_url_keeps_every_public_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_a, **_k: [
            (socket.AF_INET, 1, 6, "", ("8.8.8.8", 0)),
            (socket.AF_INET6, 1, 6, "", ("2001:4860:4860::8888", 0, 0, 0)),
        ],
    )
    target = check_url("https://dual.test/p")
    assert target.host == "dual.test"
    assert [str(a.ip) for a in target.addresses] == ["8.8.8.8", "2001:4860:4860::8888"]
