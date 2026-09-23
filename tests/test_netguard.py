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
    DnsName,
    HttpUrl,
    PublicAddress,
    PublicUrl,
    check_url,
    classify,
    parse_url,
    redact,
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
        PublicUrl(parts=parse_url("http://x.test/"), addresses=())


def test_public_url_literal_host_must_be_its_address() -> None:
    with pytest.raises(ValueError, match="literal host"):
        PublicUrl(
            parts=parse_url("http://127.0.0.1/"),
            addresses=(PublicAddress(ipaddress.ip_address("8.8.8.8")),),
        )


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


@pytest.mark.parametrize(
    ("url", "canonical"),
    [
        ("HTTP://Example.COM", "http://example.com/"),
        ("https://example.com:8443/a/b?q=1#frag", "https://example.com:8443/a/b?q=1"),
        ("http://u:p@example.com:/x", "http://example.com/x"),
        ("http://127.0.0.1/", "http://127.0.0.1/"),
        ("http://[0:0:0:0:0:ffff:7f00:1]/", "http://[::ffff:127.0.0.1]/"),
        ("http://example.com./", "http://example.com./"),
        ("http://0x7f.example.com/", "http://0x7f.example.com/"),
        ("http://127.0.0.1.nip.io/", "http://127.0.0.1.nip.io/"),
        ("http://\u24dbocalhost/", "http://localhost/"),
        ("http://\uff11\uff12\uff17.\uff10.\uff10.\uff11/", "http://127.0.0.1/"),
        ("http://127\u30020.0.1/", "http://127.0.0.1/"),
        ("http://bücher.example/", "http://xn--bcher-kva.example/"),
    ],
)
def test_parse_url_rebuilds_one_canonical_spelling(url: str, canonical: str) -> None:
    assert parse_url(url).url == canonical


@pytest.mark.parametrize(
    "host",
    [
        # octal, hex, short, integer and trailing-dot IPv4 forms
        "0177.0.0.1",
        "012.0.0.1",
        "0144.0100.0.1",
        "127.000.000.001",
        "0x7f.0.0.1",
        "0x7f000001",
        "0x",
        "2130706433",
        "017700000001",
        "127.1",
        "1.2.3",
        "0",
        "1.1.1.1.",
        "256.0.0.1",
        "example.123",
        "example.0x1f",
        # percent-encoding, zone ids, characters outside LDH, bad labels
        "%31%32%37.0.0.1",
        "[fe80::1%25lo0]",
        "[::1%lo0]",
        "[127.0.0.1]",
        "[::ffff:0177.0.0.1]",
        "a..b",
        "exa mple.com",
        "exa\\mple.com",
        "a" * 64 + ".com",
        "[::1",
        "example.com:0",
        "example.com:65536",
        "example.com:8a",
        "example.com:80:80",
        "[::1]x",
    ],
)
def test_parse_url_refuses_ambiguous_or_malformed_hosts(host: str) -> None:
    with pytest.raises(BlockedUrlError):
        parse_url(f"http://{host}/")


@pytest.mark.parametrize("url", ["http:///", "http://:80/", "http://user@/"])
def test_parse_url_refuses_missing_host(url: str) -> None:
    with pytest.raises(BlockedUrlError, match="no host"):
        parse_url(url)


def test_dns_name_rejects_numeric_names() -> None:
    with pytest.raises(ValueError, match="dotted-decimal"):
        DnsName("0177.0.0.1")


def test_http_url_rejects_zone_id() -> None:
    with pytest.raises(ValueError, match="zone id"):
        HttpUrl("http", ipaddress.IPv6Address("fe80::1%1"), None, "/")


def test_literal_host_is_checked_without_the_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(*_a: object, **_k: object) -> object:
        raise AssertionError("resolver called for an IP literal")

    monkeypatch.setattr(socket, "getaddrinfo", _fail)
    assert [str(a.ip) for a in check_url("http://[2606:4700::1111]:8080/").addresses] == [
        "2606:4700::1111"
    ]


@pytest.mark.parametrize(
    ("url", "shown"),
    [
        ("http://u:p@host.test/x", "http://host.test/x"),
        ("https://a@b@host.test/", "https://host.test/"),
        ("http://host.test/@x", "http://host.test/@x"),
        ("http://host.test/?next=//u:p@y", "http://host.test/?next=//u:p@y"),
        ("not a url", "not a url"),
    ],
)
def test_redact_drops_only_userinfo(url: str, shown: str) -> None:
    assert redact(url) == shown
