"""Fetch refusal contract, driven through the public fetch path and the CLI.

The address table is the IANA IPv4 and IPv6 Special-Purpose Address Registries
(both last updated 2025-10-09):
https://www.iana.org/assignments/iana-ipv4-special-registry/
https://www.iana.org/assignments/iana-ipv6-special-registry/
Every entry is refused at its first and last address, as is every refused IPv4
entry wrapped in each IPv6 form that embeds an IPv4 address. DNS is stubbed, so
the tests run offline and exercise the resolved-address path.
"""

from __future__ import annotations

import contextlib
import io
import ipaddress
import json
import re
import socket
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pplx_agent_tools import cli_fetch, cli_runner, errors
from pplx_agent_tools.errors import EXIT_GENERIC, AuthError, PplxError, exit_code
from pplx_agent_tools.verbs import fetch as fetch_verb
from pplx_agent_tools.verbs.fetch import FetchResult, fetch_page

IANA_V4: list[tuple[str, str]] = [
    ("0.0.0.0/8", '"This network"'),
    ("0.0.0.0/32", '"This host on this network"'),
    ("10.0.0.0/8", "Private-Use"),
    ("100.64.0.0/10", "Shared Address Space"),
    ("127.0.0.0/8", "Loopback"),
    ("169.254.0.0/16", "Link Local"),
    ("172.16.0.0/12", "Private-Use"),
    ("192.0.0.0/24", "IETF Protocol Assignments"),
    ("192.0.0.0/29", "IPv4 Service Continuity Prefix"),
    ("192.0.0.8/32", "IPv4 dummy address"),
    ("192.0.0.9/32", "Port Control Protocol Anycast"),
    ("192.0.0.10/32", "Traversal Using Relays around NAT Anycast"),
    ("192.0.0.170/32", "NAT64/DNS64 Discovery"),
    ("192.0.0.171/32", "NAT64/DNS64 Discovery"),
    ("192.0.2.0/24", "Documentation (TEST-NET-1)"),
    ("192.31.196.0/24", "AS112-v4"),
    ("192.52.193.0/24", "AMT"),
    ("192.88.99.0/24", "Deprecated (6to4 Relay Anycast)"),
    ("192.88.99.2/32", "6a44-relay anycast address"),
    ("192.168.0.0/16", "Private-Use"),
    ("192.175.48.0/24", "Direct Delegation AS112 Service"),
    ("198.18.0.0/15", "Benchmarking"),
    ("198.51.100.0/24", "Documentation (TEST-NET-2)"),
    ("203.0.113.0/24", "Documentation (TEST-NET-3)"),
    ("240.0.0.0/4", "Reserved"),
    ("255.255.255.255/32", "Limited Broadcast"),
]

IANA_V6: list[tuple[str, str]] = [
    ("::1/128", "Loopback Address"),
    ("::/128", "Unspecified Address"),
    ("::ffff:0:0/96", "IPv4-mapped Address"),
    ("64:ff9b::/96", "IPv4-IPv6 Translat."),
    ("64:ff9b:1::/48", "IPv4-IPv6 Translat."),
    ("100::/64", "Discard-Only Address Block"),
    ("100:0:0:1::/64", "Dummy IPv6 Prefix"),
    ("2001::/23", "IETF Protocol Assignments"),
    ("2001::/32", "TEREDO"),
    ("2001:1::1/128", "Port Control Protocol Anycast"),
    ("2001:1::2/128", "Traversal Using Relays around NAT Anycast"),
    ("2001:1::3/128", "DNS-SD Service Registration Protocol Anycast"),
    ("2001:2::/48", "Benchmarking"),
    ("2001:3::/32", "AMT"),
    ("2001:4:112::/48", "AS112-v6"),
    ("2001:10::/28", "Deprecated (previously ORCHID)"),
    ("2001:20::/28", "ORCHIDv2"),
    ("2001:30::/28", "Drone Remote ID Protocol Entity Tags (DETs) Prefix"),
    ("2001:db8::/32", "Documentation"),
    ("2002::/16", "6to4"),
    ("2620:4f:8000::/48", "Direct Delegation AS112 Service"),
    ("3fff::/20", "Documentation"),
    ("5f00::/16", "Segment Routing (SRv6) SIDs"),
    ("fc00::/7", "Unique-Local"),
    ("fe80::/10", "Link-Local Unicast"),
]


def _ends(cidr: str) -> list[str]:
    net = ipaddress.ip_network(cidr)
    return sorted({str(net.network_address), str(net.broadcast_address)})


def _embeddings(v4: str) -> list[tuple[str, str]]:
    """The IPv6 forms that carry `v4`: mapped, NAT64, 6to4 and IPv4-compatible."""
    n = int(ipaddress.IPv4Address(v4))
    return [
        ("mapped", str(ipaddress.IPv6Address((0xFFFF << 32) | n))),
        ("nat64", str(ipaddress.IPv6Address((0x64FF9B << 96) | n))),
        ("6to4", str(ipaddress.IPv6Address((0x2002 << 112) | (n << 80) | 1))),
        ("compat", str(ipaddress.IPv6Address(n))),
    ]


BLOCKED: list[tuple[str, str]] = (
    [(f"{cidr} {name}", addr) for cidr, name in IANA_V4 + IANA_V6 for addr in _ends(cidr)]
    + [
        (f"{kind} {cidr} {name}", addr)
        for cidr, name in IANA_V4
        for kind, addr in _embeddings(str(ipaddress.ip_network(cidr).network_address))
    ]
    + [
        ("IPv4 multicast", "224.0.0.1"),
        ("IPv4 multicast (MCAST-TEST-NET)", "233.252.0.1"),
        ("IPv6 multicast", "ff02::1"),
        ("IPv6 global multicast", "ff0e::1"),
        ("Teredo embedding a public server", "2001:0:808:808::1"),
        ("IPv4-translated (SIIT)", "::ffff:0:808:808"),
        ("IPv6 outside 2000::/3", "4000::1"),
        ("IPv6 site-local (deprecated)", "fec0::1"),
        ("IPv6 link-local with scope", "fe80::1%1"),
    ]
)

ALLOWED: list[str] = [
    "8.8.8.8",
    "1.1.1.1",
    "2606:4700::1111",
    "2001:4860:4860::8888",
    "::ffff:8.8.8.8",
    "64:ff9b::808:808",
    "2002:808:808::1",
]

HOST = "target.test"


class _Resp:
    def __init__(self, status_code: int, *, location: str | None = None, text: str = "") -> None:
        self.status_code = status_code
        self.headers: dict[str, str] = {"location": location} if location else {}
        self.text = text


class _Session:
    """curl_cffi Session stand-in: queued responses, records requested URLs."""

    def __init__(self, *responses: _Resp) -> None:
        self._responses = list(responses)
        self.requested: list[str] = []
        self.auth: list[object] = []

    def get(self, url: str, **kwargs: object) -> _Resp:
        self.requested.append(url)
        self.auth.append(kwargs.get("auth"))
        return self._responses.pop(0)


def _stub_dns(monkeypatch: pytest.MonkeyPatch, answers: dict[str, list[str]]) -> None:
    def _getaddrinfo(host: str, *_: object, **__: object) -> list[tuple[Any, ...]]:
        if host not in answers:
            raise socket.gaierror(socket.EAI_NONAME, "stub: unknown host")
        return [
            (socket.AF_INET6 if ":" in a else socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, 0))
            for a in answers[host]
        ]

    monkeypatch.setattr(socket, "getaddrinfo", _getaddrinfo)


def _page(status: int = 200) -> _Resp:
    return _Resp(status, text="<html><body><p>hello world</p></body></html>")


# ---------- (1) address table ----------


@pytest.mark.parametrize(("label", "addr"), BLOCKED, ids=[f"{a} {b}" for b, a in BLOCKED])
def test_non_public_address_is_refused_without_a_request(
    monkeypatch: pytest.MonkeyPatch, label: str, addr: str
) -> None:
    _stub_dns(monkeypatch, {HOST: [addr]})
    sess = _Session(_page())
    with pytest.raises(PplxError) as ei:
        fetch_page(f"http://{HOST}/", HOST, max_chars=None, session=sess)  # type: ignore[arg-type]
    assert type(ei.value).__name__ == "BlockedUrlError", label
    assert exit_code(ei.value) == EXIT_GENERIC
    assert sess.requested == []


@pytest.mark.parametrize("addr", ALLOWED)
def test_public_address_is_fetched(monkeypatch: pytest.MonkeyPatch, addr: str) -> None:
    _stub_dns(monkeypatch, {HOST: [addr]})
    sess = _Session(_page())
    result = fetch_page(f"http://{HOST}/", HOST, max_chars=None, session=sess)  # type: ignore[arg-type]
    assert sess.requested == [f"http://{HOST}/"]
    assert "hello world" in result.content


def test_one_internal_answer_refuses_the_host(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_dns(monkeypatch, {HOST: ["8.8.8.8", "100.64.0.1"]})
    sess = _Session(_page())
    with pytest.raises(PplxError) as ei:
        fetch_page(f"http://{HOST}/", HOST, max_chars=None, session=sess)  # type: ignore[arg-type]
    assert type(ei.value).__name__ == "BlockedUrlError"
    assert sess.requested == []


def test_every_redirect_hop_is_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_dns(monkeypatch, {HOST: ["8.8.8.8"], "inner.test": ["100.64.0.1"]})
    sess = _Session(_Resp(302, location="http://inner.test/x"), _page())
    with pytest.raises(PplxError) as ei:
        fetch_page(f"http://{HOST}/", HOST, max_chars=None, session=sess)  # type: ignore[arg-type]
    assert type(ei.value).__name__ == "BlockedUrlError"
    assert sess.requested == [f"http://{HOST}/"]


# ---------- (2) blocked URL → exit 1, error.type BlockedUrlError ----------


@pytest.fixture
def _no_cookies(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(cls: object, **_: object) -> object:
        raise AuthError("no cookies found")

    monkeypatch.setattr(cli_runner.Client, "from_default_cookies", classmethod(_raise))


@pytest.mark.usefixtures("_no_cookies")
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://100.64.0.1/",
        "http://[::ffff:127.0.0.1]/",
        "http://[64:ff9b::a9fe:a9fe]/latest/meta-data/",
        "file:///etc/passwd",
        "gopher://example.com/",
        "http:///",
        "localhost:8080",
        "http://example.com:99999/",
        "http://[::1/",
    ],
)
def test_cli_refusal_exits_1_with_blocked_type(
    url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli_fetch.main(["--json", url]) == EXIT_GENERIC
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["type"] == "BlockedUrlError"
    assert payload["error"]["exit_code"] == EXIT_GENERIC


@pytest.mark.usefixtures("_no_cookies")
@pytest.mark.parametrize(
    "url",
    [
        "http://1.1.1.1./",
        "http://127.0.0.1./",
        "http://%31%32%37.0.0.1/",
        "http://exa%6dple.com/",
        "http://0177.0.0.1/",
        "http://[fe80::1%25lo0]/",
    ],
)
def test_malformed_host_exits_1(url: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_fetch.main(["--json", url]) == EXIT_GENERIC
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["type"] == "BlockedUrlError"


# ---------- userinfo: sent as auth, never shown ----------


def test_userinfo_is_sent_as_auth_and_never_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_dns(monkeypatch, {HOST: ["8.8.8.8"]})
    sess = _Session(_page())
    result = fetch_page(f"http://u%40x:p%3Aw@{HOST}/a?b=1", HOST, max_chars=None, session=sess)  # type: ignore[arg-type]
    assert sess.requested == [f"http://{HOST}/a?b=1"]
    assert sess.auth == [("u@x", "p:w")]
    assert result.url == f"http://{HOST}/a?b=1"


@pytest.mark.parametrize("status", [404, 429, 503])
def test_userinfo_is_not_in_error_messages(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    _stub_dns(monkeypatch, {HOST: ["8.8.8.8"]})
    sess = _Session(_page(status))
    with pytest.raises(PplxError) as ei:
        fetch_page(f"http://user:secret@{HOST}/", HOST, max_chars=None, session=sess)  # type: ignore[arg-type]
    assert "secret" not in str(ei.value) and "user" not in str(ei.value)


# RFC 3986 ends the authority at the first '/', '?' or '#', so a password
# holding one unencoded reads as host "user" and a port. The prefix is a marker
# no generated text can produce by accident.
_USER = st.text(st.characters(blacklist_characters="/?#", blacklist_categories=("Cs",)))
_PASSWORD = st.builds(
    lambda a, d, b: f"S3cr{a}{d}{b}",
    st.text(st.characters(blacklist_categories=("Cs",))),
    st.sampled_from("/?#"),
    st.text(st.characters(blacklist_categories=("Cs",))),
)


@pytest.mark.usefixtures("_no_cookies")
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(user=_USER, password=_PASSWORD, as_json=st.booleans())
def test_password_with_a_delimiter_is_refused_and_never_shown(
    user: str, password: str, as_json: bool
) -> None:
    url = f"http://{user}:{password}@example.com/"
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli_fetch.main(["--json", url] if as_json else [url])
    assert code == EXIT_GENERIC
    for stream in (out.getvalue(), err.getvalue()):
        assert "S3cr" not in stream, stream
    if as_json:
        assert json.loads(out.getvalue())["error"]["type"] == "BlockedUrlError"


def test_numeric_password_with_a_delimiter_is_refused_before_dns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # "user:1234" is a valid host and port, so only the later '@' shows the password.
    _stub_dns(monkeypatch, {})
    with pytest.raises(PplxError) as ei:
        fetch_page("http://user:1234/x@example.com/", "example.com", max_chars=None)
    assert isinstance(ei.value, errors.BlockedUrlError), repr(ei.value)
    assert "1234" not in str(ei.value)


def test_userinfo_is_not_in_refusal_messages() -> None:
    with pytest.raises(PplxError) as ei:
        fetch_page("http://user:secret@127.0.0.1/", "127.0.0.1", max_chars=None)
    assert "secret" not in str(ei.value) and "user" not in str(ei.value)


# ---------- target HTTP status → exit code ----------


@pytest.mark.parametrize(
    ("status", "type_name", "code"),
    [
        (404, "TargetHttpError", 1),
        (403, "TargetHttpError", 1),
        (410, "TargetHttpError", 1),
        (408, "NetworkError", 4),
        (429, "RateLimitError", 3),
        (500, "NetworkError", 4),
        (503, "NetworkError", 4),
    ],
)
def test_target_status_maps_to_retry_semantic(
    monkeypatch: pytest.MonkeyPatch, status: int, type_name: str, code: int
) -> None:
    _stub_dns(monkeypatch, {HOST: ["8.8.8.8"]})
    with pytest.raises(PplxError) as ei:
        fetch_page(f"http://{HOST}/", HOST, max_chars=None, session=_Session(_page(status)))  # type: ignore[arg-type]
    assert type(ei.value).__name__ == type_name
    assert exit_code(ei.value) == code


def test_redirect_limit_is_a_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_dns(monkeypatch, {HOST: ["8.8.8.8"]})
    sess = _Session(*[_Resp(302, location=f"http://{HOST}/") for _ in range(6)])
    with pytest.raises(PplxError) as ei:
        fetch_page(f"http://{HOST}/", HOST, max_chars=None, session=sess)  # type: ignore[arg-type]
    assert type(ei.value).__name__ == "TargetHttpError"
    assert exit_code(ei.value) == EXIT_GENERIC
    assert len(sess.requested) == 6


# ---------- (3) every PplxError subclass has a documented exit code ----------

EXPECTED_EXIT = {
    "PplxError": 1,
    "AuthError": 2,
    "RateLimitError": 3,
    "NetworkError": 4,
    "StreamDeadlineError": 4,
    "StreamStallError": 4,
    "AntiBotError": 5,
    "SchemaError": 1,
    "BlockedUrlError": 1,
    "TargetHttpError": 1,
}


def _walk(cls: type[PplxError]) -> list[type[PplxError]]:
    out = [cls]
    for sub in cls.__subclasses__():
        out.extend(_walk(sub))
    return out


def _documented_error_codes() -> set[int]:
    skill = (Path(__file__).resolve().parent.parent / "SKILL.md").read_text()
    table = skill.split("# Exit codes", 1)[1].split("\n\n", 2)[1]
    codes = {int(m) for m in re.findall(r"^\| (\d+) \|", table, flags=re.MULTILINE)}
    # 0 is success and 6 is a partial result; neither is an error's code.
    return codes - {errors.EXIT_OK, errors.EXIT_PARTIAL}


def test_every_pplx_error_class_has_its_documented_exit_code() -> None:
    classes = [c for c in _walk(PplxError) if c.__module__ == errors.__name__]
    assert sorted(c.__name__ for c in classes) == sorted(EXPECTED_EXIT)
    documented = _documented_error_codes()
    assert documented == {1, 2, 3, 4, 5}
    for cls in classes:
        code = exit_code(cls.__new__(cls))
        assert code == EXPECTED_EXIT[cls.__name__], cls.__name__
        assert code in documented, cls.__name__


# ---------- (4) domain never carries credentials ----------


@pytest.mark.usefixtures("_no_cookies")
@pytest.mark.parametrize(
    ("url", "domain"),
    [
        ("https://u:p@host.test/", "host.test"),
        ("https://user@Host.Test:8443/x", "host.test:8443"),
        ("http://u:p@[2606:4700::1111]:8080/", "[2606:4700::1111]:8080"),
    ],
)
def test_json_domain_has_no_userinfo(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], url: str, domain: str
) -> None:
    def _fake_page(url: str, domain: str, **_: Any) -> FetchResult:
        return FetchResult(url=url, title=None, domain=domain, content="x", is_extracted=False)

    monkeypatch.setattr(fetch_verb, "fetch_page", _fake_page)
    assert cli_fetch.main(["--json", url]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["domain"] == domain
    assert "u:p" not in payload["domain"] and "user@" not in payload["domain"]
