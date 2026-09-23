"""Outbound-address policy for user-supplied fetch URLs (SSRF guard).

`classify` is the only judge of an address, and `PublicAddress` is the only
value it allows. `parse_url` turns a URL string into an `HttpUrl` whose host has
exactly one reading; `check_url` resolves it into a `PublicUrl`, which holds only
`PublicAddress`es, and the fetch path hands curl a `PublicUrl`'s rebuilt URL
alone, never the caller's string.

Address data: the IANA IPv4 and IPv6 Special-Purpose Address Registries, both
last updated 2025-10-09:
https://www.iana.org/assignments/iana-ipv4-special-registry/
https://www.iana.org/assignments/iana-ipv6-special-registry/
"""

from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass, field
from typing import Literal, TypeAlias
from urllib.parse import unquote, urljoin, urlsplit

import idna

from .errors import BlockedUrlError, NetworkError

IPAddress: TypeAlias = ipaddress.IPv4Address | ipaddress.IPv6Address
Scheme: TypeAlias = Literal["http", "https"]

_SCHEMES: dict[str, Scheme] = {"http": "http", "https": "https"}

# Every registry entry is refused, including the few the registry marks globally
# reachable (anycast services, AS112, AMT, ORCHIDv2, DETs): none of them serves
# web pages, and "in the registry means refused" needs no exceptions.
_SPECIAL_V4: tuple[tuple[ipaddress.IPv4Network, str], ...] = tuple(
    (ipaddress.IPv4Network(cidr), name)
    for cidr, name in (
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
    )
)

# The registry's three IPv4-embedding blocks (::ffff:0:0/96 IPv4-mapped,
# 64:ff9b::/96 NAT64, 2002::/16 6to4) are absent: `_embedded_ipv4` judges those
# by the IPv4 address they carry, so DNS64 answers for public hosts still work.
# Teredo (2001::/32) is covered by 2001::/23 and refused whatever it embeds.
_SPECIAL_V6: tuple[tuple[ipaddress.IPv6Network, str], ...] = tuple(
    (ipaddress.IPv6Network(cidr), name)
    for cidr, name in (
        ("::1/128", "Loopback Address"),
        ("::/128", "Unspecified Address"),
        ("64:ff9b:1::/48", "IPv4-IPv6 Translat. (local-use)"),
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
        ("2620:4f:8000::/48", "Direct Delegation AS112 Service"),
        ("3fff::/20", "Documentation"),
        ("5f00::/16", "Segment Routing (SRv6) SIDs"),
        ("fc00::/7", "Unique-Local"),
        ("fe80::/10", "Link-Local Unicast"),
    )
)

_IPV4_MAPPED = ipaddress.IPv6Network("::ffff:0:0/96")
_NAT64 = ipaddress.IPv6Network("64:ff9b::/96")
# RFC 4291 2.5.5.1, deprecated; the kernel may still route the embedded IPv4.
_IPV4_COMPATIBLE = ipaddress.IPv6Network("::/96")
_GLOBAL_UNICAST_V6 = ipaddress.IPv6Network("2000::/3")


def _registry_entry(ip: IPAddress) -> str | None:
    table = _SPECIAL_V4 if isinstance(ip, ipaddress.IPv4Address) else _SPECIAL_V6
    matches = [(net, name) for net, name in table if ip in net]
    if not matches:
        return None
    net, name = max(matches, key=lambda m: m[0].prefixlen)
    return f"IANA special-purpose {net} {name}"


def _embedded_ipv4(ip: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    if ip in _IPV4_MAPPED or ip in _NAT64:
        return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return ip.sixtofour


def _block_reason(ip: IPAddress) -> str | None:
    """Why `ip` is not a fetch target, or None when it is globally routable."""
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.scope_id:
            return "scoped (interface-local) address"
        embedded = _embedded_ipv4(ip)
        if embedded is not None:
            inner = _block_reason(embedded)
            return None if inner is None else f"embeds {embedded}: {inner}"
    return _registry_entry(ip) or _unrouted(ip)


def _unrouted(ip: IPAddress) -> str | None:
    """Reasons outside the special-purpose registry."""
    if ip.is_multicast:
        return "multicast"
    if isinstance(ip, ipaddress.IPv6Address):
        if ip in _IPV4_COMPATIBLE:
            return "IPv4-compatible (deprecated)"
        if ip not in _GLOBAL_UNICAST_V6:
            return "outside IPv6 global unicast 2000::/3"
    # Catches anything this Python's own registry knows that the tables lack.
    return None if ip.is_global else "not globally routable"


@dataclass(frozen=True)
class PublicAddress:
    """An address `classify` allows. Constructing one for any other address fails."""

    ip: IPAddress

    def __post_init__(self) -> None:
        reason = _block_reason(self.ip)
        if reason is not None:
            raise ValueError(f"{self.ip} is not public: {reason}")


@dataclass(frozen=True)
class BlockedAddress:
    ip: IPAddress
    reason: str


def classify(ip: IPAddress) -> PublicAddress | BlockedAddress:
    reason = _block_reason(ip)
    if reason is None:
        return PublicAddress(ip)
    return BlockedAddress(ip, reason)


_LABEL = re.compile(r"[a-z0-9_-]{1,63}")
# WHATWG's "ends in a number": curl and inet_aton read such a host as IPv4 in
# octal, hex, short or integer form, and platforms disagree on those readings
# (Darwin's getaddrinfo reads 0177.0.0.1 as decimal), so only the canonical
# dotted-decimal form, parsed by `ipaddress`, is accepted.
_NUMERIC_LABEL = re.compile(r"[0-9]+|0x[0-9a-f]*")
_PORT = re.compile(r"[0-9]{1,5}")
# Scheme and "//", or nothing, since curl and people read a scheme-less
# `user:pass@host` as userinfo; then the RFC 3986 authority.
_AUTHORITY = re.compile(r"((?:[A-Za-z][A-Za-z0-9+.-]*:)?//|)([^/?#]*)")


def _dns_name_error(name: str) -> str | None:
    bare = name[:-1] if name.endswith(".") else name
    labels = bare.split(".")
    if not bare or len(bare) > 253 or not all(_LABEL.fullmatch(label) for label in labels):
        return f"invalid host name {name!r}"
    if _NUMERIC_LABEL.fullmatch(labels[-1]):
        return f"numeric host {name!r} is not a dotted-decimal IPv4 address"
    return None


@dataclass(frozen=True)
class DnsName:
    """A lowercase ASCII (IDNA) host name that no parser reads as an address."""

    name: str

    def __post_init__(self) -> None:
        error = _dns_name_error(self.name)
        if error is not None:
            raise ValueError(error)


Host: TypeAlias = IPAddress | DnsName


@dataclass(frozen=True)
class HttpUrl:
    """An http(s) URL held as parts; `url` is the only string curl is given."""

    scheme: Scheme
    host: Host
    port: int | None
    path: str  # path plus "?query"
    auth: tuple[str, str] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.host, ipaddress.IPv6Address) and self.host.scope_id:
            raise ValueError(f"host [{self.host}] has a zone id")
        if self.port is not None and not 1 <= self.port <= 65535:
            raise ValueError(f"port {self.port} out of range")
        if not self.path.startswith("/") or "#" in self.path:
            raise ValueError(f"path {self.path!r} must start with '/' and hold no fragment")

    @property
    def host_text(self) -> str:
        if isinstance(self.host, DnsName):
            return self.host.name
        if isinstance(self.host, ipaddress.IPv6Address):
            return f"[{self.host}]"
        return str(self.host)

    @property
    def url(self) -> str:
        port = "" if self.port is None else f":{self.port}"
        return f"{self.scheme}://{self.host_text}{port}{self.path}"


def _userinfo_span(url: str) -> tuple[int, int, bool] | None:
    """Start and end of what in `url` may be a user:password, if anything.

    The third item is False for RFC 3986 userinfo: the authority, which ends at
    the first '/', '?' or '#', up to its last '@'. It is True when the span is
    only credential-shaped: an authority whose host:port holds a ':', then a
    later '@', as in `user:pa/ss@host` (a password with an unencoded '/'). That
    '@' may be path text, so only messages drop such a span, never results.
    """
    m = _AUTHORITY.match(url)
    if m is None:
        return None
    start, end = m.end(1), m.end(2)
    at = url.rfind("@", start, end)
    last = url.rfind("@", end)
    if last >= 0 and ":" in url[max(at + 1, start) : end]:
        return start, last + 1, True
    return (start, at + 1, False) if at >= 0 else None


def strip_userinfo(url: str) -> str:
    """`url` without its RFC 3986 userinfo, for results and requests.

    A URL with a malformed host:port gets `redact` instead: it is refused
    anyway, and its "port" may be the head of a password.
    """
    m = _AUTHORITY.match(url)
    if m is None:
        return url
    at = url.rfind("@", m.end(1), m.end(2))
    try:
        _split_hostport(url[max(at + 1, m.end(1)) : m.end(2)])
    except ValueError:
        return redact(url)
    return url if at < 0 else url[: m.end(1)] + url[at + 1 :]


def redact(url: str) -> str:
    """`url` for messages: without its userinfo or anything shaped like one."""
    span = _userinfo_span(url)
    return url if span is None else url[: span[0]] + url[span[1] :]


def _detail(error: ValueError, url: str) -> str:
    """A parser's message about `url`, which may quote its netloc, made safe to show."""
    span = _userinfo_span(url)
    if span is None:
        return str(error)
    if span[2]:
        # The parser's text may quote the password's head as a port or host.
        return "malformed authority (a password must percent-encode '/', '?' and '#')"
    return str(error).replace(url[span[0] : span[1]], "")


def check_authority(url: str) -> None:
    """Raise BlockedUrlError when `url`'s RFC 3986 host:port is malformed.

    `fetch --prompt` sends the URL on rather than parsing it, so it needs this
    much: a password holding an unencoded '/', '?' or '#' leaves a
    malformed port like the "pa" of `user:pa/ss@host`.
    """
    m = _AUTHORITY.match(url)
    if m is None or (not m.group(1) and "@" not in url):
        return
    try:
        host, bracketed, _ = _split_hostport(m.group(2).rpartition("@")[2])
        if m.group(1) and not host and not bracketed:
            raise ValueError("no host")
    except ValueError as e:
        raise BlockedUrlError(f"fetch {redact(url)}: malformed URL: {_detail(e, url)}") from e


def join_location(base: str, location: str) -> str:
    """Resolve a redirect `location` against `base`, or raise BlockedUrlError.

    A Location with a scheme or a leading '//' must name its host: urljoin
    reads `http:///x` as path /x on `base`'s host, WHATWG as host "x".
    """
    shown = f"fetch {redact(base)}: redirect to {redact(location)!r}"
    try:
        parts = urlsplit(location)
        if (parts.scheme or location.startswith("//")) and not parts.netloc:
            raise BlockedUrlError(f"{shown}: Location has no host")
        return urljoin(base, location)
    except ValueError as e:
        raise BlockedUrlError(f"{shown}: malformed Location: {_detail(e, location)}") from e


def _parse_host(text: str, bracketed: bool) -> Host:
    if "%" in text:
        # A zone id in brackets; percent-encoding otherwise, which curl decodes.
        raise ValueError(f"host {text!r} contains '%'")
    if bracketed:
        return ipaddress.IPv6Address(text)
    # UTS 46 non-transitional, as curl and browsers use, so "straße" keeps its ß.
    # Both getaddrinfo and curl get this A-label, and curl's own IDN conversion
    # never runs. ASCII is only lowercased: UTS 46 hyphen rules would refuse
    # real hosts such as r3---sn-abc.googlevideo.com.
    ascii_host = (
        text.lower()
        if text.isascii()
        else idna.encode(text, uts46=True, transitional=False).decode("ascii")
    )
    try:
        return ipaddress.IPv4Address(ascii_host)
    except ValueError:
        return DnsName(ascii_host)


def _split_hostport(hostport: str) -> tuple[str, bool, int | None]:
    if hostport.startswith("["):
        end = hostport.find("]")
        if end < 0:
            raise ValueError("unclosed '['")
        host, rest, bracketed = hostport[1:end], hostport[end + 1 :], True
    else:
        host, sep, port_text = hostport.partition(":")
        rest, bracketed = sep + port_text, False
    if rest in ("", ":"):
        return host, bracketed, None
    if not rest.startswith(":") or not _PORT.fullmatch(rest[1:]):
        raise ValueError(f"bad port {rest!r}")
    return host, bracketed, int(rest[1:])


def parse_url(url: str) -> HttpUrl:
    """Parse `url` into an `HttpUrl` or raise BlockedUrlError.

    Refuses a non-http(s) scheme, a missing host, a bad port, and any host
    other than canonical dotted-decimal IPv4, bracketed IPv6 without a zone id,
    or a DNS name whose last label is not numeric.
    """
    shown = redact(url)
    try:
        parts = urlsplit(url)
    except ValueError as e:
        raise BlockedUrlError(f"fetch {shown}: malformed URL: {_detail(e, url)}") from e
    scheme = _SCHEMES.get(parts.scheme)
    if scheme is None:
        raise BlockedUrlError(
            f"fetch {shown}: unsupported URL scheme {parts.scheme!r} (only http/https allowed)"
        )
    userinfo, _, hostport = parts.netloc.rpartition("@")
    try:
        host_text, bracketed, port = _split_hostport(hostport)
        if not host_text and not bracketed:
            raise BlockedUrlError(f"fetch {shown}: URL has no host")
        user, _, password = userinfo.partition(":")
        return HttpUrl(
            scheme=scheme,
            host=_parse_host(host_text, bracketed),
            port=port,
            path=(parts.path or "/") + (f"?{parts.query}" if parts.query else ""),
            auth=(unquote(user), unquote(password)) if userinfo else None,
        )
    except ValueError as e:  # includes UnicodeError from IDNA encoding
        raise BlockedUrlError(f"fetch {shown}: malformed host: {_detail(e, url)}") from e


@dataclass(frozen=True)
class PublicUrl:
    """An `HttpUrl` whose host had only public addresses when checked.

    An IP-literal host must be its one address. For a DNS name `addresses` is
    what the resolver answered; curl resolves the name again at connect time,
    so a DNS answer that changes in between is not covered here.
    """

    parts: HttpUrl
    addresses: tuple[PublicAddress, ...]

    def __post_init__(self) -> None:
        if not self.addresses:
            raise ValueError(f"{self.url}: a PublicUrl needs at least one address")
        host = self.parts.host
        if not isinstance(host, DnsName) and [a.ip for a in self.addresses] != [host]:
            raise ValueError(f"{self.url}: addresses must be exactly the literal host {host}")

    @property
    def url(self) -> str:
        return self.parts.url

    @property
    def host(self) -> str:
        return self.parts.host_text

    @property
    def auth(self) -> tuple[str, str] | None:
        return self.parts.auth


def _resolve(shown: str, name: str) -> list[IPAddress]:
    try:
        infos = socket.getaddrinfo(name, None)
    except ValueError as e:
        raise BlockedUrlError(f"fetch {shown}: invalid host {name!r}: {e}") from e
    except OSError as e:
        raise NetworkError(f"fetch {shown}: cannot resolve host {name!r}: {e}") from e
    addrs: list[IPAddress] = []
    for info in infos:
        raw = info[4][0]
        if not isinstance(raw, str):
            raise BlockedUrlError(f"fetch {shown}: host {name!r} resolves to non-IP {raw!r}")
        try:
            addrs.append(ipaddress.ip_address(raw))
        except ValueError as e:
            raise BlockedUrlError(f"fetch {shown}: host {name!r} resolves to {raw!r}: {e}") from e
    return addrs


def check_url(url: str) -> PublicUrl:
    """Parse and resolve `url`; return it as a `PublicUrl` or raise.

    Raises BlockedUrlError for anything `parse_url` refuses or any resolved
    address that is not public (one internal answer refuses the whole host).
    Raises NetworkError when resolution fails.
    """
    parts = parse_url(url)
    shown = redact(url)
    host = parts.host
    ips = _resolve(shown, host.name) if isinstance(host, DnsName) else [host]
    public: list[PublicAddress] = []
    for ip in ips:
        verdict = classify(ip)
        if isinstance(verdict, BlockedAddress):
            raise BlockedUrlError(
                f"fetch {shown}: host {parts.host_text!r} resolves to non-public address "
                f"{verdict.ip} ({verdict.reason})"
            )
        public.append(verdict)
    if not public:
        raise NetworkError(f"fetch {shown}: host {parts.host_text!r} resolved to no addresses")
    return PublicUrl(parts=parts, addresses=tuple(public))
