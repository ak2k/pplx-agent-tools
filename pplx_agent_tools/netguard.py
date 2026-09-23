"""Outbound-address policy for user-supplied fetch URLs (SSRF guard).

`classify` is the only judge of an address, and `PublicAddress` is the only
value it allows; `check_url` turns a URL into a `PublicUrl`, which holds only
`PublicAddress`es, and the fetch path sends requests for `PublicUrl`s alone.

Address data: the IANA IPv4 and IPv6 Special-Purpose Address Registries, both
last updated 2025-10-09:
https://www.iana.org/assignments/iana-ipv4-special-registry/
https://www.iana.org/assignments/iana-ipv6-special-registry/
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from typing import TypeAlias
from urllib.parse import urlparse

from .errors import BlockedUrlError, NetworkError

IPAddress: TypeAlias = ipaddress.IPv4Address | ipaddress.IPv6Address

_ALLOWED_SCHEMES = frozenset({"http", "https"})

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


@dataclass(frozen=True)
class PublicUrl:
    """An http(s) URL whose host resolved only to public addresses when checked.

    `addresses` is what the check saw; curl resolves the host again at connect
    time, so a DNS answer that changes in between is not covered here.
    """

    url: str
    host: str
    addresses: tuple[PublicAddress, ...]

    def __post_init__(self) -> None:
        if not self.addresses:
            raise ValueError(f"{self.url}: a PublicUrl needs at least one address")


def _resolve(url: str, host: str) -> list[IPAddress]:
    try:
        infos = socket.getaddrinfo(host, None)
    except ValueError as e:  # includes UnicodeError from IDNA encoding
        raise BlockedUrlError(f"fetch {url}: invalid host {host!r}: {e}") from e
    except OSError as e:
        raise NetworkError(f"fetch {url}: cannot resolve host {host!r}: {e}") from e
    addrs: list[IPAddress] = []
    for info in infos:
        raw = info[4][0]
        if not isinstance(raw, str):
            raise BlockedUrlError(f"fetch {url}: host {host!r} resolves to non-IP {raw!r}")
        try:
            addrs.append(ipaddress.ip_address(raw))
        except ValueError as e:
            raise BlockedUrlError(f"fetch {url}: host {host!r} resolves to {raw!r}: {e}") from e
    return addrs


def check_url(url: str) -> PublicUrl:
    """Parse and resolve `url`; return it as a `PublicUrl` or raise.

    Raises BlockedUrlError for a non-http(s) scheme, a missing or malformed
    host or port, or any resolved address that is not public (one internal
    answer refuses the whole host). Raises NetworkError when resolution fails.
    """
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        _ = parsed.port
    except ValueError as e:
        raise BlockedUrlError(f"fetch {url}: malformed URL: {e}") from e
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise BlockedUrlError(
            f"fetch {url}: unsupported URL scheme {parsed.scheme!r} (only http/https allowed)"
        )
    if not host:
        raise BlockedUrlError(f"fetch {url}: URL has no host")
    public: list[PublicAddress] = []
    for ip in _resolve(url, host):
        verdict = classify(ip)
        if isinstance(verdict, BlockedAddress):
            raise BlockedUrlError(
                f"fetch {url}: host {host!r} resolves to non-public address "
                f"{verdict.ip} ({verdict.reason})"
            )
        public.append(verdict)
    if not public:
        raise NetworkError(f"fetch {url}: host {host!r} resolved to no addresses")
    return PublicUrl(url=url, host=host, addresses=tuple(public))
