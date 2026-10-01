"""pplx auth: manage Perplexity web-session cookies.

Subcommands:
  check     — validate the session against /api/auth/session
  refresh   — keepalive ping (silent on success; designed for cron/launchd)
  import    — pull cookies from a local browser profile
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .auth import (
    SUPPORTED_BROWSERS,
    ProfileSource,
    cookie_source,
    cookie_write_path,
    describe_cookie_source,
    import_from_browser,
    save_cookies,
)
from .cli_types import PplxArgumentParser
from .errors import PplxError, exit_code
from .wire import Client


def build_parser() -> PplxArgumentParser:
    parser = PplxArgumentParser(
        prog="pplx auth",
        description="Manage Perplexity web-session cookies.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", help="validate the session against /api/auth/session")
    p_check.add_argument(
        "--profile",
        help="cookie profile (default: $PPLX_PROFILE or 'default')",
    )

    p_refresh = sub.add_parser(
        "refresh", help="ping the session endpoint to extend TTL (silent on success)"
    )
    p_refresh.add_argument("--profile", help="cookie profile")

    p_import = sub.add_parser("import", help="import cookies from a local browser profile")
    p_import.add_argument(
        "--browser",
        choices=list(SUPPORTED_BROWSERS),
        required=True,
        help="source browser (rookiepy must support it on this OS)",
    )
    p_import.add_argument(
        "--profile",
        help=(
            "destination cookie profile (default: $PPLX_PROFILE or 'default'); "
            "$PPLX_COOKIES_PATH, when set, is written instead"
        ),
    )

    return parser


def cmd_check(args: argparse.Namespace) -> int:
    try:
        client = Client.from_default_cookies(profile=args.profile)
        session = client.auth_session()
    except PplxError as e:
        print(f"pplx auth check: {e}", file=sys.stderr)
        return exit_code(e)

    user = session.get("user") or {}
    email = user.get("email") or "(no email)"
    expires = session.get("expires") or "(no expiry)"
    print(f"session valid: {email}")
    print(f"expires: {expires}")
    source = cookie_source(args.profile)
    if isinstance(source, ProfileSource):
        print(f"profile: {source.profile} ({source.path})")
    else:
        print(f"cookies: {describe_cookie_source(source)}")
    return 0


def cmd_refresh(args: argparse.Namespace) -> int:
    """Ping /api/auth/session and persist any rotated cookies back to the file
    they were loaded from ($PPLX_COOKIES_PATH, else the profile file).

    Perplexity's NextAuth uses rolling sessions — each authenticated call
    returns a fresh session-token via Set-Cookie. Without persistence the
    rotation is wasted; with it, periodic refresh keeps the session alive
    indefinitely (each refresh extends the 30-day TTL).
    """
    try:
        # Before the request: under $PPLX_COOKIES the rotated token cannot be
        # saved where the next load reads it, so nothing is sent.
        dest = cookie_write_path(args.profile, what="refreshed cookies")
        client = Client.from_default_cookies(profile=args.profile)
        client.auth_session()
        # auth_session captures rotated cookies into client.cookies; persist
        # back so the next pplx invocation reads the fresh token.
        save_cookies(client.cookies, dest=dest)
    except PplxError as e:
        print(f"pplx auth refresh: {e}", file=sys.stderr)
        return exit_code(e)
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    try:
        dest = import_from_browser(args.browser, profile=args.profile)
    except PplxError as e:
        print(f"pplx auth import: {e}", file=sys.stderr)
        return exit_code(e)
    print(f"imported {args.browser} cookies to {dest}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "check":
        return cmd_check(args)
    if args.cmd == "refresh":
        return cmd_refresh(args)
    if args.cmd == "import":
        return cmd_import(args)
    parser.error(f"unknown subcommand: {args.cmd}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
