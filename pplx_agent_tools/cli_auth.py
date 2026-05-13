"""pplx auth: manage Perplexity web-session cookies.

Subcommands:
  check     — validate the session against /api/auth/session
  refresh   — keepalive ping (silent on success; designed for cron/launchd)
  import    — pull cookies from a local browser profile (Step 3; stub for now)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .auth import (
    SUPPORTED_BROWSERS,
    default_cookies_path,
    import_from_browser,
    resolve_profile,
)
from .errors import PplxError, exit_code
from .wire import Client


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
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
        help="destination cookie profile (default: 'default')",
    )

    return parser


def cmd_check(args: argparse.Namespace) -> int:
    profile = resolve_profile(args.profile)
    try:
        client = Client.from_default_cookies(profile=args.profile)
        session = client.auth_session()
    except PplxError as e:
        print(f"pplx auth check: {e}", file=sys.stderr)
        return exit_code(e)

    user = session.get("user") or {}
    email = user.get("email") or "(no email)"
    expires = session.get("expires") or "(no expiry)"
    cookie_path = default_cookies_path(profile)
    print(f"session valid: {email}")
    print(f"expires: {expires}")
    print(f"profile: {profile} ({cookie_path})")
    return 0


def cmd_refresh(args: argparse.Namespace) -> int:
    try:
        client = Client.from_default_cookies(profile=args.profile)
        client.auth_session()
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
