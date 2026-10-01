"""pplx auth: manage Perplexity web-session cookies.

Subcommands:
  check     — validate the session against /api/auth/session
  refresh   — keepalive ping (silent on success; designed for cron/launchd)
  import    — pull cookies from a local browser profile, saved once
              /api/auth/session accepts them
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

from .auth import (
    SUPPORTED_BROWSERS,
    ProfileSource,
    cookie_file_target,
    cookie_source,
    cookie_write_path,
    describe_cookie_source,
    read_browser_cookies,
    save_cookies,
)
from .cli_types import PplxArgumentParser
from .errors import AuthError, PplxError, exit_code
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
        help=(
            "cookie profile (default: $PPLX_PROFILE or 'default'); "
            "$PPLX_COOKIES_PATH or $PPLX_COOKIES, when set, is read instead"
        ),
    )

    p_refresh = sub.add_parser(
        "refresh", help="ping the session endpoint to extend TTL (silent on success)"
    )
    p_refresh.add_argument(
        "--profile",
        help=(
            "cookie profile (default: $PPLX_PROFILE or 'default'); "
            "$PPLX_COOKIES_PATH, when set, is read and written instead"
        ),
    )

    p_import = sub.add_parser(
        "import",
        help="import cookies from a local browser profile once perplexity.ai accepts the session",
    )
    p_import.add_argument(
        "--browser",
        choices=list(SUPPORTED_BROWSERS),
        required=True,
        help="browser to read perplexity.ai cookies from (safari: macOS only)",
    )
    p_import.add_argument(
        "--browser-profile",
        metavar="NAME_OR_PATH",
        help=(
            "browser profile to read: its directory name (e.g. Default, Profile 1, "
            "xxxx.default-release) or path; for safari, the path of a Cookies.binarycookies "
            "file (default: the profile whose cookies changed last, which may be another "
            "account's)"
        ),
    )
    p_import.add_argument(
        "--profile",
        help=(
            "destination cookie profile (default: $PPLX_PROFILE or 'default'); "
            "$PPLX_COOKIES_PATH, when set, is written instead"
        ),
    )
    p_import.add_argument(
        "--no-verify",
        action="store_true",
        help=(
            "save without asking perplexity.ai whether the session works (offline use); "
            "only a missing session cookie is caught"
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

    expires = session.get("expires") or "(no expiry)"
    print(f"session valid: {_account(session)}")
    print(f"expires: {expires}")
    source = cookie_source(args.profile)
    if isinstance(source, ProfileSource):
        print(f"profile: {source.profile} ({source.path})")
    else:
        print(f"cookies: {describe_cookie_source(source)}")
    return 0


def cmd_refresh(args: argparse.Namespace) -> int:
    """Ping /api/auth/session and save rotated cookies to the file they came from.

    That file is $PPLX_COOKIES_PATH, else the profile file. Perplexity's
    NextAuth uses rolling sessions — each authenticated call returns a fresh
    session-token via Set-Cookie. Without persistence the rotation is wasted;
    with it, periodic refresh keeps the session alive indefinitely (each
    refresh extends the 30-day TTL).
    """
    try:
        # Before the request: under $PPLX_COOKIES the rotated token cannot be
        # saved where the next load reads it, so nothing is sent.
        dest = cookie_write_path(
            args.profile,
            inline_refusal=(
                "cannot refresh cookies held in $PPLX_COOKIES; "
                "unset it, or set $PPLX_COOKIES_PATH to a file"
            ),
        )
        # Resolved before the load: if a symlink at `dest` is repointed while
        # the request runs, the save refuses rather than write this session's
        # cookies into another account's file.
        expected = cookie_file_target(dest)
        client = Client.from_default_cookies(profile=args.profile)
        client.auth_session()
        # auth_session captures rotated cookies into client.cookies; persist
        # back so the next pplx invocation reads the fresh token.
        save_cookies(client.cookies, dest=dest, expected=expected)
    except PplxError as e:
        print(f"pplx auth refresh: {e}", file=sys.stderr)
        return exit_code(e)
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    """Read the browser's cookies and save them, as `refresh` saves, once
    /api/auth/session accepts them; with --no-verify, without asking."""
    browser: str = args.browser
    session: dict[str, Any] | None = None
    try:
        dest = cookie_write_path(
            args.profile,
            inline_refusal=(
                "$PPLX_COOKIES is set and overrides any cookie file, so an import "
                "would not be used; it was not changed. Replace or unset $PPLX_COOKIES"
            ),
        )
        # Pinned before the request, as `cmd_refresh` pins it.
        expected = cookie_file_target(dest)
        cookies = read_browser_cookies(browser, args.browser_profile)
        if args.no_verify:
            print(
                f"warning: the {browser} session was not verified with perplexity.ai (--no-verify)",
                file=sys.stderr,
            )
        else:
            client = Client(cookies)
            try:
                session = client.auth_session()
            except AuthError as e:
                raise AuthError(
                    f"perplexity.ai did not accept the {browser} session: it is missing or "
                    f"expired; sign in at perplexity.ai in {browser}, then import again"
                ) from e
            except PplxError as e:
                # Only a rejection calls for signing in again; any other failure
                # keeps its own exit code.
                print(
                    f"pplx auth import: cannot verify the {browser} session: {e}; nothing was "
                    "saved. Retry, or pass --no-verify to save the cookies unverified",
                    file=sys.stderr,
                )
                return exit_code(e)
            # The check can rotate the session token.
            cookies = client.cookies
        written = save_cookies(cookies, dest=dest, expected=expected)
    except PplxError as e:
        print(f"pplx auth import: {e}", file=sys.stderr)
        return exit_code(e)
    print(f"imported {browser} cookies to {written}")
    if session is not None:
        print(f"session valid: {_account(session)}")
    return 0


def _account(session: dict[str, Any]) -> str:
    """The account an authenticated /api/auth/session answer names."""
    user = session.get("user") or {}
    named = (user.get(k) for k in ("email", "username", "name", "id"))
    return next((v for v in named if v), "(no email)")


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
