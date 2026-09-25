"""How an ask-family run ended, as the lifecycle reports it to a verb.

Each verb narrows `Outcome` with an exhaustive match: the stream ends a verb
returns a result for (`StreamEnd`, plus `SettledWithoutTerminal` for ask)
and the ends it raises for (`Lost`, `ServerFailed`, `Rejected`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias, final

from pplx_agent_tools.errors import PplxError

CutCause = Literal["stall", "deadline", "first_content"]
# Read only to pick the error for an empty answer.
EndedBy = Literal["server", "auth"]
SettleBy = Literal["settle", "stall", "deadline", "server", "auth"]


@final
@dataclass(frozen=True, slots=True)
class Completed:
    reconnects: int


@final
@dataclass(frozen=True, slots=True)
class Cut:
    cause: CutCause
    seconds: float
    reconnects: int


@final
@dataclass(frozen=True, slots=True)
class EndedEarly:
    """The server ended the stream before completing it; the partial is kept."""

    reconnects: int
    by: EndedBy


@final
@dataclass(frozen=True, slots=True)
class SettledWithoutTerminal:
    """Ask only: the answer was whole (`text_completed`) but the terminal
    frame with the final sources never came."""

    reconnects: int
    by: SettleBy


@final
@dataclass(frozen=True, slots=True)
class Lost:
    msg: str


@final
@dataclass(frozen=True, slots=True)
class ServerFailed:
    raw_status: str | None


@final
@dataclass(frozen=True, slots=True)
class Rejected:
    error: PplxError


@final
@dataclass(frozen=True, slots=True)
class NotStreamed:
    """Plain fetch: no ask-family stream was involved."""


StreamEnd: TypeAlias = Completed | Cut | EndedEarly
AskEnd: TypeAlias = Completed | Cut | EndedEarly | SettledWithoutTerminal
FetchStream: TypeAlias = NotStreamed | Completed | Cut | EndedEarly
Outcome: TypeAlias = (
    Completed | SettledWithoutTerminal | Cut | EndedEarly | Lost | ServerFailed | Rejected
)
