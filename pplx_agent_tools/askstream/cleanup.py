"""What the driver sends when a run ends, on every exit path: Terminate (stop
a run that may still be spending quota) and Delete (remove the thread).

`cleanup_plan` is pure. It reads the ids from `Done.ids`, or from `live.ids`
for a Live state interrupted before `Done`, never from the path that led
there. Each leg states why it is or is not sent, so diagnostics can report
it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, final

from typing_extensions import assert_never

from pplx_agent_tools.askstream.fsm import (
    Done,
    Ids,
    Known,
    NoIds,
    ReconnectBackoff,
    Reconnecting,
    StartBackoff,
    Starting,
    State,
    Streaming,
    UuidOnly,
    ids_uuid,
)
from pplx_agent_tools.askstream.ids import BackendUuid, ContextUuid, ThreadRef
from pplx_agent_tools.askstream.outcome import (
    Completed,
    Cut,
    EndedEarly,
    Lost,
    Rejected,
    ServerFailed,
    SettledWithoutTerminal,
)


@final
@dataclass(frozen=True, slots=True)
class TerminateRef:
    """Every field the web client's stop button sends; no token is needed."""

    uuid: BackendUuid
    context: ContextUuid
    model_preference: str


@final
@dataclass(frozen=True, slots=True)
class Terminate:
    ref: TerminateRef


@final
@dataclass(frozen=True, slots=True)
class TerminateNotNeeded:
    """The run is over on the server, or no thread was created."""


@final
@dataclass(frozen=True, slots=True)
class TerminateUnsupported:
    """A run that may still be going, without the context uuid or the
    display model the request needs."""


@final
@dataclass(frozen=True, slots=True)
class Delete:
    ref: ThreadRef


@final
@dataclass(frozen=True, slots=True)
class DeleteNotNeeded:
    """No thread was created."""


@final
@dataclass(frozen=True, slots=True)
class DeleteKept:
    """The caller asked to keep the thread."""


@final
@dataclass(frozen=True, slots=True)
class DeleteNoToken:
    """A thread exists but its read-write token never arrived."""


TerminateLeg: TypeAlias = Terminate | TerminateNotNeeded | TerminateUnsupported
DeleteLeg: TypeAlias = Delete | DeleteNotNeeded | DeleteKept | DeleteNoToken


def _run_may_be_live(last: State) -> bool:
    match last:
        case Done(outcome=outcome):
            match outcome:
                case Completed() | SettledWithoutTerminal() | ServerFailed():
                    return False
                case Cut() | Lost() | EndedEarly() | Rejected():
                    return True
                case _:
                    assert_never(outcome)
        case Streaming() | Reconnecting() | ReconnectBackoff():
            return True
        case Starting() | StartBackoff():
            return False
        case _:
            assert_never(last)


def _ids(last: State) -> Ids:
    match last:
        case Done(ids=ids):
            return ids
        case Streaming(live=live) | Reconnecting(live=live) | ReconnectBackoff(live=live):
            return live.ids
        case Starting() | StartBackoff():
            return NoIds()
        case _:
            assert_never(last)


def _terminate(ids: Ids, display_model: str | None) -> TerminateLeg:
    match ids:
        case NoIds():
            return TerminateNotNeeded()
        case UuidOnly(context=context) | Known(context=context):
            if context is None or display_model is None:
                return TerminateUnsupported()
            return Terminate(TerminateRef(ids_uuid(ids), context, display_model))
        case _:
            assert_never(ids)


def _delete(ids: Ids, keep_thread: bool) -> DeleteLeg:
    match ids:
        case NoIds():
            return DeleteNotNeeded()
        case _ if keep_thread:
            return DeleteKept()
        case UuidOnly():
            return DeleteNoToken()
        case Known(ref=ref):
            return Delete(ref)
        case _:
            assert_never(ids)


def cleanup_plan(
    last: State, keep_thread: bool, display_model: str | None
) -> tuple[TerminateLeg, DeleteLeg]:
    """Terminate when the run may still be going on the server (a cut, a
    loss, an early end, a rejection after the first byte, or an interrupt
    of a Live state); `keep_thread` gates only Delete. `display_model` is the
    last non-null one the run's frames carried: the terminate request names
    the model the server ran, not the one requested."""
    ids = _ids(last)
    terminate = _terminate(ids, display_model) if _run_may_be_live(last) else TerminateNotNeeded()
    return terminate, _delete(ids, keep_thread)
