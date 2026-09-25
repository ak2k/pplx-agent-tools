"""Per-verb lifecycle policy: the bounds the FSM enforces, set once per run.

Every union here stands for a choice that used to be a bool or an Optional,
so the FSM matches on what a verb asked for rather than on flags.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypeAlias, final

from pplx_agent_tools.askstream.patch import Limits


@final
@dataclass(frozen=True, slots=True)
class At:
    """A bounded deadline. In a `Policy` it is seconds after the run starts;
    in a lifecycle state it is the absolute clock time."""

    t: float


@final
@dataclass(frozen=True, slots=True)
class Unbounded:
    pass


Deadline: TypeAlias = At | Unbounded


@final
@dataclass(frozen=True, slots=True)
class SettleAfterText:
    """Ask: COMPLETED completes; after `text_completed`, wait at most
    `settle_s` for it."""

    settle_s: float


@final
@dataclass(frozen=True, slots=True)
class AtTextComplete:
    """Fetch: COMPLETED or `text_completed` completes."""


@final
@dataclass(frozen=True, slots=True)
class AtCompleted:
    """Research: only COMPLETED completes."""


Completion: TypeAlias = SettleAfterText | AtTextComplete | AtCompleted


@final
@dataclass(frozen=True, slots=True)
class FirstContentOff:
    pass


@final
@dataclass(frozen=True, slots=True)
class FirstContentWithin:
    s: float


FirstContent: TypeAlias = FirstContentOff | FirstContentWithin


@final
@dataclass(frozen=True, slots=True)
class Off:
    pass


@final
@dataclass(frozen=True, slots=True)
class Bounded:
    """At most `consecutive` reconnects without progress between them, and
    `total` in the run."""

    consecutive: int
    total: int


Reconnect: TypeAlias = Off | Bounded

# Which answer paths a verb's projection reads; research's workflow text items
# are step summaries, never its answer.
AnswerPaths = Literal["ask_text_or_workflow", "ask_text_only"]

Verb = Literal["ask", "fetch", "research"]

RATE_LIMIT_DEFAULT_S = 5.0  # a 429 without retry-after
RATE_LIMIT_CAP_S = 60.0  # a hostile retry-after cannot park the run longer
JITTER_LOW = 0.85
JITTER_HIGH = 1.15  # parallel callers do not wake in lockstep

OFF = Off()
FIRST_CONTENT_OFF = FirstContentOff()
DEFAULT_LIMITS = Limits()


@final
@dataclass(frozen=True, slots=True)
class PolicyError:
    reason: str


@final
@dataclass(frozen=True, slots=True)
class Policy:
    """Build with `Policy.make` (validated) or `for_verb`."""

    deadline: Deadline
    stall_s: float
    completion: Completion
    answer_paths: AnswerPaths
    first_content: FirstContent
    silence_s: float
    open_s: float
    grace_s: float
    reconnect: Reconnect
    backoff_base_s: float
    backoff_cap_s: float
    rate_limit_attempts: int
    min_useful_s: float
    limits: Limits

    @staticmethod
    def make(
        *,
        deadline: Deadline,
        stall_s: float,
        completion: Completion,
        answer_paths: AnswerPaths,
        first_content: FirstContent = FIRST_CONTENT_OFF,
        silence_s: float = 25.0,
        open_s: float = 30.0,
        grace_s: float = 30.0,
        reconnect: Reconnect = OFF,
        backoff_base_s: float = 1.0,
        backoff_cap_s: float = 8.0,
        rate_limit_attempts: int = 3,
        min_useful_s: float = 5.0,
        limits: Limits = DEFAULT_LIMITS,
    ) -> Policy | PolicyError:
        seconds = {
            "stall_s": stall_s,
            "silence_s": silence_s,
            "open_s": open_s,
            "grace_s": grace_s,
            "backoff_base_s": backoff_base_s,
            "backoff_cap_s": backoff_cap_s,
            "min_useful_s": min_useful_s,
        }
        match deadline:
            case At(t):
                seconds["deadline"] = t
            case Unbounded():
                pass
        match completion:
            case SettleAfterText(settle_s):
                seconds["settle_s"] = settle_s
            case AtTextComplete() | AtCompleted():
                pass
        match first_content:
            case FirstContentWithin(s):
                seconds["first_content"] = s
            case FirstContentOff():
                pass
        for name, v in seconds.items():
            if not (math.isfinite(v) and v > 0):
                return PolicyError(f"{name} must be a positive finite number")
        if isinstance(deadline, At) and stall_s > deadline.t:
            return PolicyError("stall_s must not exceed the deadline")
        if backoff_base_s > backoff_cap_s:
            return PolicyError("backoff_base_s must not exceed backoff_cap_s")
        if rate_limit_attempts < 1:
            return PolicyError("rate_limit_attempts must be at least 1")
        match reconnect:
            case Bounded(consecutive, total):
                if consecutive < 1 or total < consecutive:
                    return PolicyError("reconnect bounds need 1 <= consecutive <= total")
            case Off():
                pass
        return Policy(
            deadline=deadline,
            stall_s=stall_s,
            completion=completion,
            answer_paths=answer_paths,
            first_content=first_content,
            silence_s=silence_s,
            open_s=open_s,
            grace_s=grace_s,
            reconnect=reconnect,
            backoff_base_s=backoff_base_s,
            backoff_cap_s=backoff_cap_s,
            rate_limit_attempts=rate_limit_attempts,
            min_useful_s=min_useful_s,
            limits=limits,
        )

    @property
    def low_speed_s(self) -> float:
        """curl's low-speed abort; a backstop only, so it trails the FSM's
        own open and silence timers."""
        return max(self.open_s, self.silence_s) + 5


def for_verb(
    verb: Verb,
    *,
    deadline_s: float | None = None,
    stall_s: float | None = None,
    reconnect: Reconnect = OFF,
    first_content: FirstContent = FIRST_CONTENT_OFF,
) -> Policy | PolicyError:
    """The measured defaults for one verb; `deadline_s` and `stall_s` are the
    CLI's `--timeout` and `--stall-timeout`."""
    match verb:
        case "ask":
            return Policy.make(
                deadline=At(540.0 if deadline_s is None else deadline_s),
                stall_s=480.0 if stall_s is None else stall_s,
                completion=SettleAfterText(15.0),
                answer_paths="ask_text_or_workflow",
                first_content=first_content,
                reconnect=reconnect,
            )
        case "fetch":
            return Policy.make(
                deadline=At(540.0 if deadline_s is None else deadline_s),
                stall_s=480.0 if stall_s is None else stall_s,
                completion=AtTextComplete(),
                answer_paths="ask_text_or_workflow",
                first_content=first_content,
                reconnect=reconnect,
            )
        case "research":
            return Policy.make(
                deadline=At(3600.0 if deadline_s is None else deadline_s),
                stall_s=240.0 if stall_s is None else stall_s,
                completion=AtCompleted(),
                answer_paths="ask_text_only",
                first_content=first_content,
                reconnect=reconnect,
            )


def jitter(u: float) -> float:
    """The backoff multiplier for a uniform draw `u` in [0, 1)."""
    return JITTER_LOW + (JITTER_HIGH - JITTER_LOW) * u


def rate_limit_delay(retry_after: float | None, remaining: float, u: float) -> float:
    """Wait before re-sending the initial POST after a 429; never past the
    deadline, so the retry cannot outlive the run."""
    base = RATE_LIMIT_DEFAULT_S if retry_after is None else retry_after
    return max(0.0, min(min(base, RATE_LIMIT_CAP_S) * jitter(u), remaining))


def reconnect_delay(policy: Policy, consecutive: int, retry_after: float | None, u: float) -> float:
    """Backoff before reconnect attempt `consecutive + 1`; a 429 waits at
    least its (capped) retry-after."""
    delay = min(policy.backoff_cap_s, policy.backoff_base_s * 2.0**consecutive * jitter(u))
    if retry_after is not None:
        delay = max(delay, min(max(retry_after, 0.0), RATE_LIMIT_CAP_S))
    return delay
