"""The model version lifecycle (State Machine): states, legal transitions and alias changes per operation.

Pure: no I/O. Every plan is computed through `apply`, so an illegal combination raises before anything is
written. The registry adapter executes a `Plan`'s alias operations in order, then writes its states.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from modelport.errors import IllegalTransitionError, PromotionRefusedError
from modelport.naming import Alias


class ModelState(StrEnum):
    """State of one model version (tag `modelport.state`)."""

    REGISTERED = "registered"
    CANDIDATE = "candidate"
    REJECTED = "rejected"
    CHAMPION = "champion"
    PREVIOUS = "previous"
    RETIRED = "retired"


class Event(StrEnum):
    """What happened to a version."""

    GATE_PASSED = "gate_passed"
    GATE_FAILED = "gate_failed"
    CANARY_PASSED = "canary_passed"
    CANARY_FAILED = "canary_failed"
    MANUAL_PROMOTE = "manual_promote"
    SUPERSEDED = "superseded"
    DISPLACED = "displaced"
    CANDIDATE_REPLACED = "candidate_replaced"
    ROLLBACK_RESTORE = "rollback_restore"
    ROLLBACK_DEMOTE = "rollback_demote"


_S, _E = ModelState, Event
TRANSITIONS: Final[Mapping[tuple[ModelState, Event], ModelState]] = MappingProxyType(
    {
        (_S.REGISTERED, _E.GATE_PASSED): _S.CANDIDATE,
        (_S.REGISTERED, _E.GATE_FAILED): _S.REJECTED,
        (_S.CANDIDATE, _E.CANARY_PASSED): _S.CHAMPION,
        (_S.CANDIDATE, _E.CANARY_FAILED): _S.REJECTED,
        (_S.CANDIDATE, _E.MANUAL_PROMOTE): _S.CHAMPION,
        (_S.CANDIDATE, _E.CANDIDATE_REPLACED): _S.RETIRED,
        (_S.CHAMPION, _E.SUPERSEDED): _S.PREVIOUS,
        (_S.CHAMPION, _E.ROLLBACK_DEMOTE): _S.RETIRED,
        (_S.PREVIOUS, _E.MANUAL_PROMOTE): _S.CHAMPION,
        (_S.PREVIOUS, _E.DISPLACED): _S.RETIRED,
        (_S.PREVIOUS, _E.ROLLBACK_RESTORE): _S.CHAMPION,
        (_S.RETIRED, _E.MANUAL_PROMOTE): _S.CHAMPION,
    }
)


def apply(state: ModelState, event: Event) -> ModelState:
    """The state after `event`.

    Raises:
        IllegalTransitionError: the event is not allowed in this state.
    """
    try:
        return TRANSITIONS[(state, event)]
    except KeyError:
        raise IllegalTransitionError(
            what=f"A {state} version cannot be moved by {event.value.replace('_', ' ')}",
            fix="Check the version's state with `modelport status <name>`.",
            details={"state": state.value, "event": event.value},
        ) from None


@dataclass(frozen=True)
class AliasSnapshot:
    """The three aliases of one model and the states of the versions they name."""

    model: str
    champion: str | None
    previous: str | None
    candidate: str | None
    states: Mapping[str, ModelState] = field(default_factory=dict)  # version -> state

    def state_of(self, version: str, default: ModelState) -> ModelState:
        """The recorded state of `version`, or `default` for a version no alias names."""
        return self.states.get(version, default)


class AliasOpKind(StrEnum):
    """Set or delete one alias."""

    SET = "set"
    DELETE = "delete"


@dataclass(frozen=True)
class AliasOp:
    """One alias operation; `version` is None for DELETE."""

    kind: AliasOpKind
    alias: Alias
    version: str | None


@dataclass(frozen=True)
class Plan:
    """Ordered alias operations and resulting version states."""

    alias_ops: tuple[AliasOp, ...]
    states: Mapping[str, ModelState]


def _set(alias: Alias, version: str) -> AliasOp:
    return AliasOp(AliasOpKind.SET, alias, version)


def _delete(alias: Alias) -> AliasOp:
    return AliasOp(AliasOpKind.DELETE, alias, None)


def plan_gate(snapshot: AliasSnapshot, version: str, passed: bool) -> Plan:
    """Gate passed: alias candidate (an older candidate retires). Gate failed: the version is rejected."""
    current = snapshot.state_of(version, ModelState.REGISTERED)
    if not passed:
        return Plan(alias_ops=(), states={version: apply(current, Event.GATE_FAILED)})
    states: dict[str, ModelState] = {}
    old = snapshot.candidate
    if old is not None and old != version:
        states[old] = apply(snapshot.state_of(old, ModelState.CANDIDATE), Event.CANDIDATE_REPLACED)
    states[version] = apply(current, Event.GATE_PASSED)
    return Plan(alias_ops=(_set(Alias.CANDIDATE, version),), states=states)


def plan_promote(snapshot: AliasSnapshot, version: str, *, event: Event, current_state: ModelState) -> Plan:
    """Make `version` the champion; the old champion becomes previous, an older previous retires.

    The order keeps a champion visible at every step: previous first, then champion, then candidate removed.

    Raises:
        IllegalTransitionError: `version` is already the champion, or cannot be promoted from its state.
    """
    if snapshot.champion == version:
        raise IllegalTransitionError(what=f"v{version} is already the champion", fix="Nothing to do.")
    states: dict[str, ModelState] = {}
    ops: list[AliasOp] = []
    if snapshot.previous is not None and snapshot.previous != version:
        states[snapshot.previous] = apply(
            snapshot.state_of(snapshot.previous, ModelState.PREVIOUS), Event.DISPLACED
        )
    if snapshot.champion is not None:
        ops.append(_set(Alias.PREVIOUS, snapshot.champion))
        states[snapshot.champion] = apply(
            snapshot.state_of(snapshot.champion, ModelState.CHAMPION), Event.SUPERSEDED
        )
    ops.append(_set(Alias.CHAMPION, version))
    states[version] = apply(current_state, event)
    if snapshot.candidate == version:
        ops.append(_delete(Alias.CANDIDATE))
    return Plan(alias_ops=tuple(ops), states=states)


def plan_canary_failed(snapshot: AliasSnapshot, version: str) -> Plan:
    """The canary failed: the candidate alias goes (if it names this version) and the version is rejected."""
    ops = (_delete(Alias.CANDIDATE),) if snapshot.candidate == version else ()
    return Plan(
        alias_ops=ops,
        states={version: apply(snapshot.state_of(version, ModelState.CANDIDATE), Event.CANARY_FAILED)},
    )


def plan_rollback(snapshot: AliasSnapshot) -> Plan:
    """Restore the previous champion; the current champion retires.

    Raises:
        PromotionRefusedError: there is no previous version.
    """
    previous = snapshot.previous
    if previous is None:
        raise PromotionRefusedError(
            what=f"{snapshot.model} has no previous version to roll back to",
            fix="Promote a specific version with `modelport promote <name> <version>`.",
        )
    states = {previous: apply(snapshot.state_of(previous, ModelState.PREVIOUS), Event.ROLLBACK_RESTORE)}
    if snapshot.champion is not None:
        states[snapshot.champion] = apply(
            snapshot.state_of(snapshot.champion, ModelState.CHAMPION), Event.ROLLBACK_DEMOTE
        )
    return Plan(alias_ops=(_set(Alias.CHAMPION, previous), _delete(Alias.PREVIOUS)), states=states)
