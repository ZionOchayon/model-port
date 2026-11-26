import pytest

from modelport.errors import IllegalTransitionError, PromotionRefusedError
from modelport.lifecycle import (
    AliasOp,
    AliasOpKind,
    AliasSnapshot,
    Event,
    ModelState,
    plan_canary_failed,
    plan_gate,
    plan_promote,
    plan_rollback,
)
from modelport.naming import Alias

S = ModelState
SET, DELETE = AliasOpKind.SET, AliasOpKind.DELETE


def snap(champion=None, previous=None, candidate=None, **states):
    return AliasSnapshot(
        model="image-classifier",
        champion=champion,
        previous=previous,
        candidate=candidate,
        states={v.removeprefix("v"): s for v, s in states.items()},
    )


def test_first_gate_pass_sets_candidate():
    plan = plan_gate(snap(), "1", passed=True)
    assert plan.alias_ops == (AliasOp(SET, Alias.CANDIDATE, "1"),)
    assert plan.states == {"1": S.CANDIDATE}


def test_gate_pass_retires_old_candidate():
    plan = plan_gate(snap(champion="1", candidate="2", v1=S.CHAMPION, v2=S.CANDIDATE), "3", passed=True)
    assert plan.alias_ops == (AliasOp(SET, Alias.CANDIDATE, "3"),)
    assert plan.states == {"2": S.RETIRED, "3": S.CANDIDATE}


def test_gate_fail_no_alias_ops():
    plan = plan_gate(snap(champion="1", v1=S.CHAMPION), "2", passed=False)
    assert plan.alias_ops == ()
    assert plan.states == {"2": S.REJECTED}


def test_promote_order_previous_before_champion():
    plan = plan_promote(
        snap(champion="2", previous="1", candidate="3", v1=S.PREVIOUS, v2=S.CHAMPION, v3=S.CANDIDATE),
        "3",
        event=Event.CANARY_PASSED,
        current_state=S.CANDIDATE,
    )
    assert plan.alias_ops == (
        AliasOp(SET, Alias.PREVIOUS, "2"),
        AliasOp(SET, Alias.CHAMPION, "3"),
        AliasOp(DELETE, Alias.CANDIDATE, None),
    )
    assert plan.states == {"1": S.RETIRED, "2": S.PREVIOUS, "3": S.CHAMPION}


def test_promote_first_model_no_previous():
    plan = plan_promote(
        snap(candidate="1", v1=S.CANDIDATE), "1", event=Event.CANARY_PASSED, current_state=S.CANDIDATE
    )
    assert plan.alias_ops == (AliasOp(SET, Alias.CHAMPION, "1"), AliasOp(DELETE, Alias.CANDIDATE, None))
    assert plan.states == {"1": S.CHAMPION}


def test_promote_displaces_old_previous():
    plan = plan_promote(
        snap(champion="2", previous="1", v1=S.PREVIOUS, v2=S.CHAMPION),
        "4",
        event=Event.MANUAL_PROMOTE,
        current_state=S.RETIRED,
    )
    assert plan.states["1"] is S.RETIRED
    assert plan.alias_ops[0] == AliasOp(SET, Alias.PREVIOUS, "2")
    assert Alias.CANDIDATE not in {op.alias for op in plan.alias_ops}


def test_manual_promote_of_previous_swaps_the_two():
    plan = plan_promote(
        snap(champion="2", previous="1", v1=S.PREVIOUS, v2=S.CHAMPION),
        "1",
        event=Event.MANUAL_PROMOTE,
        current_state=S.PREVIOUS,
    )
    assert plan.alias_ops == (AliasOp(SET, Alias.PREVIOUS, "2"), AliasOp(SET, Alias.CHAMPION, "1"))
    assert plan.states == {"2": S.PREVIOUS, "1": S.CHAMPION}


def test_promote_current_champion_refused():
    with pytest.raises(IllegalTransitionError) as info:
        plan_promote(
            snap(champion="7", v7=S.CHAMPION), "7", event=Event.MANUAL_PROMOTE, current_state=S.CHAMPION
        )
    assert info.value.what == "v7 is already the champion"
    assert info.value.fix == "Nothing to do."


def test_promote_rejected_version_refused():
    with pytest.raises(IllegalTransitionError):
        plan_promote(
            snap(champion="1", v1=S.CHAMPION), "2", event=Event.MANUAL_PROMOTE, current_state=S.REJECTED
        )


def test_rollback_without_previous_refused():
    with pytest.raises(PromotionRefusedError) as info:
        plan_rollback(snap(champion="1", v1=S.CHAMPION))
    assert info.value.what == "image-classifier has no previous version to roll back to"
    assert "modelport promote" in info.value.fix


def test_rollback_ops_and_states():
    plan = plan_rollback(snap(champion="2", previous="1", v1=S.PREVIOUS, v2=S.CHAMPION))
    assert plan.alias_ops == (AliasOp(SET, Alias.CHAMPION, "1"), AliasOp(DELETE, Alias.PREVIOUS, None))
    assert plan.states == {"1": S.CHAMPION, "2": S.RETIRED}


def test_canary_failed_deletes_candidate():
    plan = plan_canary_failed(snap(champion="1", candidate="2", v1=S.CHAMPION, v2=S.CANDIDATE), "2")
    assert plan.alias_ops == (AliasOp(DELETE, Alias.CANDIDATE, None),)
    assert plan.states == {"2": S.REJECTED}


def test_canary_failed_for_other_version_keeps_alias():
    plan = plan_canary_failed(snap(candidate="3", v3=S.CANDIDATE), "2")
    assert plan.alias_ops == ()
    assert plan.states == {"2": S.REJECTED}
