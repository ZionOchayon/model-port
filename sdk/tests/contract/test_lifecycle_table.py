"""Every state x event pair equals the transition table of the data contract (written out literally)."""

import pytest

from modelport.errors import IllegalTransitionError
from modelport.lifecycle import Event, ModelState, apply

S, E = ModelState, Event
EXPECTED = {
    (S.REGISTERED, E.GATE_PASSED): S.CANDIDATE,
    (S.REGISTERED, E.GATE_FAILED): S.REJECTED,
    (S.CANDIDATE, E.CANARY_PASSED): S.CHAMPION,
    (S.CANDIDATE, E.CANARY_FAILED): S.REJECTED,
    (S.CANDIDATE, E.MANUAL_PROMOTE): S.CHAMPION,
    (S.CANDIDATE, E.CANDIDATE_REPLACED): S.RETIRED,
    (S.CHAMPION, E.SUPERSEDED): S.PREVIOUS,
    (S.CHAMPION, E.ROLLBACK_DEMOTE): S.RETIRED,
    (S.PREVIOUS, E.MANUAL_PROMOTE): S.CHAMPION,
    (S.PREVIOUS, E.DISPLACED): S.RETIRED,
    (S.PREVIOUS, E.ROLLBACK_RESTORE): S.CHAMPION,
    (S.RETIRED, E.MANUAL_PROMOTE): S.CHAMPION,
}
PAIRS = [(s, e) for s in ModelState for e in Event]


def test_table_has_six_states_and_ten_events():
    assert (len(ModelState), len(Event)) == (6, 10)


@pytest.mark.parametrize(("state", "event"), PAIRS, ids=[f"{s}-{e}" for s, e in PAIRS])
def test_every_state_event_pair(state, event):
    if (state, event) in EXPECTED:
        assert apply(state, event) is EXPECTED[(state, event)]
    else:
        with pytest.raises(IllegalTransitionError):
            apply(state, event)
