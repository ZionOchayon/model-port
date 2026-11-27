from modelport.spec import parse_spec
from modelport_platform.canary.core import CanaryPolicy, CanaryStats, judge_canary

POLICY = CanaryPolicy(max_error_rate=0.02, latency_p95_ms=500, max_latency_increase=0.2, min_requests=50)
HEALTHY = CanaryStats(requests=100, errors=0, p95_ms=40.0)
CHAMPION = CanaryStats(requests=900, errors=9, p95_ms=50.0)


def test_healthy_candidate_passes():
    verdict = judge_canary(CHAMPION, HEALTHY, POLICY)
    assert verdict.passed
    assert verdict.reasons == ()


def test_too_few_requests_fails():
    verdict = judge_canary(CHAMPION, CanaryStats(12, 0, 40.0), POLICY)
    assert verdict.reasons == ("only 12 requests reached the candidate in the window (need 50)",)


def test_error_rate_above_max_fails():
    verdict = judge_canary(None, CanaryStats(100, 5, 40.0), POLICY)
    assert verdict.reasons == ("error rate 0.050 > max_error_rate 0.020",)


def test_error_rate_vs_champion_margin():
    champion = CanaryStats(requests=1000, errors=0, p95_ms=50.0)
    worse = CanaryStats(requests=100, errors=2, p95_ms=40.0)  # 0.02: at the max, but 0.02 > 0.00 + 0.01
    assert judge_canary(champion, worse, POLICY).reasons == (
        "error rate 0.020 > champion 0.000 + margin 0.010",
    )
    few = CanaryStats(requests=10, errors=0, p95_ms=50.0)  # champion below min_requests: no comparison
    assert judge_canary(few, worse, POLICY).passed


def test_no_latency_data_fails():
    verdict = judge_canary(CHAMPION, CanaryStats(100, 0, None), POLICY)
    assert verdict.reasons == ("no latency data for the candidate in the window",)
    slow = judge_canary(None, CanaryStats(100, 0, 612.0), POLICY)
    assert slow.reasons == ("p95 latency 612.0 ms > latency_p95_ms 500.0 ms",)


def test_latency_vs_champion_fails():
    verdict = judge_canary(CHAMPION, CanaryStats(100, 0, 61.0), POLICY)
    assert verdict.reasons == ("p95 latency 61.0 ms > champion 50.0 ms + 20%",)


def test_first_model_without_champion():
    assert judge_canary(None, HEALTHY, POLICY).passed


def test_all_reasons_reported():
    verdict = judge_canary(CHAMPION, CanaryStats(10, 5, None), POLICY)
    assert not verdict.passed
    assert len(verdict.reasons) == 4  # too few, max error rate, vs champion, no latency
    assert CanaryStats(0, 0, None).error_rate == 0.0


def test_policy_from_spec():
    spec = parse_spec(
        {
            "name": "tiny",
            "class": "m:Tiny",
            "framework": "xgboost",
            "labels": ["a", "b"],
            "signature": {"inputs": [{"name": "x", "type": "double"}]},
            "data": {"dataset": "s3://datasets/tiny"},
            "serving": {"max_error_rate": 0.05, "latency_p95_ms": 300, "min_canary_requests": 20},
            "evaluation": {"max_latency_increase": 0.5},
        }
    )
    assert CanaryPolicy.from_spec(spec) == CanaryPolicy(0.05, 300.0, 0.5, 20)
