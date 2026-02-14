from datetime import datetime, timedelta, timezone
import pytest

from mod1_launch_hunter.module import LaunchHunterModule, EnrichedEvent, OnchainData
from mod5_kol_radar.module import KOLRadarModule, KOLEvaluationContext, PushRecord


@pytest.fixture
def mod1():
    return LaunchHunterModule("mod1_launch_hunter/config.yaml")


@pytest.fixture
def mod5():
    return KOLRadarModule("mod5_kol_radar/config.yaml")


def _assert_protocol(out):
    for k in ["module_name", "final_score", "confidence_score", "factors", "base_breakdown", "hit_rules", "explain", "observability"]:
        assert k in out
    for f in ["base", "source", "market", "cross", "decay", "propagation"]:
        assert f in out["factors"]


def test_mod1_normal(mod1):
    now = datetime.now(timezone.utc)
    ev = EnrichedEvent(
        event_id="e1",
        t0_first_seen=now - timedelta(seconds=20),
        t1_captured=now,
        source_type="twitter_builder",
        source_account="@a",
        raw_text="new primitive launch tge",
        onchain=OnchainData(contract_verified=True),
        contract_address="0x1",
        project_id="p1",
        cross_evidence={"original_sources_count": 2, "platforms": ["twitter", "onchain"], "cross_bonus_key": "cross_twitter_onchain"},
        official_confirmed=True,
    )
    out = mod1.evaluate(ev)
    _assert_protocol(out)
    assert out["final_score"] >= 0
    assert "registry_updates" in out


def test_mod1_missing_fields_negative(mod1):
    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError):
        mod1.evaluate(EnrichedEvent(event_id="", t0_first_seen=now, t1_captured=now, source_type="twitter_builder", source_account="@a", raw_text="x"))


def test_mod5_dedup(mod5):
    now = datetime.now(timezone.utc)
    pushes = [
        PushRecord("a1", now - timedelta(hours=2), now - timedelta(hours=1), "useful", 1.0, True, 80, now - timedelta(days=1)),
        PushRecord("a1", now - timedelta(hours=2), now - timedelta(hours=1), "useful", 1.0, True, 81, now - timedelta(days=2)),
    ] + [
        PushRecord(f"a{i}", now - timedelta(hours=2), now - timedelta(hours=1), "useful", 1.0, True, 80, now - timedelta(days=1))
        for i in range(2,12)
    ]
    out = mod5.evaluate(KOLEvaluationContext("@k", "mod1", "twitter_kol_s", pushes))
    _assert_protocol(out)


def test_mod5_noise_extreme(mod5):
    now = datetime.now(timezone.utc)
    pushes = [
        PushRecord(f"a{i}", now, now - timedelta(days=1), "bad", None, False, 9999, now)
        for i in range(12)
    ]
    out = mod5.evaluate(KOLEvaluationContext("@k", "mod1", "twitter_kol_c", pushes))
    _assert_protocol(out)


def test_fail_modes(mod1, mod5):
    now = datetime.now(timezone.utc)
    mod1.config["enabled"] = False
    out1 = mod1.evaluate(EnrichedEvent(event_id="e", t0_first_seen=now, t1_captured=now, source_type="twitter_builder", source_account="@a", raw_text="x"))
    assert "module_disabled" in out1["hit_rules"]

    mod5.config["enabled"] = False
    out2 = mod5.evaluate(KOLEvaluationContext("@k", "mod1", "twitter_kol_s", []))
    assert "disabled" in out2["explain"]["module_status"]

    mod5.config["enabled"] = True
    mod5.config["emergency_mode"]["enabled"] = True
    out3 = mod5.evaluate(KOLEvaluationContext("@k", "mod1", "twitter_kol_s", []))
    assert out3["explain"]["module_status"] == "emergency_mode"

    mod5.config["emergency_mode"]["enabled"] = False
    out4 = mod5.evaluate(KOLEvaluationContext("@k", "mod1", "twitter_kol_s", []))
    assert out4["explain"]["module_status"] == "insufficient_data"


def test_structure_alignment_formula_presence(mod1):
    now = datetime.now(timezone.utc)
    ev = EnrichedEvent(
        event_id="e2",
        t0_first_seen=now - timedelta(minutes=1),
        t1_captured=now,
        source_type="twitter_builder",
        source_account="@a",
        raw_text="launch",
        project_id="p2",
    )
    out = mod1.evaluate(ev)
    sp = out["observability"]["scoring_process"]
    assert set(["base", "source", "market", "cross", "decay", "propagation", "final"]).issubset(set(sp.keys()))


def test_mod1_stage_progress_path(mod1):
    class R:
        def get_project(self, project_id):
            return {"project_id": project_id, "has_token": True, "stage": "announced"}

        def concept_exists(self, concept):
            return False

    mod1.registry = R()
    now = datetime.now(timezone.utc)
    ev = EnrichedEvent(
        event_id="e3",
        t0_first_seen=now - timedelta(minutes=2),
        t1_captured=now,
        source_type="twitter_builder",
        source_account="@a",
        raw_text="project mainnet live now",
        project_id="p3",
    )
    out = mod1.evaluate(ev)
    assert "novelty_new_progress" in out["hit_rules"]


def test_mod5_discovery_and_registry_update(mod5):
    candidates = mod5.discover_kol_multi_channel(
        seed_kols=["@s1", "@s2", "@s3"],
        following_map={
            "@s1": [{"handle": "@x", "followers_count": 2000}],
            "@s2": [{"handle": "@x", "followers_count": 2000}],
            "@s3": [{"handle": "@x", "followers_count": 2000}],
        },
        early_repliers=[{"handle": "@y", "early_reply_count": 6}],
        quoted_accounts=[{"handle": "@z", "quoter_count": 2}],
        community_contributors=[{"handle": "@f", "alpha_rate": 0.5, "message_count": 20}],
    )
    assert len(candidates) >= 1
    scan = mod5.scan_kol_following(
        following_accounts=[{"handle": "@a", "bio": "builder", "followers": 20000}],
        known_cross_count={"@a": 2},
    )
    assert scan["total_matched"] == 1
    confirmed = mod5.confirm_kol_candidates(["@a"])
    assert confirmed[0]["tier"] == "C"
    upd = mod5.update_kol_registry({"@a": {"mod1": 90, "mod2": 80, "mod3": 70, "mod4": 60}})
    assert "kol_registry_updates" in upd and "source_weights_suggestion" in upd


def test_mod5_tier_action_uses_current_tier(mod5):
    action = mod5._suggest_tier_action(score=90, sample_count=35, current_tier="B")
    assert action["action"] == "upgrade"


def test_mod5_trend_cluster_6h_report(mod5):
    now = datetime.now(timezone.utc)
    events = [
        {"concept": "AI Infra", "source": "twitter_a", "event_time": now - timedelta(hours=4)},
        {"concept": "AI Infra", "source": "twitter_b", "event_time": now - timedelta(hours=6)},
        {"concept": "AI Infra", "source": "onchain", "event_time": now - timedelta(hours=8)},
        {"concept": "AI Infra", "source": "twitter_a", "event_time": now - timedelta(days=8)},
    ]
    out = mod5.trend_cluster_report_6h(events, known_concepts=set())
    assert out["job"] == "trend_cluster_6h"
    assert len(out["push_reports"]) == 1
    assert out["push_reports"][0]["trend_signal"] is True


def test_mod5_lead_time_faster_scores_higher(mod5):
    now = datetime.now(timezone.utc)
    fast = [
        PushRecord(f"f{i}", now - timedelta(minutes=5), now - timedelta(minutes=4), "useful", 1.0, True, 80, now - timedelta(days=1))
        for i in range(12)
    ]
    slow = [
        PushRecord(f"s{i}", now - timedelta(hours=2), now - timedelta(minutes=1), "useful", 1.0, True, 80, now - timedelta(days=1))
        for i in range(12)
    ]
    out_fast = mod5.evaluate(KOLEvaluationContext("@k", "mod1", "twitter_kol_s", fast))
    out_slow = mod5.evaluate(KOLEvaluationContext("@k", "mod1", "twitter_kol_s", slow))
    assert out_fast["base_breakdown"]["lead_time"] >= out_slow["base_breakdown"]["lead_time"]


def test_mod5_score_to_tier_threshold_edges(mod5):
    assert mod5._score_to_tier(80) == "S"
    assert mod5._score_to_tier(60) == "A"
    assert mod5._score_to_tier(40) == "B"
    assert mod5._score_to_tier(20) == "C"
