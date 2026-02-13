"""
打新猎手 直接测试 (无pytest依赖)
"""

import sys
import os
import traceback
import yaml
import tempfile
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(__file__))
from launch_hunter import (
    LaunchHunter, EnrichedEvent, OnchainData, CrossEvidence, RiskData,
    ModuleOutput, RegistryReader, NullRegistryReader,
    evaluate_launch, _null_sonnet_caller
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'launch_hunter.yaml')

# ── Registry / Sonnet Mock ─────────────────

class MockRegistryReader(RegistryReader):
    def __init__(self):
        self.projects = {}
        self.concepts = {}
        self.propagations = {}
    def project_exists(self, pid): return pid in self.projects
    def get_project(self, pid): return self.projects.get(pid)
    def concept_exists(self, c): return c in self.concepts
    def get_concept(self, c): return self.concepts.get(c)
    def get_propagation(self, f): return self.propagations.get(f)

def mock_sonnet_high(prompt):
    return {"track_score": 5, "pain_point_score": 4, "uniqueness_score": 5,
            "credibility_score": 4, "narrative_core": "DeFi新范式", "narrative_tags": ["DeFi", "创新"]}

def mock_sonnet_low(prompt):
    return {"track_score": 1, "pain_point_score": 1, "uniqueness_score": 1,
            "credibility_score": 1, "narrative_core": "普通", "narrative_tags": []}

def mock_sonnet_error(prompt):
    raise TimeoutError("timeout")

# ── Factory ─────────────────

def make_event(**ov):
    now = datetime.now(timezone.utc)
    d = dict(
        event_id="test-001", t0_first_seen=now - timedelta(seconds=30), t1_captured=now,
        source_type="twitter_builder", source_account="@test", event_type="launch",
        raw_text="New project launch: ProjectX is a breakthrough DeFi protocol. Free mint live now!",
        detected_projects=["ProjectX"], contract_address="0xabc", project_id="projectx",
        onchain=OnchainData(contract_verified=True, deployer="0xd", has_liquidity=True,
                            liquidity_usd=50000, status="success"),
        cross_evidence=CrossEvidence(original_sources_count=2, platforms=["twitter", "feishu"]),
        risk=RiskData(risk_level="green"), is_original=True,
    )
    d.update(ov)
    return EnrichedEvent(**d)

# ── Test Runner ─────────────────

passed = 0
failed = 0
errors = []

def run_test(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  ✅ {name}")
    except Exception as e:
        failed += 1
        errors.append((name, e))
        print(f"  ❌ {name}: {e}")

# ── Tests ─────────────────

def test_config_loads():
    h = LaunchHunter(CONFIG_PATH)
    assert h.is_enabled is True
    assert h.config['version'] == '1.0'

def test_max_score_sum_100():
    h = LaunchHunter(CONFIG_PATH)
    t = sum(h.config[d]['max_score'] for d in ['novelty','verifiability','narrative','innovation','timeliness'])
    assert t == 100, f"sum={t}"

def test_config_hash_stable():
    assert LaunchHunter(CONFIG_PATH).config_hash == LaunchHunter(CONFIG_PATH).config_hash

def test_wrong_event_type():
    h = LaunchHunter(CONFIG_PATH)
    try:
        h.evaluate(make_event(event_type="meme_signal"))
        assert False, "should raise"
    except ValueError as e:
        assert "launch" in str(e)

def test_missing_event_id():
    h = LaunchHunter(CONFIG_PATH)
    try:
        h.evaluate(make_event(event_id=""))
        assert False, "should raise"
    except ValueError:
        pass

def test_novelty_first_mention():
    h = LaunchHunter(CONFIG_PATH, registry=NullRegistryReader())
    r = h.evaluate(make_event())
    assert r.components['novelty'] == 30, f"got {r.components['novelty']}"
    assert 'novelty_first_mention' in r.hit_rules

def test_novelty_new_progress():
    reg = MockRegistryReader()
    reg.projects['projectx'] = {'has_token': False, 'stage': 'announced'}
    h = LaunchHunter(CONFIG_PATH, registry=reg)
    r = h.evaluate(make_event(contract_address="0xnew"))
    assert r.components['novelty'] == 15, f"got {r.components['novelty']}"

def test_novelty_known_repeat():
    reg = MockRegistryReader()
    reg.projects['projectx'] = {'has_token': True, 'stage': 'mainnet'}
    h = LaunchHunter(CONFIG_PATH, registry=reg)
    r = h.evaluate(make_event())
    assert r.components['novelty'] == 5

def test_novelty_no_project_id():
    h = LaunchHunter(CONFIG_PATH)
    r = h.evaluate(make_event(project_id="", detected_projects=[]))
    assert r.components['novelty'] == 30

def test_verifiability_verified():
    h = LaunchHunter(CONFIG_PATH)
    r = h.evaluate(make_event(onchain=OnchainData(contract_verified=True, status="success")))
    assert r.components['verifiability'] == 20

def test_verifiability_unverified():
    h = LaunchHunter(CONFIG_PATH)
    r = h.evaluate(make_event(onchain=OnchainData(contract_verified=False, status="success")))
    assert r.components['verifiability'] == 10

def test_verifiability_no_contract():
    h = LaunchHunter(CONFIG_PATH)
    r = h.evaluate(make_event(contract_address="", onchain=OnchainData(status="success")))
    assert r.components['verifiability'] == 5

def test_verifiability_timeout_degrades():
    h = LaunchHunter(CONFIG_PATH)
    r = h.evaluate(make_event(onchain=OnchainData(contract_verified=True, status="timeout")))
    assert r.components['verifiability'] == 10  # degraded
    assert 'onchain_api_timeout' in r.observability.get('warnings', [])

def test_narrative_high():
    h = LaunchHunter(CONFIG_PATH, sonnet_caller=mock_sonnet_high)
    r = h.evaluate(make_event())
    assert r.components['narrative'] == 18  # 5+4+5+4

def test_narrative_low():
    h = LaunchHunter(CONFIG_PATH, sonnet_caller=mock_sonnet_low)
    r = h.evaluate(make_event())
    assert r.components['narrative'] == 4

def test_narrative_error_degrades():
    h = LaunchHunter(CONFIG_PATH, sonnet_caller=mock_sonnet_error)
    r = h.evaluate(make_event())
    assert r.components['narrative'] == 8  # 2+2+2+2 default
    assert 'sonnet_degraded' in r.observability.get('warnings', [])

def test_narrative_clamp():
    def overflow(p):
        return {"track_score": 10, "pain_point_score": -1, "uniqueness_score": 5,
                "credibility_score": 5, "narrative_core": "t", "narrative_tags": []}
    h = LaunchHunter(CONFIG_PATH, sonnet_caller=overflow)
    r = h.evaluate(make_event())
    assert r.components['narrative'] == 15  # 5+0+5+5

def test_innovation_multi_keywords():
    h = LaunchHunter(CONFIG_PATH)
    r = h.evaluate(make_event(raw_text="breakthrough new paradigm zero to one project"))
    assert r.components['innovation'] == 20

def test_innovation_one_keyword():
    h = LaunchHunter(CONFIG_PATH)
    r = h.evaluate(make_event(raw_text="A breakthrough in yield farming.", detected_projects=[]))
    assert r.components['innovation'] == 10

def test_innovation_no_keywords():
    h = LaunchHunter(CONFIG_PATH)
    r = h.evaluate(make_event(raw_text="Just another normal token launch.", detected_projects=[]))
    assert r.components['innovation'] == 3

def test_innovation_chinese():
    h = LaunchHunter(CONFIG_PATH)
    r = h.evaluate(make_event(raw_text="这是一个新范式的颠覆性项目，首创DeFi机制"))
    assert r.components['innovation'] == 20, f"got {r.components['innovation']}"  # 3 zh hits → tier_s

def test_timeliness_fast():
    now = datetime.now(timezone.utc)
    h = LaunchHunter(CONFIG_PATH)
    r = h.evaluate(make_event(t0_first_seen=now - timedelta(seconds=30), t1_captured=now))
    assert r.components['timeliness'] == 10

def test_timeliness_medium():
    now = datetime.now(timezone.utc)
    h = LaunchHunter(CONFIG_PATH)
    r = h.evaluate(make_event(t0_first_seen=now - timedelta(seconds=120), t1_captured=now))
    assert r.components['timeliness'] == 7

def test_timeliness_slow():
    now = datetime.now(timezone.utc)
    h = LaunchHunter(CONFIG_PATH)
    r = h.evaluate(make_event(t0_first_seen=now - timedelta(seconds=600), t1_captured=now))
    assert r.components['timeliness'] == 4

def test_timeliness_very_slow():
    now = datetime.now(timezone.utc)
    h = LaunchHunter(CONFIG_PATH)
    r = h.evaluate(make_event(t0_first_seen=now - timedelta(seconds=7200), t1_captured=now))
    assert r.components['timeliness'] == 1

def test_full_score_scenario():
    now = datetime.now(timezone.utc)
    h = LaunchHunter(CONFIG_PATH, sonnet_caller=mock_sonnet_high)
    r = h.evaluate(make_event(
        t0_first_seen=now - timedelta(seconds=20), t1_captured=now,
        raw_text="Breakthrough new paradigm: zero to one DeFi protocol. Free mint live now!",
        onchain=OnchainData(contract_verified=True, has_liquidity=True, status="success"),
    ))
    assert r.score >= 90, f"got {r.score}"

def test_low_score_scenario():
    now = datetime.now(timezone.utc)
    reg = MockRegistryReader()
    reg.projects['projectx'] = {'has_token': True, 'stage': 'mainnet'}
    reg.concepts['ProjectX'] = {'concept': 'ProjectX'}  # not a new concept
    h = LaunchHunter(CONFIG_PATH, registry=reg, sonnet_caller=mock_sonnet_low)
    r = h.evaluate(make_event(
        t0_first_seen=now - timedelta(seconds=7200), t1_captured=now,
        raw_text="Another update from ProjectX.", contract_address="",
        onchain=OnchainData(status="success"),
    ))
    assert r.score <= 25, f"got {r.score}"

def test_output_format():
    h = LaunchHunter(CONFIG_PATH, sonnet_caller=mock_sonnet_high)
    r = h.evaluate(make_event())
    assert isinstance(r.score, (int, float)) and 0 <= r.score <= 100
    assert isinstance(r.components, dict)
    assert isinstance(r.hit_rules, list)
    assert isinstance(r.observability, dict)
    for dim in ['novelty','verifiability','narrative','innovation','timeliness']:
        assert dim in r.components, f"missing {dim}"
    assert r.observability['module'] == 'mod1'
    assert 'config_hash' in r.observability

def test_disabled_module():
    cfg = yaml.safe_load(open(CONFIG_PATH))
    cfg['enabled'] = False
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(cfg, f)
        path = f.name
    try:
        h = LaunchHunter(path)
        r = h.evaluate(make_event())
        assert r.score == 0
        assert r.observability['module_status'] == 'disabled'
    finally:
        os.unlink(path)

def test_no_push_decision():
    h = LaunchHunter(CONFIG_PATH, sonnet_caller=mock_sonnet_high)
    r = h.evaluate(make_event())
    assert not hasattr(r, 'push')
    assert not hasattr(r, 'decision')
    assert 'push' not in r.observability
    assert 'decision' not in r.observability

def test_registry_updates_are_suggestions():
    h = LaunchHunter(CONFIG_PATH, sonnet_caller=mock_sonnet_high)
    r = h.evaluate(make_event())
    if r.registry_updates:
        allowed = {'project-registry', 'concept-tracker'}
        assert set(r.registry_updates.keys()).issubset(allowed)

def test_convenience_function():
    r = evaluate_launch(make_event(), CONFIG_PATH, sonnet_caller=mock_sonnet_high)
    assert isinstance(r, ModuleOutput)
    assert r.score > 0

def test_novelty_stage_change():
    """stage变化视为新进展 → 15分"""
    reg = MockRegistryReader()
    reg.projects['projectx'] = {
        'has_token': True, 'stage': 'testnet', 'current_stage': 'mainnet'
    }
    h = LaunchHunter(CONFIG_PATH, registry=reg)
    r = h.evaluate(make_event())
    assert r.components['novelty'] == 15, f"got {r.components['novelty']}"
    assert 'novelty_new_progress' in r.hit_rules

def test_innovation_new_concept_not_in_scoring():
    """新概念只记录不参与分档 (v1.1合规)"""
    h = LaunchHunter(CONFIG_PATH)
    r = h.evaluate(make_event(raw_text="Just another normal project.", detected_projects=["BrandNewConcept"]))
    assert r.components['innovation'] == 3  # 0 keyword hits → default
    detail = r.observability['scoring_process']['dimension_details']['innovation']
    assert 'BrandNewConcept' in detail['new_concepts']  # 记录但不加分

def test_enriched_event_no_extra_fields():
    allowed = {
        'event_id','t0_first_seen','t1_captured','source_type','source_account','raw_text',
        't2_second_mention','t3_market_reaction','detected_projects','detected_tokens',
        'contract_address','project_id','event_type','onchain','cross_evidence','risk',
        'is_original','official_confirmed','feishu_unconfirmed','cross_validated',
        'cross_validated_new','event_type_changed','onchain_status',
    }
    actual = set(EnrichedEvent.__dataclass_fields__.keys())
    extra = actual - allowed
    assert extra == set(), f"Extra fields: {extra}"

# ── Run ─────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("打新猎手 (mod1) 测试")
    print("=" * 60)

    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]

    print(f"\n运行 {len(tests)} 个测试...\n")
    for fn in tests:
        run_test(fn.__name__, fn)

    print(f"\n{'=' * 60}")
    print(f"结果: {passed} 通过, {failed} 失败 / 共 {passed + failed} 个")
    print(f"{'=' * 60}")

    if errors:
        print("\n失败详情:")
        for name, e in errors:
            print(f"\n  {name}:")
            traceback.print_exception(type(e), e, e.__traceback__)

    sys.exit(0 if failed == 0 else 1)
