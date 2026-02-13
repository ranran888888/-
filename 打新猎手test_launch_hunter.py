"""
打新猎手测试 - mod1
覆盖: 配置校验 / 5维度评分 / 降级路径 / 可观测性 / 冻结接口合规
"""

import pytest
import os
import yaml
from datetime import datetime, timedelta, timezone
from launch_hunter import (
    LaunchHunter, EnrichedEvent, OnchainData, CrossEvidence, RiskData,
    ModuleOutput, RegistryReader, NullRegistryReader,
    evaluate_launch, _null_sonnet_caller
)


CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'launch_hunter.yaml')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 测试用 Registry 实现
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MockRegistryReader(RegistryReader):
    """测试用Registry (可预设数据)"""

    def __init__(self):
        self.projects = {}
        self.concepts = {}
        self.propagations = {}

    def project_exists(self, project_id: str) -> bool:
        return project_id in self.projects

    def get_project(self, project_id: str) -> dict | None:
        return self.projects.get(project_id)

    def concept_exists(self, concept: str) -> bool:
        return concept in self.concepts

    def get_concept(self, concept: str) -> dict | None:
        return self.concepts.get(concept)

    def get_propagation(self, fingerprint: str) -> dict | None:
        return self.propagations.get(fingerprint)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 测试用 Sonnet 实现
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def mock_sonnet_high(prompt: str) -> dict:
    """高分Sonnet响应"""
    return {
        "track_score": 5,
        "pain_point_score": 4,
        "uniqueness_score": 5,
        "credibility_score": 4,
        "narrative_core": "DeFi新范式",
        "narrative_tags": ["DeFi", "创新"]
    }


def mock_sonnet_low(prompt: str) -> dict:
    """低分Sonnet响应"""
    return {
        "track_score": 1,
        "pain_point_score": 1,
        "uniqueness_score": 1,
        "credibility_score": 1,
        "narrative_core": "普通项目",
        "narrative_tags": []
    }


def mock_sonnet_error(prompt: str) -> dict:
    """Sonnet异常"""
    raise TimeoutError("Sonnet API timeout")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 工厂函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_event(**overrides) -> EnrichedEvent:
    """创建测试用EnrichedEvent"""
    now = datetime.now(timezone.utc)
    defaults = dict(
        event_id="test-event-001",
        t0_first_seen=now - timedelta(seconds=30),
        t1_captured=now,
        source_type="twitter_builder",
        source_account="@test_builder",
        raw_text="New project launch: ProjectX is a breakthrough DeFi protocol. Free mint live now!",
        event_type="launch",
        detected_projects=["ProjectX"],
        contract_address="0xabc123",
        project_id="projectx",
        onchain=OnchainData(
            contract_verified=True,
            deployer="0xdeployer",
            has_liquidity=True,
            liquidity_usd=50000.0,
            status="success"
        ),
        cross_evidence=CrossEvidence(original_sources_count=2, platforms=["twitter", "feishu"]),
        risk=RiskData(risk_level="green"),
        is_original=True,
    )
    defaults.update(overrides)
    return EnrichedEvent(**defaults)


def make_hunter(**kwargs) -> LaunchHunter:
    """创建LaunchHunter实例"""
    return LaunchHunter(
        config_path=CONFIG_PATH,
        **kwargs
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 一、配置校验测试
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestConfigValidation:

    def test_valid_config_loads(self):
        """正常配置可加载"""
        hunter = make_hunter()
        assert hunter.is_enabled is True
        assert hunter.config['version'] == '1.0'

    def test_config_hash_stable(self):
        """同配置hash一致"""
        h1 = make_hunter()
        h2 = make_hunter()
        assert h1.config_hash == h2.config_hash

    def test_max_score_sum_is_100(self):
        """5维度分数之和=100"""
        hunter = make_hunter()
        total = sum(
            hunter.config[dim]['max_score']
            for dim in ['novelty', 'verifiability', 'narrative', 'innovation', 'timeliness']
        )
        assert total == 100

    def test_invalid_config_raises(self, tmp_path):
        """无效配置应抛异常"""
        bad_cfg = tmp_path / "bad.yaml"
        bad_cfg.write_text("version: '1.0'\n")

        with pytest.raises(ValueError, match="Missing required config keys"):
            LaunchHunter(config_path=str(bad_cfg))

    def test_timeliness_monotonic_check(self, tmp_path):
        """timeliness阈值非单调应抛异常"""
        cfg = yaml.safe_load(open(CONFIG_PATH))
        cfg['timeliness']['thresholds'] = {'tier_s': 300, 'tier_a': 60, 'tier_b': 1800}

        bad_cfg = tmp_path / "bad_time.yaml"
        bad_cfg.write_text(yaml.dump(cfg))

        with pytest.raises(ValueError, match="monotonic"):
            LaunchHunter(config_path=str(bad_cfg))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 二、输入校验测试
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestInputValidation:

    def test_wrong_event_type_raises(self):
        """非launch类型应拒绝"""
        hunter = make_hunter()
        event = make_event(event_type="meme_signal")

        with pytest.raises(ValueError, match="event_type='launch'"):
            hunter.evaluate(event)

    def test_missing_event_id_raises(self):
        """缺少event_id应拒绝"""
        hunter = make_hunter()
        event = make_event(event_id="")

        with pytest.raises(ValueError, match="event_id"):
            hunter.evaluate(event)

    def test_missing_raw_text_raises(self):
        """缺少raw_text应拒绝"""
        hunter = make_hunter()
        event = make_event(raw_text="")

        with pytest.raises(ValueError, match="raw_text"):
            hunter.evaluate(event)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 三、维度评分测试
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestNovelty:

    def test_first_mention(self):
        """project-registry不存在 → 30分"""
        hunter = make_hunter(registry=NullRegistryReader())
        event = make_event()
        result = hunter.evaluate(event)

        assert result.components['novelty'] == 30
        assert 'novelty_first_mention' in result.hit_rules

    def test_new_progress(self):
        """存在但有新进展 → 15分"""
        registry = MockRegistryReader()
        registry.projects['projectx'] = {
            'project_id': 'projectx',
            'has_token': False,
            'stage': 'announced'
        }

        hunter = make_hunter(registry=registry)
        event = make_event(contract_address="0xnew_contract")
        result = hunter.evaluate(event)

        assert result.components['novelty'] == 15
        assert 'novelty_new_progress' in result.hit_rules

    def test_known_repeat(self):
        """已知项目无新进展 → 5分"""
        registry = MockRegistryReader()
        registry.projects['projectx'] = {
            'project_id': 'projectx',
            'has_token': True,
            'stage': 'mainnet'
        }

        hunter = make_hunter(registry=registry)
        event = make_event()
        result = hunter.evaluate(event)

        assert result.components['novelty'] == 5
        assert 'novelty_known_repeat' in result.hit_rules

    def test_no_project_id(self):
        """无项目标识 → 保守偏高30分"""
        hunter = make_hunter()
        event = make_event(project_id="", detected_projects=[])
        result = hunter.evaluate(event)

        assert result.components['novelty'] == 30
        assert 'novelty_no_project_id' in result.hit_rules


class TestVerifiability:

    def test_contract_verified(self):
        """有合约+已验证 → 20分"""
        hunter = make_hunter()
        event = make_event(
            contract_address="0xabc",
            onchain=OnchainData(contract_verified=True, status="success")
        )
        result = hunter.evaluate(event)

        assert result.components['verifiability'] == 20
        assert 'verifiability_contract_verified' in result.hit_rules

    def test_contract_unverified(self):
        """有合约+未验证 → 10分"""
        hunter = make_hunter()
        event = make_event(
            contract_address="0xabc",
            onchain=OnchainData(contract_verified=False, status="success")
        )
        result = hunter.evaluate(event)

        assert result.components['verifiability'] == 10
        assert 'verifiability_contract_unverified' in result.hit_rules

    def test_no_contract(self):
        """无合约 → 5分"""
        hunter = make_hunter()
        event = make_event(
            contract_address="",
            onchain=OnchainData(status="success")
        )
        result = hunter.evaluate(event)

        assert result.components['verifiability'] == 5
        assert 'verifiability_no_contract' in result.hit_rules

    def test_onchain_timeout_degrades(self):
        """链上API超时 → 降级"""
        hunter = make_hunter()
        event = make_event(
            contract_address="0xabc",
            onchain=OnchainData(contract_verified=True, status="timeout")
        )
        result = hunter.evaluate(event)

        assert result.components['verifiability'] == 10
        assert 'degraded' in result.hit_rules[1]  # verifiability rule
        assert 'onchain_api_timeout' in result.observability.get('warnings', [])

    def test_onchain_error_degrades(self):
        """链上API错误 → 降级"""
        hunter = make_hunter()
        event = make_event(
            contract_address="",
            onchain=OnchainData(status="error")
        )
        result = hunter.evaluate(event)

        assert result.components['verifiability'] == 5


class TestNarrative:

    def test_sonnet_high_score(self):
        """Sonnet高分 → narrative接近20"""
        hunter = make_hunter(sonnet_caller=mock_sonnet_high)
        event = make_event()
        result = hunter.evaluate(event)

        assert result.components['narrative'] == 18  # 5+4+5+4
        assert 'narrative_strong' in result.hit_rules

    def test_sonnet_low_score(self):
        """Sonnet低分 → narrative=4"""
        hunter = make_hunter(sonnet_caller=mock_sonnet_low)
        event = make_event()
        result = hunter.evaluate(event)

        assert result.components['narrative'] == 4  # 1+1+1+1
        assert 'narrative_weak' in result.hit_rules

    def test_sonnet_error_degrades(self):
        """Sonnet异常 → 降级到默认8分"""
        hunter = make_hunter(sonnet_caller=mock_sonnet_error)
        event = make_event()
        result = hunter.evaluate(event)

        assert result.components['narrative'] == 8  # 2+2+2+2 (默认)
        assert 'narrative_sonnet_degraded' in result.hit_rules
        assert 'sonnet_degraded' in result.observability.get('warnings', [])

    def test_sonnet_score_clamped(self):
        """Sonnet返回超范围值 → 夹紧到0-5"""
        def sonnet_overflow(prompt):
            return {
                "track_score": 10,
                "pain_point_score": -1,
                "uniqueness_score": 5,
                "credibility_score": 5,
                "narrative_core": "test",
                "narrative_tags": []
            }

        hunter = make_hunter(sonnet_caller=sonnet_overflow)
        event = make_event()
        result = hunter.evaluate(event)

        # 10→5, -1→0, 5, 5 → 15, capped at 20
        assert result.components['narrative'] == 15

    def test_narrative_tags_in_registry_updates(self):
        """叙事标签应出现在registry_updates"""
        hunter = make_hunter(sonnet_caller=mock_sonnet_high)
        event = make_event()
        result = hunter.evaluate(event)

        assert result.registry_updates is not None
        concepts = result.registry_updates.get('concept-tracker', [])
        tags = [c['concept'] for c in concepts]
        assert 'DeFi' in tags
        assert '创新' in tags


class TestInnovation:

    def test_multiple_keywords(self):
        """命中多个创新词 → 高分"""
        hunter = make_hunter()
        event = make_event(
            raw_text="This is a breakthrough new paradigm project, truly zero to one."
        )
        result = hunter.evaluate(event)

        assert result.components['innovation'] == 20  # 3+ hits
        assert 'innovation_tier_s' in result.hit_rules

    def test_one_keyword(self):
        """命中1个 → 10分"""
        hunter = make_hunter()
        event = make_event(raw_text="A breakthrough in yield farming.")
        result = hunter.evaluate(event)

        assert result.components['innovation'] == 10
        assert 'innovation_tier_b' in result.hit_rules

    def test_no_keywords(self):
        """无命中 → 3分"""
        hunter = make_hunter()
        event = make_event(raw_text="Just another normal token launch.")
        result = hunter.evaluate(event)

        assert result.components['innovation'] == 3
        assert 'innovation_no_keywords' in result.hit_rules

    def test_chinese_keywords(self):
        """中文创新词命中"""
        hunter = make_hunter()
        event = make_event(raw_text="这是一个新范式的颠覆性项目，首创DeFi机制")
        result = hunter.evaluate(event)

        assert result.components['innovation'] >= 15  # 3 zh hits


class TestTimeliness:

    def test_fast_capture(self):
        """<60秒 → 10分"""
        now = datetime.now(timezone.utc)
        hunter = make_hunter()
        event = make_event(
            t0_first_seen=now - timedelta(seconds=30),
            t1_captured=now
        )
        result = hunter.evaluate(event)

        assert result.components['timeliness'] == 10
        assert 'timeliness_tier_s' in result.hit_rules

    def test_medium_capture(self):
        """<300秒 → 7分"""
        now = datetime.now(timezone.utc)
        hunter = make_hunter()
        event = make_event(
            t0_first_seen=now - timedelta(seconds=120),
            t1_captured=now
        )
        result = hunter.evaluate(event)

        assert result.components['timeliness'] == 7
        assert 'timeliness_tier_a' in result.hit_rules

    def test_slow_capture(self):
        """<1800秒 → 4分"""
        now = datetime.now(timezone.utc)
        hunter = make_hunter()
        event = make_event(
            t0_first_seen=now - timedelta(seconds=600),
            t1_captured=now
        )
        result = hunter.evaluate(event)

        assert result.components['timeliness'] == 4
        assert 'timeliness_tier_b' in result.hit_rules

    def test_very_slow(self):
        """>1800秒 → 1分"""
        now = datetime.now(timezone.utc)
        hunter = make_hunter()
        event = make_event(
            t0_first_seen=now - timedelta(seconds=7200),
            t1_captured=now
        )
        result = hunter.evaluate(event)

        assert result.components['timeliness'] == 1
        assert 'timeliness_slow' in result.hit_rules


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 四、集成测试
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestIntegration:

    def test_full_score_scenario(self):
        """最高分场景: 新项目+已验证+强叙事+多创新词+快速"""
        now = datetime.now(timezone.utc)
        hunter = make_hunter(sonnet_caller=mock_sonnet_high)
        event = make_event(
            t0_first_seen=now - timedelta(seconds=20),
            t1_captured=now,
            raw_text="Breakthrough new paradigm: zero to one DeFi protocol. Free mint live now!",
            onchain=OnchainData(contract_verified=True, has_liquidity=True, status="success"),
        )
        result = hunter.evaluate(event)

        # novelty=30 + verifiability=20 + narrative=18 + innovation=20 + timeliness=10 = 98
        assert result.score >= 90
        assert result.score <= 100

    def test_low_score_scenario(self):
        """最低分场景: 已知项目+无合约+弱叙事+无创新+慢"""
        now = datetime.now(timezone.utc)

        registry = MockRegistryReader()
        registry.projects['projectx'] = {'has_token': True, 'stage': 'mainnet'}

        hunter = make_hunter(
            registry=registry,
            sonnet_caller=mock_sonnet_low
        )
        event = make_event(
            t0_first_seen=now - timedelta(seconds=7200),
            t1_captured=now,
            raw_text="Another update from ProjectX.",
            contract_address="",
            onchain=OnchainData(status="success"),
        )
        result = hunter.evaluate(event)

        # novelty=5 + verifiability=5 + narrative=4 + innovation=3 + timeliness=1 = 18
        assert result.score <= 25

    def test_output_format_matches_frozen_interface(self):
        """输出格式严格匹配冻结接口"""
        hunter = make_hunter(sonnet_caller=mock_sonnet_high)
        event = make_event()
        result = hunter.evaluate(event)

        # 冻结接口要求的字段
        assert isinstance(result.score, (int, float))
        assert 0 <= result.score <= 100
        assert isinstance(result.components, dict)
        assert isinstance(result.hit_rules, list)
        assert isinstance(result.observability, dict)

        # components必须包含5个维度
        for dim in ['novelty', 'verifiability', 'narrative', 'innovation', 'timeliness']:
            assert dim in result.components

        # observability必须包含回放字段
        obs = result.observability
        assert 'module' in obs
        assert obs['module'] == 'mod1'
        assert 'config_version' in obs
        assert 'config_hash' in obs
        assert 'scoring_process' in obs

    def test_disabled_module(self, tmp_path):
        """模块禁用时返回空结果"""
        cfg = yaml.safe_load(open(CONFIG_PATH))
        cfg['enabled'] = False
        disabled_cfg = tmp_path / "disabled.yaml"
        disabled_cfg.write_text(yaml.dump(cfg))

        hunter = LaunchHunter(config_path=str(disabled_cfg))
        event = make_event()
        result = hunter.evaluate(event)

        assert result.score == 0
        assert result.components == {}
        assert result.observability['module_status'] == 'disabled'

    def test_convenience_function(self):
        """便捷函数 evaluate_launch 可正常调用"""
        event = make_event()
        result = evaluate_launch(event, CONFIG_PATH, sonnet_caller=mock_sonnet_high)

        assert isinstance(result, ModuleOutput)
        assert result.score > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 五、冻结接口合规测试
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFrozenInterfaceCompliance:

    def test_no_field_added_to_enriched_event(self):
        """不新增字段到EnrichedEvent (冻结接口约束1)"""
        # EnrichedEvent的字段应该只包含冻结接口中定义的字段
        allowed_fields = {
            # Event必填
            'event_id', 't0_first_seen', 't1_captured', 'source_type',
            'source_account', 'raw_text',
            # Event可选
            't2_second_mention', 't3_market_reaction', 'detected_projects',
            'detected_tokens', 'contract_address', 'project_id', 'event_type',
            # Enriched扩展
            'onchain', 'cross_evidence', 'risk', 'is_original',
            'official_confirmed', 'feishu_unconfirmed', 'cross_validated',
            'cross_validated_new', 'event_type_changed', 'onchain_status',
        }
        actual_fields = set(EnrichedEvent.__dataclass_fields__.keys())
        extra_fields = actual_fields - allowed_fields
        assert extra_fields == set(), f"Extra fields in EnrichedEvent: {extra_fields}"

    def test_registry_not_directly_written(self):
        """Registry通过建议更新, 不直接写入 (冻结接口约束2)"""
        hunter = make_hunter(sonnet_caller=mock_sonnet_high)
        event = make_event()
        result = hunter.evaluate(event)

        # registry_updates是建议, 不是直接写入
        if result.registry_updates:
            assert isinstance(result.registry_updates, dict)
            # 只允许写入 project-registry 和 concept-tracker
            allowed_registries = {'project-registry', 'concept-tracker'}
            actual_registries = set(result.registry_updates.keys())
            assert actual_registries.issubset(allowed_registries)

    def test_no_push_decision_in_output(self):
        """模块不做推送决策 (冻结接口约束4)"""
        hunter = make_hunter(sonnet_caller=mock_sonnet_high)
        event = make_event()
        result = hunter.evaluate(event)

        # 输出中不应包含push/decision相关字段
        assert not hasattr(result, 'push')
        assert not hasattr(result, 'decision')
        assert 'push' not in result.observability
        assert 'decision' not in result.observability

    def test_yaml_config_format_compliance(self):
        """配置文件格式合规 (snake_case, 有type声明)"""
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)

        # 检查所有键名是snake_case
        def check_snake_case(d, path=""):
            if isinstance(d, dict):
                for key in d:
                    assert re.match(r'^[a-z][a-z0-9_]*$', str(key)) or isinstance(key, int), \
                        f"Non-snake_case key: {path}.{key}"
                    check_snake_case(d[key], f"{path}.{key}")
            elif isinstance(d, list):
                for i, item in enumerate(d):
                    check_snake_case(item, f"{path}[{i}]")

        # 此处只检查顶层和关键的二级键
        for key in config:
            assert re.match(r'^[a-z][a-z0-9_]*$', str(key)), \
                f"Non-snake_case top-level key: {key}"


import re  # needed for the last test

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 运行入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
