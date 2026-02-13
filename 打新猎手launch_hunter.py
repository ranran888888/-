"""
打新猎手 - mod1
版本: 1.0
功能: 评估新项目发行事件（launch），输出Base Score + 可观测性
严格遵循: 冻结接口清单 v1.1

五维度评分:
  - novelty      (30分): 是否首次提及/新进展
  - verifiability (20分): 链上可验证性
  - narrative     (20分): Sonnet叙事分析【补丁P10】
  - innovation    (20分): 创新关键词匹配
  - timeliness    (10分): T1-T0 捕获延迟
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Callable
from datetime import datetime, timezone
import yaml
import hashlib
import json
import re


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 数据结构 (严格对齐冻结接口 EnrichedEvent)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class OnchainData:
    """链上数据 (冻结接口: EnrichedEvent.onchain)"""
    contract_verified: bool = False
    deployer: str = ""
    deployer_history: Optional[Dict] = None
    has_liquidity: bool = False
    liquidity_usd: float = 0.0
    status: str = "success"  # success/timeout/error


@dataclass
class CrossEvidence:
    """跨源证据 (冻结接口: EnrichedEvent.cross_evidence)"""
    original_sources_count: int = 0
    platforms: Optional[List[str]] = None
    cross_bonus_key: str = ""


@dataclass
class RiskData:
    """风控数据 (冻结接口: EnrichedEvent.risk)"""
    risk_level: str = "green"  # red/yellow/green
    risk_evidence: Optional[List[str]] = None


@dataclass
class EnrichedEvent:
    """
    富化后事件 (严格对齐冻结接口 EnrichedEvent Schema)
    模块只读取, 不修改, 不新增字段
    """
    # === Event 必填字段 ===
    event_id: str = ""
    t0_first_seen: Optional[datetime] = None
    t1_captured: Optional[datetime] = None
    source_type: str = ""
    source_account: str = ""
    raw_text: str = ""

    # === Event 可选字段 ===
    t2_second_mention: Optional[datetime] = None
    t3_market_reaction: Optional[datetime] = None
    detected_projects: Optional[List[str]] = None
    detected_tokens: Optional[List[str]] = None
    contract_address: str = ""
    project_id: str = ""
    event_type: str = ""

    # === Enriched 扩展字段 ===
    onchain: Optional[OnchainData] = None
    cross_evidence: Optional[CrossEvidence] = None
    risk: Optional[RiskData] = None
    is_original: bool = True
    official_confirmed: bool = False
    feishu_unconfirmed: bool = False
    cross_validated: bool = False
    cross_validated_new: bool = False
    event_type_changed: bool = False
    onchain_status: str = ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 模块输出 (严格对齐冻结接口 "模块输出格式")
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ModuleOutput:
    """
    冻结接口标准模块输出:
    {
        "score": float (0-100),
        "components": { ... },
        "hit_rules": [ ... ],
        "observability": { ... },
        "registry_updates": { ... }   # 可选
    }
    """
    score: float
    components: Dict[str, float]
    hit_rules: List[str]
    observability: Dict[str, Any]
    registry_updates: Optional[Dict[str, Any]] = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Registry 只读接口 (抽象, 由外部注入)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RegistryReader:
    """
    Registry只读接口

    冻结接口约束:
      - 模块只能读取Registry, 不能直接写入
      - 写入由pipeline层统一执行
      - 模块只输出 registry_updates 建议
    """

    def project_exists(self, project_id: str) -> bool:
        """project-registry: 项目是否已存在"""
        raise NotImplementedError

    def get_project(self, project_id: str) -> Optional[Dict]:
        """project-registry: 获取项目信息"""
        raise NotImplementedError

    def concept_exists(self, concept: str) -> bool:
        """concept-tracker: 概念是否已存在"""
        raise NotImplementedError

    def get_concept(self, concept: str) -> Optional[Dict]:
        """concept-tracker: 获取概念信息"""
        raise NotImplementedError

    def get_propagation(self, fingerprint: str) -> Optional[Dict]:
        """propagation-tracker: 获取传播信息"""
        raise NotImplementedError


class NullRegistryReader(RegistryReader):
    """空实现 (用于测试/降级)"""

    def project_exists(self, project_id: str) -> bool:
        return False

    def get_project(self, project_id: str) -> Optional[Dict]:
        return None

    def concept_exists(self, concept: str) -> bool:
        return False

    def get_concept(self, concept: str) -> Optional[Dict]:
        return None

    def get_propagation(self, fingerprint: str) -> Optional[Dict]:
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sonnet叙事分析 接口 (抽象, 由外部注入)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 【补丁P10】Sonnet叙事分析prompt模板
NARRATIVE_PROMPT_TEMPLATE = """你是 Web3 打新分析师。分析以下新项目发行信息。

推文：{raw_text}
项目：{project_name}
链：{chain}
类型：{launch_type}

请从以下4个维度打分（每个0-5分，总分0-20）：
1. 赛道定位：是否在有前景的赛道，叙事是否清晰
2. 痛点清晰度：解决什么问题，是否真实需求
3. 独特性：与同类项目有何区别，创新点在哪
4. 可信度：信息来源、项目方背景、技术可行性

请仅返回JSON：
{{
  "track_score": 0-5,
  "pain_point_score": 0-5,
  "uniqueness_score": 0-5,
  "credibility_score": 0-5,
  "narrative_core": "一句话叙事（15字内）",
  "narrative_tags": ["标签1", "标签2"]
}}"""


# Sonnet调用回调类型: (prompt: str) -> dict
# 返回值示例: {"track_score": 3, "pain_point_score": 2, ...}
SonnetCallable = Callable[[str], Dict[str, Any]]


def _null_sonnet_caller(prompt: str) -> Dict[str, Any]:
    """
    降级用Sonnet调用 (Sonnet不可用时返回保守默认值)
    每维度给2分, 总计8分
    """
    return {
        "track_score": 2,
        "pain_point_score": 2,
        "uniqueness_score": 2,
        "credibility_score": 2,
        "narrative_core": "sonnet_unavailable",
        "narrative_tags": [],
        "_degraded": True
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主类: LaunchHunter (mod1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LaunchHunter:
    """
    打新猎手 (mod1)

    职责边界 (冻结接口约束):
      ✅ 计算 Base Score (0-100)
      ✅ 输出 Base Components (评分贡献分解)
      ✅ 输出命中规则列表
      ✅ 输出可观测性字段
      ✅ 建议更新 Registry (不直接写)
      ✅ 读取配置文件 (YAML)
      ✅ 读取 Registry (只读)

      ❌ 不做推送决策 (scorer统一执行)
      ❌ 不做冷却期判定 (scorer统一执行)
      ❌ 不做衰减计算 (scorer统一执行)
      ❌ 不做跨源加成 (scorer统一执行)
      ❌ 不做市场状态判定 (market_state_job执行)
      ❌ 不调用外部API (enricher层已传入)
      ❌ 不新增字段到 Event/EnrichedEvent/Alert
    """

    def __init__(
        self,
        config_path: str,
        registry: Optional[RegistryReader] = None,
        sonnet_caller: Optional[SonnetCallable] = None
    ):
        """
        初始化打新猎手

        Args:
            config_path: 配置文件路径 (launch_hunter.yaml)
            registry: Registry只读接口 (可选, 默认NullRegistryReader)
            sonnet_caller: Sonnet调用回调 (可选, 默认降级实现)
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
            self.config = yaml.safe_load(config_content)

        # 配置校验
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a YAML mapping (dict), got None or invalid type")

        self._validate_config()

        # 配置hash (用于回放/审计)
        self.config_hash = hashlib.md5(config_content.encode()).hexdigest()
        self.config_path = config_path

        # 依赖注入
        self.registry = registry or NullRegistryReader()
        self.sonnet_caller = sonnet_caller or _null_sonnet_caller

        # 模块开关
        self.is_enabled = self.config.get('enabled', True)

    # ─────────────────────────────────────────────
    # 配置校验
    # ─────────────────────────────────────────────

    def _validate_config(self) -> None:
        """校验配置完整性和合法性"""
        required_keys = [
            'version',
            'novelty',
            'verifiability',
            'narrative',
            'innovation',
            'timeliness',
            'launch_action_keywords',
            'innovation_keywords'
        ]
        missing = [k for k in required_keys if k not in self.config]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")

        # 校验各维度配置
        for dim in ['novelty', 'verifiability', 'timeliness']:
            dim_cfg = self.config[dim]
            if 'max_score' not in dim_cfg:
                raise ValueError(f"Missing {dim}.max_score")

        # 校验narrative配置
        nar_cfg = self.config['narrative']
        if 'max_score' not in nar_cfg:
            raise ValueError("Missing narrative.max_score")
        if 'sub_dimensions' not in nar_cfg:
            raise ValueError("Missing narrative.sub_dimensions")
        if 'timeout_seconds' not in nar_cfg:
            raise ValueError("Missing narrative.timeout_seconds")

        # 校验innovation配置
        inn_cfg = self.config['innovation']
        if 'max_score' not in inn_cfg:
            raise ValueError("Missing innovation.max_score")

        # 校验timeliness阈值单调性 (更小的delay对应更高分)
        time_cfg = self.config['timeliness']
        thresholds = time_cfg.get('thresholds', {})
        if thresholds:
            t1 = thresholds.get('tier_s', 60)
            t2 = thresholds.get('tier_a', 300)
            t3 = thresholds.get('tier_b', 1800)
            if not (t1 < t2 < t3):
                raise ValueError(
                    f"timeliness.thresholds must be monotonic: "
                    f"tier_s < tier_a < tier_b, got {t1} < {t2} < {t3}"
                )

        # 校验launch_action_keywords非空
        kw_cfg = self.config['launch_action_keywords']
        if not kw_cfg.get('en') and not kw_cfg.get('zh'):
            raise ValueError("launch_action_keywords must have at least en or zh keywords")

        # 分数总和校验 (5维度应 = 100)
        total_max = sum(
            self.config[dim]['max_score']
            for dim in ['novelty', 'verifiability', 'narrative', 'innovation', 'timeliness']
        )
        if total_max != 100:
            raise ValueError(
                f"Sum of max_score across 5 dimensions must be 100, got {total_max}"
            )

    # ─────────────────────────────────────────────
    # 主入口
    # ─────────────────────────────────────────────

    def evaluate(self, enriched: EnrichedEvent) -> ModuleOutput:
        """
        评估一个launch事件

        Args:
            enriched: 富化后的事件 (event_type=launch)

        Returns:
            ModuleOutput: 标准模块输出
        """
        # 模块禁用
        if not self.is_enabled:
            return self._disabled_result(enriched)

        # 输入校验
        self._validate_input(enriched)

        # 计算5个维度
        components = {}
        hit_rules = []
        dimension_details = {}

        # 1. Novelty (30分)
        nov_score, nov_rule, nov_detail = self._compute_novelty(enriched)
        components['novelty'] = nov_score
        hit_rules.append(nov_rule)
        dimension_details['novelty'] = nov_detail

        # 2. Verifiability (20分)
        ver_score, ver_rule, ver_detail = self._compute_verifiability(enriched)
        components['verifiability'] = ver_score
        hit_rules.append(ver_rule)
        dimension_details['verifiability'] = ver_detail

        # 3. Narrative (20分) - Sonnet叙事分析【补丁P10】
        nar_score, nar_rule, nar_detail = self._compute_narrative(enriched)
        components['narrative'] = nar_score
        hit_rules.append(nar_rule)
        dimension_details['narrative'] = nar_detail

        # 4. Innovation (20分)
        inn_score, inn_rule, inn_detail = self._compute_innovation(enriched)
        components['innovation'] = inn_score
        hit_rules.append(inn_rule)
        dimension_details['innovation'] = inn_detail

        # 5. Timeliness (10分)
        time_score, time_rule, time_detail = self._compute_timeliness(enriched)
        components['timeliness'] = time_score
        hit_rules.append(time_rule)
        dimension_details['timeliness'] = time_detail

        # 总分 (cap 100)
        total_score = min(sum(components.values()), 100)

        # Registry更新建议 (不直接写入)
        registry_updates = self._build_registry_updates(enriched, components, nar_detail)

        # 可观测性
        observability = self._build_observability(
            enriched, total_score, components, hit_rules, dimension_details
        )

        return ModuleOutput(
            score=total_score,
            components=components,
            hit_rules=hit_rules,
            observability=observability,
            registry_updates=registry_updates
        )

    # ─────────────────────────────────────────────
    # 输入校验
    # ─────────────────────────────────────────────

    def _validate_input(self, enriched: EnrichedEvent) -> None:
        """校验输入事件"""
        if not enriched.event_id:
            raise ValueError("event_id is required")

        if enriched.event_type != 'launch':
            raise ValueError(
                f"LaunchHunter only handles event_type='launch', got '{enriched.event_type}'"
            )

        if not enriched.raw_text:
            raise ValueError("raw_text is required")

        if enriched.t0_first_seen is None:
            raise ValueError("t0_first_seen is required")

        if enriched.t1_captured is None:
            raise ValueError("t1_captured is required")

    # ─────────────────────────────────────────────
    # 维度1: Novelty (30分)
    # ─────────────────────────────────────────────

    def _compute_novelty(self, enriched: EnrichedEvent) -> Tuple[float, str, Dict]:
        """
        新颖度评分

        规则 (来自v1.1规格):
          - is_first_mention → 30分
          - is_new_progress  → 15分
          - else (已知重复)  → 5分

        判定逻辑:
          - is_first_mention: project-registry中不存在该项目
          - is_new_progress: project-registry中存在但有新信息
                             (合约地址/流动性/stage变化)
        """
        max_score = self.config['novelty']['max_score']  # 30
        scores_cfg = self.config['novelty']['scores']

        project_id = enriched.project_id
        detected = enriched.detected_projects or []

        # 确定查询用的项目标识
        lookup_key = project_id or (detected[0] if detected else "")

        detail = {
            'lookup_key': lookup_key,
            'project_id': project_id,
            'detected_projects': detected,
            'is_original': enriched.is_original,
        }

        # 无项目标识 → 视为首次提及 (保守偏高, 避免漏掉alpha)
        if not lookup_key:
            score = scores_cfg.get('no_project_id', scores_cfg['first_mention'])
            detail['reason'] = 'no_project_id_treat_as_first'
            return score, 'novelty_no_project_id', detail

        # 查询 project-registry
        existing = self.registry.get_project(lookup_key)

        if existing is None:
            # 首次提及
            score = scores_cfg['first_mention']  # 30
            detail['reason'] = 'first_mention'
            detail['registry_hit'] = False
            return score, 'novelty_first_mention', detail

        # 已存在 → 判断是否有新进展
        detail['registry_hit'] = True
        detail['existing_stage'] = existing.get('stage', 'unknown')

        is_new_progress = self._check_new_progress(enriched, existing)
        detail['is_new_progress'] = is_new_progress

        if is_new_progress:
            score = scores_cfg['new_progress']  # 15
            detail['reason'] = 'new_progress'
            return score, 'novelty_new_progress', detail
        else:
            score = scores_cfg['known_repeat']  # 5
            detail['reason'] = 'known_repeat'
            return score, 'novelty_known_repeat', detail

    def _check_new_progress(self, enriched: EnrichedEvent, existing: Dict) -> bool:
        """
        判断是否有新进展:
          - 原来没合约, 现在有了
          - 原来没流动性, 现在有了
          - stage变化 (如 announced → testnet → mainnet)
        """
        onchain = enriched.onchain or OnchainData()

        # 新增合约地址
        if enriched.contract_address and not existing.get('has_token', False):
            return True

        # 新增流动性
        if onchain.has_liquidity and not existing.get('has_token', False):
            return True

        # stage变化
        existing_stage = existing.get('stage', '')
        if enriched.detected_projects:
            # 如果enriched带有不同的stage信息, 视为进展
            # 实际stage由enricher填充, 这里通过对比判断
            pass

        return False

    # ─────────────────────────────────────────────
    # 维度2: Verifiability (20分)
    # ─────────────────────────────────────────────

    def _compute_verifiability(self, enriched: EnrichedEvent) -> Tuple[float, str, Dict]:
        """
        可验证性评分

        规则 (来自v1.1规格):
          - 有合约 + contract_verified → 20分
          - 有合约 + 未验证            → 10分
          - 无合约                     → 5分

        额外考虑 (onchain.status降级):
          - onchain.status = timeout/error → 降级处理
        """
        scores_cfg = self.config['verifiability']['scores']
        onchain = enriched.onchain or OnchainData()

        detail = {
            'has_contract': bool(enriched.contract_address),
            'contract_verified': onchain.contract_verified,
            'has_liquidity': onchain.has_liquidity,
            'liquidity_usd': onchain.liquidity_usd,
            'onchain_status': onchain.status,
        }

        # onchain API超时/错误 → 降级到"有合约但未验证"或"无合约"
        if onchain.status in ('timeout', 'error'):
            detail['degraded'] = True
            detail['degrade_reason'] = f'onchain_api_{onchain.status}'

            if enriched.contract_address:
                # 有合约但链上API不可用 → 给中间分
                score = scores_cfg['contract_unverified']  # 10
                return score, 'verifiability_contract_api_degraded', detail
            else:
                score = scores_cfg['no_contract']  # 5
                return score, 'verifiability_no_contract_api_degraded', detail

        # 正常路径
        if enriched.contract_address and onchain.contract_verified:
            score = scores_cfg['contract_verified']  # 20

            # 流动性加成信息 (记录但不额外加分, base满分20)
            detail['liquidity_bonus_info'] = (
                f"liquidity_usd={onchain.liquidity_usd}"
                if onchain.has_liquidity else "no_liquidity"
            )
            return score, 'verifiability_contract_verified', detail

        elif enriched.contract_address:
            score = scores_cfg['contract_unverified']  # 10
            return score, 'verifiability_contract_unverified', detail

        else:
            score = scores_cfg['no_contract']  # 5
            return score, 'verifiability_no_contract', detail

    # ─────────────────────────────────────────────
    # 维度3: Narrative (20分) 【补丁P10】
    # ─────────────────────────────────────────────

    def _compute_narrative(self, enriched: EnrichedEvent) -> Tuple[float, str, Dict]:
        """
        Sonnet叙事分析评分

        规则 (来自v1.1补丁P10):
          - 调用Sonnet分析4个子维度 (各0-5分)
          - track_score + pain_point_score + uniqueness_score + credibility_score
          - 总分cap到max_score (20)
          - Sonnet超时/错误 → 降级到默认分

        冻结接口约束:
          - 这里不调用外部API, Sonnet调用通过注入的callable实现
          - callable由pipeline层封装超时/降级逻辑
        """
        nar_cfg = self.config['narrative']
        max_score = nar_cfg['max_score']  # 20
        sub_dims = nar_cfg['sub_dimensions']  # ["track", "pain_point", "uniqueness", "credibility"]
        default_per_dim = nar_cfg.get('default_per_dimension', 2)

        # 构造prompt
        project_info = {}
        if enriched.detected_projects:
            project_info['name'] = enriched.detected_projects[0]
        project_info['chain'] = ''  # 由enricher填充, 若无则为空
        project_info['launch_type'] = ''

        prompt = NARRATIVE_PROMPT_TEMPLATE.format(
            raw_text=enriched.raw_text[:1500],  # 截断防止prompt过长
            project_name=project_info.get('name', '未知'),
            chain=project_info.get('chain', '未知'),
            launch_type=project_info.get('launch_type', '未知')
        )

        detail = {
            'prompt_length': len(prompt),
            'raw_text_truncated': len(enriched.raw_text) > 1500,
        }

        # 调用Sonnet (通过注入的callable, 已封装超时/降级)
        try:
            result = self.sonnet_caller(prompt)
            detail['sonnet_status'] = 'success'
            detail['sonnet_degraded'] = result.get('_degraded', False)
        except Exception as e:
            # 调用失败 → 降级
            result = _null_sonnet_caller(prompt)
            detail['sonnet_status'] = 'error'
            detail['sonnet_error'] = str(e)[:200]
            detail['sonnet_degraded'] = True

        # 提取4个子维度分数
        track = self._clamp(result.get('track_score', default_per_dim), 0, 5)
        pain = self._clamp(result.get('pain_point_score', default_per_dim), 0, 5)
        unique = self._clamp(result.get('uniqueness_score', default_per_dim), 0, 5)
        cred = self._clamp(result.get('credibility_score', default_per_dim), 0, 5)

        total = min(track + pain + unique + cred, max_score)

        detail['sub_scores'] = {
            'track_score': track,
            'pain_point_score': pain,
            'uniqueness_score': unique,
            'credibility_score': cred,
        }
        detail['narrative_core'] = result.get('narrative_core', '')
        detail['narrative_tags'] = result.get('narrative_tags', [])

        # 生成hit_rule
        if detail.get('sonnet_degraded'):
            rule = 'narrative_sonnet_degraded'
        elif total >= 16:
            rule = 'narrative_strong'
        elif total >= 10:
            rule = 'narrative_moderate'
        else:
            rule = 'narrative_weak'

        return total, rule, detail

    # ─────────────────────────────────────────────
    # 维度4: Innovation (20分)
    # ─────────────────────────────────────────────

    def _compute_innovation(self, enriched: EnrichedEvent) -> Tuple[float, str, Dict]:
        """
        创新度评分

        规则 (来自v1.1规格):
          - 匹配创新关键词库 (en + zh)
          - 多个关键词命中 → 更高分
          - 同时检查concept-tracker是否有新概念

        评分逻辑:
          - 命中3+ 创新词 → 20分
          - 命中2个       → 15分
          - 命中1个       → 10分
          - 命中0个       → 3分 (基础分)
        """
        inn_cfg = self.config['innovation']
        max_score = inn_cfg['max_score']  # 20
        scores_cfg = inn_cfg['scores']

        # 加载关键词
        kw_cfg = self.config['innovation_keywords']
        en_keywords = [kw.lower() for kw in kw_cfg.get('en', [])]
        zh_keywords = kw_cfg.get('zh', [])

        text_lower = enriched.raw_text.lower()

        # 匹配英文关键词
        en_hits = [kw for kw in en_keywords if kw in text_lower]

        # 匹配中文关键词
        zh_hits = [kw for kw in zh_keywords if kw in enriched.raw_text]

        all_hits = en_hits + zh_hits
        hit_count = len(all_hits)

        # concept-tracker新概念检查
        new_concepts = []
        if enriched.detected_projects:
            for proj in enriched.detected_projects:
                if not self.registry.concept_exists(proj):
                    new_concepts.append(proj)

        detail = {
            'en_keyword_hits': en_hits,
            'zh_keyword_hits': zh_hits,
            'total_keyword_hits': hit_count,
            'new_concepts': new_concepts,
        }

        # 评分
        if hit_count >= 3 or len(new_concepts) >= 2:
            score = scores_cfg.get('tier_s', 20)
            rule = 'innovation_tier_s'
        elif hit_count >= 2 or len(new_concepts) >= 1:
            score = scores_cfg.get('tier_a', 15)
            rule = 'innovation_tier_a'
        elif hit_count >= 1:
            score = scores_cfg.get('tier_b', 10)
            rule = 'innovation_tier_b'
        else:
            score = scores_cfg.get('default', 3)
            rule = 'innovation_no_keywords'

        score = min(score, max_score)
        detail['reason'] = rule

        return score, rule, detail

    # ─────────────────────────────────────────────
    # 维度5: Timeliness (10分)
    # ─────────────────────────────────────────────

    def _compute_timeliness(self, enriched: EnrichedEvent) -> Tuple[float, str, Dict]:
        """
        时效性评分

        规则 (来自v1.1规格):
          - delay = T1_captured - T0_first_seen (秒)
          - <60s  → 10分
          - <300s → 7分
          - <1800s → 4分
          - else  → 1分
        """
        time_cfg = self.config['timeliness']
        thresholds = time_cfg['thresholds']
        scores_cfg = time_cfg['scores']

        # 计算延迟
        delay_seconds = (enriched.t1_captured - enriched.t0_first_seen).total_seconds()

        # 防御负值 (T1 < T0 异常)
        if delay_seconds < 0:
            delay_seconds = abs(delay_seconds)
            abnormal = True
        else:
            abnormal = False

        detail = {
            'delay_seconds': delay_seconds,
            't0': enriched.t0_first_seen.isoformat(),
            't1': enriched.t1_captured.isoformat(),
            'time_abnormal': abnormal,
        }

        # 分档
        tier_s = thresholds.get('tier_s', 60)
        tier_a = thresholds.get('tier_a', 300)
        tier_b = thresholds.get('tier_b', 1800)

        if delay_seconds < tier_s:
            score = scores_cfg.get('tier_s', 10)
            rule = 'timeliness_tier_s'
        elif delay_seconds < tier_a:
            score = scores_cfg.get('tier_a', 7)
            rule = 'timeliness_tier_a'
        elif delay_seconds < tier_b:
            score = scores_cfg.get('tier_b', 4)
            rule = 'timeliness_tier_b'
        else:
            score = scores_cfg.get('default', 1)
            rule = 'timeliness_slow'

        detail['reason'] = rule

        return score, rule, detail

    # ─────────────────────────────────────────────
    # Registry更新建议
    # ─────────────────────────────────────────────

    def _build_registry_updates(
        self,
        enriched: EnrichedEvent,
        components: Dict[str, float],
        narrative_detail: Dict
    ) -> Dict[str, Any]:
        """
        构建Registry更新建议

        冻结接口约束:
          - 模块只输出"建议写入"的数据
          - 实际写入由pipeline层统一执行
        """
        updates = {}

        # project-registry建档建议
        project_id = enriched.project_id or (
            enriched.detected_projects[0] if enriched.detected_projects else None
        )
        if project_id:
            updates['project-registry'] = {
                'project_id': project_id,
                'name': project_id,
                'has_token': bool(enriched.contract_address),
                'confirmed': enriched.official_confirmed,
                'updated_at': datetime.now(timezone.utc).isoformat(),
            }

        # concept-tracker标签建议
        tags = narrative_detail.get('narrative_tags', [])
        if tags:
            updates['concept-tracker'] = [
                {
                    'concept': tag,
                    'source': enriched.source_account,
                    'platform': self._extract_platform(enriched.source_type),
                }
                for tag in tags
            ]

        return updates if updates else None

    # ─────────────────────────────────────────────
    # 可观测性
    # ─────────────────────────────────────────────

    def _build_observability(
        self,
        enriched: EnrichedEvent,
        total_score: float,
        components: Dict[str, float],
        hit_rules: List[str],
        dimension_details: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """
        构建完整可观测性字段 (支持回放/审计)
        """
        obs = {
            # 基本信息
            'module': 'mod1',
            'module_name': 'launch_hunter',
            'event_id': enriched.event_id,
            'eval_timestamp': datetime.now(timezone.utc).isoformat(),
            'module_status': 'normal',

            # 输入摘要
            'input_summary': {
                'event_type': enriched.event_type,
                'source_type': enriched.source_type,
                'source_account': enriched.source_account,
                'project_id': enriched.project_id,
                'contract_address': enriched.contract_address or None,
                'detected_projects': enriched.detected_projects,
                'is_original': enriched.is_original,
                'risk_level': enriched.risk.risk_level if enriched.risk else 'unknown',
                'onchain_status': (enriched.onchain.status
                                   if enriched.onchain else 'no_onchain'),
                'cross_validated': enriched.cross_validated,
                'official_confirmed': enriched.official_confirmed,
            },

            # 评分过程
            'scoring_process': {
                'total_score': total_score,
                'components': components,
                'hit_rules': hit_rules,
                'dimension_details': dimension_details,
            },

            # 配置快照 (支持回放)
            'config_version': self.config.get('version', 'unknown'),
            'config_hash': self.config_hash,
            'config_snapshot': {
                'novelty': self.config['novelty'],
                'verifiability': self.config['verifiability'],
                'narrative': {
                    'max_score': self.config['narrative']['max_score'],
                    'timeout_seconds': self.config['narrative']['timeout_seconds'],
                },
                'innovation': self.config['innovation'],
                'timeliness': self.config['timeliness'],
            },
        }

        # 告警标记
        warnings = []
        onchain = enriched.onchain or OnchainData()
        if onchain.status in ('timeout', 'error'):
            warnings.append(f'onchain_api_{onchain.status}')
        if dimension_details.get('narrative', {}).get('sonnet_degraded'):
            warnings.append('sonnet_degraded')
        if not enriched.is_original:
            warnings.append('relay_not_original')
        if enriched.feishu_unconfirmed:
            warnings.append('feishu_unconfirmed')

        if warnings:
            obs['warnings'] = warnings

        return obs

    # ─────────────────────────────────────────────
    # 禁用/降级结果
    # ─────────────────────────────────────────────

    def _disabled_result(self, enriched: EnrichedEvent) -> ModuleOutput:
        """模块禁用时的返回"""
        return ModuleOutput(
            score=0,
            components={},
            hit_rules=[],
            observability={
                'module': 'mod1',
                'module_name': 'launch_hunter',
                'event_id': enriched.event_id,
                'eval_timestamp': datetime.now(timezone.utc).isoformat(),
                'module_status': 'disabled',
                'config_version': self.config.get('version', 'unknown'),
                'config_hash': self.config_hash,
            },
            registry_updates=None
        )

    # ─────────────────────────────────────────────
    # 工具函数
    # ─────────────────────────────────────────────

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        """夹紧值到范围"""
        return max(low, min(high, value))

    @staticmethod
    def _extract_platform(source_type: str) -> str:
        """从source_type提取平台"""
        if source_type.startswith('twitter'):
            return 'twitter'
        elif source_type.startswith('feishu'):
            return 'feishu'
        elif source_type.startswith('helius'):
            return 'onchain'
        return 'unknown'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 便捷函数 (对标 kol_evaluator 的 evaluate_kol)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evaluate_launch(
    enriched: EnrichedEvent,
    config_path: str,
    registry: Optional[RegistryReader] = None,
    sonnet_caller: Optional[SonnetCallable] = None
) -> ModuleOutput:
    """
    评估launch事件 (便捷函数)

    Args:
        enriched: 富化后的事件
        config_path: 配置文件路径
        registry: Registry只读接口
        sonnet_caller: Sonnet调用回调

    Returns:
        标准模块输出
    """
    hunter = LaunchHunter(
        config_path=config_path,
        registry=registry,
        sonnet_caller=sonnet_caller
    )
    return hunter.evaluate(enriched)
