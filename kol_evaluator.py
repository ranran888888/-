"""
KOL评估器 - mod5
版本: 1.0-fixed
功能: 评估KOL在各模块的表现，输出分级建议
"""

from dataclasses import dataclass, replace
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timedelta, timezone
from statistics import mean, stdev
import yaml
import hashlib


@dataclass
class PushRecord:
    """单条推送记录"""
    alert_id: str
    t0_first_seen: datetime
    t3_market_reaction: Optional[datetime]
    user_feedback: Optional[str]  # useful/useless/null
    outcome_value: Optional[float]
    is_original: bool
    final_score: float
    created_at: datetime


@dataclass
class KOLEvaluationContext:
    """KOL评估输入上下文"""
    kol_handle: str
    module: str
    pushes: List[PushRecord]
    current_score: Optional[float] = None
    current_tier: Optional[str] = None


@dataclass
class KOLEvaluationResult:
    """KOL评估输出结果"""
    module: str
    score: float
    components: Dict[str, int]
    stats: Dict[str, Any]
    hit_rules: List[str]
    observability: Dict[str, Any]
    suggested_actions: List[Dict[str, str]]


class KOLEvaluator:
    """KOL评估器"""

    def __init__(self, config_path: str):
        """
        初始化评估器

        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
            self.config = yaml.safe_load(config_content)

        # 【修复点1】校验配置类型
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a YAML mapping (dict), got None or invalid type")

        # 必填键校验
        required_keys = [
            'version', 'comparison_semantics', 'data_cleaning',
            'lead_time', 'hit_rate', 'accuracy', 'independence', 'consistency',
            'tier_thresholds', 'min_samples', 'evaluation_window_days'
        ]

        missing_keys = [k for k in required_keys if k not in self.config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

        # 校验嵌套必填键
        for dimension in ['lead_time', 'hit_rate', 'accuracy', 'independence', 'consistency']:
            if 'thresholds' not in self.config[dimension]:
                raise ValueError(f"Missing {dimension}.thresholds")
            if 'scores' not in self.config[dimension]:
                raise ValueError(f"Missing {dimension}.scores")

        # 【修复点2】类型和值域校验
        # 校验evaluation_window_days
        window_days = self.config['evaluation_window_days']
        if not isinstance(window_days, int) or window_days <= 0:
            raise ValueError(f"evaluation_window_days must be a positive integer, got {window_days}")

        # 校验tier_thresholds单调性
        tier_thresholds = self.config['tier_thresholds']
        if not (tier_thresholds['S'] > tier_thresholds['A'] > tier_thresholds['B'] > tier_thresholds['C'] >= 0):
            raise ValueError(f"tier_thresholds must be monotonic: S > A > B > C >= 0, got {tier_thresholds}")

        # 校验min_samples
        min_samples = self.config['min_samples']
        for key in ['for_evaluation', 'for_upgrade', 'for_downgrade']:
            if key not in min_samples:
                raise ValueError(f"Missing min_samples.{key}")
            if not isinstance(min_samples[key], int) or min_samples[key] < 0:
                raise ValueError(f"min_samples.{key} must be a non-negative integer, got {min_samples[key]}")

        # 【修复点3】策略枚举校验
        allowed_strategies = {
            't3_before_t0_policy': {'drop', 'nullify'},
            'invalid_feedback_policy': {'drop', 'nullify'},
            'score_out_of_range_policy': {'drop', 'clamp'}
        }
        data_cleaning = self.config['data_cleaning']
        for key, valid_values in allowed_strategies.items():
            actual = data_cleaning.get(key)
            if actual not in valid_values:
                raise ValueError(
                    f"Invalid data_cleaning.{key}='{actual}', expected one of {sorted(valid_values)}"
                )

        # 比较语义一致性校验
        expected_semantics = {
            'lead_time': 'gt',
            'hit_rate': 'gt',
            'accuracy': 'gt',
            'independence': 'gt',
            'consistency': 'lt'
        }

        actual_semantics = self.config['comparison_semantics']

        for dim, expected in expected_semantics.items():
            actual = actual_semantics.get(dim)
            if actual != expected:
                raise ValueError(
                    f"Comparison semantics mismatch for {dim}: "
                    f"config has '{actual}', but code implements '{expected}'. "
                    "To change comparison logic, modify BOTH config and code."
                )

        # 计算配置hash（用于回放）
        self.config_hash = hashlib.md5(config_content.encode()).hexdigest()
        self.config_path = config_path

        # 禁用检查
        self.is_enabled = self.config.get('enabled', True)

    def evaluate(self, ctx: KOLEvaluationContext) -> KOLEvaluationResult:
        """
        评估KOL

        Args:
            ctx: 评估上下文

        Returns:
            评估结果
        """
        # 禁用时返回hold + module_status
        if not self.is_enabled:
            return self._disabled_result(ctx)

        # 紧急模式返回hold + module_status
        if self.config.get('emergency_mode', {}).get('enabled', False):
            return self._emergency_mode_result(ctx)

        # 验证输入
        self._validate_input(ctx)

        # 应用时间窗口过滤
        windowed_pushes, window_info = self._apply_time_window(ctx.pushes)

        # 数据清洗：移除脏数据（无副作用）
        cleaned_pushes, cleaning_info = self._clean_data(windowed_pushes)

        # 去重并排序
        unique_pushes, dedup_info = self._deduplicate_and_sort(cleaned_pushes)

        # 样本数检查
        if len(unique_pushes) < self.config['min_samples']['for_evaluation']:
            return self._insufficient_data_result(
                ctx,
                len(ctx.pushes),
                len(windowed_pushes),
                len(cleaned_pushes),
                len(unique_pushes),
                window_info,
                cleaning_info,
                dedup_info
            )

        # 计算各维度分数
        components = {}
        stats = {}
        hit_rules = []

        # Lead Time
        lt_score, lt_stats, lt_rule = self._compute_lead_time(unique_pushes)
        components['lead_time'] = lt_score
        stats.update(lt_stats)
        hit_rules.append(lt_rule)

        # Hit Rate
        hr_score, hr_stats, hr_rule = self._compute_hit_rate(unique_pushes)
        components['hit_rate'] = hr_score
        stats.update(hr_stats)
        hit_rules.append(hr_rule)

        # Accuracy
        acc_score, acc_stats, acc_rule = self._compute_accuracy(unique_pushes)
        components['accuracy'] = acc_score
        stats.update(acc_stats)
        hit_rules.append(acc_rule)

        # Independence
        ind_score, ind_stats, ind_rule = self._compute_independence(unique_pushes)
        components['independence'] = ind_score
        stats.update(ind_stats)
        hit_rules.append(ind_rule)

        # Consistency
        cons_score, cons_stats, cons_rule = self._compute_consistency(unique_pushes)
        components['consistency'] = cons_score
        stats.update(cons_stats)
        hit_rules.append(cons_rule)

        # 总分
        total_score = sum(components.values())

        # 建议动作
        suggested_actions = self._suggest_actions(
            total_score,
            len(unique_pushes),
            ctx.current_tier
        )

        # 可观测性（含完整回放信息）
        observability = self._build_observability(
            ctx,
            len(ctx.pushes),
            len(windowed_pushes),
            len(cleaned_pushes),
            len(unique_pushes),
            window_info,
            cleaning_info,
            dedup_info,
            stats,
            total_score,
            components,
            hit_rules
        )

        return KOLEvaluationResult(
            module=ctx.module,
            score=total_score,
            components=components,
            stats=stats,
            hit_rules=hit_rules,
            observability=observability,
            suggested_actions=suggested_actions
        )

    def _validate_input(self, ctx: KOLEvaluationContext) -> None:
        """验证输入"""
        if not ctx.kol_handle:
            raise ValueError("kol_handle is required")

        if ctx.module not in ['mod1', 'mod2', 'mod3', 'mod4']:
            raise ValueError(f"module must be one of [mod1, mod2, mod3, mod4], got {ctx.module}")

        if not isinstance(ctx.pushes, list):
            raise ValueError("pushes must be a list")

    def _apply_time_window(self, pushes: List[PushRecord]) -> Tuple[List[PushRecord], Dict]:
        """
        应用时间窗口过滤

        Returns:
            (窗口内记录, 窗口信息)
        """
        window_days = self.config['evaluation_window_days']
        # 【修复点6】使用UTC时区
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=window_days)

        windowed = [p for p in pushes if p.created_at >= cutoff_time]

        window_info = {
            "window_days": window_days,
            "cutoff_time": cutoff_time.isoformat(),
            "original_count": len(pushes),
            "windowed_count": len(windowed),
            "filtered_out_count": len(pushes) - len(windowed)
        }

        return windowed, window_info

    def _clean_data(self, pushes: List[PushRecord]) -> Tuple[List[PushRecord], Dict]:
        """
        数据清洗：移除脏数据（无副作用 + 策略配置化）

        Returns:
            (清洗后记录, 清洗信息)
        """
        cfg = self.config['data_cleaning']

        if not cfg['enabled']:
            # 【修复点4】返回浅拷贝
            return list(pushes), {
                "applied": False,
                "original_count": len(pushes),
                "cleaned_count": len(pushes),
                "issues_found": 0,
                "issues": [],
                "reason": "cleaning_disabled"
            }

        if not pushes:
            return [], {
                "applied": False,
                "original_count": 0,
                "cleaned_count": 0,
                "issues_found": 0,
                "issues": []
            }

        cleaned: List[PushRecord] = []
        issues = []

        clamp_min = cfg.get('score_clamp_min', 0)
        clamp_max = cfg.get('score_clamp_max', 100)

        for i, push in enumerate(pushes):
            # 只引用，不修改原对象
            new_push = push
            dropped = False

            # 检查1: T3早于T0（时间倒流）
            if new_push.t3_market_reaction and new_push.t3_market_reaction < new_push.t0_first_seen:
                policy = cfg.get('t3_before_t0_policy', 'nullify')

                if policy == 'drop':
                    issues.append({
                        "index": i,
                        "alert_id": push.alert_id,
                        "issue": "t3_before_t0",
                        "action": "dropped",
                        "t0": push.t0_first_seen.isoformat(),
                        "t3": push.t3_market_reaction.isoformat()
                    })
                    dropped = True
                else:  # nullify
                    issues.append({
                        "index": i,
                        "alert_id": push.alert_id,
                        "issue": "t3_before_t0",
                        "action": "nullified",
                        "t0": push.t0_first_seen.isoformat(),
                        "t3": push.t3_market_reaction.isoformat()
                    })
                    new_push = replace(new_push, t3_market_reaction=None)

            if dropped:
                continue

            # 检查2: user_feedback非法值
            if new_push.user_feedback and new_push.user_feedback not in ['useful', 'useless']:
                policy = cfg.get('invalid_feedback_policy', 'nullify')

                if policy == 'drop':
                    issues.append({
                        "index": i,
                        "alert_id": push.alert_id,
                        "issue": "invalid_feedback",
                        "action": "dropped",
                        "value": new_push.user_feedback
                    })
                    continue
                else:  # nullify
                    issues.append({
                        "index": i,
                        "alert_id": push.alert_id,
                        "issue": "invalid_feedback",
                        "action": "nullified",
                        "value": new_push.user_feedback
                    })
                    new_push = replace(new_push, user_feedback=None)

            # 检查3: final_score超范围
            if new_push.final_score < clamp_min or new_push.final_score > clamp_max:
                policy = cfg.get('score_out_of_range_policy', 'clamp')

                if policy == 'drop':
                    issues.append({
                        "index": i,
                        "alert_id": push.alert_id,
                        "issue": "score_out_of_range",
                        "action": "dropped",
                        "value": new_push.final_score
                    })
                    continue
                else:  # clamp
                    clamped = max(clamp_min, min(clamp_max, new_push.final_score))
                    issues.append({
                        "index": i,
                        "alert_id": push.alert_id,
                        "issue": "score_out_of_range",
                        "action": "clamped",
                        "original_value": new_push.final_score,
                        "clamped_value": clamped
                    })
                    new_push = replace(new_push, final_score=clamped)

            cleaned.append(new_push)

        return cleaned, {
            "applied": len(issues) > 0,
            "original_count": len(pushes),
            "cleaned_count": len(cleaned),
            "issues_found": len(issues),
            "issues": issues[:10]
        }

    def _deduplicate_and_sort(self, pushes: List[PushRecord]) -> Tuple[List[PushRecord], Dict]:
        """
        按alert_id去重并排序

        Returns:
            (去重并排序后列表, 去重信息)
        """
        if not pushes:
            return [], {"applied": False, "original_count": 0, "unique_count": 0}

        seen = {}
        for push in pushes:
            if push.alert_id not in seen:
                seen[push.alert_id] = push
            else:
                # 保留created_at更早的
                if push.created_at < seen[push.alert_id].created_at:
                    seen[push.alert_id] = push

        # 按created_at排序
        unique = sorted(seen.values(), key=lambda x: x.created_at)

        return unique, {
            "applied": len(unique) < len(pushes),
            "original_count": len(pushes),
            "unique_count": len(unique),
            "strategy": "keep_earliest_by_alert_id",
            "sort_key": "created_at_asc"
        }

    def _compute_lead_time(self, pushes: List[PushRecord]) -> Tuple[int, Dict, str]:
        """
        计算Lead Time分数

        Returns:
            (分数, 统计信息, 命中规则)
        """
        config = self.config['lead_time']

        # 过滤有T3的记录
        valid_records = [p for p in pushes if p.t3_market_reaction is not None]

        stats = {
            'lead_time_samples': len(valid_records),
            'lead_time_missing_t3': len(pushes) - len(valid_records)
        }

        # 全部缺失T3时返回明确规则
        if not valid_records:
            stats['lead_time_avg_seconds'] = None
            return config['scores']['default'], stats, 'lead_time_all_t3_missing_default'

        # 计算平均延迟
        lead_times = [(p.t3_market_reaction - p.t0_first_seen).total_seconds()
                      for p in valid_records]
        avg_lead_time = mean(lead_times)
        stats['lead_time_avg_seconds'] = avg_lead_time

        # 分档（严格大于）
        thresholds = config['thresholds']
        scores = config['scores']

        if avg_lead_time > thresholds['tier_s']:
            return scores['tier_s'], stats, 'lead_time_tier_s'
        elif avg_lead_time > thresholds['tier_a']:
            return scores['tier_a'], stats, 'lead_time_tier_a'
        elif avg_lead_time > thresholds['tier_b']:
            return scores['tier_b'], stats, 'lead_time_tier_b'
        elif avg_lead_time > thresholds['tier_c']:
            return scores['tier_c'], stats, 'lead_time_tier_c'
        else:
            return scores['default'], stats, 'lead_time_below_tier_c_default'

    def _compute_hit_rate(self, pushes: List[PushRecord]) -> Tuple[int, Dict, str]:
        """计算Hit Rate分数"""
        config = self.config['hit_rate']

        # 统计反馈
        feedback_records = [p for p in pushes if p.user_feedback is not None]
        useful_count = sum(1 for p in feedback_records if p.user_feedback == 'useful')

        stats = {
            'hit_rate_useful': useful_count,
            'hit_rate_total_feedback': len(feedback_records)
        }

        # 无反馈时返回明确规则
        if not feedback_records:
            stats['hit_rate_ratio'] = 0.0
            return config['scores']['default'], stats, 'hit_rate_no_feedback_default'

        hit_rate = useful_count / len(feedback_records)
        stats['hit_rate_ratio'] = hit_rate

        # 分档（严格大于）
        thresholds = config['thresholds']
        scores = config['scores']

        if hit_rate > thresholds['tier_s']:
            return scores['tier_s'], stats, 'hit_rate_tier_s'
        elif hit_rate > thresholds['tier_a']:
            return scores['tier_a'], stats, 'hit_rate_tier_a'
        elif hit_rate > thresholds['tier_b']:
            return scores['tier_b'], stats, 'hit_rate_tier_b'
        else:
            return scores['default'], stats, 'hit_rate_below_tier_b_default'

    def _compute_accuracy(self, pushes: List[PushRecord]) -> Tuple[int, Dict, str]:
        """计算Accuracy分数"""
        config = self.config['accuracy']

        # 统计有outcome的记录
        has_outcome = sum(1 for p in pushes if p.outcome_value is not None)

        stats = {
            'accuracy_has_outcome': has_outcome,
            'accuracy_total': len(pushes)
        }

        # 无outcome时返回明确规则
        if not pushes or has_outcome == 0:
            stats['accuracy_ratio'] = 0.0
            return config['scores']['default'], stats, 'accuracy_all_outcome_missing_default'

        accuracy = has_outcome / len(pushes)
        stats['accuracy_ratio'] = accuracy

        # 分档（严格大于）
        thresholds = config['thresholds']
        scores = config['scores']

        if accuracy > thresholds['tier_s']:
            return scores['tier_s'], stats, 'accuracy_tier_s'
        elif accuracy > thresholds['tier_a']:
            return scores['tier_a'], stats, 'accuracy_tier_a'
        elif accuracy > thresholds['tier_b']:
            return scores['tier_b'], stats, 'accuracy_tier_b'
        else:
            return scores['default'], stats, 'accuracy_below_tier_b_default'

    def _compute_independence(self, pushes: List[PushRecord]) -> Tuple[int, Dict, str]:
        """计算Independence分数"""
        config = self.config['independence']

        # 统计原创记录
        original_count = sum(1 for p in pushes if p.is_original)

        stats = {
            'independence_original': original_count,
            'independence_total': len(pushes)
        }

        if not pushes:
            stats['independence_ratio'] = 0.0
            return config['scores']['default'], stats, 'independence_no_samples_default'

        independence = original_count / len(pushes)
        stats['independence_ratio'] = independence

        # 分档（严格大于）
        thresholds = config['thresholds']
        scores = config['scores']

        if independence > thresholds['tier_s']:
            return scores['tier_s'], stats, 'independence_tier_s'
        elif independence > thresholds['tier_a']:
            return scores['tier_a'], stats, 'independence_tier_a'
        elif independence > thresholds['tier_b']:
            return scores['tier_b'], stats, 'independence_tier_b'
        else:
            return scores['default'], stats, 'independence_below_tier_b_default'

    def _compute_consistency(self, pushes: List[PushRecord]) -> Tuple[int, Dict, str]:
        """计算Consistency分数"""
        config = self.config['consistency']

        # 样本不足时返回明确规则
        if len(pushes) < 2:
            stats = {'consistency_stddev': 0.0}
            return config['scores']['default'], stats, 'consistency_insufficient_samples_default'

        # 计算分数标准差
        scores = [p.final_score for p in pushes]
        std = stdev(scores)

        stats = {'consistency_stddev': std}

        # 分档（严格小于）
        thresholds = config['thresholds']
        score_config = config['scores']

        if std < thresholds['tier_s']:
            return score_config['tier_s'], stats, 'consistency_tier_s'
        elif std < thresholds['tier_a']:
            return score_config['tier_a'], stats, 'consistency_tier_a'
        elif std < thresholds['tier_b']:
            return score_config['tier_b'], stats, 'consistency_tier_b'
        else:
            return score_config['default'], stats, 'consistency_above_tier_b_default'

    def _suggest_actions(
        self,
        total_score: float,
        sample_count: int,
        current_tier: Optional[str]
    ) -> List[Dict[str, str]]:
        """生成建议动作"""
        tier_thresholds = self.config['tier_thresholds']
        min_samples = self.config['min_samples']

        # 判定应有tier
        # 严格对齐v1.1：分级使用严格大于（>）
        if total_score > tier_thresholds['S']:
            suggested_tier = 'S'
        elif total_score > tier_thresholds['A']:
            suggested_tier = 'A'
        elif total_score > tier_thresholds['B']:
            suggested_tier = 'B'
        else:
            suggested_tier = 'C'

        # 无当前tier时返回maintain
        if current_tier is None:
            return [{
                'action': 'maintain',
                'current': suggested_tier,
                'reason': f'new_kol: initial tier suggestion {suggested_tier}, score {total_score}, sample_count {sample_count}'
            }]

        # 升级
        if self._tier_rank(suggested_tier) > self._tier_rank(current_tier):
            if sample_count >= min_samples['for_upgrade']:
                return [{
                    'action': 'upgrade',
                    'from': current_tier,
                    'to': suggested_tier,
                    'reason': f'score {total_score} > tier_{suggested_tier.lower()}_threshold {tier_thresholds[suggested_tier]}, sample_count {sample_count} >= min_upgrade {min_samples["for_upgrade"]}'
                }]
            else:
                return [{
                    'action': 'hold',
                    'current': current_tier,
                    'reason': f'score {total_score} qualifies for {suggested_tier}, but sample_count {sample_count} < min_upgrade {min_samples["for_upgrade"]}'
                }]

        # 降级
        elif self._tier_rank(suggested_tier) < self._tier_rank(current_tier):
            if sample_count >= min_samples['for_downgrade']:
                return [{
                    'action': 'downgrade',
                    'from': current_tier,
                    'to': suggested_tier,
                    'reason': f'score {total_score} < tier_{current_tier.lower()}_threshold {tier_thresholds[current_tier]}, sample_count {sample_count} >= min_downgrade {min_samples["for_downgrade"]}'
                }]
            else:
                return [{
                    'action': 'hold',
                    'current': current_tier,
                    'reason': f'score {total_score} suggests downgrade to {suggested_tier}, but sample_count {sample_count} < min_downgrade {min_samples["for_downgrade"]}'
                }]

        # 维持
        else:
            tier_name = current_tier.lower()
            return [{
                'action': 'maintain',
                'current': current_tier,
                'reason': f'score {total_score} within tier_{tier_name} range, sample_count {sample_count} >= min_evaluation {min_samples["for_evaluation"]}'
            }]

    def _tier_rank(self, tier: str) -> int:
        """Tier排名（S最高）"""
        ranks = {'S': 4, 'A': 3, 'B': 2, 'C': 1}
        return ranks.get(tier, 0)

    def _build_config_snapshot(self) -> Dict[str, Any]:
        """
        【修复点5辅助】构建配置快照（复用代码）
        """
        return {
            'tier_thresholds': self.config['tier_thresholds'],
            'min_samples': self.config['min_samples'],
            'evaluation_window_days': self.config['evaluation_window_days'],
            'comparison_semantics': self.config['comparison_semantics'],
            'data_cleaning': self.config['data_cleaning'],
            'lead_time': self.config['lead_time'],
            'hit_rate': self.config['hit_rate'],
            'accuracy': self.config['accuracy'],
            'independence': self.config['independence'],
            'consistency': self.config['consistency']
        }

    def _disabled_result(self, ctx: KOLEvaluationContext) -> KOLEvaluationResult:
        """模块禁用结果 - 统一使用hold"""
        return KOLEvaluationResult(
            module=ctx.module,
            score=0,
            components={},
            stats={},
            hit_rules=[],
            observability={
                'kol_handle': ctx.kol_handle,
                'module': ctx.module,
                'eval_timestamp': datetime.now(timezone.utc).isoformat(),
                'module_status': 'disabled',
                'config_version': self.config.get('version', 'unknown'),
                # 【修复点5】补齐回放字段
                'config_hash': self.config_hash,
                'config_snapshot': self._build_config_snapshot()
            },
            suggested_actions=[{
                'action': 'hold',
                'current': ctx.current_tier,
                'reason': 'module disabled by config (enabled=false)'
            }]
        )

    def _emergency_mode_result(self, ctx: KOLEvaluationContext) -> KOLEvaluationResult:
        """紧急模式结果 - 统一使用hold"""
        fallback_strategy = self.config.get('emergency_mode', {}).get('fallback_strategy', 'use_current_tiers')

        return KOLEvaluationResult(
            module=ctx.module,
            score=ctx.current_score or 0,
            components={},
            stats={},
            hit_rules=[],
            observability={
                'kol_handle': ctx.kol_handle,
                'module': ctx.module,
                'eval_timestamp': datetime.now(timezone.utc).isoformat(),
                'module_status': 'emergency_mode',
                'emergency_strategy': fallback_strategy,
                'config_version': self.config.get('version', 'unknown'),
                # 【修复点5】补齐回放字段
                'config_hash': self.config_hash,
                'config_snapshot': self._build_config_snapshot()
            },
            suggested_actions=[{
                'action': 'hold',
                'current': ctx.current_tier,
                'reason': f'emergency mode enabled, strategy={fallback_strategy}'
            }]
        )

    def _insufficient_data_result(
        self,
        ctx: KOLEvaluationContext,
        original_count: int,
        windowed_count: int,
        cleaned_count: int,
        unique_count: int,
        window_info: Dict,
        cleaning_info: Dict,
        dedup_info: Dict
    ) -> KOLEvaluationResult:
        """样本不足结果"""
        return KOLEvaluationResult(
            module=ctx.module,
            score=0,
            components={},
            stats={},
            hit_rules=[],
            observability={
                'kol_handle': ctx.kol_handle,
                'module': ctx.module,
                'eval_timestamp': datetime.now(timezone.utc).isoformat(),
                'sample_count': original_count,
                'sample_count_windowed': windowed_count,
                'sample_count_cleaned': cleaned_count,
                'sample_count_after_dedup': unique_count,
                'warning': 'insufficient_samples',
                'evaluation_window': window_info,
                'data_cleaning': cleaning_info,
                'deduplication': dedup_info,
                'config_version': self.config.get('version', 'unknown'),
                'config_hash': self.config_hash,
                # 【修复点5】补齐回放字段
                'config_snapshot': self._build_config_snapshot()
            },
            suggested_actions=[{
                'action': 'insufficient_data',
                'reason': f'sample_count {unique_count} < min_evaluation {self.config["min_samples"]["for_evaluation"]}'
            }]
        )

    def _build_observability(
        self,
        ctx: KOLEvaluationContext,
        original_count: int,
        windowed_count: int,
        cleaned_count: int,
        unique_count: int,
        window_info: Dict,
        cleaning_info: Dict,
        dedup_info: Dict,
        stats: Dict,
        total_score: float,
        components: Dict[str, int],
        hit_rules: List[str]
    ) -> Dict[str, Any]:
        """
        构建完整可观测性字段（支持完全回放）
        """
        obs = {
            # 基本信息
            'kol_handle': ctx.kol_handle,
            'module': ctx.module,
            'eval_timestamp': datetime.now(timezone.utc).isoformat(),
            'module_status': 'normal',

            # 样本统计
            'sample_count': original_count,
            'sample_count_windowed': windowed_count,
            'sample_count_cleaned': cleaned_count,
            'sample_count_after_dedup': unique_count,

            # 时间窗口信息
            'evaluation_window': window_info,

            # 数据清洗信息
            'data_cleaning': cleaning_info,

            # 去重信息
            'deduplication': dedup_info,

            # 配置版本与快照
            'config_version': self.config['version'],
            'config_hash': self.config_hash,
            'config_snapshot': self._build_config_snapshot(),

            # 完整评分过程
            'scoring_process': {
                'total_score': total_score,
                'components': components,
                'hit_rules': hit_rules,
                'component_stats': stats
            }
        }

        # 检查warnings
        warnings = []
        if stats.get('lead_time_missing_t3', 0) == unique_count and unique_count > 0:
            warnings.append('all_t3_missing')
        if stats.get('accuracy_has_outcome', 0) == 0 and unique_count > 0:
            warnings.append('all_outcome_missing')
        if cleaning_info.get('issues_found', 0) > 0:
            warnings.append('data_quality_issues')

        if warnings:
            obs['warnings'] = warnings

        return obs


def evaluate_kol(
    kol_handle: str,
    module: str,
    pushes: List[PushRecord],
    config_path: str,
    current_score: Optional[float] = None,
    current_tier: Optional[str] = None
) -> KOLEvaluationResult:
    """
    评估KOL（便捷函数）

    Args:
        kol_handle: KOL handle
        module: 模块名 (mod1/mod2/mod3/mod4)
        pushes: 推送记录列表
        config_path: 配置文件路径
        current_score: 当前分数（可选）
        current_tier: 当前分级（可选）

    Returns:
        评估结果
    """
    evaluator = KOLEvaluator(config_path)

    ctx = KOLEvaluationContext(
        kol_handle=kol_handle,
        module=module,
        pushes=pushes,
        current_score=current_score,
        current_tier=current_tier
    )

    return evaluator.evaluate(ctx)
