from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple
import json

from scoring_protocol import finalize


@dataclass
class PushRecord:
    alert_id: str
    t0_first_seen: datetime
    t3_market_reaction: Optional[datetime]
    user_feedback: Optional[str]
    outcome_value: Optional[float]
    is_original: bool
    final_score: float
    created_at: datetime


@dataclass
class KOLEvaluationContext:
    kol_handle: str
    module: str
    source_type: str
    pushes: List[PushRecord]
    current_score: Optional[float] = None
    current_tier: Optional[str] = None


class KOLRadarModule:
    def __init__(self, config_path: str):
        self.config = json.load(open(config_path, "r", encoding="utf-8"))
        self._validate_config()

    def _validate_config(self) -> None:
        for k in ["lead_time", "hit_rate", "accuracy", "independence", "consistency", "tier_thresholds", "min_samples"]:
            if k not in self.config:
                raise ValueError(f"Missing required config: {k}")
        if not (self.config["comparison_semantics"]["consistency"] == "lt"):
            raise ValueError("comparison semantics mismatch")

    def evaluate(self, ctx: KOLEvaluationContext) -> Dict[str, Any]:
        if not self.config.get("enabled", True):
            return self._failure(ctx, "disabled", "module_disabled")
        if self.config.get("emergency_mode", {}).get("enabled", False):
            return self._failure(ctx, "emergency_mode", "emergency_mode")

        windowed, window_info = self._apply_time_window(ctx.pushes)
        cleaned, cleaning_info = self._clean_data(windowed)
        unique, dedup = self._deduplicate_and_sort(cleaned)

        if len(unique) < self.config["min_samples"]["for_evaluation"]:
            return self._failure(ctx, "insufficient_data", "insufficient_data", {
                "evaluation_window": window_info,
                "data_cleaning": cleaning_info,
                "deduplication": dedup,
            })

        breakdown: Dict[str, float] = {}
        rules: List[str] = []

        lt, ltr = self._compute_lead_time(unique)
        hr, hrr = self._compute_hit_rate(unique)
        acc, accr = self._compute_accuracy(unique)
        ind, indr = self._compute_independence(unique)
        cons, consr = self._compute_consistency(unique)

        breakdown.update({"lead_time": lt, "hit_rate": hr, "accuracy": acc, "independence": ind, "consistency": cons})
        rules.extend([ltr, hrr, accr, indr, consr])
        base = min(sum(breakdown.values()), 100)

        enriched = {
            "source_type": ctx.source_type,
            "official_confirmed": False,
            "cross_evidence": {"original_sources_count": 1, "platforms": ["twitter"], "cross_bonus_key": ""},
            "onchain": {"contract_verified": False},
            "is_original": True,
            "t1_captured": datetime.now(timezone.utc),
        }
        explain_extra = {
            "module_status": "normal",
            "kol_handle": ctx.kol_handle,
            "window_info": window_info,
            "cleaning_info": cleaning_info,
            "dedup_info": dedup,
            "tier_action": self._suggest_tier_action(base, len(unique), ctx.current_tier),
        }
        return finalize(
            module_name=self.config["module_name"],
            enriched=enriched,
            base_score=base,
            base_breakdown=breakdown,
            hit_rules=rules,
            config=self.config,
            explain_extra=explain_extra,
        ).to_dict()

    def discover_kol_multi_channel(
        self,
        seed_kols: List[str],
        following_map: Dict[str, List[Dict[str, Any]]],
        early_repliers: List[Dict[str, Any]],
        quoted_accounts: List[Dict[str, Any]],
        community_contributors: List[Dict[str, Any]],
        known_handles: Optional[set[str]] = None,
    ) -> List[Dict[str, Any]]:
        known = known_handles or set()
        candidates: List[Dict[str, Any]] = []

        # channel1
        counts: Dict[str, int] = {}
        followers: Dict[str, int] = {}
        for seed in seed_kols:
            for acc in following_map.get(seed, []):
                h = acc["handle"]
                counts[h] = counts.get(h, 0) + 1
                followers[h] = max(followers.get(h, 0), int(acc.get("followers_count", 0)))
        for h, c in counts.items():
            if c >= 3 and followers.get(h, 0) > 1000 and h not in known:
                candidates.append({"handle": h, "source": "cross_following", "confidence": "high", "cross_count": c})

        # channel2
        for r in early_repliers:
            h = r["handle"]
            if int(r.get("early_reply_count", 0)) >= 5 and h not in known:
                candidates.append({"handle": h, "source": "early_replier", "confidence": "medium", "reply_count": int(r.get("early_reply_count", 0))})

        # channel3
        for q in quoted_accounts:
            h = q["handle"]
            if int(q.get("quoter_count", 0)) >= 2 and h not in known:
                candidates.append({"handle": h, "source": "quoted_by_kol", "confidence": "high", "quoter_count": int(q.get("quoter_count", 0))})

        # channel4
        for c in community_contributors:
            h = c["handle"]
            if float(c.get("alpha_rate", 0)) > 0.3 and int(c.get("message_count", 0)) >= 10 and h not in known:
                candidates.append({"handle": h, "source": "community_active", "confidence": "medium", "alpha_rate": float(c.get("alpha_rate", 0))})

        uniq: Dict[str, Dict[str, Any]] = {}
        for c in candidates:
            if c["handle"] not in uniq:
                uniq[c["handle"]] = c
        return list(uniq.values())

    def trend_cluster_report_6h(
        self,
        concept_events: List[Dict[str, Any]],
        known_concepts: Optional[set[str]] = None,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        趋势聚类6h报告：48h内≥3独立来源→趋势信号→新概念→升温→推送报告
        concept_events item:
          {"concept": str, "source": str, "event_time": datetime}
        """
        now = now or datetime.now(timezone.utc)
        known_concepts = known_concepts or set()
        window_48h = now - timedelta(hours=48)
        week_start = now - timedelta(days=7)
        prev_week_start = now - timedelta(days=14)

        grouped: Dict[str, Dict[str, Any]] = {}
        for e in concept_events:
            concept = str(e.get("concept", "")).strip()
            source = str(e.get("source", "")).strip()
            event_time = e.get("event_time")
            if not concept or not source or not isinstance(event_time, datetime):
                continue
            if concept not in grouped:
                grouped[concept] = {"sources_48h": set(), "week_mentions": 0, "prev_week_mentions": 0}
            if event_time >= window_48h:
                grouped[concept]["sources_48h"].add(source)
            if event_time >= week_start:
                grouped[concept]["week_mentions"] += 1
            elif prev_week_start <= event_time < week_start:
                grouped[concept]["prev_week_mentions"] += 1

        reports: List[Dict[str, Any]] = []
        for concept, st in grouped.items():
            independent_sources = len(st["sources_48h"])
            trend_signal = independent_sources >= 3
            if not trend_signal:
                continue
            is_new_concept = concept not in known_concepts
            week_mentions = st["week_mentions"]
            prev_week_mentions = st["prev_week_mentions"]
            warming = week_mentions > (prev_week_mentions * 3 if prev_week_mentions > 0 else 0)
            reports.append({
                "concept": concept,
                "independent_sources_48h": independent_sources,
                "trend_signal": trend_signal,
                "new_concept": is_new_concept,
                "warming": warming,
                "push_report": True,
                "window_hours": 48,
            })

        return {
            "job": "trend_cluster_6h",
            "generated_at": now.isoformat(),
            "reports": reports,
            "push_reports": [r for r in reports if r["push_report"]],
        }

    def scan_kol_following(self, following_accounts: List[Dict[str, Any]], known_cross_count: Dict[str, int]) -> Dict[str, Any]:
        candidates: List[Dict[str, Any]] = []
        for acc in following_accounts:
            handle = acc["handle"]
            bio = str(acc.get("bio", "")).lower()
            followers = int(acc.get("followers", 0))
            relevance = 0
            reasons: List[str] = []
            if any(k in bio for k in ["builder", "research", "dev", "alpha"]):
                relevance += 30
                reasons.append("builder关键词")
            if followers > 10000:
                relevance += 20
                reasons.append("高粉丝数")
            cross = int(known_cross_count.get(handle, 0))
            if cross >= 2:
                relevance += 25
                reasons.append(f"被{cross}个已知KOL关注")
            if relevance >= 50:
                candidates.append({
                    "handle": handle,
                    "relevance_score": relevance,
                    "bio": acc.get("bio", ""),
                    "followers": followers,
                    "reason": " + ".join(reasons),
                })
        candidates.sort(key=lambda x: x["relevance_score"], reverse=True)
        return {"candidates": candidates[:50], "total_scanned": len(following_accounts), "total_matched": len(candidates)}

    def confirm_kol_candidates(self, confirmed_handles: List[str]) -> List[Dict[str, Any]]:
        return [{"handle": h, "tier": "C", "overall_score": 50, "discovered_from": "user_scan"} for h in confirmed_handles]

    def update_kol_registry(self, module_scores: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        updates = []
        source_weights: Dict[str, float] = {}
        for handle, m in module_scores.items():
            vals = [m.get(x, 0.0) for x in ["mod1", "mod2", "mod3", "mod4"]]
            overall = sum(vals) / 4.0
            tier = self._score_to_tier(overall)
            updates.append({"handle": handle, "overall": overall, "tier": tier, "scores": m})
        for u in updates:
            source_weights[f"twitter_kol_{u['tier'].lower()}"] = {"S": 0.95, "A": 0.7, "B": 0.5, "C": 0.3}[u["tier"]]
        return {"kol_registry_updates": updates, "source_weights_suggestion": source_weights}

    def _score_to_tier(self, score: float) -> str:
        th = self.config["tier_thresholds"]
        if score >= th["S"]:
            return "S"
        if score >= th["A"]:
            return "A"
        if score >= th["B"]:
            return "B"
        if score >= th["C"]:
            return "C"
        return "C"

    def _suggest_tier_action(self, score: float, sample_count: int, current_tier: Optional[str]) -> Dict[str, Any]:
        suggested = self._score_to_tier(score)
        if current_tier is None:
            return {"action": "maintain", "current": suggested}
        rank = {"S": 4, "A": 3, "B": 2, "C": 1}
        if rank[suggested] > rank[current_tier]:
            if sample_count >= self.config["min_samples"]["for_upgrade"]:
                return {"action": "upgrade", "from": current_tier, "to": suggested}
            return {"action": "hold", "current": current_tier}
        if rank[suggested] < rank[current_tier]:
            if sample_count >= self.config["min_samples"]["for_downgrade"]:
                return {"action": "downgrade", "from": current_tier, "to": suggested}
            return {"action": "hold", "current": current_tier}
        return {"action": "maintain", "current": current_tier}

    def _failure(self, ctx: KOLEvaluationContext, status: str, rule: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        enriched = {
            "source_type": ctx.source_type,
            "official_confirmed": False,
            "cross_evidence": {"original_sources_count": 0, "platforms": [], "cross_bonus_key": ""},
            "onchain": {"contract_verified": False},
            "is_original": True,
            "t1_captured": datetime.now(timezone.utc),
        }
        ex = {"module_status": status, "kol_handle": ctx.kol_handle}
        if extra:
            ex.update(extra)
        return finalize(
            module_name=self.config["module_name"],
            enriched=enriched,
            base_score=0,
            base_breakdown={},
            hit_rules=[rule],
            config=self.config,
            explain_extra=ex,
        ).to_dict()

    def _apply_time_window(self, pushes: List[PushRecord]) -> Tuple[List[PushRecord], Dict[str, Any]]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.config["evaluation_window_days"])
        windowed = [p for p in pushes if p.created_at >= cutoff]
        return windowed, {"cutoff": cutoff.isoformat(), "input": len(pushes), "output": len(windowed)}

    def _clean_data(self, pushes: List[PushRecord]) -> Tuple[List[PushRecord], Dict[str, Any]]:
        cfg = self.config["data_cleaning"]
        if not cfg.get("enabled", True):
            return list(pushes), {"applied": False, "issues_found": 0}
        out, issues = [], []
        for p in pushes:
            newp = p
            if p.t3_market_reaction and p.t3_market_reaction < p.t0_first_seen:
                if cfg["t3_before_t0_policy"] == "drop":
                    issues.append("t3_before_t0_drop")
                    continue
                newp = replace(newp, t3_market_reaction=None)
                issues.append("t3_before_t0_nullify")
            if newp.user_feedback and newp.user_feedback not in ["useful", "useless"]:
                if cfg["invalid_feedback_policy"] == "drop":
                    issues.append("invalid_feedback_drop")
                    continue
                newp = replace(newp, user_feedback=None)
                issues.append("invalid_feedback_nullify")
            if (newp.final_score < cfg["score_clamp_min"] or newp.final_score > cfg["score_clamp_max"]):
                if cfg["score_out_of_range_policy"] == "drop":
                    issues.append("score_drop")
                    continue
                newp = replace(newp, final_score=max(cfg["score_clamp_min"], min(cfg["score_clamp_max"], newp.final_score)))
                issues.append("score_clamp")
            out.append(newp)
        return out, {"applied": True, "issues_found": len(issues), "issues": issues[:10]}

    def _deduplicate_and_sort(self, pushes: List[PushRecord]) -> Tuple[List[PushRecord], Dict[str, Any]]:
        d: Dict[str, PushRecord] = {}
        for p in pushes:
            if p.alert_id not in d or p.created_at < d[p.alert_id].created_at:
                d[p.alert_id] = p
        uniq = sorted(d.values(), key=lambda x: x.created_at)
        return uniq, {"input": len(pushes), "output": len(uniq), "strategy": "keep_earliest_by_alert_id"}

    def _compute_lead_time(self, pushes: List[PushRecord]) -> Tuple[float, str]:
        cfg = self.config["lead_time"]
        valid = [p for p in pushes if p.t3_market_reaction]
        if not valid:
            return float(cfg["scores"]["default"]), "lead_time_all_t3_missing_default"
        avg = mean([(p.t3_market_reaction - p.t0_first_seen).total_seconds() for p in valid])
        th, sc = cfg["thresholds"], cfg["scores"]
        if avg <= th["tier_c"]:
            return float(sc["tier_s"]), "lead_time_tier_s"
        if avg <= th["tier_b"]:
            return float(sc["tier_a"]), "lead_time_tier_a"
        if avg <= th["tier_a"]:
            return float(sc["tier_b"]), "lead_time_tier_b"
        if avg <= th["tier_s"]:
            return float(sc["tier_c"]), "lead_time_tier_c"
        return float(sc["default"]), "lead_time_below_tier_c_default"

    def _compute_hit_rate(self, pushes: List[PushRecord]) -> Tuple[float, str]:
        cfg = self.config["hit_rate"]
        fb = [p for p in pushes if p.user_feedback is not None]
        if not fb:
            return float(cfg["scores"]["default"]), "hit_rate_no_feedback_default"
        rate = sum(1 for p in fb if p.user_feedback == "useful") / len(fb)
        th, sc = cfg["thresholds"], cfg["scores"]
        if rate > th["tier_s"]:
            return float(sc["tier_s"]), "hit_rate_tier_s"
        if rate > th["tier_a"]:
            return float(sc["tier_a"]), "hit_rate_tier_a"
        if rate > th["tier_b"]:
            return float(sc["tier_b"]), "hit_rate_tier_b"
        return float(sc["default"]), "hit_rate_below_tier_b_default"

    def _compute_accuracy(self, pushes: List[PushRecord]) -> Tuple[float, str]:
        cfg = self.config["accuracy"]
        if not pushes:
            return float(cfg["scores"]["default"]), "accuracy_all_outcome_missing_default"
        ratio = sum(1 for p in pushes if p.outcome_value is not None) / len(pushes)
        if ratio == 0:
            return float(cfg["scores"]["default"]), "accuracy_all_outcome_missing_default"
        th, sc = cfg["thresholds"], cfg["scores"]
        if ratio > th["tier_s"]:
            return float(sc["tier_s"]), "accuracy_tier_s"
        if ratio > th["tier_a"]:
            return float(sc["tier_a"]), "accuracy_tier_a"
        if ratio > th["tier_b"]:
            return float(sc["tier_b"]), "accuracy_tier_b"
        return float(sc["default"]), "accuracy_below_tier_b_default"

    def _compute_independence(self, pushes: List[PushRecord]) -> Tuple[float, str]:
        cfg = self.config["independence"]
        if not pushes:
            return float(cfg["scores"]["default"]), "independence_no_samples_default"
        ratio = sum(1 for p in pushes if p.is_original) / len(pushes)
        th, sc = cfg["thresholds"], cfg["scores"]
        if ratio > th["tier_s"]:
            return float(sc["tier_s"]), "independence_tier_s"
        if ratio > th["tier_a"]:
            return float(sc["tier_a"]), "independence_tier_a"
        if ratio > th["tier_b"]:
            return float(sc["tier_b"]), "independence_tier_b"
        return float(sc["default"]), "independence_below_tier_b_default"

    def _compute_consistency(self, pushes: List[PushRecord]) -> Tuple[float, str]:
        cfg = self.config["consistency"]
        if len(pushes) < 2:
            return float(cfg["scores"]["default"]), "consistency_insufficient_samples_default"
        std = stdev([p.final_score for p in pushes])
        th, sc = cfg["thresholds"], cfg["scores"]
        if std < th["tier_s"]:
            return float(sc["tier_s"]), "consistency_tier_s"
        if std < th["tier_a"]:
            return float(sc["tier_a"]), "consistency_tier_a"
        if std < th["tier_b"]:
            return float(sc["tier_b"]), "consistency_tier_b"
        return float(sc["default"]), "consistency_above_tier_b_default"
