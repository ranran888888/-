from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List
import hashlib
import json


SOURCE_WEIGHTS = {
    "twitter_builder": 1.0,
    "twitter_project_official": 1.0,
    "twitter_kol_s": 0.95,
    "twitter_kol_a": 0.7,
    "twitter_kol_b": 0.5,
    "twitter_kol_c": 0.3,
    "community_elite": 0.8,
    "community_blogger": 0.5,
    "community_monitor": 0.25,
    "keyword_search": 0.3,
    "onchain": 0.9,
}

CROSS_BONUS = {
    "cross_twitter_community": 1.3,
    "cross_twitter_onchain": 1.4,
    "cross_all_three": 1.5,
}
FINAL_CROSS_CAP = 1.2

DECAY_RULES = {
    "first_mention": 1.0,
    "within_6h": 0.5,
    "within_24h": 0.3,
    "within_72h": 0.15,
}


@dataclass
class UnifiedScoreOutput:
    module_name: str
    final_score: float
    confidence_score: float
    factors: Dict[str, float]
    base_breakdown: Dict[str, float]
    hit_rules: List[str]
    explain: Dict[str, Any]
    observability: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_name": self.module_name,
            "final_score": self.final_score,
            "confidence_score": self.confidence_score,
            "factors": self.factors,
            "base_breakdown": self.base_breakdown,
            "hit_rules": self.hit_rules,
            "explain": self.explain,
            "observability": self.observability,
        }


def _hash_config(cfg: Dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(payload.encode()).hexdigest()


def calc_confidence(enriched: Dict[str, Any]) -> tuple[float, List[str], Dict[str, int]]:
    score = 0
    missing: List[str] = []
    comp = {
        "source_grade": 0,
        "onchain_verify": 0,
        "multi_source": 0,
        "cross_platform": 0,
        "official_confirm": 0,
    }

    src = enriched.get("source_type", "")
    if src in ["twitter_builder", "twitter_project_official", "onchain"]:
        comp["source_grade"] = 30
    elif src in ["twitter_kol_s", "community_elite"]:
        comp["source_grade"] = 25
    elif src in ["twitter_kol_a"]:
        comp["source_grade"] = 20
    elif src in ["twitter_kol_b", "community_blogger"]:
        comp["source_grade"] = 15
    else:
        comp["source_grade"] = 10

    onchain = enriched.get("onchain", {}) or {}
    if onchain.get("contract_verified"):
        comp["onchain_verify"] = 25
    else:
        missing.append("onchain_verify")

    cross_e = enriched.get("cross_evidence", {}) or {}
    if cross_e.get("original_sources_count", 0) >= 2:
        comp["multi_source"] = 25
    else:
        missing.append("multi_source")

    platforms = set(cross_e.get("platforms") or [])
    if len(platforms) >= 2:
        comp["cross_platform"] = 10
    else:
        missing.append("cross_platform")

    if enriched.get("official_confirmed"):
        comp["official_confirm"] = 10
    else:
        missing.append("official_confirm")

    score = min(sum(comp.values()), 100)
    return float(score), missing, comp


def apply_cross_bonus(confidence: float, enriched: Dict[str, Any]) -> tuple[float, float, bool]:
    cross_e = enriched.get("cross_evidence", {}) or {}
    key = cross_e.get("cross_bonus_key", "")
    raw_mult = CROSS_BONUS.get(key, 1.0)
    conf_boosted = min(confidence * raw_mult, 100)
    final_mult = min(raw_mult, FINAL_CROSS_CAP)
    return conf_boosted, final_mult, raw_mult > FINAL_CROSS_CAP


def compute_decay(event_time: datetime, now: datetime | None = None) -> tuple[float, str]:
    now = now or datetime.now(timezone.utc)
    delta_h = (now - event_time).total_seconds() / 3600
    if delta_h <= 6:
        return DECAY_RULES["within_6h"], "within_6h"
    if delta_h <= 24:
        return DECAY_RULES["within_24h"], "within_24h"
    if delta_h <= 72:
        return DECAY_RULES["within_72h"], "within_72h"
    return DECAY_RULES["within_72h"], "within_72h"


def compute_source(source_type: str) -> float:
    return SOURCE_WEIGHTS.get(source_type, 0.15)


def finalize(
    module_name: str,
    enriched: Dict[str, Any],
    base_score: float,
    base_breakdown: Dict[str, float],
    hit_rules: List[str],
    config: Dict[str, Any],
    explain_extra: Dict[str, Any] | None = None,
) -> UnifiedScoreOutput:
    confidence_raw, conf_missing, conf_comp = calc_confidence(enriched)
    confidence_boosted, cross_mult, cap_hit = apply_cross_bonus(confidence_raw, enriched)

    source_mult = compute_source(enriched.get("source_type", ""))
    market_mult = 1.0
    event_time = enriched.get("t1_captured") or datetime.now(timezone.utc)
    decay_mult, decay_key = compute_decay(event_time)
    propagation_mult = 1.0 if enriched.get("is_original", True) else 0.3

    final = base_score * source_mult * market_mult * cross_mult * decay_mult * propagation_mult
    final = max(0.0, min(final, 100.0))

    explain = {
        "score_breakdown": {
            "base_components": base_breakdown,
            "confidence_components": conf_comp,
            "propagation_effect": "original" if propagation_mult == 1.0 else "relay",
            "decay_key": decay_key,
            "cooldown": "miss",
            "cross_cap_hit": cap_hit,
        },
        "warning_reasons": [
            "single_source" if "multi_source" in conf_missing else "",
            "no_onchain_verify" if "onchain_verify" in conf_missing else "",
        ],
        "confidence_missing": conf_missing,
    }
    explain["warning_reasons"] = [x for x in explain["warning_reasons"] if x]
    if explain_extra:
        explain.update(explain_extra)

    config_hash = _hash_config(config)
    observability = {
        "config_hash": config_hash,
        "config_snapshot": config,
        "scoring_process": {
            "base": base_score,
            "source": source_mult,
            "market": market_mult,
            "cross": cross_mult,
            "decay": decay_mult,
            "propagation": propagation_mult,
            "final": final,
        },
        "input_keys": sorted(list(enriched.keys())),
    }

    return UnifiedScoreOutput(
        module_name=module_name,
        final_score=round(final, 4),
        confidence_score=round(confidence_boosted, 4),
        factors={
            "base": float(base_score),
            "source": source_mult,
            "market": market_mult,
            "cross": cross_mult,
            "decay": decay_mult,
            "propagation": propagation_mult,
        },
        base_breakdown=base_breakdown,
        hit_rules=hit_rules,
        explain=explain,
        observability=observability,
    )
