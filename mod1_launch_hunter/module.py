from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Callable
import json

from scoring_protocol import finalize


@dataclass
class OnchainData:
    contract_verified: bool = False
    has_liquidity: bool = False
    liquidity_usd: float = 0.0
    status: str = "success"


@dataclass
class EnrichedEvent:
    event_id: str
    t0_first_seen: datetime
    t1_captured: datetime
    source_type: str
    source_account: str
    raw_text: str
    event_type: str = "launch"
    detected_projects: Optional[List[str]] = None
    contract_address: str = ""
    project_id: str = ""
    onchain: Optional[OnchainData] = None
    official_confirmed: bool = False
    is_original: bool = True
    cross_evidence: Optional[Dict[str, Any]] = None


class RegistryReader:
    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        return None

    def concept_exists(self, concept: str) -> bool:
        return False


class NullRegistryReader(RegistryReader):
    pass


SonnetCallable = Callable[[str], Dict[str, Any]]


def _null_sonnet(_: str) -> Dict[str, Any]:
    return {
        "track_score": 2,
        "pain_point_score": 2,
        "uniqueness_score": 2,
        "credibility_score": 2,
        "narrative_core": "sonnet_unavailable",
        "narrative_tags": [],
        "_degraded": True,
    }


class LaunchHunterModule:
    def __init__(self, config_path: str, registry: Optional[RegistryReader] = None, sonnet_caller: Optional[SonnetCallable] = None):
        self.config = json.load(open(config_path, "r", encoding="utf-8"))
        self.registry = registry or NullRegistryReader()
        self.sonnet_caller = sonnet_caller or _null_sonnet
        self._validate_config()

    def _validate_config(self) -> None:
        required = ["novelty", "verifiability", "narrative", "innovation", "timeliness", "launch_action_keywords", "innovation_keywords"]
        for k in required:
            if k not in self.config:
                raise ValueError(f"Missing required config key: {k}")
        total = sum(self.config[d]["max_score"] for d in ["novelty", "verifiability", "narrative", "innovation", "timeliness"])
        if total != 100:
            raise ValueError("mod1 total max score must be 100")

    def evaluate(self, enriched: EnrichedEvent) -> Dict[str, Any]:
        if not self.config.get("enabled", True):
            return finalize(
                module_name=self.config["module_name"],
                enriched=enriched.__dict__,
                base_score=0,
                base_breakdown={},
                hit_rules=["module_disabled"],
                config=self.config,
                explain_extra={"module_status": "disabled"},
            ).to_dict()

        self._validate_input(enriched)

        b, r = {}, []
        n, nr = self._novelty(enriched)
        v, vr = self._verifiability(enriched)
        na, nar = self._narrative(enriched)
        inn, ir = self._innovation(enriched)
        t, tr = self._timeliness(enriched)
        b.update({"novelty": n, "verifiability": v, "narrative": na, "innovation": inn, "timeliness": t})
        r.extend([nr, vr, nar, ir, tr])
        base = min(sum(b.values()), 100)

        enriched_dict = {**enriched.__dict__, "onchain": (enriched.onchain.__dict__ if enriched.onchain else {}), "cross_evidence": enriched.cross_evidence or {}}
        out = finalize(
            module_name=self.config["module_name"],
            enriched=enriched_dict,
            base_score=base,
            base_breakdown=b,
            hit_rules=r,
            config=self.config,
            explain_extra={"module_status": "normal"},
        ).to_dict()
        out["registry_updates"] = self._build_registry_updates(enriched)
        return out

    def _validate_input(self, e: EnrichedEvent) -> None:
        if not e.event_id or not e.raw_text:
            raise ValueError("event_id and raw_text are required")
        if e.event_type != "launch":
            raise ValueError("event_type must be launch")

    def _novelty(self, e: EnrichedEvent) -> Tuple[float, str]:
        cfg = self.config["novelty"]["scores"]
        key = e.project_id or ((e.detected_projects or [""])[0])
        if not key:
            return float(cfg["no_project_id"]), "novelty_no_project_id"
        existing = self.registry.get_project(key)
        if existing is None:
            return float(cfg["first_mention"]), "novelty_first_mention"
        onchain = e.onchain or OnchainData()
        if (e.contract_address and not existing.get("has_token", False)) or (onchain.has_liquidity and not existing.get("has_token", False)):
            return float(cfg["new_progress"]), "novelty_new_progress"
        if self._is_stage_progress(e.raw_text, existing.get("stage", "")):
            return float(cfg["new_progress"]), "novelty_new_progress"
        return float(cfg["known_repeat"]), "novelty_known_repeat"

    @staticmethod
    def _infer_stage_from_text(raw_text: str) -> str:
        text = raw_text.lower()
        if any(k in text for k in ["launched", "live now", "tge", "token generation", "minting now", "public sale"]):
            return "launched"
        if any(k in text for k in ["mainnet", "main net", "production"]):
            return "mainnet"
        if any(k in text for k in ["testnet", "test net", "devnet", "dev net"]):
            return "testnet"
        if any(k in text for k in ["announced", "upcoming", "coming soon"]):
            return "announced"
        return ""

    def _is_stage_progress(self, raw_text: str, existing_stage: str) -> bool:
        implied = self._infer_stage_from_text(raw_text)
        if not implied or not existing_stage:
            return False
        rank = {"announced": 0, "testnet": 1, "mainnet": 2, "launched": 3}
        return rank.get(implied, -1) > rank.get(existing_stage, -1)

    def _build_registry_updates(self, e: EnrichedEvent) -> Dict[str, Any]:
        updates: Dict[str, Any] = {}
        key = e.project_id or ((e.detected_projects or [""])[0])
        if key:
            updates["project-registry"] = {
                "project_id": key,
                "has_token": bool(e.contract_address),
                "confirmed": bool(e.official_confirmed),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        updates["concept-tracker"] = {
            "source": e.source_account,
            "keywords": [k for k in self.config["innovation_keywords"]["en"] if k.lower() in e.raw_text.lower()]
        }
        return updates

    def _verifiability(self, e: EnrichedEvent) -> Tuple[float, str]:
        cfg = self.config["verifiability"]["scores"]
        onchain = e.onchain or OnchainData()
        if onchain.status in ("timeout", "error"):
            return (float(cfg["contract_unverified"]) if e.contract_address else float(cfg["no_contract"]), "verifiability_api_degraded")
        if e.contract_address and onchain.contract_verified:
            return float(cfg["contract_verified"]), "verifiability_contract_verified"
        if e.contract_address:
            return float(cfg["contract_unverified"]), "verifiability_contract_unverified"
        return float(cfg["no_contract"]), "verifiability_no_contract"

    def _narrative(self, e: EnrichedEvent) -> Tuple[float, str]:
        res = self.sonnet_caller(e.raw_text)
        total = min(float(res.get("track_score", 2) + res.get("pain_point_score", 2) + res.get("uniqueness_score", 2) + res.get("credibility_score", 2)), 20)
        if res.get("_degraded"):
            return total, "narrative_sonnet_degraded"
        if total >= 16:
            return total, "narrative_strong"
        if total >= 10:
            return total, "narrative_moderate"
        return total, "narrative_weak"

    def _innovation(self, e: EnrichedEvent) -> Tuple[float, str]:
        cfg = self.config["innovation"]["scores"]
        txt = e.raw_text.lower()
        en = [k.lower() for k in self.config["innovation_keywords"]["en"]]
        zh = self.config["innovation_keywords"]["zh"]
        hit = sum(1 for k in en if k in txt) + sum(1 for k in zh if k in e.raw_text)
        if hit >= 3:
            return float(cfg["tier_s"]), "innovation_tier_s"
        if hit >= 2:
            return float(cfg["tier_a"]), "innovation_tier_a"
        if hit >= 1:
            return float(cfg["tier_b"]), "innovation_tier_b"
        return float(cfg["default"]), "innovation_no_keywords"

    def _timeliness(self, e: EnrichedEvent) -> Tuple[float, str]:
        tcfg, scfg = self.config["timeliness"]["thresholds"], self.config["timeliness"]["scores"]
        delay = abs((e.t1_captured - e.t0_first_seen).total_seconds())
        if delay < tcfg["tier_s"]:
            return float(scfg["tier_s"]), "timeliness_tier_s"
        if delay < tcfg["tier_a"]:
            return float(scfg["tier_a"]), "timeliness_tier_a"
        if delay < tcfg["tier_b"]:
            return float(scfg["tier_b"]), "timeliness_tier_b"
        return float(scfg["default"]), "timeliness_slow"
