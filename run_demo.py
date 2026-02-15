from datetime import datetime, timedelta, timezone
import json

from mod1_launch_hunter.module import LaunchHunterModule, EnrichedEvent, OnchainData
from mod5_kol_radar.module import KOLRadarModule, KOLEvaluationContext, PushRecord


def main() -> None:
    mod1 = LaunchHunterModule("mod1_launch_hunter/config.yaml")
    now = datetime.now(timezone.utc)
    ev = EnrichedEvent(
        event_id="evt-1",
        t0_first_seen=now - timedelta(seconds=50),
        t1_captured=now,
        source_type="twitter_builder",
        source_account="@builder",
        raw_text="New primitive launch, mainnet live now, TGE",
        detected_projects=["AlphaX"],
        project_id="alphax",
        contract_address="0xabc",
        onchain=OnchainData(contract_verified=True, has_liquidity=True, liquidity_usd=10000),
        cross_evidence={"original_sources_count": 2, "platforms": ["twitter", "onchain"], "cross_bonus_key": "cross_twitter_onchain"},
        official_confirmed=True,
        is_original=True,
    )
    out1 = mod1.evaluate(ev)

    mod5 = KOLRadarModule("mod5_kol_radar/config.yaml")
    pushes = [
        PushRecord(
            alert_id=f"a{i//2}",
            t0_first_seen=now - timedelta(hours=2, minutes=i),
            t3_market_reaction=now - timedelta(hours=1, minutes=i),
            user_feedback="useful" if i % 3 != 0 else "useless",
            outcome_value=1.2 if i % 2 == 0 else None,
            is_original=(i % 2 == 0),
            final_score=70 + i,
            created_at=now - timedelta(days=i),
        )
        for i in range(12)
    ]
    ctx = KOLEvaluationContext(kol_handle="@kol", module="mod1", source_type="twitter_kol_a", pushes=pushes)
    out5 = mod5.evaluate(ctx)

    print(json.dumps({"mod1": out1, "mod5": out5}, ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
