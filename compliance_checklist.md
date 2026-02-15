# v1.1 对照清单（mod1 + mod5）

| v1.1要求 | 代码位置 | 状态 |
|---|---|---|
| 统一评分主线字段存在（base/source/market/cross/decay/propagation） | `scoring_protocol.py::finalize` | ✅ |
| 输出 `module_name/final_score/confidence_score/factors/base_breakdown/hit_rules/explain/observability` | `scoring_protocol.py::UnifiedScoreOutput.to_dict` | ✅ |
| cross封顶1.2 | `scoring_protocol.py::FINAL_CROSS_CAP/apply_cross_bonus` | ✅ |
| confidence独立100分制输出 | `scoring_protocol.py::calc_confidence` | ✅ |
| mod1 五维Base 30/20/20/20/10 | `mod1_launch_hunter/module.py` 五个 `_xxx` | ✅ |
| mod1 P1 动作词配置化 | `mod1_launch_hunter/config.yaml::launch_action_keywords` | ✅ |
| mod1 P10 叙事维度接入 | `mod1_launch_hunter/module.py::_narrative` | ✅ |
| mod1 stage新进展判定 | `mod1_launch_hunter/module.py::_infer_stage_from_text/_is_stage_progress` | ✅ |
| mod1 registry_updates输出 | `mod1_launch_hunter/module.py::_build_registry_updates` | ✅ |
| mod5 五维评估维度（lead_time快者高分） | `mod5_kol_radar/module.py` `_compute_*` | ✅ |
| mod5 时间窗口/清洗/去重/样本策略/紧急模式 | `mod5_kol_radar/module.py` | ✅ |
| mod5 分级阈值语义（>=，边界值命中） | `mod5_kol_radar/module.py::_score_to_tier` | ✅ |
| mod5 P5 四渠道自动发现 | `mod5_kol_radar/module.py::discover_kol_multi_channel` | ✅ |
| mod5 P6 关注列表筛选 | `mod5_kol_radar/module.py::scan_kol_following` | ✅ |
| 趋势聚类6h报告（48h≥3独立来源→趋势信号→新概念→升温→推送） | `mod5_kol_radar/module.py::trend_cluster_report_6h` | ✅ |
| mod5 用户确认入库对象输出 | `mod5_kol_radar/module.py::confirm_kol_candidates` | ✅ |
| mod5 周期更新与source_weights建议 | `mod5_kol_radar/module.py::update_kol_registry` | ✅ |
| mod5 tier动作使用 current_tier/min_samples | `mod5_kol_radar/module.py::_suggest_tier_action` | ✅ |
| 结构对齐测试（输出字段+公式主线） | `tests/test_acceptance.py::test_structure_alignment_formula_presence` | ✅ |
| 正常/负面/去重/极端值/失败模式测试 | `tests/test_acceptance.py` | ✅ |

## 差异声明
- 无差异。
