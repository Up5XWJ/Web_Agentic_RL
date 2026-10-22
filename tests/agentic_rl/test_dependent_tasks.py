from copy import deepcopy
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import OmegaConf
import pytest

from open_apps.agentic_rl.dependent_tasks import (
    TEMPLATES, build_case, resolve_graph, validate_case,
)
from open_apps.agentic_rl.task_scenes import (
    SceneBuilder, business_state, hydra_literal, merge_content,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def builder():
    return SceneBuilder(ROOT)


def test_hydra_literals_preserve_multilingual_text_quotes_and_real_newlines():
    value = {"Alice": [["确认\nMeet at C:\\notes\\new; he said \"yes\". Grüße 中文\\", False, "Alice", "2026-10-10 09:00"]]}
    encoded = hydra_literal(value)
    decoded = OverridesParser.create().parse_overrides(["value=" + encoded])[0].value()
    assert decoded == value


def test_additive_yaml_content_is_merged_without_reapplying_defaults():
    base = {"chat_history": {"Alice": [["old", False, "Alice", "time"]]}, "users": ["Alice"]}
    patch = {"defaults": ["default"], "+chat_history": {"Alice": [["extra", False, "Alice", "time"]], "Bob": []},
             "+users": ["Bob"]}
    merged = merge_content(base, patch)
    assert [m[0] for m in merged["chat_history"]["Alice"]] == ["old", "extra"]
    assert merged["users"] == ["Alice", "Bob"]
    assert "defaults" not in merged and not any(key.startswith("+") for key in merged)
    assert len(base["chat_history"]["Alice"]) == 1


@pytest.mark.parametrize("template", TEMPLATES)
@pytest.mark.parametrize("tier", range(5))
def test_all_dependency_templates_and_complexity_tiers_have_valid_outcomes(builder, template, tier):
    case = build_case(builder, template, tier * 10)
    checks = validate_case(case)
    assert checks["source_removal_checks"] == case.metadata["read_count"]
    assert checks["counterfactual_answer_changes"] >= 1
    assert checks["wrong_final_state_rejected"]
    expected_counts = ({0}, {1}, {1}, {2}, {3, 4})
    assert case.metadata["variant_count"] in expected_counts[tier]


def test_required_sources_are_unique_and_missing_confirmations_fail(builder):
    case = build_case(builder, "messenger_calendar_maps_to_todo", 41)
    location = case.metadata["source_locations"]["notice"]
    rows = case.apps["messenger"]["chat_history"][location[2]]
    confirmed = deepcopy(rows[location[-1]])
    rows.append(confirmed)
    with pytest.raises(ValueError, match="Expected one source for notice, got 2"):
        resolve_graph(case.apps, case.metadata["dependency_graph"])
    rows.pop()
    rows.pop(location[-1])
    with pytest.raises(ValueError, match="Expected one source for notice, got 0"):
        resolve_graph(case.apps, case.metadata["dependency_graph"])


def test_answer_is_derived_from_seeded_facts_and_not_silently_recomputed(builder):
    case = build_case(builder, "calendar_to_todo", 10)
    location = case.metadata["source_locations"]["event"]
    case.apps["calendar"]["events"][location[-1]]["location"] = "Changed venue"
    resolved, _, _ = resolve_graph(case.apps, case.metadata["dependency_graph"])
    assert "Changed venue" in resolved["title"]
    with pytest.raises(ValueError, match="Private answer does not match"):
        validate_case(case)


def test_overrides_compose_to_the_same_real_initial_data_and_ui(builder):
    case = build_case(builder, "todo_messenger_calendar_to_maps", 48)
    repeated = build_case(builder, "todo_messenger_calendar_to_maps", 48)
    assert case == repeated
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        cfg = compose(config_name="config", overrides=[*case.task.overrides,
                      "apps.onlineshop.enable=False", "seed=42",
                      "logs_dir=/tmp/openapps_offline_scene", "databases_dir=/tmp/openapps_offline_scene/databases"])
    actual = OmegaConf.to_container(cfg.apps, resolve=True)
    assert business_state(actual) == business_state(case.apps)
    for source in case.metadata["variants"]:
        assert actual[source["app"]] == case.apps[source["app"]]
    assert all("database_path=" not in override for override in case.task.overrides)


def test_large_map_yaml_is_sampled_and_adversarial_message_is_materialized(builder):
    import random
    sources = [
        {"app": "maps", "path": "config/apps/maps/content/long_descriptions.yaml"},
        {"app": "messenger", "path": "config/apps/messenger/content/adversarial_descriptions.yaml"},
    ]
    apps, metadata = builder.background(sources, 4, random.Random(17))
    assert metadata["before_sampling"]["maps"] >= 1000
    assert len(apps["maps"]["saved_places"]) == 34
    assert any("you have achieved your goal" in message[0]
               for message in apps["messenger"]["chat_history"]["Alice"])
    assert not any(key.startswith("+") for app in ("maps", "messenger") for key in apps[app])
