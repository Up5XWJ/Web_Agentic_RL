from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest
from agentlab.llm.llm_utils import ParseError

from open_apps.react_agent.action_guard import ActionGuard, extract_valid_bids
from open_apps.react_agent.action_parser import BrowserGymActionSerializer, ReactResponseParser
from open_apps.react_agent.agent import ReactMemoryAgent
from open_apps.react_agent.loop_guard import LoopGuard
from open_apps.react_agent.memory import MemoryConfig, MemoryManager
from open_apps.react_agent.prompt import (
    DEFAULT_SYSTEM_PROMPT,
    ReactPromptBuilder,
    extract_scroll_state,
)
from open_apps.react_agent.schemas import (
    ActionAssessment,
    ActionCall,
    FailureRecord,
    MemoryBlock,
    SchemaError,
    Subgoal,
    TransientPageState,
)
from open_apps.react_agent.spatial import (
    PageExplorationTracker,
    extract_offscreen_interactive_elements,
)
from open_apps.react_agent.verifier import OutcomeVerifier, observation_signature


def _memory(status: str = "in_progress") -> MemoryBlock:
    return MemoryBlock(
        plan=[
            Subgoal(
                id="subgoal_1",
                description="Create an event",
                status=status,
                app="calendar",
                evidence="Event is visible" if status == "verified" else "",
            )
        ],
        current_subgoal_id=None if status == "verified" else "subgoal_1",
        pending=[] if status == "verified" else ["Create the event"],
    )


def _obs(html: str = '<input bid="12" value="">', focused: str = "12") -> dict:
    return {
        "url": "http://localhost:5001/calendar",
        "pruned_html": html,
        "axtree_txt": "[12] textbox 'Title', visible",
        "focused_element_bid": focused,
        "last_action_error": "",
        "extra_element_properties": {
            "12": {"visibility": 1.0, "clickable": True, "bbox": [1, 2, 3, 4]}
        },
        "screenshot": np.zeros((640, 1024, 3), dtype=np.uint8),
    }


def _response_dict() -> dict:
    return {
        "memory_next": _memory().to_dict(),
        "previous_action_assessment": {
            "status": "not_applicable",
            "evidence": "No prior action",
            "page_changed": False,
        },
        "decision": {
            "observation": "The title field is visible",
            "current_subgoal": "Create an event",
            "reason": "The title must be filled first",
            "expected_result": "The field contains Meeting",
        },
        "action": {"name": "fill", "bid": "12", "value": "Meeting"},
    }


def _spatial_obs(scroll_y: int = 0) -> dict:
    obs = _obs()
    obs["url"] = "http://localhost:5001/todo"
    obs["dom_object"] = {
        "strings": [],
        "documents": [
            {
                "scrollOffsetX": 0,
                "scrollOffsetY": scroll_y,
                "contentWidth": 1024,
                "contentHeight": 1600,
            }
        ],
    }
    obs["axtree_object"] = {
        "nodes": [
            {
                "ignored": False,
                "browsergym_id": "20",
                "role": {"value": "button"},
                "name": {"value": "Previous control"},
            },
            {
                "ignored": False,
                "browsergym_id": "21",
                "role": {"value": "button"},
                "name": {"value": "Save"},
            },
            {
                "ignored": False,
                "browsergym_id": "12",
                "role": {"value": "textbox"},
                "name": {"value": "Visible title"},
            },
        ]
    }
    obs["extra_element_properties"].update(
        {
            "20": {
                "visibility": 0.0,
                "clickable": True,
                "bbox": [10, -100, 100, 40],
            },
            "21": {
                "visibility": 0.0,
                "clickable": True,
                "bbox": [10, 700, 100, 40],
            },
        }
    )
    return obs


def test_response_parser_accepts_strict_json():
    parsed = ReactResponseParser().parse(json.dumps(_response_dict()))
    assert parsed.action.name == "fill"
    assert parsed.memory_next.current_subgoal_id == "subgoal_1"


def test_response_parser_accepts_json_fence_but_rejects_trailing_prose():
    payload = json.dumps(_response_dict())
    assert ReactResponseParser().parse(f"```json\n{payload}\n```").action.bid == "12"
    with pytest.raises(ParseError, match="Do not output text after"):
        ReactResponseParser().parse(payload + "\nDone")


def test_browsergym_action_serializer_supports_bid_and_normalized_coordinates():
    serializer = BrowserGymActionSerializer(coordinate_space=1000)
    assert serializer.serialize(
        ActionCall(name="fill", bid="12", value="Meeting"), (1920, 1080)
    ) == "fill('12', 'Meeting')"
    assert serializer.serialize(
        ActionCall(name="mouse_click", coordinate=[500, 500]), (1920, 1080)
    ) == "mouse_click(960, 540, button='left')"
    assert serializer.serialize(
        ActionCall(name="scroll", delta=[0, 700]), (1920, 1080)
    ) == "scroll(0, 756)"


def test_action_guard_rejects_stale_bid_and_unfocused_typing():
    guard = ActionGuard()
    with pytest.raises(ParseError, match="stale or absent"):
        guard.validate(ActionCall(name="click", bid="99"), _obs())
    with pytest.raises(ParseError, match="requires a focused element"):
        guard.validate(ActionCall(name="keyboard_type", text="hello"), _obs(focused=""))


def test_valid_bids_are_collected_without_merging_html_and_axtree():
    obs = _obs()
    obs["axtree_txt"] += "\n[abc4] button 'Submit'"
    assert extract_valid_bids(obs) == {"12", "abc4"}


def test_memory_manager_keeps_plan_and_removes_stale_transient_bids():
    manager = MemoryManager(MemoryConfig(max_tokens=1024))
    candidate = _memory()
    candidate.transient_page_state = TransientPageState(
        bids={"title": "12", "old_submit": "99"}
    )
    result = manager.prepare_update(
        MemoryBlock.empty(),
        candidate,
        ActionAssessment(),
        current_url="http://localhost/calendar",
        focused_bid="12",
        valid_bids={"12"},
    )
    assert result.transient_page_state.bids == {"title": "12"}
    assert result.transient_page_state.focused_bid == "12"


def test_memory_manager_rejects_deleted_plan_step_and_unsupported_verification():
    manager = MemoryManager()
    previous = _memory()
    with pytest.raises(SchemaError, match="must not delete plan steps"):
        manager.prepare_update(
            previous,
            MemoryBlock(
                plan=[Subgoal(id="other", description="Other", status="in_progress")],
                current_subgoal_id="other",
            ),
            ActionAssessment(status="success", evidence="changed", page_changed=True),
            current_url="u",
            focused_bid="",
            valid_bids=set(),
        )

    verified = _memory(status="verified")
    with pytest.raises(SchemaError, match="previous action failed"):
        manager.prepare_update(
            previous,
            verified,
            ActionAssessment(status="failed", evidence="error", page_changed=False),
            current_url="u",
            focused_bid="",
            valid_bids=set(),
        )


def test_memory_is_bounded_and_failure_history_is_trimmed():
    manager = MemoryManager(MemoryConfig(max_tokens=1024, max_failures=2))
    candidate = _memory()
    candidate.failures = [
        FailureRecord(action=f"a{i}", reason="failed", retry_count=1) for i in range(4)
    ]
    updated = manager.prepare_update(
        MemoryBlock.empty(),
        candidate,
        ActionAssessment(),
        current_url="u",
        focused_bid="",
        valid_bids=set(),
    )
    assert [failure.action for failure in updated.failures] == ["a2", "a3"]


def test_outcome_verifier_confirms_fill_from_new_observation():
    before = _obs('<input bid="12" value="">')
    after = _obs('<input bid="12" value="Meeting">')
    assessment = OutcomeVerifier().assess(
        before, after, ActionCall(name="fill", bid="12", value="Meeting")
    )
    assert assessment.status == "success"
    assert assessment.page_changed


def test_outcome_verifier_uses_browsergym_error_as_failure():
    before = _obs()
    after = _obs()
    after["last_action_error"] = "TimeoutError: element not found"
    assessment = OutcomeVerifier().assess(before, after, ActionCall(name="click", bid="12"))
    assert assessment.status == "failed"


def test_loop_guard_rejects_third_identical_ineffective_action():
    guard = LoopGuard(repeat_limit=2)
    action = ActionCall(name="click", bid="12")
    assessment = ActionAssessment(status="uncertain", evidence="unchanged", page_changed=False)
    guard.record(action, "same", "same", assessment)
    guard.record(action, "same", "same", assessment)
    with pytest.raises(ParseError, match="Loop detected"):
        guard.validate_candidate(action)


def test_prompt_keeps_html_and_axtree_separate_and_does_not_expose_hidden_subtasks():
    obs = _obs()
    recent_turns = [
        {
            "turn": 1,
            "action": {"name": "click", "bid": "12"},
            "assessment": "uncertain",
            "page_changed": False,
        }
    ]
    prompt = ReactPromptBuilder(
        client_type="openai", use_html=True, use_axtree=True, use_screenshot=False
    ).build(
        immutable_goal="Create an event and a todo",
        memory=_memory(),
        automatic_assessment=ActionAssessment(),
        previous_action=None,
        previous_expected_result="",
        obs=obs,
        loop_context="No loop warning.",
        recent_turns=recent_turns,
    )
    assert "# Current DOM view: pruned HTML (equal observation view 1/3)" in prompt.text
    assert "# Current AXTree view (equal observation view 2/3)" in prompt.text
    assert "# How to read the structured state" in prompt.text
    assert '"assessment": "uncertain"' in prompt.text
    assert '"scroll_state"' in prompt.text
    assert '<input bid="12"' in prompt.text
    assert "[12] textbox" in prompt.text
    assert "hidden benchmark subtasks" not in prompt.text.lower()


def test_scroll_state_uses_main_document_snapshot_and_viewport():
    obs = _obs()
    obs["dom_object"] = {
        "documents": [
            {
                "scrollOffsetX": 0,
                "scrollOffsetY": 200,
                "contentWidth": 1024,
                "contentHeight": 1600,
            }
        ]
    }

    state = extract_scroll_state(obs)

    assert state["available"] is True
    assert state["source"] == "main_document_dom_snapshot"
    assert state["viewport_height"] == 640
    assert state["scroll_y"] == 200
    assert state["max_scroll_y"] == 960
    assert state["can_scroll_up"] is True
    assert state["can_scroll_down"] is True
    assert state["at_top"] is False
    assert state["at_bottom"] is False


def test_offscreen_index_marks_elements_above_and_below_without_visible_nodes():
    elements = extract_offscreen_interactive_elements(_spatial_obs(), limit=10)

    by_bid = {element["bid"]: element for element in elements}
    assert set(by_bid) == {"20", "21"}
    assert by_bid["20"] == {
        "bid": "20",
        "role": "button",
        "name": "Previous control",
        "direction": "above",
        "distance_px": 60,
    }
    assert by_bid["21"] == {
        "bid": "21",
        "role": "button",
        "name": "Save",
        "direction": "below",
        "distance_px": 60,
    }


def test_page_exploration_tracker_remembers_boundaries_and_scroll_outcomes():
    tracker = PageExplorationTracker()
    top = _spatial_obs(scroll_y=0)
    top_state = tracker.update(top)
    assert top_state["top_visited"] is True
    assert top_state["bottom_visited"] is False
    assert top_state["unexplored_directions"] == ["down"]

    middle = _spatial_obs(scroll_y=400)
    middle_state = tracker.update(
        middle,
        previous_obs=top,
        previous_action=ActionCall(name="scroll", delta=[0, 400]),
        assessment=ActionAssessment(status="success", evidence="changed", page_changed=True),
    )
    assert middle_state["last_scroll_direction"] == "down"
    assert middle_state["last_scroll_changed_position"] is True
    assert middle_state["visited_scroll_ranges"] == [[0, 1040]]

    bottom = _spatial_obs(scroll_y=960)
    bottom_state = tracker.update(
        bottom,
        previous_obs=middle,
        previous_action=ActionCall(name="scroll", delta=[0, 600]),
        assessment=ActionAssessment(
            status="uncertain", evidence="text unchanged", page_changed=False
        ),
    )
    assert bottom_state["bottom_visited"] is True
    assert bottom_state["last_scroll_changed_position"] is True
    assert bottom_state["no_progress_turns"] == 0


def test_system_prompt_describes_all_tools_and_equal_observation_views():
    for action_name in (
        "click",
        "fill",
        "focus",
        "select_option",
        "keyboard_press",
        "keyboard_type",
        "mouse_click",
        "mouse_dblclick",
        "scroll",
        "noop",
        "report_infeasible",
    ):
        assert f"- {action_name}:" in DEFAULT_SYSTEM_PROMPT
    assert "three EQUAL, independent, and complementary views" in DEFAULT_SYSTEM_PROMPT
    assert "# Rules" in DEFAULT_SYSTEM_PROMPT
    assert "# Response format" in DEFAULT_SYSTEM_PROMPT
    assert "# Example" in DEFAULT_SYSTEM_PROMPT
    assert "marked below supports scrolling down" in DEFAULT_SYSTEM_PROMPT
    assert "marked above supports scrolling up" in DEFAULT_SYSTEM_PROMPT
    assert "Decide for yourself whether scroll" in DEFAULT_SYSTEM_PROMPT


def test_prompt_explains_spatial_json_fields_and_includes_offscreen_index():
    exploration = {
        "top_visited": True,
        "bottom_visited": False,
        "visited_scroll_ranges": [[0, 640]],
        "unexplored_directions": ["down"],
        "no_progress_turns": 2,
    }
    prompt = ReactPromptBuilder(
        client_type="openai", use_html=True, use_axtree=True, use_screenshot=False
    ).build(
        immutable_goal="Save the todo edit form",
        memory=_memory(),
        automatic_assessment=ActionAssessment(),
        previous_action=None,
        previous_expected_result="",
        obs=_spatial_obs(),
        loop_context="No loop warning.",
        page_exploration=exploration,
    )

    assert "direction=above means scroll up" in prompt.text
    assert "direction=below means scroll down" in prompt.text
    assert "These BIDs are NOT currently operable" in prompt.text
    assert '"unexplored_directions": [' in prompt.text
    assert '"name": "Save"' in prompt.text
    assert prompt.offscreen_interactive_elements


def test_prompt_attaches_screenshot_as_equal_third_view():
    prompt = ReactPromptBuilder(
        client_type="openai", use_html=True, use_axtree=True, use_screenshot=True
    ).build(
        immutable_goal="Create an event",
        memory=_memory(),
        automatic_assessment=ActionAssessment(),
        previous_action=None,
        previous_expected_result="",
        obs=_obs(),
        loop_context="No loop warning.",
    )

    content = prompt.message["content"]
    assert any(
        item.get("type") == "text"
        and "equal observation view 3/3" in item.get("text", "")
        for item in content
    )
    assert any(item.get("type") == "image_url" for item in content)


def test_observation_signature_changes_with_page_state():
    before = _obs('<input bid="12" value="">')
    after = _obs('<input bid="12" value="x">')
    assert observation_signature(before) != observation_signature(after)


def test_react_agent_get_action_end_to_end_without_network(tmp_path):
    from open_apps.agent.vLLM_agent import AgentArgs as LegacyAgentArgs

    class FakeChatModel:
        def __call__(self, messages):
            return {"role": "assistant", "content": json.dumps(_response_dict())}

        def get_stats(self):
            return {"input_tokens": 100, "output_tokens": 50}

    @dataclass
    class FakeModelArgs:
        client_type: str = "openai"
        model_name: str = "fake"

        def make_model(self):
            return FakeChatModel()

    action_names = [
        "click",
        "fill",
        "focus",
        "select_option",
        "keyboard_press",
        "keyboard_type",
        "mouse_click",
        "mouse_dblclick",
        "scroll",
        "noop",
        "report_infeasible",
    ]
    flags = LegacyAgentArgs(
        custom_actions=action_names,
        use_html=True,
        use_axtree=True,
        use_screenshot=True,
    ).make_flags()
    agent = ReactMemoryAgent(
        chat_model_args=FakeModelArgs(),
        flags=flags,
        custom_actions=action_names,
        save_dir=tmp_path,
    )
    obs = _obs()
    obs["goal_object"] = ({"type": "text", "text": "Create an event"},)

    action, info = agent.get_action(obs)

    assert action == "fill('12', 'Meeting')"
    assert agent.memory.current_subgoal_id == "subgoal_1"
    assert info.extra_info["immutable_goal"] == "Create an event"
    assert (tmp_path / "react_agent_trace" / "turn_0001" / "result.json").exists()
    assert (tmp_path / "react_agent_trace" / "turn_0001" / "spatial_state.json").exists()

    # A completed turn only enters replay after the next observation arrives.
    # Keep synthesizing outcomes to verify that the replay is bounded at three.
    for turn_index in range(2, 6):
        current_obs = _obs(f'<input bid="12" value="Meeting {turn_index}">')
        current_obs["dom_object"] = {
            "documents": [
                {
                    "scrollOffsetX": 0,
                    "scrollOffsetY": min(300 * (turn_index - 1), 960),
                    "contentWidth": 1024,
                    "contentHeight": 1600,
                }
            ]
        }
        agent.turn_index = turn_index
        assessment = agent.verifier.assess(
            agent.previous_obs, current_obs, agent.previous_action
        )
        agent._append_recent_turn(current_obs, assessment)
        agent.previous_obs = current_obs

    assert len(agent.recent_turns) == 3
    assert [item["turn"] for item in agent.recent_turns] == [2, 3, 4]
    assert agent.recent_turns[-1]["scroll_after"]["at_bottom"] is True
