from types import SimpleNamespace

import pytest

from open_apps.agent.vLLM_agent import VLLMChatModel
from run_openapps_category import (
    command_for_task,
    list_category_tasks,
    normalize_category,
)


@pytest.mark.parametrize(
    "category",
    ["calendar", "messenger", "maps", "todo", "code_editor", "shop"],
)
def test_each_openapps_category_has_a_task(category):
    assert list_category_tasks(category)


def test_code_editor_and_shop_have_targeted_smoke_tasks():
    assert list_category_tasks("code_editor") == ["navigate_to_codeeditor"]
    assert list_category_tasks("shop") == ["navigate_to_onlineshop"]


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("OpenCalendar", "calendar"),
        ("OpenMessenger", "messenger"),
        ("OpenMaps", "maps"),
        ("OpenToDo", "todo"),
        ("OpenCodeEditor", "code_editor"),
        ("OpenShop", "shop"),
    ],
)
def test_openapps_names_are_valid_category_aliases(alias, expected):
    assert normalize_category(alias) == expected


def test_shop_command_enables_optional_app():
    args = SimpleNamespace(category="shop", headed=False, max_steps=7)
    command = command_for_task(args, "navigate_to_onlineshop")
    assert "agent=qwen36-siliconflow" in command
    assert "browsergym_env_args.max_steps=7" in command
    assert "apps.onlineshop.enable=True" in command


def test_provider_specific_chat_request_fields_are_forwarded():
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="ok"), logprobs=None
                    )
                ],
            )

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    model = VLLMChatModel(
        model_name="Qwen/Qwen3.6-35B-A3B",
        hostname=None,
        port="",
        api_key="test-key",
        temperature=0.0,
        max_tokens=1024,
        client_class=FakeClient,
        request_kwargs={"enable_thinking": False},
        max_tokens_param="max_tokens",
    )
    response = model([{"role": "user", "content": "test"}])
    assert response["role"] == "assistant"
    assert response["content"] == "ok"
    assert captured["max_tokens"] == 1024
    assert "max_completion_tokens" not in captured
    assert captured["extra_body"] == {"enable_thinking": False}
