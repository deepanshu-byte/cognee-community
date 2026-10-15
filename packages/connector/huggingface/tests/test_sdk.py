"""Exercise the real official SDK against offline HTTP response fixtures."""

from datetime import UTC, datetime
from unittest.mock import Mock

import httpx
import pytest
from huggingface_hub import HfApi
from huggingface_hub.community import deserialize_event
from huggingface_hub.errors import EntryNotFoundError

from cognee_community_connector_huggingface.huggingface import (
    _list_repos,
    _read_card,
    discussion_to_row,
    huggingface_source,
    sync_hub,
)

WHEN = "2026-01-01T00:00:00.000Z"


def test_sdk_repository_and_discussion_pagination(tmp_path, monkeypatch):
    seen = []

    def discussion(num):
        return {
            "num": num,
            "title": f"Discussion {num}",
            "author": {"name": "reader"},
            "createdAt": WHEN,
            "status": "open",
            "repo": {"name": "owner/first", "type": "model"},
            "isPullRequest": False,
        }

    def handler(request):
        seen.append(str(request.url))
        path = request.url.path
        if path == "/api/models":
            if request.url.params.get("cursor") == "next":
                return httpx.Response(200, json=[{"id": "owner/second", "lastModified": WHEN}])
            return httpx.Response(
                200,
                json=[{"id": "owner/first", "lastModified": WHEN}],
                headers={"Link": '<https://huggingface.co/api/models?cursor=next>; rel="next"'},
            )
        if path == "/api/models/owner/second/discussions":
            return httpx.Response(200, json={"count": 0, "start": 0, "discussions": []})
        if path == "/api/models/owner/first/discussions":
            page = int(request.url.params.get("p", "0"))
            return httpx.Response(
                200, json={"count": 2, "start": page, "discussions": [discussion(page + 1)]}
            )
        if path.startswith("/api/models/owner/first/discussions/"):
            num = int(path.rsplit("/", 1)[-1])
            data = discussion(num)
            data["events"] = [
                {
                    "id": f"event-{num}",
                    "type": "comment",
                    "createdAt": WHEN,
                    "author": {"name": "reader"},
                    "data": {
                        "edited": True,
                        "hidden": False,
                        "latest": {"raw": f"Comment {num}", "updatedAt": WHEN},
                        "history": [],
                    },
                }
            ]
            return httpx.Response(200, json=data)
        raise AssertionError(f"Unexpected request: {request.url}")

    session = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("huggingface_hub.hf_api.get_session", lambda: session)
    monkeypatch.setattr("huggingface_hub.utils._pagination.get_session", lambda: session)
    monkeypatch.setattr("huggingface_hub.utils._http.get_session", lambda: session)
    client = HfApi(token=False)
    readme = tmp_path / "README.md"
    readme.write_text("---\nlicense: mit\n---\nSmall card", encoding="utf-8")
    monkeypatch.setattr(client, "hf_hub_download", lambda **kwargs: str(readme))
    state = {}
    try:
        rows = list(sync_hub(client, state, owners=("owner",)))
    finally:
        session.close()
    assert {r["id"] for r in rows} == {
        "model:owner/first:card",
        "model:owner/second:card",
        "model:owner/first:discussion:1",
        "model:owner/first:discussion:2",
        "huggingface:scope",
    }
    assert any("cursor=next" in url for url in seen)
    assert any("discussions?p=1" in url for url in seen)
    assert state["model"]["repos"]["owner/first"]["discussion_activity"]["2"]


@pytest.mark.parametrize(
    "kind, data, expected",
    [
        ("status-change", {"status": "closed"}, "new_status: closed"),
        ("commit", {"subject": "Fix tokenizer", "oid": "123"}, "summary: Fix tokenizer"),
        ("title-change", {"from": "Old", "to": "New"}, "new_title: New"),
    ],
)
def test_official_event_types_are_rendered(kind, data, expected):
    event = deserialize_event(
        {"id": "event", "type": kind, "createdAt": WHEN, "author": {"name": "reader"}, "data": data}
    )
    row = discussion_to_row(
        "model",
        "owner/model",
        {
            "num": 1,
            "title": "PR",
            "status": "merged",
            "author": "reader",
            "is_pull_request": True,
            "events": [event],
        },
    )
    assert expected in row["content"]
    assert "Pull request: True" in row["content"]


def test_missing_readme_keeps_repository_identity():
    client = Mock()
    client.hf_hub_download.side_effect = EntryNotFoundError("No README")
    row = _read_card(client, "space", "owner/demo")
    assert row["id"] == "space:owner/demo:card"
    assert "owner owns the space repository owner/demo" in row["content"]


@pytest.mark.parametrize(
    "explicit, env, expected",
    [
        ("explicit", "environment", "explicit"),
        (None, "environment", "environment"),
        (None, None, None),
    ],
)
def test_auth_precedence(monkeypatch, explicit, env, expected):
    if env:
        monkeypatch.setenv("HF_TOKEN", env)
    else:
        monkeypatch.delenv("HF_TOKEN", raising=False)
    factory = Mock()
    monkeypatch.setattr("cognee_community_connector_huggingface.huggingface.HfApi", factory)
    huggingface_source(repo_ids=["owner/model"], token=explicit)
    factory.assert_called_once_with(token=expected)


def test_no_volatile_metadata_in_document_content():
    from cognee_community_connector_huggingface.huggingface import _metadata, card_to_row

    metadata = _metadata(
        {"downloads": 12345, "likes": 67890, "created_at": datetime(2026, 1, 1, tzinfo=UTC)}
    )
    assert metadata["downloads"] == 12345
    row = card_to_row("dataset", "owner/data", "Dataset documentation")
    assert "12345" not in row["content"]
    assert "67890" not in row["content"]


@pytest.mark.parametrize(
    "repo_type, method, task_key",
    [
        ("model", "list_models", "pipeline_tag"),
        ("dataset", "list_datasets", "task_categories"),
        ("space", "list_spaces", None),
    ],
)
def test_task_filters_follow_each_sdk_contract(repo_type, method, task_key):
    client = Mock()
    getattr(client, method).return_value = []
    assert (
        _list_repos(
            client,
            repo_type,
            ("owner",),
            None,
            ("custom",),
            task="text-generation",
            library="transformers",
        )
        == []
    )
    kwargs = getattr(client, method).call_args.kwargs
    if task_key:
        assert kwargs[task_key] == "text-generation"
        assert kwargs["filter"] == ["custom", "transformers"]
    else:
        assert kwargs["filter"] == ["custom", "transformers", "text-generation"]
