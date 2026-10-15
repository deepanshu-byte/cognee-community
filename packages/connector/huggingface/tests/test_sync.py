"""Offline Hub fixtures, real dlt merges, and cognee document routing."""

import asyncio
import copy
import importlib
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import NAMESPACE_OID, uuid5

import httpx
import pytest
from huggingface_hub import DatasetInfo, ModelInfo, SpaceInfo
from huggingface_hub.errors import HfHubHTTPError, LocalTokenNotFoundError

from cognee_community_connector_huggingface.huggingface import (
    _owner_access,
    _request,
    _retry_after,
    card_to_row,
    discussion_to_row,
    huggingface_source,
    sync_hub,
)

WHEN = "2026-01-01T00:00:00Z"
LATER = "2026-02-01T00:00:00Z"
CARD = "---\nlicense: mit\nbase_model: owner/base\ndatasets: [owner/data]\n---\n# Example\nBody"


def http_error(status, headers=None):
    response = httpx.Response(
        status,
        headers=headers or {},
        request=httpx.Request("GET", "https://huggingface.co/api/models"),
    )
    return HfHubHTTPError(f"HTTP {status}", response=response)


def info(repo_id="owner/model", repo_type="model", when=WHEN):
    cls = {"model": ModelInfo, "dataset": DatasetInfo, "space": SpaceInfo}[repo_type]
    return cls(id=repo_id, lastModified=when, createdAt=WHEN, likes=5, downloads=10)


def thread(num=1, content="First comment", status="open", hidden=False):
    return SimpleNamespace(
        num=num,
        title=f"Thread {num}",
        status=status,
        author="reader",
        is_pull_request=False,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        events=[
            SimpleNamespace(
                type="comment", author="reader", hidden=hidden, content=content, created_at=None
            )
        ],
    )


class FakeHub:
    """Implements the HfApi read surface; listing generators model SDK pagination."""

    def __init__(self, tmp_path):
        self.root = tmp_path
        self.repos = {"model": {"owner/model": info()}, "dataset": {}, "space": {}}
        self.cards = {}
        self.threads = {("model", "owner/model"): [thread()]}
        self.calls = []
        self.redirects = {}
        self.failures = {}
        self.identity = {"name": "owner", "orgs": [], "auth": {"accessToken": {"role": "read"}}}

    def _check(self, method, key):
        self.calls.append((method, key))
        failure = self.failures.get((method, key))
        if failure:
            raise failure

    def _listing(self, repo_type, kwargs):
        self._check("listing", repo_type)
        for repo_id, repo in self.repos[repo_type].items():
            if kwargs.get("author") and not repo_id.startswith(kwargs["author"] + "/"):
                continue
            yield repo

    def list_models(self, **kwargs):
        self.calls.append(("model_kwargs", kwargs))
        return self._listing("model", kwargs)

    def list_datasets(self, **kwargs):
        return self._listing("dataset", kwargs)

    def list_spaces(self, **kwargs):
        return self._listing("space", kwargs)

    def repo_info(self, repo_id, repo_type):
        self._check("info", (repo_type, repo_id))
        canonical = self.redirects.get(repo_id, repo_id)
        if canonical not in self.repos[repo_type]:
            raise http_error(404)
        return self.repos[repo_type][canonical]

    def hf_hub_download(self, repo_id, filename, repo_type):
        assert filename == "README.md"
        self._check("card", (repo_type, repo_id))
        path = self.root / f"{repo_type}-{repo_id.replace('/', '-')}.md"
        path.write_text(self.cards.get((repo_type, repo_id), CARD), encoding="utf-8")
        return str(path)

    def get_repo_discussions(self, repo_id, repo_type):
        self._check("threads", (repo_type, repo_id))
        yield from self.threads.get((repo_type, repo_id), [])

    def get_discussion_details(self, repo_id, repo_type, discussion_num):
        self._check("details", (repo_type, repo_id, discussion_num))
        return next(
            t for t in self.threads.get((repo_type, repo_id), []) if t.num == discussion_num
        )

    def whoami(self):
        self._check("whoami", "owner")
        return self.identity


@pytest.fixture
def hub(tmp_path):
    return FakeHub(tmp_path)


def sync(hub, state, **kwargs):
    return list(sync_hub(hub, state, owners=("owner",), **kwargs))


def test_first_sync_and_unchanged_sync(hub):
    state = {}
    rows = sync(hub, state)
    assert {r["id"] for r in rows} == {
        "model:owner/model:card",
        "model:owner/model:discussion:1",
        "huggingface:scope",
    }
    assert state["model"]["last_modified"] == "2026-01-01T00:00:00+00:00"
    assert state["model"]["repos"]["owner/model"]["discussion_activity"]["1"]
    hub.calls.clear()
    assert sync(hub, state) == []
    assert not any(method == "card" for method, _ in hub.calls)


def test_card_edit_emits_once(hub):
    state = {}
    sync(hub, state)
    hub.repos["model"]["owner/model"] = info(when=LATER)
    hub.cards[("model", "owner/model")] = CARD + "\nEdited"
    rows = sync(hub, state)
    assert len(rows) == 1 and rows[0]["content"].endswith("Edited")
    assert sync(hub, state) == []


def test_old_new_repo_below_global_cursor_is_ingested(hub):
    state = {}
    hub.repos["model"]["owner/model"] = info(when=LATER)
    sync(hub, state)
    hub.repos["model"]["owner/older"] = info("owner/older")
    assert [r["id"] for r in sync(hub, state)] == ["model:owner/older:card"]


def test_discussion_activity_independent_of_card(hub):
    state = {}
    sync(hub, state)
    hub.threads[("model", "owner/model")] = [thread(content="New comment", status="closed")]
    rows = sync(hub, state)
    assert [r["id"] for r in rows] == ["model:owner/model:discussion:1"]
    assert "New comment" in rows[0]["content"]
    assert "First comment" not in rows[0]["content"]
    assert "Status: closed" in rows[0]["content"]
    assert sync(hub, state) == []


def test_hidden_comment_text_is_removed(hub):
    state = {}
    sync(hub, state)
    hub.threads[("model", "owner/model")] = [thread(hidden=True)]
    row = sync(hub, state)[0]
    assert "First comment" not in row["content"]
    assert "Hidden comment" in row["content"]


def test_removed_thread_tombstoned(hub):
    state = {}
    sync(hub, state)
    hub.threads.clear()
    assert sync(hub, state) == [{"id": "model:owner/model:discussion:1", "_deleted": True}]


def test_delete_last_repo_removes_card_and_threads(hub):
    state = {}
    sync(hub, state)
    hub.repos["model"].clear()
    rows = sync(hub, state)
    assert {r["id"] for r in rows} == {
        "model:owner/model:card",
        "model:owner/model:discussion:1",
    }
    assert all(r["_deleted"] for r in rows)
    assert state["model"]["repos"] == {}


@pytest.mark.parametrize("status", [401, 403])
def test_gated_read_preserves_previous_state(hub, status):
    state = {}
    sync(hub, state)
    before = copy.deepcopy(state)
    hub.repos["model"]["owner/model"] = info(when=LATER)
    hub.failures[("card", ("model", "owner/model"))] = http_error(status)
    assert sync(hub, state) == []
    assert state == before


def test_lost_owner_access_does_not_tombstone(hub):
    state = {}
    sync(hub, state)
    before = copy.deepcopy(state)
    hub.repos["model"].clear()
    hub.identity["name"] = "unrelated"
    assert sync(hub, state) == []
    assert state == before


def test_anonymous_missing_repo_is_retained(hub):
    state = {}
    sync(hub, state)
    hub.repos["model"].clear()
    hub.failures[("whoami", "owner")] = LocalTokenNotFoundError("No token")
    assert sync(hub, state) == []
    assert "owner/model" in state["model"]["repos"]


def test_finegrained_token_does_not_prove_owner_access(hub):
    hub.identity["auth"]["accessToken"]["role"] = "fineGrained"
    assert _owner_access(hub, "owner") is False


def test_org_membership_proves_owner_access(hub):
    hub.identity["name"] = "member"
    hub.identity["orgs"] = [{"name": "owner", "roleInOrg": "read"}]
    assert _owner_access(hub, "owner") is True


@pytest.mark.parametrize("explicit", [False, True])
def test_rename_ingests_new_identity_and_tombstones_old(hub, explicit):
    state = {}
    kwargs = {"repo_ids": ("owner/model",)} if explicit else {"owners": ("owner",)}
    list(sync_hub(hub, state, **kwargs))
    hub.repos["model"] = {"owner/renamed": info("owner/renamed")}
    hub.redirects["owner/model"] = "owner/renamed"
    hub.threads[("model", "owner/renamed")] = [thread()]
    rows = list(sync_hub(hub, state, **kwargs))
    assert {r["id"] for r in rows if r["_deleted"]} == {
        "model:owner/model:card",
        "model:owner/model:discussion:1",
    }
    assert {r["id"] for r in rows if not r["_deleted"]} == {
        "model:owner/renamed:card",
        "model:owner/renamed:discussion:1",
    }


@pytest.mark.parametrize("failure_point", ["listing", "card", "threads", "details"])
def test_server_failure_publishes_no_rows_or_state(hub, failure_point):
    state = {}
    sync(hub, state)
    before = copy.deepcopy(state)
    hub.repos["model"]["owner/model"] = info(when=LATER)
    key = {
        "listing": "model",
        "card": ("model", "owner/model"),
        "threads": ("model", "owner/model"),
        "details": ("model", "owner/model", 1),
    }[failure_point]
    hub.failures[(failure_point, key)] = http_error(503)
    emitted = []
    with pytest.raises(HfHubHTTPError):
        for row in sync_hub(hub, state, owners=("owner",)):
            emitted.append(row)
    assert emitted == []
    assert state == before


def test_metadata_updates_do_not_emit_graph_documents(hub):
    state = {}
    sync(hub, state)
    hub.repos["model"]["owner/model"].downloads = 999
    hub.repos["model"]["owner/model"].likes = 100
    assert sync(hub, state) == []
    assert state["model"]["repos"]["owner/model"]["metadata"]["downloads"] == 999


def test_all_repo_types_have_distinct_ids_and_urls(hub):
    hub.repos["dataset"]["owner/model"] = info(repo_type="dataset")
    hub.repos["space"]["owner/model"] = info(repo_type="space")
    state = {}
    rows = sync(hub, state, repo_types=("model", "dataset", "space"))
    cards = {r["id"]: r for r in rows if r["id"].endswith(":card")}
    assert len(cards) == 3
    assert cards["dataset:owner/model:card"]["url"] == "https://huggingface.co/datasets/owner/model"
    assert cards["space:owner/model:card"]["url"] == "https://huggingface.co/spaces/owner/model"
    assert all(repo_type in state for repo_type in ("model", "dataset", "space"))


def test_filters_and_all_listing_pages_consumed(hub):
    def pages(**kwargs):
        assert kwargs["author"] == "owner"
        assert kwargs["search"] == "example"
        assert kwargs["filter"] == ["text-generation", "transformers"]
        assert "cardData" in kwargs["expand"]
        yield info("owner/first")
        yield info("owner/second")

    hub.list_models = pages
    rows = sync(hub, {}, search="example", filters=("text-generation", "transformers"))
    assert len([r for r in rows if r["id"].endswith(":card")]) == 2


def test_later_page_failure_cannot_delete(hub):
    state = {}
    sync(hub, state)
    before = copy.deepcopy(state)

    def broken(**kwargs):
        yield info("owner/first")
        raise http_error(500)

    hub.list_models = broken
    with pytest.raises(HfHubHTTPError):
        sync(hub, state)
    assert state == before


def test_rate_limit_honors_retry_after_on_later_page(monkeypatch):
    calls = []
    delays = []

    def pages():
        calls.append(1)
        yield "first"
        if len(calls) == 1:
            raise http_error(429, {"Retry-After": "2"})
        yield "second"

    monkeypatch.setattr(
        "cognee_community_connector_huggingface.huggingface.time.sleep", delays.append
    )
    assert _request(pages, materialize=True) == ["first", "second"]
    assert delays == [2.0]


def test_retry_exhaustion_and_retry_date(monkeypatch):
    monkeypatch.setattr(
        "cognee_community_connector_huggingface.huggingface.time.sleep", lambda _: None
    )
    method = Mock(side_effect=http_error(429))
    with pytest.raises(HfHubHTTPError):
        _request(method)
    assert method.call_count == 5
    assert (
        _retry_after(SimpleNamespace(headers={"Retry-After": "Thu, 01 Jan 1970 00:00:00 GMT"}), 0)
        == 0
    )


def test_selection_change_rejected(hub):
    state = {}
    sync(hub, state)
    with pytest.raises(ValueError, match="Selection changed"):
        list(sync_hub(hub, state, repo_ids=("other/model",)))


def test_relationships_survive_document_conversion():
    from cognee.tasks.ingestion.resolve_dlt_sources import _build_document_data_item

    row = card_to_row("model", "owner/model", CARD)
    item = _build_document_data_item(
        SimpleNamespace(row_data=row), uuid5(NAMESPACE_OID, "card"), "huggingface"
    )
    assert item.external_metadata["source"] == "huggingface"
    assert item.external_metadata["external_id"] == "model:owner/model:card"
    assert "owner owns the model repository owner/model" in item.data
    assert "owner/model is based on model owner/base" in item.data
    assert "owner/model uses dataset owner/data" in item.data
    discussion = discussion_to_row("model", "owner/model", thread())
    assert "belongs to model repository owner/model" in discussion["content"]


def test_real_dlt_merge_edit_and_delete_last_repo(hub, tmp_path, monkeypatch):
    import dlt

    monkeypatch.setenv("RUNTIME__DLTHUB_TELEMETRY", "false")
    pipeline = dlt.pipeline(
        pipeline_name="hub_test",
        pipelines_dir=str(tmp_path / "state"),
        destination=dlt.destinations.sqlalchemy(f"sqlite:///{tmp_path / 'hub.db'}"),
        dataset_name="wiki",
    )

    def run():
        pipeline.run(huggingface_source(owners=["owner"], client=hub))
        with (
            pipeline.sql_client() as sql,
            sql.execute_query("SELECT id, content FROM huggingface_documents") as cursor,
        ):
            return dict(cursor.fetchall())

    initial = run()
    assert "model:owner/model:card" in initial
    hub.repos["model"]["owner/model"] = info(when=LATER)
    hub.cards[("model", "owner/model")] = CARD + "\nNew body"
    assert run()["model:owner/model:card"].endswith("New body")
    hub.repos["model"].clear()
    assert set(run()) == {"huggingface:scope"}


def test_cognee_document_routing_and_deferred_orphan_cleanup(hub, monkeypatch):
    module = importlib.import_module("cognee.tasks.ingestion.resolve_dlt_sources")
    row = card_to_row("model", "owner/model", CARD)
    staged = SimpleNamespace(
        row_data=row,
        table_name="huggingface_documents",
        primary_key_value=row["id"],
        content_hash="stable",
    )
    calls = []

    async def ingest(*args, **kwargs):
        assert kwargs["write_disposition"] == "merge"
        assert kwargs["max_rows_per_table"] == 0
        return [staged]

    async def unique(identifier, user):
        return uuid5(NAMESPACE_OID, identifier)

    async def cleanup(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(module, "ingest_dlt_source", ingest)
    monkeypatch.setattr(module, "get_unique_data_id", unique)
    monkeypatch.setattr(module, "_delete_dlt_orphans", cleanup)

    async def check():
        items, deferred = await module.resolve_dlt_sources(
            huggingface_source(owners=["owner"], client=hub),
            "wiki",
            SimpleNamespace(),
            write_disposition="merge",
        )
        assert len(items) == 1 and items[0].external_metadata["source"] == "huggingface"
        assert calls == []
        await deferred()
        assert calls[0][1]["sources"] == ("huggingface",)

    asyncio.run(check())
