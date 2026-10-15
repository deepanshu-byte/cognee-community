"""Hugging Face cards and discussions as incrementally merged cognee documents."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import dlt
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR
from huggingface_hub import HfApi, RepoCard
from huggingface_hub.errors import EntryNotFoundError, LocalTokenNotFoundError
from huggingface_hub.utils import validate_repo_id

logger = logging.getLogger(__name__)
_TYPES = {"model": "models", "dataset": "datasets", "space": "spaces"}
_CARD_FIELDS = (
    "license",
    "base_model",
    "datasets",
    "language",
    "tags",
    "pipeline_tag",
    "library_name",
)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def _stamp(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _status(exc: Exception) -> int | None:
    return getattr(getattr(exc, "response", None), "status_code", None)


def _retry_after(response: Any, attempt: int) -> float:
    header = (getattr(response, "headers", {}) or {}).get("Retry-After")
    if header is not None:
        try:
            return max(0.0, float(header))
        except ValueError:
            try:
                return max(0.0, (parsedate_to_datetime(header) - datetime.now(UTC)).total_seconds())
            except (ValueError, TypeError):
                pass
    return float(2**attempt)


def _request(method, *, materialize: bool = False, **kwargs):
    """Retry paginated reads from the start so later-page 429s cannot truncate a sweep."""
    for attempt in range(5):
        try:
            result = method(**kwargs)
            return list(result) if materialize else result
        except Exception as exc:
            if _status(exc) != 429 or attempt == 4:
                raise
            delay = _retry_after(exc.response, attempt)
            logger.warning("Hub rate limit; retrying in %.1f seconds", delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def _url(repo_type: str, repo_id: str) -> str:
    prefix = "" if repo_type == "model" else f"{_TYPES[repo_type]}/"
    return f"https://huggingface.co/{prefix}{repo_id}"


def _readable_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return ", ".join(item for item in value if isinstance(item, str))
    return ""


def card_to_row(repo_type: str, repo_id: str, raw_card: str) -> dict[str, Any]:
    """Render allowlisted front matter and explicit relationships, never raw YAML."""
    card = RepoCard(raw_card)
    metadata = card.data.to_dict()
    owner = repo_id.split("/", 1)[0]
    lines = [
        f"Repository: {repo_id} ({repo_type})",
        f"Owner: {owner}",
        f"{owner} owns the {repo_type} repository {repo_id}.",
    ]
    for field in _CARD_FIELDS:
        value = _readable_value(metadata.get(field))
        if value:
            lines.append(f"{field}: {value}")
    for field, relation, target_type in (
        ("base_model", "is based on model", "model"),
        ("datasets", "uses dataset", "dataset"),
    ):
        values = metadata.get(field, [])
        refs = [values] if isinstance(values, str) else values if isinstance(values, list) else []
        for ref in refs:
            if isinstance(ref, str):
                lines.append(f"{repo_id} {relation} {ref} ({_url(target_type, ref)}).")
    if card.text.strip():
        lines.extend(["", card.text.strip()])
    return {
        "id": f"{repo_type}:{repo_id}:card",
        "title": repo_id,
        "url": _url(repo_type, repo_id),
        "content": "\n".join(lines),
        "_deleted": False,
    }


def model_card_to_row(repo_id: str, raw_card: str) -> dict[str, Any]:
    """Render a model card (convenience wrapper)."""
    return card_to_row("model", repo_id, raw_card)


def _read_card(client: Any, repo_type: str, repo_id: str) -> dict[str, Any]:
    try:
        filename = _request(
            client.hf_hub_download, repo_id=repo_id, filename="README.md", repo_type=repo_type
        )
        raw = Path(filename).read_text(encoding="utf-8")
    except EntryNotFoundError:
        # A valid repo can have no card. Retain its identity and owner relation.
        raw = ""
    return card_to_row(repo_type, repo_id, raw)


def iter_model_cards(client: Any, repo_ids: Sequence[str]) -> Iterator[dict[str, Any]]:
    for repo_id in dict.fromkeys(repo_ids):
        yield _read_card(client, "model", repo_id)


def _event_text(event: Any) -> str:
    if _get(event, "hidden", False):
        return "Hidden comment"
    parts = [
        str(_get(event, "type", "event")),
        str(_get(event, "author", "unknown")),
        _stamp(_get(event, "created_at")),
    ]
    # Exclude raw payloads and edit history: removed comments must not survive
    # in a previous-version field on the re-rendered thread document.
    for key in ("content", "new_status", "old_title", "new_title", "summary", "oid"):
        value = _get(event, key)
        if value is not None:
            parts.append(f"{key}: {value}")
    return " | ".join(parts)


def discussion_to_row(repo_type: str, repo_id: str, discussion: Any) -> dict[str, Any]:
    number = _get(discussion, "num")
    title = _get(discussion, "title", "")
    lines = [
        f"Thread #{number} belongs to {repo_type} repository {repo_id} "
        f"({_url(repo_type, repo_id)}).",
        f"Title: {title}",
        f"Status: {_get(discussion, 'status', '')}",
        f"Author: {_get(discussion, 'author', '')}",
        f"Pull request: {bool(_get(discussion, 'is_pull_request', False))}",
    ]
    lines.extend(_event_text(event) for event in _get(discussion, "events", []))
    return {
        "id": f"{repo_type}:{repo_id}:discussion:{number}",
        "title": f"{repo_id} #{number}: {title}",
        "url": f"{_url(repo_type, repo_id)}/discussions/{number}",
        "content": "\n".join(lines),
        "_deleted": False,
    }


def _fingerprint(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def _metadata(info: Any) -> dict:
    # State metadata is kept outside document rows so counts/timestamps cannot
    # change cognee's row hashes or become graph entities.
    return {
        "downloads": _get(info, "downloads"),
        "likes": _get(info, "likes"),
        "created_at": _stamp(_get(info, "created_at")),
        "last_modified": _stamp(_get(info, "last_modified")),
    }


def _owner_access(client: Any, owner: str) -> bool:
    """Prove continuing owner access using broad read/write credentials.

    Anonymous/fine-grained tokens cannot prove owner access from a missing repo
    response. Their ambiguous deletions are conservatively deferred.
    """
    try:
        identity = _request(client.whoami)
    except LocalTokenNotFoundError:
        return False
    except Exception as exc:
        if _status(exc) in (401, 403):
            return False
        raise
    token_info = identity.get("auth", {}).get("accessToken", {})
    if token_info.get("role") not in ("read", "write"):
        return False
    return identity.get("name") == owner or any(
        org.get("name") == owner
        and org.get("roleInOrg") in ("read", "contributor", "write", "admin")
        for org in identity.get("orgs", [])
    )


def _repo_info(client: Any, repo_type: str, repo_id: str):
    return _request(client.repo_info, repo_id=repo_id, repo_type=repo_type)


def _list_repos(
    client: Any,
    repo_type: str,
    owners: tuple,
    search: str | None,
    filters: tuple,
    task: str | None = None,
    library: str | None = None,
) -> list:
    method = getattr(client, f"list_{_TYPES[repo_type]}")
    expand = ["cardData", "lastModified", "createdAt", "likes"]
    if repo_type != "space":
        expand.append("downloads")
    result = []
    for owner in owners or (None,):
        kwargs: dict[str, Any] = {"expand": expand}
        if owner:
            kwargs["author"] = owner
        if search:
            kwargs["search"] = search
        selected_filters = list(filters)
        if library:
            selected_filters.append(library)
        if task:
            if repo_type == "model":
                kwargs["pipeline_tag"] = task
            elif repo_type == "dataset":
                kwargs["task_categories"] = task
            else:
                selected_filters.append(task)
        if selected_filters:
            kwargs["filter"] = selected_filters
        result.extend(_request(method, materialize=True, **kwargs))
    return result


def _sync_repo(
    client: Any, repo_type: str, repo_id: str, info: Any, old: dict, include_discussions: bool
) -> tuple[dict, list[dict]]:
    entry = copy.deepcopy(old)
    documents = dict(entry.get("documents", {}))
    pending = []
    when = _stamp(_get(info, "last_modified"))
    if not old or not when or when != old.get("last_modified"):
        card = _read_card(client, repo_type, repo_id)
        digest = _fingerprint(card)
        if documents.get(card["id"]) != digest:
            pending.append(card)
        documents[card["id"]] = digest
    if include_discussions:
        threads = _request(
            client.get_repo_discussions, materialize=True, repo_id=repo_id, repo_type=repo_type
        )
        live_threads = set()
        activity = {}
        for thread in threads:
            # Listings have no reliable last-activity field. Details must be
            # checked even when the card has not changed; emit changed text only.
            details = _request(
                client.get_discussion_details,
                repo_id=repo_id,
                repo_type=repo_type,
                discussion_num=_get(thread, "num"),
            )
            row = discussion_to_row(repo_type, repo_id, details)
            live_threads.add(row["id"])
            times = [_stamp(_get(details, "created_at"))]
            for event in _get(details, "events", []):
                times.append(_stamp(_get(event, "created_at")))
                if _get(event, "edited", False):
                    times.append(_stamp(_get(event, "last_edited_at")))
            activity[str(_get(thread, "num"))] = max(times)
            digest = _fingerprint(row)
            if documents.get(row["id"]) != digest:
                pending.append(row)
            documents[row["id"]] = digest
        for doc_id in list(documents):
            if ":discussion:" in doc_id and doc_id not in live_threads:
                pending.append({"id": doc_id, "_deleted": True})
                del documents[doc_id]
        entry["discussion_activity"] = activity
    entry.update(documents=documents, last_modified=when, metadata=_metadata(info))
    return entry, pending


def sync_hub(
    client: Any,
    state: dict,
    *,
    repo_ids: tuple = (),
    owners: tuple = (),
    repo_types: tuple = ("model",),
    search: str | None = None,
    filters: tuple = (),
    include_discussions: bool = True,
    task: str | None = None,
    library: str | None = None,
) -> Iterator[dict]:
    """Stage a full run before emitting changes; failures leave state unmodified."""
    scope = json.dumps(
        [repo_ids, owners, repo_types, search, filters, include_discussions, task, library]
    )
    if state.get("scope", scope) != scope:
        raise ValueError("Selection changed; use a separate dataset/pipeline for each selection.")
    next_state = copy.deepcopy(state)
    next_state["scope"] = scope
    pending: list[dict] = []
    for repo_type in repo_types:
        previous = state.get(repo_type, {}).get("repos", {})
        current = copy.deepcopy(previous)
        listed = (
            _list_repos(client, repo_type, owners, search, filters, task, library)
            if (owners or search or filters or task or library)
            else []
        )
        infos = {_get(info, "id"): info for info in listed}
        protected: set[str] = set()
        redirects: set[str] = set()
        for repo_id in repo_ids:
            try:
                info = _repo_info(client, repo_type, repo_id)
                canonical = _get(info, "id")
                infos[canonical] = info
                if canonical != repo_id:
                    redirects.add(repo_id)
            except Exception as exc:
                status = _status(exc)
                if status not in (401, 403, 404):
                    raise
                logger.warning("Hub: cannot read %s %s (HTTP %s)", repo_type, repo_id, status)
                if status != 404 or not _owner_access(client, repo_id.split("/")[0]):
                    protected.add(repo_id)
        for repo_id, info in infos.items():
            try:
                entry, rows = _sync_repo(
                    client, repo_type, repo_id, info, previous.get(repo_id, {}), include_discussions
                )
            except Exception as exc:
                if _status(exc) not in (401, 403):
                    raise
                protected.add(repo_id)
                logger.warning(
                    "Hub: skipping inaccessible %s %s (HTTP %s)", repo_type, repo_id, _status(exc)
                )
                continue
            current[repo_id] = entry
            pending.extend(rows)
        for missing in set(previous) - set(infos) - protected:
            renamed = missing in redirects
            if renamed:
                # Do not delete the old identity if fetching the target failed.
                canonical = _get(_repo_info(client, repo_type, missing), "id")
                if canonical in protected:
                    continue
            else:
                try:
                    info = _repo_info(client, repo_type, missing)
                    canonical = _get(info, "id")
                    if canonical == missing:
                        # No longer matching a search/tag is not a deletion.
                        continue
                    if canonical in protected:
                        continue
                    if canonical not in infos:
                        entry, rows = _sync_repo(
                            client,
                            repo_type,
                            canonical,
                            info,
                            previous.get(canonical, {}),
                            include_discussions,
                        )
                        current[canonical] = entry
                        pending.extend(rows)
                except Exception as exc:
                    if _status(exc) in (401, 403):
                        logger.warning("Hub: retaining inaccessible repo %s", missing)
                        continue
                    if _status(exc) != 404:
                        raise
                    if not _owner_access(client, missing.split("/")[0]):
                        logger.warning(
                            "Hub: retaining %s; continuing owner access is unproven", missing
                        )
                        continue
            for doc_id in previous[missing].get("documents", {}):
                pending.append({"id": doc_id, "_deleted": True})
            current.pop(missing, None)
        next_state[repo_type] = {
            "repos": current,
            "last_modified": max(
                (entry.get("last_modified", "") for entry in current.values()), default=""
            ),
        }
    # Cognee 1.4.0 skips orphan cleanup for empty readbacks. A stable scope
    # document keeps the readback non-empty so deleting the final repo works.
    if not state.get("manifest"):
        pending.append(
            {
                "id": "huggingface:scope",
                "title": "Hugging Face sync scope",
                "url": "https://huggingface.co",
                "content": f"Selected Hub scope: {scope}",
                "_deleted": False,
            }
        )
    next_state["manifest"] = True
    yield from pending
    state.clear()
    state.update(next_state)


def huggingface_source(
    repo_ids: Sequence[str] = (),
    *,
    owners: Sequence[str] = (),
    repo_types: Sequence[str] = ("model",),
    search: str | None = None,
    task: str | None = None,
    library: str | None = None,
    tags: Sequence[str] = (),
    include_discussions: bool = True,
    token: str | None = None,
    client: Any = None,
):
    """Create a document-mode dlt resource with merge, state and hard deletes.

    Select explicit IDs, owners, or search/filter criteria. Explicit IDs apply
    to each selected type; selectors form a union. Use a separate dataset for
    each selection. Optional token defaults to HF_TOKEN then Hub credentials.
    """
    for name, value in (
        ("repo_ids", repo_ids),
        ("owners", owners),
        ("repo_types", repo_types),
        ("tags", tags),
    ):
        if isinstance(value, str):
            raise ValueError(f"{name} must be a sequence, not a string.")
    repo_ids, owners, repo_types, tags = (
        tuple(sorted(set(items))) for items in (repo_ids, owners, repo_types, tags)
    )
    if not repo_types or set(repo_types) - _TYPES.keys():
        raise ValueError("repo_types must contain model, dataset, or space.")
    if not (repo_ids or owners or search or task or library or tags):
        raise ValueError("Select at least one repo ID, owner, search, task, library, or tag.")
    for repo_id in repo_ids:
        validate_repo_id(repo_id)
    for owner in owners:
        validate_repo_id(owner)
        if "/" in owner:
            raise ValueError("owners must be user or organisation names.")
    filters = tags

    if client is None:
        client = HfApi(token=token or os.environ.get("HF_TOKEN"))

    @dlt.resource(
        name="huggingface_documents",
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def huggingface_documents():
        yield from sync_hub(
            client,
            dlt.current.resource_state(),
            repo_ids=repo_ids,
            owners=owners,
            repo_types=repo_types,
            search=search,
            filters=filters,
            include_discussions=include_discussions,
            task=task,
            library=library,
        )

    resource = huggingface_documents()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "huggingface")
    return resource
