"""Offline model-card rendering and document-source contract tests."""

from unittest.mock import Mock

import pytest

from cognee_community_connector_huggingface.huggingface import (
    huggingface_source,
    iter_model_cards,
    model_card_to_row,
)

CARD = """---
license: mit
base_model: owner/base
datasets: [owner/training]
language: [en, fr]
tags: [text-generation]
pipeline_tag: text-generation
library_name: transformers
widget:
  - text: secret-widget-example
model-index:
  - name: ignored-evaluation
---
# Small model
A useful model card.
"""


def test_card_renders_allowed_fields_and_body_only():
    row = model_card_to_row("owner/model", CARD)
    assert row["id"] == "model:owner/model:card"
    assert row["url"] == "https://huggingface.co/owner/model"
    for expected in (
        "Owner: owner",
        "license: mit",
        "base_model: owner/base",
        "datasets: owner/training",
        "language: en, fr",
        "tags: text-generation",
        "pipeline_tag: text-generation",
        "library_name: transformers",
        "A useful model card.",
    ):
        assert expected in row["content"]
    assert "secret-widget-example" not in row["content"]
    assert "ignored-evaluation" not in row["content"]
    assert "---" not in row["content"]


def test_plain_markdown_card():
    assert model_card_to_row("owner/model", "# Hello\nBody")["content"].endswith("# Hello\nBody")


def test_nested_allowed_metadata_is_not_dumped():
    row = model_card_to_row("owner/model", "---\nbase_model:\n  nested: hidden\n---\nBody")
    assert "hidden" not in row["content"]
    assert "Body" in row["content"]


def test_only_readme_downloaded_and_duplicate_ids_deduplicated(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(CARD, encoding="utf-8")
    client = Mock()
    client.hf_hub_download.return_value = str(readme)
    rows = list(iter_model_cards(client, ["owner/model", "owner/model"]))
    assert len(rows) == 1
    client.hf_hub_download.assert_called_once_with(
        repo_id="owner/model", filename="README.md", repo_type="model"
    )


def test_download_error_propagates():
    client = Mock()
    client.hf_hub_download.side_effect = RuntimeError("unavailable")
    with pytest.raises(RuntimeError, match="unavailable"):
        list(iter_model_cards(client, ["owner/model"]))


@pytest.mark.parametrize("repo_ids", [[], "owner/model", ["invalid/repo/name"]])
def test_invalid_selection_rejected(repo_ids):
    with pytest.raises(ValueError):
        huggingface_source(repo_ids, client=Mock())


def test_source_is_merge_resource_with_document_marker(tmp_path):
    from cognee.tasks.ingestion.dlt_utils import document_source_tag

    readme = tmp_path / "README.md"
    readme.write_text(CARD, encoding="utf-8")
    client = Mock()
    client.hf_hub_download.return_value = str(readme)
    client.repo_info.return_value = {"id": "owner/model"}
    client.get_repo_discussions.return_value = []
    resource = huggingface_source(["owner/model"], client=client)
    assert document_source_tag(resource) == "huggingface"
    schema = resource.compute_table_schema()
    disposition = schema["write_disposition"]
    if isinstance(disposition, dict):
        disposition = disposition["disposition"]
    assert disposition == "merge"
    assert schema["columns"]["id"]["primary_key"] is True
    rows = list(resource)
    assert rows[0]["content"].endswith("A useful model card.")
    assert schema["columns"]["_deleted"]["hard_delete"] is True
