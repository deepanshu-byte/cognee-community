"""Run with LLM_API_KEY set; HF_TOKEN is optional for public model cards."""

import argparse
import asyncio

import cognee

from cognee_community_connector_huggingface import huggingface_source


async def main(args):
    source = huggingface_source(
        repo_ids=args.repos,
        owners=args.owners,
        repo_types=args.repo_types,
        search=args.search,
        task=args.task,
        library=args.library,
        tags=args.tags,
        include_discussions=not args.no_discussions,
    )
    await cognee.remember(
        source,
        dataset_name=args.dataset,
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )
    answer = await cognee.search(
        query_text=args.query,
        query_type=cognee.SearchType.GRAPH_COMPLETION,
        datasets=[args.dataset],
    )
    print(answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos", nargs="+", default=[])
    parser.add_argument("--owners", nargs="+", default=[])
    parser.add_argument(
        "--repo-types", nargs="+", default=["model"], choices=["model", "dataset", "space"]
    )
    parser.add_argument("--search")
    parser.add_argument("--task")
    parser.add_argument("--library")
    parser.add_argument("--tags", nargs="+", default=[])
    parser.add_argument("--no-discussions", action="store_true")
    parser.add_argument("--dataset", default="huggingface_docs")
    parser.add_argument("--query", default="Summarize the repositories and their intended uses.")
    options = parser.parse_args()
    if not (
        options.repos
        or options.owners
        or options.search
        or options.task
        or options.library
        or options.tags
    ):
        parser.error("Select --repos, --owners, or search/filter criteria.")
    asyncio.run(main(options))
