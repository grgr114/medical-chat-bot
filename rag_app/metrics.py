from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rag_app.bm25 import BM25Index
from rag_app.config import Settings, get_settings
from rag_app.documents import ChunkDocument, load_chunks
from rag_app.embeddings import EmbeddingModel
from rag_app.llm import LLMClient
from rag_app.qdrant_store import DenseHit, QdrantStore
from rag_app.retrieval import Candidate, RAGService


DEFAULT_TOP_K = (1, 3, 5, 10, 20)


@dataclass(frozen=True)
class QuestionCase:
    question: str
    chunk_id: int
    title: str = ""
    url: str = ""


@dataclass(frozen=True)
class RankedHit:
    rank: int
    chunk_id: int
    page_id: int | None
    title: str
    source_file: str
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None


@dataclass(frozen=True)
class QueryResult:
    profile: str
    case: QuestionCase
    hits: list[RankedHit]
    latency_ms: float
    relevant_total: int


def page_id_from_source_file(source_file: str) -> int | None:
    stem = Path(source_file).stem
    try:
        return int(stem)
    except ValueError:
        return None


def load_page_index(path: Path) -> dict[int, dict[str, str]]:
    pages: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            try:
                page_id = int(str(row.get("page_id") or "").strip())
            except ValueError:
                continue
            pages[page_id] = {
                "title": str(row.get("title") or "").strip(),
                "url": str(row.get("url") or "").strip(),
            }
    return pages


def _normalize_header(name: str) -> str:
    return name.strip().lstrip("\ufeff").lower().replace(" ", "_")


def _column_lookup(fieldnames: list[str] | None) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for raw in fieldnames or []:
        aliases[_normalize_header(raw)] = raw
    return aliases


def _cell(row: dict[str, str | None], aliases: dict[str, str], *normalized_names: str) -> str:
    for name in normalized_names:
        key = aliases.get(name)
        if key is not None:
            return str(row.get(key) or "").strip()
    return ""


def load_questions(
    path: Path,
    chunks_by_id: dict[int, ChunkDocument],
    pages: dict[int, dict[str, str]],
) -> list[QuestionCase]:
    raw_text = path.read_text(encoding="utf-8-sig")
    if not raw_text.strip():
        raise ValueError(f"Empty file: {path}")
    first_line = raw_text.splitlines()[0]
    delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
    reader = csv.DictReader(io.StringIO(raw_text), delimiter=delimiter)
    fieldnames = list(reader.fieldnames or [])
    aliases = _column_lookup(fieldnames)
    if "chunk_id" not in aliases:
        raise ValueError(
            f"{path} must include a 'chunk_id' column (found columns: {fieldnames}). "
            "Save as UTF-8 CSV with comma or semicolon delimiter."
        )

    cases: list[QuestionCase] = []
    skipped_empty = 0
    for row_number, row in enumerate(reader, start=2):
        question = _cell(row, aliases, "question")
        raw_chunk_id = _cell(row, aliases, "chunk_id")
        if not question or not raw_chunk_id:
            skipped_empty += 1
            continue
        try:
            chunk_id = int(raw_chunk_id)
        except ValueError as exc:
            raise ValueError(f"Invalid chunk_id on {path}:{row_number}: {raw_chunk_id!r}") from exc
        doc = chunks_by_id.get(chunk_id)
        if doc is None:
            raise ValueError(f"Unknown chunk_id {chunk_id} in {path}:{row_number} (not in chunks file)")
        page_id = page_id_from_source_file(doc.source_file)
        page_meta = pages.get(page_id, {}) if page_id is not None else {}
        title = str(page_meta.get("title") or doc.doc_title or "").strip()
        url = str(page_meta.get("url") or "").strip()
        cases.append(
            QuestionCase(
                question=question,
                chunk_id=chunk_id,
                title=title,
                url=url,
            )
        )

    if not cases:
        raise ValueError(
            f"No usable questions in {path}. Columns: {fieldnames}. "
            f"Rows skipped (missing question or chunk_id): {skipped_empty}. "
            "Rebuild the metrics image after changing rag_app (e.g. docker compose build metrics)."
        )
    return cases


def _candidate_to_hit(candidate: Candidate, rank: int) -> RankedHit:
    return RankedHit(
        rank=rank,
        chunk_id=candidate.doc.chunk_id,
        page_id=page_id_from_source_file(candidate.doc.source_file),
        title=candidate.doc.doc_title,
        source_file=candidate.doc.source_file,
        score=candidate.rerank_score
        if candidate.rerank_score is not None
        else candidate.fused_score,
        dense_score=candidate.dense_score,
        sparse_score=candidate.sparse_score,
        rerank_score=candidate.rerank_score,
    )


def _dense_to_hit(hit: DenseHit, rank: int) -> RankedHit:
    return RankedHit(
        rank=rank,
        chunk_id=hit.doc.chunk_id,
        page_id=page_id_from_source_file(hit.doc.source_file),
        title=hit.doc.doc_title,
        source_file=hit.doc.source_file,
        score=hit.score,
        dense_score=hit.score,
    )


async def retrieve_profile(
    *,
    profile: str,
    question: str,
    service: RAGService,
    max_k: int,
) -> list[RankedHit]:
    if profile == "dense":
        vector = service.embeddings.encode_queries([question])[0]
        return [_dense_to_hit(hit, rank) for rank, hit in enumerate(service.qdrant.search(vector, max_k), start=1)]

    if profile == "sparse":
        hits = service.bm25.search(question, max_k)
        return [
            RankedHit(
                rank=rank,
                chunk_id=hit.doc.chunk_id,
                page_id=page_id_from_source_file(hit.doc.source_file),
                title=hit.doc.doc_title,
                source_file=hit.doc.source_file,
                score=hit.score,
                sparse_score=hit.score,
            )
            for rank, hit in enumerate(hits, start=1)
        ]

    if profile == "hybrid":
        original_settings = service.settings
        service.settings = original_settings.model_copy(
            update={"enable_query_rewrite": False, "enable_llm_rerank": False}
        )
        try:
            candidates = await service.retrieve(question)
        finally:
            service.settings = original_settings
        return [_candidate_to_hit(candidate, rank) for rank, candidate in enumerate(candidates[:max_k], start=1)]

    if profile == "pipeline":
        original_settings = service.settings
        service.settings = original_settings.model_copy(
            update={"enable_query_rewrite": True, "enable_llm_rerank": True}
        )
        try:
            candidates = await service.retrieve(question)
        finally:
            service.settings = original_settings
        return [_candidate_to_hit(candidate, rank) for rank, candidate in enumerate(candidates[:max_k], start=1)]

    if profile == "full":
        candidates = await service.retrieve(question)
        return [_candidate_to_hit(candidate, rank) for rank, candidate in enumerate(candidates[:max_k], start=1)]

    raise ValueError(f"Unknown profile: {profile}")


async def evaluate(
    *,
    settings: Settings,
    questions_path: Path,
    texts_path: Path,
    chunks_path: Path,
    profiles: list[str],
    top_k: tuple[int, ...],
) -> tuple[list[QueryResult], dict[int, dict[str, str]]]:
    pages = load_page_index(texts_path)
    docs = load_chunks(chunks_path)
    chunks_by_id = {doc.chunk_id: doc for doc in docs}
    cases = load_questions(questions_path, chunks_by_id, pages)

    embeddings = EmbeddingModel(settings)
    qdrant = QdrantStore(settings)
    qdrant.wait_until_ready()
    qdrant_count = qdrant.count()
    if qdrant_count <= 0:
        raise RuntimeError(
            f"Qdrant collection {settings.qdrant_collection!r} is empty. "
            "Start docker compose and index the chunks first."
        )

    service = RAGService(
        settings=settings,
        docs=docs,
        embeddings=embeddings,
        qdrant=qdrant,
        bm25=BM25Index(docs),
        llm=LLMClient(settings),
    )

    max_k = max(top_k)
    results: list[QueryResult] = []
    for profile in profiles:
        for index, case in enumerate(cases, start=1):
            started = time.perf_counter()
            hits = await retrieve_profile(
                profile=profile,
                question=case.question,
                service=service,
                max_k=max(settings.candidate_limit, max_k),
            )
            latency_ms = (time.perf_counter() - started) * 1000
            results.append(
                QueryResult(
                    profile=profile,
                    case=case,
                    hits=hits[:max_k],
                    latency_ms=latency_ms,
                    relevant_total=1,
                )
            )
            print(
                f"[{profile}] {index}/{len(cases)} chunk={case.chunk_id} "
                f"rank={first_relevant_rank(hits, case.chunk_id) or '-'} "
                f"{latency_ms:.0f} ms"
            )
    return results, pages


def first_relevant_rank(hits: list[RankedHit], chunk_id: int) -> int | None:
    for hit in hits:
        if hit.chunk_id == chunk_id:
            return hit.rank
    return None


def dcg(gains: list[float]) -> float:
    return sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))


def average_precision(hits: list[RankedHit], chunk_id: int, relevant_total: int) -> float:
    if relevant_total <= 0:
        return 0.0
    relevant_seen = 0
    precision_sum = 0.0
    for index, hit in enumerate(hits, start=1):
        if hit.chunk_id == chunk_id:
            relevant_seen += 1
            precision_sum += relevant_seen / index
    return precision_sum / min(relevant_total, len(hits))


def metrics_for_result(result: QueryResult, top_k: tuple[int, ...]) -> dict[str, float | int | str | None]:
    chunk_id = result.case.chunk_id
    first_rank = first_relevant_rank(result.hits, chunk_id)
    row: dict[str, float | int | str | None] = {
        "profile": result.profile,
        "question": result.case.question,
        "expected_chunk_id": chunk_id,
        "expected_title": result.case.title,
        "first_relevant_rank": first_rank,
        "reciprocal_rank": 1 / first_rank if first_rank else 0.0,
        "average_precision": average_precision(result.hits, chunk_id, result.relevant_total),
        "latency_ms": result.latency_ms,
        "relevant_total": result.relevant_total,
        "top1_chunk_id": result.hits[0].chunk_id if result.hits else None,
        "top1_title": result.hits[0].title if result.hits else "",
        "top1_score": result.hits[0].score if result.hits else None,
        "unique_pages_at_max_k": len({hit.page_id for hit in result.hits if hit.page_id is not None}),
    }
    relevant_scores = [hit.score for hit in result.hits if hit.chunk_id == chunk_id]
    non_relevant_scores = [hit.score for hit in result.hits if hit.chunk_id != chunk_id]
    row["best_relevant_score"] = max(relevant_scores) if relevant_scores else None
    row["best_non_relevant_score"] = max(non_relevant_scores) if non_relevant_scores else None
    if relevant_scores and non_relevant_scores:
        row["score_margin"] = max(relevant_scores) - max(non_relevant_scores)
    else:
        row["score_margin"] = None

    for k in top_k:
        window = result.hits[:k]
        relevant_count = sum(1 for hit in window if hit.chunk_id == chunk_id)
        gains = [1.0 if hit.chunk_id == chunk_id else 0.0 for hit in window]
        ideal_relevant = min(result.relevant_total, k)
        ideal = [1.0] * ideal_relevant + [0.0] * (k - ideal_relevant)
        ideal_dcg = dcg(ideal)
        row[f"hit@{k}"] = 1.0 if relevant_count else 0.0
        row[f"precision@{k}"] = relevant_count / k
        row[f"recall@{k}"] = relevant_count / result.relevant_total if result.relevant_total else 0.0
        row[f"ndcg@{k}"] = dcg(gains) / ideal_dcg if ideal_dcg else 0.0
        row[f"relevant_chunks@{k}"] = relevant_count
        row[f"unique_pages@{k}"] = len({hit.page_id for hit in window if hit.page_id is not None})
    return row


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - index) + ordered[high] * (index - low)


def aggregate(rows: list[dict[str, float | int | str | None]], top_k: tuple[int, ...]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, float | int | str | None]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["profile"])].append(row)

    summary: dict[str, dict[str, float]] = {}
    for profile, items in grouped.items():
        latencies = [float(item["latency_ms"] or 0.0) for item in items]
        ranks = [float(item["first_relevant_rank"]) for item in items if item["first_relevant_rank"]]
        margins = [float(item["score_margin"]) for item in items if item["score_margin"] is not None]
        metrics: dict[str, float] = {
            "questions": float(len(items)),
            "mrr": statistics.fmean(float(item["reciprocal_rank"] or 0.0) for item in items),
            "map": statistics.fmean(float(item["average_precision"] or 0.0) for item in items),
            "mean_first_rank": statistics.fmean(ranks) if ranks else 0.0,
            "median_first_rank": statistics.median(ranks) if ranks else 0.0,
            "latency_mean_ms": statistics.fmean(latencies),
            "latency_p50_ms": percentile(latencies, 0.50),
            "latency_p95_ms": percentile(latencies, 0.95),
            "mean_score_margin": statistics.fmean(margins) if margins else 0.0,
        }
        for k in top_k:
            for name in ("hit", "precision", "recall", "ndcg"):
                key = f"{name}@{k}"
                metrics[key] = statistics.fmean(float(item[key] or 0.0) for item in items)
        summary[profile] = metrics
    return summary


def write_outputs(
    *,
    output_dir: Path,
    rows: list[dict[str, float | int | str | None]],
    results: list[QueryResult],
    summary: dict[str, dict[str, float]],
    top_k: tuple[int, ...],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "rag_metrics_details.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with details_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    rankings = []
    for result in results:
        rankings.append(
            {
                "profile": result.profile,
                "question": result.case.question,
                "expected_chunk_id": result.case.chunk_id,
                "expected_title": result.case.title,
                "latency_ms": result.latency_ms,
                "hits": [hit.__dict__ for hit in result.hits],
            }
        )
    report = {
        "top_k": top_k,
        "summary": summary,
        "details_csv": str(details_path),
        "rankings": rankings,
    }
    (output_dir / "rag_metrics_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_summary(summary: dict[str, dict[str, float]], top_k: tuple[int, ...]) -> None:
    columns = ["profile", "mrr", "map"]
    for k in top_k:
        columns.extend([f"hit@{k}", f"ndcg@{k}"])
    columns.extend(["latency_p50_ms", "latency_p95_ms"])

    widths = {column: len(column) for column in columns}
    rendered: list[dict[str, str]] = []
    for profile, metrics in summary.items():
        row = {"profile": profile}
        for column in columns[1:]:
            value = metrics.get(column, 0.0)
            row[column] = f"{value:.3f}" if not column.endswith("_ms") else f"{value:.0f}"
        rendered.append(row)
        for column, value in row.items():
            widths[column] = max(widths[column], len(value))

    print("\nRAG metrics (retrieval / pipeline profiles)")
    print(" ".join(column.ljust(widths[column]) for column in columns))
    print(" ".join("-" * widths[column] for column in columns))
    for row in rendered:
        print(" ".join(row[column].ljust(widths[column]) for column in columns))


def parse_top_k(value: str) -> tuple[int, ...]:
    parsed = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    if not parsed or parsed[0] <= 0:
        raise argparse.ArgumentTypeError("--top-k must contain positive integers")
    return parsed


def parse_profiles(value: str) -> list[str]:
    profiles = [part.strip() for part in value.split(",") if part.strip()]
    allowed = {"dense", "sparse", "hybrid", "pipeline", "full"}
    unknown = [profile for profile in profiles if profile not in allowed]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown profile(s): {', '.join(unknown)}")
    if not profiles:
        raise argparse.ArgumentTypeError("At least one profile is required")
    return profiles


def print_worst_rows(rows: list[dict[str, float | int | str | None]], top_k: tuple[int, ...], limit: int) -> None:
    if limit <= 0:
        return
    largest_k = max(top_k)
    misses = [
        row
        for row in rows
        if float(row.get(f"hit@{largest_k}") or 0.0) == 0.0
        or float(row.get("reciprocal_rank") or 0.0) < 0.2
    ]
    misses.sort(key=lambda item: (float(item.get("reciprocal_rank") or 0.0), str(item.get("profile"))))
    if not misses:
        print("\nNo hard misses at the largest k. Nice.")
        return
    print(f"\nWorst {min(limit, len(misses))} cases")
    for row in misses[:limit]:
        print(
            f"- [{row['profile']}] chunk {row['expected_chunk_id']} rank={row['first_relevant_rank'] or '-'} "
            f"top1={row['top1_chunk_id'] or '-'} :: {row['question']}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval (and optional full pipeline with rewrite + rerank) "
        "against questions.csv chunk_id labels."
    )
    parser.add_argument("--questions", type=Path, default=Path("pipeline/questions.csv"))
    parser.add_argument("--texts", type=Path, default=Path("pipeline/texts.csv"))
    parser.add_argument("--chunks", type=Path, default=Path("pipeline/output/chunks.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("pipeline/output/metrics"))
    parser.add_argument(
        "--profiles",
        type=parse_profiles,
        default=parse_profiles("dense,sparse,hybrid,full"),
        help="Comma-separated profiles: dense, sparse, hybrid (RRF only, no LLM), "
        "pipeline (hybrid + forced query rewrite + LLM rerank; needs LLM), "
        "full (same as app: uses .env enable_query_rewrite / enable_llm_rerank).",
    )
    parser.add_argument("--top-k", type=parse_top_k, default=DEFAULT_TOP_K)
    parser.add_argument("--dense-limit", type=int, default=None)
    parser.add_argument("--sparse-limit", type=int, default=None)
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--collection", type=str, default="")
    parser.add_argument("--qdrant-url", type=str, default="")
    parser.add_argument("--worst", type=int, default=10, help="How many weak cases to print.")
    return parser


async def async_main(args: argparse.Namespace) -> int:
    settings = get_settings()
    updates: dict[str, object] = {}
    if args.collection:
        updates["qdrant_collection"] = args.collection
    if args.qdrant_url:
        updates["qdrant_url"] = args.qdrant_url
    if args.dense_limit:
        updates["dense_limit"] = args.dense_limit
    if args.sparse_limit:
        updates["sparse_limit"] = args.sparse_limit
    if args.candidate_limit:
        updates["candidate_limit"] = args.candidate_limit
    if max(args.top_k) > settings.candidate_limit and not args.candidate_limit:
        updates["candidate_limit"] = max(args.top_k)
    if updates:
        settings = settings.model_copy(update=updates)

    results, _pages = await evaluate(
        settings=settings,
        questions_path=args.questions,
        texts_path=args.texts,
        chunks_path=args.chunks,
        profiles=args.profiles,
        top_k=args.top_k,
    )
    rows = [metrics_for_result(result, args.top_k) for result in results]
    summary = aggregate(rows, args.top_k)
    write_outputs(
        output_dir=args.output_dir,
        rows=rows,
        results=results,
        summary=summary,
        top_k=args.top_k,
    )
    print_summary(summary, args.top_k)
    print_worst_rows(rows, args.top_k, args.worst)
    print(f"\nWrote metrics to {args.output_dir}")
    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
