from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.lib.caption_images import run_caption_job
from pipeline.lib.expand_abbr import run_expand
from pipeline.lib.fill_context import run_fill_context
from pipeline.paths import PIPELINE_ROOT, default_layout
from pipeline.sources.base import PipelineContext
from pipeline.sources.registry import instantiate


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, value: str | None, fallback: Path) -> Path:
    if value is None or value == "":
        return fallback
    p = Path(value)
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def build_context(cfg: dict, root: Path) -> PipelineContext:
    layout = default_layout()
    paths = cfg.get("paths") or {}
    out_dir = _resolve(root, paths.get("output_dir"), layout["output_dir"])
    return PipelineContext(
        root=root.resolve(),
        texts_dir=_resolve(root, paths.get("texts"), layout["texts"]),
        images_dir=_resolve(root, paths.get("images"), layout["images"]),
        output_dir=out_dir,
        image_descriptions=_resolve(
            root,
            paths.get("image_descriptions"),
            layout["image_descriptions"],
        ),
    )


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def max_chunk_id(rows: list[dict]) -> int:
    m = 0
    for r in rows:
        try:
            m = max(m, int(r.get("chunk_id") or 0))
        except (TypeError, ValueError):
            continue
    return m


def assign_chunk_ids(rows: list[dict], start: int = 1) -> list[dict]:
    out: list[dict] = []
    n = start
    for r in rows:
        d = dict(r)
        d["chunk_id"] = n
        n += 1
        out.append(d)
    return out


def cmd_caption(args: argparse.Namespace, cfg: dict, ctx: PipelineContext) -> int:
    paths = cfg.get("paths") or {}
    layout = default_layout()
    out_dir = ctx.output_dir
    return run_caption_job(
        images_dir=ctx.images_dir,
        output_jsonl=_resolve(ctx.root, paths.get("image_descriptions"), layout["image_descriptions"]),
        checkpoint_log=out_dir / "caption_checkpoint.log",
        progress_json=out_dir / "caption_progress.json",
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
        limit=args.limit,
    )


def cmd_build(args: argparse.Namespace, cfg: dict, ctx: PipelineContext) -> int:
    paths = cfg.get("paths") or {}
    layout = default_layout()
    chunks_path = _resolve(ctx.root, paths.get("chunks"), layout["chunks"])

    sources_cfg = cfg.get("sources") or []
    selected: list[dict] = []
    if args.sources:
        want = {s.strip() for s in args.sources.split(",") if s.strip()}
        for entry in sources_cfg:
            sid = entry.get("id") or ""
            if sid in want:
                selected.append(entry)
        missing = want - {e.get("id") or "" for e in selected}
        if missing:
            print(f"Unknown source id(s): {missing}", file=sys.stderr)
            return 1
    else:
        for entry in sources_cfg:
            if entry.get("enabled", True):
                selected.append(entry)

    if not selected:
        print("No enabled sources (check config.json and --sources).", file=sys.stderr)
        return 1

    existing: list[dict] = []
    next_id = 1
    if args.append:
        if not (args.sources or "").strip():
            print(
                "Ошибка: с флагом --append нужно указать --sources id1,id2, "
                "чтобы не продублировать весь корпус.",
                file=sys.stderr,
            )
            return 1
        existing = read_jsonl(chunks_path)
        next_id = max_chunk_id(existing) + 1
        if not existing and next_id == 1:
            print(
                f"Append mode: {chunks_path} missing or empty; writing only new chunks.",
                file=sys.stderr,
            )

    combined: list[dict] = []
    for entry in selected:
        class_path = entry.get("class")
        if not class_path:
            print(f"Source {entry.get('id')!r} has no 'class' field.", file=sys.stderr)
            return 1
        src = instantiate(class_path)
        opts = dict(entry.get("options") or {})
        rows = src.collect(ctx, opts)
        combined.extend(rows)

    new_part = assign_chunk_ids(combined, start=next_id)
    if args.append:
        final = existing + new_part
    else:
        final = assign_chunk_ids(combined, start=1)

    write_jsonl(chunks_path, final)
    print(f"Wrote {len(final)} chunks ({len(new_part)} from this run) -> {chunks_path}")
    return 0


def cmd_expand(args: argparse.Namespace, cfg: dict, ctx: PipelineContext) -> int:
    paths = cfg.get("paths") or {}
    layout = default_layout()
    terms = _resolve(ctx.root, paths.get("terms"), layout["terms"])
    if not terms.is_file():
        print(f"terms.csv not found: {terms}", file=sys.stderr)
        return 1
    inp = _resolve(ctx.root, paths.get("chunks"), layout["chunks"])
    out = _resolve(ctx.root, paths.get("chunks_expanded"), layout["chunks_expanded"])
    if args.input:
        inp = Path(args.input).resolve()
    if args.output:
        out = Path(args.output).resolve()
    n = run_expand(terms, inp, out)
    print(f"Expanded {n} rows -> {out}")
    return 0


def cmd_context(args: argparse.Namespace, cfg: dict, ctx: PipelineContext) -> int:
    paths = cfg.get("paths") or {}
    layout = default_layout()
    inp = _resolve(ctx.root, paths.get("chunks_expanded"), layout["chunks_expanded"])
    if not inp.is_file():
        inp = _resolve(ctx.root, paths.get("chunks"), layout["chunks"])
    if args.input:
        inp = Path(args.input).resolve()
    out = Path(args.output).resolve() if args.output else inp
    ck = ctx.output_dir / "checkpoints"
    run_fill_context(
        input_path=inp,
        output_path=out,
        checkpoint_dir=ck,
        only_empty=not args.all,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        checkpoint_every=args.checkpoint_every,
        max_chunks=args.max_chunks,
    )
    return 0


def main() -> int:
    root = PIPELINE_ROOT
    p = argparse.ArgumentParser(description="RAG chunk pipeline: captions, build, expand, context.")
    p.add_argument(
        "--config",
        type=Path,
        default=root / "config.json",
        help="Path to config.json (default: pipeline/config.json)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("captions", help="Vision captions: pipeline/images -> output/image_descriptions.jsonl")
    pc.add_argument("--base-url", default="http://127.0.0.1:1488/v1")
    pc.add_argument("--model", default="lmstudio-community/gemma-4-e4b-it")
    pc.add_argument("--api-key", default="lm-studio")
    pc.add_argument("--max-tokens", type=int, default=4096)
    pc.add_argument("--temperature", type=float, default=0.2)
    pc.add_argument("--retries", type=int, default=3)
    pc.add_argument("--retry-sleep", type=float, default=2.0)
    pc.add_argument("--limit", type=int, default=0)

    pb = sub.add_parser("build", help="Run enabled chunk sources from config (writes output/chunks.jsonl)")
    pb.add_argument(
        "--append",
        action="store_true",
        help="Keep existing chunks.jsonl and append new rows with new chunk_id (use with --sources)",
    )
    pb.add_argument(
        "--sources",
        type=str,
        default="",
        help="Comma-separated source ids to run (default: all enabled in config)",
    )

    pe = sub.add_parser("expand", help="Expand abbreviations (terms.csv): chunks -> chunks_expanded.jsonl")
    pe.add_argument("--input", type=Path, default=None)
    pe.add_argument("--output", type=Path, default=None)

    pf = sub.add_parser("context", help="Fill chunk_context via LLM (OpenAI-compatible API)")
    pf.add_argument("--input", type=Path, default=None)
    pf.add_argument("--output", type=Path, default=None)
    pf.add_argument("--all", action="store_true", help="Regenerate all chunk_context (default: empty only)")
    pf.add_argument("--base-url", default="http://127.0.0.1:1488/v1")
    pf.add_argument("--model", default="lmstudio-community/gemma-4-e4b-it")
    pf.add_argument("--api-key", default="lm-studio")
    pf.add_argument("--checkpoint-every", type=int, default=10)
    pf.add_argument("--max-chunks", type=int, default=0)

    pr = sub.add_parser("run", help="Run several steps in order: captions, build, expand[, context]")
    pr.add_argument(
        "--steps",
        type=str,
        default="captions,build,expand",
        help="Comma-separated: captions, build, expand, context",
    )
    pr.add_argument("--skip-captions", action="store_true")
    pr.add_argument("--with-context", action="store_true", help="Include LLM context step (slow, needs API)")
    pr.add_argument("--build-append", action="store_true")
    pr.add_argument("--build-sources", type=str, default="")
    # forward common API args
    pr.add_argument("--base-url", default="http://127.0.0.1:1488/v1")
    pr.add_argument("--model", default="lmstudio-community/gemma-4-e4b-it")
    pr.add_argument("--api-key", default="lm-studio")

    args = p.parse_args()
    cfg = load_config(args.config)
    ctx = build_context(cfg, root)

    if args.cmd == "captions":
        return cmd_caption(args, cfg, ctx)
    if args.cmd == "build":
        return cmd_build(args, cfg, ctx)
    if args.cmd == "expand":
        return cmd_expand(args, cfg, ctx)
    if args.cmd == "context":
        return cmd_context(args, cfg, ctx)
    if args.cmd == "run":
        steps = [s.strip() for s in args.steps.split(",") if s.strip()]
        if args.with_context and "context" not in steps:
            steps.append("context")
        if args.skip_captions:
            steps = [s for s in steps if s != "captions"]
        ns_cap = argparse.Namespace(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            max_tokens=4096,
            temperature=0.2,
            retries=3,
            retry_sleep=2.0,
            limit=0,
        )
        ns_build = argparse.Namespace(
            append=args.build_append,
            sources=args.build_sources,
        )
        ns_exp = argparse.Namespace(input=None, output=None)
        ns_ctx = argparse.Namespace(
            input=None,
            output=None,
            all=False,
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            checkpoint_every=10,
            max_chunks=0,
        )
        for step in steps:
            if step == "captions":
                r = cmd_caption(ns_cap, cfg, ctx)
                if r != 0:
                    return r
            elif step == "build":
                r = cmd_build(ns_build, cfg, ctx)
                if r != 0:
                    return r
            elif step == "expand":
                r = cmd_expand(ns_exp, cfg, ctx)
                if r != 0:
                    return r
            elif step == "context":
                r = cmd_context(ns_ctx, cfg, ctx)
                if r != 0:
                    return r
            else:
                print(f"Unknown step: {step}", file=sys.stderr)
                return 1
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
