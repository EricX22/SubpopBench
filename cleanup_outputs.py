#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import time
from pathlib import Path
from typing import Tuple, List

DONE_MARKERS = ("done", "final_results.pkl")
VALUE_FILES = ("out.txt", "err.txt", "args.json", "results.json", "model.pkl", "final_results.pkl", "done")
EVENT_PREFIX = "events.out.tfevents"

TRACE_PAT = re.compile(r"(Traceback \(most recent call last\):|CalledProcessError|ValueError:|RuntimeError:|AssertionError:)", re.MULTILINE)

def count_done(tag_dir: Path) -> int:
    if not tag_dir.is_dir():
        return 0
    c = 0
    for p in tag_dir.iterdir():
        if p.is_dir() and any((p / m).exists() for m in DONE_MARKERS):
            c += 1
    return c

def newest_mtime(root: Path) -> float:
    mt = 0.0
    if not root.exists():
        return mt
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            try:
                mt = max(mt, (Path(dirpath) / fn).stat().st_mtime)
            except FileNotFoundError:
                pass
    return mt

def age_hours(p: Path) -> float:
    mt = newest_mtime(p)
    if mt <= 0:
        return 1e9
    return (time.time() - mt) / 3600.0

def safe_move(src: Path, dst: Path, execute: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  MOVE: {src} -> {dst}")
    if execute:
        shutil.move(str(src), str(dst))

def safe_rmtree(p: Path, execute: bool):
    print(f"  RM:   {p}")
    if execute:
        shutil.rmtree(p)

def is_truly_empty_dir(p: Path) -> bool:
    """Strict: no files and no subdirs."""
    if not p.is_dir():
        return False
    try:
        next(p.iterdir())
        return False
    except StopIteration:
        return True

def archive_tag_folder(src: Path, archive_root: Path, execute: bool, reason: str = "dupe_tag"):
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    dest = archive_root / "DUPES__archive" / f"{src.name}__{reason}__arch_{ts}"
    print(f"  ARCHIVE TAG: {src} -> {dest}")
    if execute:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))

def choose_canonical(a: Path, b: Path) -> Tuple[Path, Path]:
    da, db = count_done(a), count_done(b)
    if da != db:
        return (a, b) if da > db else (b, a)
    na, nb = newest_mtime(a), newest_mtime(b)
    return (a, b) if na >= nb else (b, a)

def merge_runs(keep: Path, src: Path, execute: bool):
    """
    Move run dirs from src -> keep when they don't exist in keep.
    If a run dir name exists in both, we keep the canonical one and log a collision.
    """
    if not src.is_dir():
        return
    keep.mkdir(parents=True, exist_ok=True)

    for child in sorted(src.iterdir()):
        if not child.is_dir():
            continue
        target = keep / child.name
        if target.exists():
            src_score = run_quality_score(child)
            tgt_score = run_quality_score(target)

            if src_score > tgt_score:
                # archive the existing target, then move src into place
                ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                dest = (keep.parent / "DEBUG__archive" / "COLLISIONS__archive" /
                        keep.name / f"{target.name}__arch_{ts}")
                print(f"  COLLISION (src better, swap): archive {target} -> {dest}, then move {child} -> {target}")
                if execute:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(target), str(dest))
                    shutil.move(str(child), str(target))
            else:
                print(f"  COLLISION (kept existing): {target}  (src had: {child})")
            continue

        safe_move(child, target, execute)

def maybe_cleanup_dupe_tag_folder(src: Path, archive_root: Path, execute: bool):
    """
    After merging, either delete if *truly empty*, else archive the leftover tag folder.
    Never blind-rm a non-empty tag folder.
    """
    if not src.is_dir():
        return
    if is_truly_empty_dir(src):
        safe_rmtree(src, execute)
    else:
        archive_tag_folder(src, archive_root, execute, reason="post_merge_nonempty")

def find_dupe_pairs(output_root: Path) -> List[Tuple[Path, Path]]:
    names = {p.name for p in output_root.iterdir() if p.is_dir()}
    pairs = []

    for n in list(names):
        if n.endswith("_attrNo_attrNo"):
            base = n[:-len("_attrNo")]
            if base in names:
                pairs.append((output_root / base, output_root / n))
        if n.endswith("_attrYes_attrYes"):
            base = n[:-len("_attrYes")]
            if base in names:
                pairs.append((output_root / base, output_root / n))

    for n in list(names):
        if n.endswith("_attrNo") and not n.endswith("_attrNo_attrNo"):
            base = n[:-len("_attrNo")]
            if base in names:
                pairs.append((output_root / base, output_root / n))

    uniq, seen = [], set()
    for a, b in pairs:
        key = tuple(sorted([a.name, b.name]))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((a, b))
    return uniq

def has_debug_value(run_dir: Path) -> bool:
    """If it has logs/args, it's worth archiving."""
    if (run_dir / "out.txt").exists() or (run_dir / "err.txt").exists() or (run_dir / "args.json").exists():
        return True
    for p in run_dir.iterdir():
        if p.is_file() and not p.name.startswith(EVENT_PREFIX):
            return True
    return False

def looks_truly_empty(run_dir: Path) -> bool:
    """No done/final, and contains only event files (or nothing)."""
    if any((run_dir / m).exists() for m in DONE_MARKERS):
        return False
    files = [p for p in run_dir.iterdir() if p.is_file()]
    if not files:
        return True
    return all(p.name.startswith(EVENT_PREFIX) for p in files)

def extract_failure_snippet(run_dir: Path, max_lines: int = 40) -> str:
    outp = run_dir / "out.txt"
    errp = run_dir / "err.txt"
    txt = ""
    for p in (outp, errp):
        if p.exists():
            try:
                txt += p.read_text(errors="ignore") + "\n"
            except Exception:
                pass
    lines = txt.strip().splitlines()
    tail = "\n".join(lines[-max_lines:]) if lines else ""
    return tail

def cleanup_tag(tag_dir: Path, archive_root: Path, min_age_hours: float, execute: bool) -> Tuple[int,int,int]:
    """
    Returns (archived, deleted, kept_young)
    """
    archived = deleted = kept_young = 0
    if not tag_dir.is_dir():
        return (0,0,0)

    for run_dir in sorted([p for p in tag_dir.iterdir() if p.is_dir()]):
        # skip completed
        if any((run_dir / m).exists() for m in DONE_MARKERS):
            continue

        age = age_hours(run_dir)
        if age < min_age_hours:
            print(f"  KEEP (young {age:.2f}h): {run_dir}")
            kept_young += 1
            continue

        # Delete only if truly empty
        if looks_truly_empty(run_dir):
            safe_rmtree(run_dir, execute)
            deleted += 1
            continue

        # Otherwise archive
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        dest = archive_root / tag_dir.name / f"{run_dir.name}__arch_{ts}"
        snippet = extract_failure_snippet(run_dir)
        print(f"  ARCHIVE: {run_dir} -> {dest}")
        if execute:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(run_dir), str(dest))
            if snippet:
                (dest / "FAILURE_TAIL.txt").write_text(snippet)
        archived += 1

    return (archived, deleted, kept_young)


def run_quality_score(d: Path) -> Tuple[int, int, float]:
    """
    Higher is better.
    (done_present, num_value_files_present, newest_mtime)
    """
    done = int(any((d / m).exists() for m in DONE_MARKERS))
    value = 0
    for f in VALUE_FILES:
        if (d / f).exists():
            value += 1
    mt = newest_mtime(d)
    return (done, value, mt)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", type=str, default="/scratch/jrg4wx/subpop_bench/output")
    ap.add_argument("--archive_root", type=str, default="/scratch/jrg4wx/subpop_bench/output/DEBUG__archive")
    ap.add_argument("--min_age_hours", type=float, default=6.0)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--only_prefix", type=str, default="")
    args = ap.parse_args()

    out_root = Path(args.output_root).resolve()
    arch_root = Path(args.archive_root).resolve()

    print(f"OUTPUT_ROOT:  {out_root}")
    print(f"ARCHIVE_ROOT: {arch_root}")
    print(f"MODE: {'EXECUTE' if args.execute else 'DRY-RUN'}")
    print(f"min_age_hours={args.min_age_hours}")
    if args.only_prefix:
        print(f"only_prefix={args.only_prefix}")

    # 1) Merge duped tag folders
    pairs = find_dupe_pairs(out_root)
    if args.only_prefix:
        pairs = [(a,b) for (a,b) in pairs if a.name.startswith(args.only_prefix) or b.name.startswith(args.only_prefix)]

    print("\n=== MERGE DUPLICATE TAG FOLDERS ===")
    for a, b in pairs:
        keep, src = choose_canonical(a, b)
        print(f"\nPAIR: {a.name} <-> {b.name}")
        print(f"  done_count: {a.name}={count_done(a)}  {b.name}={count_done(b)}")
        print(f"  choose keep={keep.name}, merge_from={src.name}")
        merge_runs(keep, src, args.execute)
        maybe_cleanup_dupe_tag_folder(src, arch_root, args.execute)

    # 2) Cleanup runs (archive/delete)
    print("\n=== CLEANUP RUN DIRS (archive failures, delete truly empty) ===")
    tags = [p for p in out_root.iterdir() if p.is_dir()]
    if args.only_prefix:
        tags = [p for p in tags if p.name.startswith(args.only_prefix)]

    total_arch = total_del = total_keep = 0
    for tag in sorted(tags):
        # Skip the archive folder itself to avoid recursion
        if tag.resolve() == arch_root.resolve():
            continue
        arch, dele, keep = cleanup_tag(tag, arch_root, args.min_age_hours, args.execute)
        if arch or dele or keep:
            print(f"[{tag.name}] archived={arch} deleted={dele} kept_young={keep}")
        total_arch += arch
        total_del += dele
        total_keep += keep

    print(f"\nDONE. archived={total_arch} deleted={total_del} kept_young={total_keep}")
    print("Re-run summarize_progress after EXECUTE.")

if __name__ == "__main__":
    main()
