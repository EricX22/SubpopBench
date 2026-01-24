import argparse
import re
from pathlib import Path
from collections import defaultdict

RUN_RE = re.compile(r"^(?P<dataset>.+)_(?P<algo>.+)_hparams(?P<hp>\d+)_seed(?P<seed>\d+)$")

# Matches the training table rows you pasted, e.g.:
# 11000         7.2986        0.0620        0.9555        0.3611        0.9531        0.3297
ROW_RE = re.compile(
    r"^\s*(?P<step>\d+)\s+"
    r"(?P<epoch>[0-9.]+)\s+"
    r"(?P<loss>[0-9.eE+-]+)\s+"
    r"(?P<te_avg>[0-9.]+)\s+"
    r"(?P<te_worst>[0-9.]+)\s+"
    r"(?P<va_avg>[0-9.]+)\s+"
    r"(?P<va_worst>[0-9.]+)\s*$"
)

def find_logs(run_dir: Path):
    # common filenames in SubpopBench runs
    candidates = [
        run_dir / "out.txt",
        run_dir / "stdout.txt",
        run_dir / "log.txt",
        run_dir / "err.txt",
        run_dir / "stderr.txt",
    ]
    # plus: slurm logs might be inside run dir in some setups
    candidates += list(run_dir.glob("*.out")) + list(run_dir.glob("*.err"))
    return [p for p in candidates if p.exists() and p.is_file()]

def parse_last_metrics_from_text(text: str):
    last = None
    for line in text.splitlines():
        m = ROW_RE.match(line)
        if m:
            d = {k: float(v) if k != "step" else int(v) for k, v in m.groupdict().items()}
            last = d
    return last

def parse_run(run_dir: Path):
    # "done markers" (SubpopBench varies)
    done = (run_dir / "done").exists() or (run_dir / "final_results.pkl").exists()

    # parse logs for latest metrics
    best = None
    for logp in find_logs(run_dir):
        try:
            txt = logp.read_text(errors="ignore")
        except Exception:
            continue
        m = parse_last_metrics_from_text(txt)
        if m:
            best = m  # take latest found; often same across logs
    return done, best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--select", default="va_worst", choices=["va_worst", "te_worst", "va_avg", "te_avg"])
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    root = Path(args.root)
    run_dirs = []
    for p in root.rglob("*"):
        if p.is_dir() and RUN_RE.match(p.name):
            run_dirs.append(p)

    print(f"ROOT: {root}")
    print(f"Run dirs (recursive): {len(run_dirs)}")

    status_counts = defaultdict(int)
    per_algo = defaultdict(list)

    for rd in run_dirs:
        m = RUN_RE.match(rd.name)
        dataset, algo = m.group("dataset"), m.group("algo")
        hp, seed = int(m.group("hp")), int(m.group("seed"))
        done, metrics = parse_run(rd)

        if done:
            status = "done"
        elif metrics is not None:
            status = "incomplete"
        else:
            status = "empty"
        status_counts[status] += 1

        rec = {
            "dir": str(rd),
            "dataset": dataset,
            "algo": algo,
            "hp": hp,
            "seed": seed,
            "status": status,
            "metrics": metrics,
        }
        per_algo[algo].append(rec)

    print("Status counts:")
    for k in ["done", "incomplete", "empty"]:
        if status_counts[k]:
            print(f"  {k}: {status_counts[k]}")

    key = args.select
    def score(rec):
        m = rec["metrics"]
        if not m:
            return float("-inf")
        return float(m[key])

    print("\n=== Best-so-far per algorithm (selected by %s) ===" % key)
    print("ALGO       N   DONE  BEST te_worst  BEST te_avg  BEST va_worst  RUN")
    print("-"*78)
    for algo, recs in sorted(per_algo.items()):
        done_n = sum(r["status"] == "done" for r in recs)
        best_rec = max(recs, key=score)
        m = best_rec["metrics"] or {}
        print(f"{algo:8s} {len(recs):3d} {done_n:6d} "
              f"{m.get('te_worst', float('nan')):12.4f} "
              f"{m.get('te_avg', float('nan')):10.4f} "
              f"{m.get('va_worst', float('nan')):12.4f} "
              f"{Path(best_rec['dir']).name}")

    # optionally: topk per algo
    for algo, recs in sorted(per_algo.items()):
        ranked = sorted(recs, key=score, reverse=True)
        ranked = [r for r in ranked if r["metrics"] is not None][:args.topk]
        if not ranked:
            continue
        print(f"\n=== Top {args.topk} {algo} runs by {key} ===")
        for r in ranked:
            m = r["metrics"]
            print(f"  {key}={m[key]:.4f} te_worst={m['te_worst']:.4f} te_avg={m['te_avg']:.4f} "
                  f"va_worst={m['va_worst']:.4f} hp={r['hp']} seed={r['seed']} "
                  f"state={r['status'].upper()} dir={Path(r['dir']).name}")

if __name__ == "__main__":
    main()
