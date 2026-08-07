# corpus/bakeoff/harness/run_one.py
"""Ein Kandidat, ein Dokument → output.md + metrics.json (dateibasiert).

Aufruf (im passenden Kandidaten-Env):
    python harness/run_one.py --candidate eigenbau --class 01 [--role gold]
                              [--force] [--timeout 3600]

Ergebnisse landen unter results/<kandidat>/<klasse>[.gold]/ — ein
existierendes Ergebnis wird ohne --force nicht ueberschrieben, damit ein
abgebrochener Feld-Lauf nichts verwirft und Kandidaten nachtraeglich
ergaenzt werden koennen. Fehler sind ein Ergebnis: metrics.json mit
``error`` statt output.md.

Kennzahlen: Laufzeit (wall), Speicher-Spitze (ru_maxrss self+children,
plattformkorrigiert), Modell-Calls/Tokens/Kosten (vom Adapter), Warnungen.
"""

import argparse
import json
import platform
import resource
import signal
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters import ADAPTERS, Ctx  # noqa: E402
from budget import Ledger, BudgetExceeded  # noqa: E402
from manifest import CLASSES, RESULTS, input_path, result_dir  # noqa: E402


def _max_rss_mb() -> dict:
    """Peak-RSS in MB. macOS liefert Bytes, Linux KB."""
    div = 1024 * 1024 if platform.system() == "Darwin" else 1024
    return {
        "self": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / div, 1),
        "children": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / div, 1),
    }


class Timeout(RuntimeError):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, choices=sorted(ADAPTERS))
    ap.add_argument("--class", dest="class_id", required=True, choices=sorted(CLASSES))
    ap.add_argument("--role", choices=["main", "gold"], default="main")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--timeout", type=int, default=3600, help="Sekunden, 0 = aus")
    args = ap.parse_args()

    spec = ADAPTERS[args.candidate]
    cls = CLASSES[args.class_id]
    if cls["format"] not in spec["formats"]:
        print(f"SKIP: {args.candidate} kann Format {cls['format']} nicht "
              f"(deklariert: {sorted(spec['formats'])})")
        return 0

    src = input_path(args.class_id, args.role)
    if not src.exists():
        print(f"FEHLER: Eingabe fehlt: {src}", file=sys.stderr)
        return 2

    out_dir = result_dir(args.candidate, args.class_id, args.role)
    if (out_dir / "metrics.json").exists() and not args.force:
        print(f"SKIP: {out_dir} existiert (nutze --force zum Ueberschreiben)")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)

    ledger = Ledger(RESULTS / "LEDGER.json")
    ctx = Ctx(candidate=args.candidate, class_id=args.class_id, ledger=ledger)

    metrics = {
        "candidate": args.candidate,
        "class": args.class_id,
        "role": args.role,
        "input": f"{src.parent.name}/{src.name}",
        "input_bytes": src.stat().st_size,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": platform.node(),
        "platform": platform.system(),
    }

    rss_before = _max_rss_mb()
    t0 = time.monotonic()
    error = None
    result = None

    if args.timeout:
        def _on_alarm(signum, frame):
            raise Timeout(f"Timeout nach {args.timeout}s")
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(args.timeout)

    try:
        result = spec["run"](str(src), ctx)
    except BudgetExceeded as e:
        error = {"type": "budget", "message": str(e)}
    except Timeout as e:
        error = {"type": "timeout", "message": str(e)}
    except Exception as e:
        error = {"type": e.__class__.__name__, "message": str(e),
                 "trace_tail": traceback.format_exc()[-2000:]}
    finally:
        if args.timeout:
            signal.alarm(0)

    metrics["runtime_s"] = round(time.monotonic() - t0, 2)
    metrics["max_rss_mb"] = _max_rss_mb()
    metrics["rss_baseline_mb"] = rss_before

    if result is not None:
        metrics.update({
            "model_calls": result.model_calls,
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
            "cost_usd": result.cost_usd,
            "warnings": result.warnings,
            "adapter_meta": result.meta,
            "output_chars": len(result.markdown),
        })
        tmp = out_dir / "output.md.tmp"
        tmp.write_text(result.markdown, encoding="utf-8")
        tmp.replace(out_dir / "output.md")
    else:
        metrics["error"] = error
        # Teil-Kosten sind trotzdem im Ledger — hier nur der Vollstaendigkeit:
        metrics["warnings"] = ["Lauf fehlgeschlagen — Kosten ggf. im LEDGER.json"]

    tmp = out_dir / "metrics.json.tmp"
    tmp.write_text(json.dumps(metrics, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out_dir / "metrics.json")

    status = "FEHLER" if error else "OK"
    print(f"{status}: {args.candidate} × {args.class_id}({args.role}) "
          f"in {metrics['runtime_s']}s → {out_dir}")
    if error:
        print(f"  {error['type']}: {error['message'][:300]}")
    return 1 if error else 0


if __name__ == "__main__":
    sys.exit(main())
