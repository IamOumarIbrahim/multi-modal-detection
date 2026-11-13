#   MMSAR - regenerate every table, figure and the PDF from the committed confidence cache
#        Oumar Ibrahim U22200741

#   python scripts/reproduce.py            # tables -> figures -> tests -> PDF (CPU, about 4 min)
#   python scripts/reproduce.py --verify   # same, then fail if any committed result changed (git diff)
#   python scripts/reproduce.py --no-pdf   # skip LaTeX
#   python scripts/reproduce.py --no-figures
#
# Only the committed runs/detect/unified_confidences_cache.json is needed. Rebuilding that cache
# from the videos (harvest -> thermal -> frames -> split -> train -> extract) is described in
# scripts/README.md and needs the raw videos and a GPU.

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(title, cmd, cwd=ROOT):
    print(f"\n=== {title}\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    if subprocess.run(cmd, cwd=cwd).returncode != 0:
        sys.exit(f"FAILED: {title}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="fail if committed results change")
    ap.add_argument("--no-pdf", action="store_true", help="skip the LaTeX build")
    ap.add_argument("--no-figures", action="store_true", help="skip the figure scripts")
    a = ap.parse_args()

    run("Unit tests", [PY, "-m", "pytest", "-q"])
    run("Tables III and IV (fixed tau*)", [PY, "scripts/evaluate_locked_thresholds.py"])
    run("Validation-F1 protocol, window ablation", [PY, "scripts/evaluate_mmsar_unified.py"])
    for objective in ["f2", "cost"]:
        run(f"Table V ({objective}, unconstrained)", [PY, "scripts/fbeta_threshold_sensitivity.py",
                                                      "--objective", objective])
        run(f"Table V ({objective}, precision >= 0.95)", [PY, "scripts/fbeta_threshold_sensitivity.py",
                                                          "--objective", objective, "--min_precision", "0.95"])
    if not a.no_figures:
        run("Figures (charts, frames, demo clip)", [PY, "scripts/make_figures.py"])

    if not a.no_pdf:
        if shutil.which("latexmk") is None:
            print("latexmk not found: skipping the PDF build")
        else:
            run("Paper PDF", ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                cwd=ROOT / "manuscript")

    if a.verify:
        run("Committed results unchanged", ["git", "diff", "--exit-code", "--stat", "--", "runs/detect"])
    print("\nAll steps passed.")


if __name__ == "__main__":
    main()
