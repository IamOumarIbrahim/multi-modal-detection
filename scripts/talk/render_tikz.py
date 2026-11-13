#   MMSAR - render every TikZ figure in manuscript/ as a stand-alone PNG for slides
#        Oumar Ibrahim U22200741

# five of these never made it into the 6-page paper; they are still useful in the talk
# needs pdflatex and pdftoppm (both ship with MiKTeX and TeX Live)
# output  figures/tikz_<name>.png (300 dpi)

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MS = ROOT / "manuscript"
BUILD = ROOT / "figures" / "_tikz" / "build"

PREAMBLE = r"""\documentclass[border=6pt]{standalone}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{groupplots}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,calc,fit,backgrounds,decorations.pathreplacing,plotmarks,patterns}
\pgfdeclareplotmark{pentagon*}{%
  \pgfpathmoveto{\pgfpointpolar{90}{\pgfplotmarksize}}%
  \pgfpathlineto{\pgfpointpolar{162}{\pgfplotmarksize}}%
  \pgfpathlineto{\pgfpointpolar{234}{\pgfplotmarksize}}%
  \pgfpathlineto{\pgfpointpolar{306}{\pgfplotmarksize}}%
  \pgfpathlineto{\pgfpointpolar{18}{\pgfplotmarksize}}%
  \pgfpathclose
  \pgfusepathqfillstroke
}
\setlength{\columnwidth}{252pt}
\setlength{\textwidth}{516pt}
\graphicspath{{MSDIR/}}
\begin{document}
\input{MSDIR/NAME.tex}
\end{document}
"""


def main():
    for tool in ("pdflatex", "pdftoppm"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} not found")
    BUILD.mkdir(parents=True, exist_ok=True)
    ok = []
    for tex in sorted(MS.glob("figure_*.tex")):
        name = tex.stem
        src = BUILD / f"{name}.tex"
        src.write_text(PREAMBLE.replace("MSDIR", MS.as_posix()).replace("NAME", name), encoding="utf-8")
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", src.name], cwd=BUILD,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAILED {name}: see {BUILD / (name + '.log')}")
            continue
        out = ROOT / "figures" / f"tikz_{name.replace('figure_', '')}"
        subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", str(BUILD / f"{name}.pdf"), str(out)],
                       check=True)
        ok.append(out.name + ".png")
        print("saved figures/" + out.name + ".png")
    print(f"{len(ok)} figures")


if __name__ == "__main__":
    main()
