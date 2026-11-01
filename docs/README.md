# Manuscript & Research Documentation

This directory contains the final conference paper manuscript, peer review rebuttal materials, and publication assets.

## Primary Manuscript

- **[Causal Temporal Filtering.pdf](./Causal%20Temporal%20Filtering.pdf)**: Official camera-ready conference paper PDF (6 pages, IEEE Conference double-column format).
- **[manuscript/](./manuscript/)**: Full LaTeX source package:
  - `main.tex`: Clean final manuscript LaTeX source.
  - `main-highlighted.tex`: Manuscript source with peer-review revisions highlighted.
  - `references.bib`: Audited, cross-referenced BibTeX database.
  - `IEEEtran.cls` & `IEEEtran.bst`: Official IEEE formatting files.
  - `figures/`: Vector and raster figures used in the manuscript.
  - `AUTHOR_RESPONSE_LETTER.md` / `response_letter.pdf`: Comprehensive itemized reviewer responses.

## Compiling the Manuscript

To compile the paper from source:

```bash
cd docs/manuscript
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
