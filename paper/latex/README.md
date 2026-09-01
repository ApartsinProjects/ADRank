# LaTeX two-column build

Professional two-column PDF (`docs/adrank-2col.pdf`), built from the paper HTML.

Regenerate:
1. `python paper/build_html.py` (docs/index.html)
2. `python <html2tex skill>/scripts/convert_to_tex.py --input docs/index.html --out-dir _tex/ --columns 2`
3. `python paper/latex/build_latex.py` (wraps wide figures full-width, builds the bibliography, writes main.tex)
4. `pdflatex main.tex` (twice), from `_tex/` — needs `html2tex_compat.tex` and `figures/`.

main.tex is self-contained apart from `html2tex_compat.tex` (float/table fixes) and `figures/`.
