# Two-column LaTeX build (elsarticle)

Professional Elsevier-style two-column PDF (`docs/adrank-2col-latex.pdf`), built
from `docs/index.html` via the html2tex skill (TwoColPaper route A).

Regenerate:
1. `python paper/build_html.py`            # docs/index.html
2. `python paper/latex/build_elsarticle.py`  # -> docs/adrank-2col-latex.pdf

`build_elsarticle.py` runs `convert_to_tex.py --columns 2` + `pack_tmlr_bundle.py
--template elsarticle`, then grafts the front matter: the abstract into the
elsarticle frontmatter, the real author block (Apartsin corresponding,
apartsin@gmail.com), `\journal{Neurocomputing}`, a manual `thebibliography` from
the 22 references, and full-width promotion of the two wide figures. `main.tex`
is the generated bundle (needs `html2tex_compat.tex` + `figures/`); compiled with
MiKTeX pdflatex.
