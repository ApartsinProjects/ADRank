# -*- coding: utf-8 -*-
"""Build the elsarticle two-column LaTeX PDF for ADRank (TwoColPaper route A).

Runs html2tex (convert --columns 2 -> pack elsarticle), then grafts the
paper-specific front matter onto _tex/main.tex: abstract into the frontmatter,
real author/affiliation block with corresponding author, journal name, a
manual thebibliography from the 22 references (the house-style div.references
is not auto-extracted), full-width promotion of the two wide figures, and
Unicode/​badge cleanup. Output: _tex/main.pdf -> docs/adrank-2col.pdf.
"""
import os, re, io, json, subprocess, sys, shutil

ROOT = r"E:\Projects\Submitted\ADRank"
SKILL = r"C:\Users\apart\.claude\skills\html2tex"
SCRATCH = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\scratchpad"
PY = sys.executable


def sh(cmd):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


UNI = {'\u00fc': r'\"u', '\u00f6': r'\"o', '\u00e4': r'\"a', '\u00df': r'\ss{}',
       '\u00e9': r"\'e", '\u00e8': r'\`e', '\u00e1': r"\'a", '\u00ed': r"\'i",
       '\u00f3': r"\'o", '\u00fa': r"\'u", '\u00f1': r'\~n', '\u00e7': r'\c{c}',
       '\u00fd': r"\'y", '\u00f8': r'\o{}', '\u010d': r'\v{c}', '\u0161': r'\v{s}',
       '\u017e': r'\v{z}', '\u2013': '--', '\u2014': '---', '\u2019': "'",
       '\u2018': "'", '\u00d7': r'$\times$', '\u2208': r'$\in$', '\u2193': ''}


def esc(s):
    s = s.replace('\\', r'\textbackslash{}').replace('&', r'\&').replace('%', r'\%')
    s = s.replace('#', r'\#').replace('_', r'\_').replace('~', r'\textasciitilde{}')
    for k, v in UNI.items():
        s = s.replace(k, v)
    return s


def uni_body(s):
    for k, v in UNI.items():
        s = s.replace(k, v)
    return s


def main():
    sh([PY, os.path.join(SKILL, "scripts", "convert_to_tex.py"),
        "--input", "docs/index.html", "--out-dir", "_tex", "--columns", "2"])
    sh([PY, os.path.join(SKILL, "scripts", "pack_tmlr_bundle.py"),
        "--in-dir", "_tex", "--template", "elsarticle"])

    # references (HTML order = citation numbering) -> thebibliography
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(io.open(os.path.join(ROOT, "docs", "index.html"), encoding="utf-8").read(), "html.parser")
    refs = []
    for p in soup.select("div.references p"):
        for a in p.find_all("a"):
            a.replace_with(a.get_text())
        refs.append(" ".join(p.get_text().split()))
    assert len(refs) == 22, f"expected 22 refs, got {len(refs)}"
    biblio = ["\\begin{thebibliography}{99}\n\\setlength{\\itemsep}{2pt}"]
    for i, r in enumerate(refs, 1):
        biblio.append(f"\\bibitem{{ref{i}}} {esc(r)}")
    biblio.append("\\end{thebibliography}")
    biblio = "\n".join(biblio)

    main_p = os.path.join(ROOT, "_tex", "main.tex")
    tex = io.open(main_p, encoding="utf-8").read()

    fm_end = tex.index("\\end{frontmatter}") + len("\\end{frontmatter}")
    doc_end = tex.rindex("\\end{document}")
    body_full = tex[fm_end:doc_end]

    # abstract sits in the body under \section{Abstract}; carve it out
    m = re.search(r"\\section\{Abstract\}\\label\{abstract\}\s*(.*?)\s*\\section\{Introduction\}", body_full, re.S)
    abstract = m.group(1).strip()
    assert len(abstract) > 500, "abstract extraction failed"

    intro = body_full.index("\\section{Introduction}")
    refs_pos = body_full.index("\\section{References}")
    body_main = body_full[intro:refs_pos].rstrip()

    # strip leaked download-badge hrefs; unicode fixes
    body_main = re.sub(r"\\href\{adrank[^}]*\}\{[^}]*\}", "", body_main)
    body_main = uni_body(body_main)
    abstract = uni_body(abstract)

    # promote the two WIDE figures (fig1 two-panel, fig4 bars) to full-width figure*
    def promote(mobj):
        blk = mobj.group(0)
        if "fig1_rho" in blk or "fig4_regret" in blk:
            blk = blk.replace(r"\begin{figure}[tbp]", r"\begin{figure*}[t]").replace(r"\end{figure}", r"\end{figure*}")
        return blk
    body_main = re.sub(r"\\begin\{figure\}\[tbp\].*?\\end\{figure\}", promote, body_main, flags=re.S)

    # frontmatter grafting
    tex = tex.replace("\\begin{abstract}\n\n\\end{abstract}",
                      "\\begin{abstract}\n" + abstract + "\n\\end{abstract}")
    tex = tex.replace(
        "\\author{Anonymous Authors}\n\\address{Anonymous Affiliations}",
        "\\author[hit]{Alexander Apartsin\\corref{cor1}}\n"
        "\\ead{apartsin@gmail.com}\n"
        "\\author[afeka]{Yehudit Aperstein}\n"
        "\\cortext[cor1]{Corresponding author}\n"
        "\\address[hit]{School of Computer Science, Faculty of Sciences, Holon Institute of Technology (HIT), Holon, Israel}\n"
        "\\address[afeka]{Intelligent Systems, Afeka Academic College of Engineering, Tel-Aviv, Israel}")
    tex = tex.replace("__JOURNAL__", "Neurocomputing")
    tex = tex.replace("\\title{Ranking Anomaly Detectors from Normal Data Alone}",
                      "\\title{\\vspace*{-2\\baselineskip}Ranking Anomaly Detectors from Normal Data Alone}")

    # recompute head (frontmatter) after replacements, then splice body + biblio
    fm_end = tex.index("\\end{frontmatter}") + len("\\end{frontmatter}")
    doc_end = tex.rindex("\\end{document}")
    tex = tex[:fm_end] + "\n\n" + body_main + "\n\n" + biblio + "\n\n" + tex[doc_end:]

    io.open(main_p, "w", encoding="utf-8").write(tex)
    print(f"grafted: abstract {len(abstract)}c, 22 refs, figure* = {tex.count(chr(92)+'begin{figure*}')}")

    sh([PY, os.path.join(SKILL, "scripts", "compile_local.py"), "--in-dir", "_tex", "--auto-patch"])
    shutil.copy(os.path.join(ROOT, "_tex", "main.pdf"), os.path.join(ROOT, "docs", "adrank-2col-latex.pdf"))
    shutil.copy(main_p, os.path.join(ROOT, "paper", "latex", "main.tex"))
    print("done -> docs/adrank-2col-latex.pdf")


if __name__ == "__main__":
    main()
