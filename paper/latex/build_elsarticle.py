# -*- coding: utf-8 -*-
"""Build the elsarticle two-column LaTeX PDF for ADRank (TwoColPaper route A).

Runs html2tex (convert --columns 2 -> pack elsarticle), then grafts the
paper-specific front matter onto _tex/main.tex: abstract into the frontmatter,
real author/affiliation block with corresponding author, journal name,
full-width promotion of the two wide figures, journal back-matter conventions,
and Unicode/badge cleanup. The bibliography is built by the converter itself
(the div.references fix now landed in convert_to_tex.py). Output:
_tex/main.pdf -> docs/adrank-2col-latex.pdf.
"""
import os, re, io, subprocess, sys, shutil

ROOT = r"E:\Projects\Submitted\ADRank"
SKILL = r"C:\Users\apart\.claude\skills\html2tex"
PY = sys.executable

UNI = {'\u2208': r'$\in$', '\u2193': '', '\u00d7': r'$\times$'}


def sh(cmd):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def uni(s):
    for k, v in UNI.items():
        s = s.replace(k, v)
    return s


def main():
    sh([PY, os.path.join(SKILL, "scripts", "convert_to_tex.py"),
        "--input", "docs/index.html", "--out-dir", "_tex", "--columns", "2"])
    sh([PY, os.path.join(SKILL, "scripts", "pack_tmlr_bundle.py"),
        "--in-dir", "_tex", "--template", "elsarticle"])

    main_p = os.path.join(ROOT, "_tex", "main.tex")
    tex = io.open(main_p, encoding="utf-8").read()

    # carve the abstract out of the body (house-style div.abstract comes through
    # as a \section{Abstract}); re-insert it into elsarticle's frontmatter.
    m = re.search(r"\\section\{Abstract\}\\label\{abstract\}\s*(.*?)\s*\\section\{Introduction\}", tex, re.S)
    abstract = uni(m.group(1).strip())
    assert len(abstract) > 500, "abstract extraction failed"

    # drop the pre-Introduction body cruft (web badges, HTML byline, the
    # \section{Abstract} block) by splicing frontmatter straight to Introduction.
    fm_end = tex.index("\\end{frontmatter}") + len("\\end{frontmatter}")
    intro = tex.index("\\section{Introduction}")
    tex = tex[:fm_end] + "\n\n" + tex[intro:]

    # the converter's thebibliography prints its own heading; remove the phantom
    # \section{References} left by the <h2>References</h2> whose <div> was extracted.
    tex = re.sub(r"\\section\{References\}\\label\{references\}\n?", "", tex)
    # journals NUMBER Limitations (converter stars it per the ACL convention).
    tex = tex.replace("\\section*{Limitations}", "\\section{Limitations}")
    # strip any leaked download-badge hrefs; map stray unicode
    tex = re.sub(r"\\href\{adrank[^}]*\}\{[^}]*\}", "", tex)
    tex = uni(tex)

    # make long bibliography URLs breakable: xurl breaks anywhere, and inside
    # thebibliography normalize \href{U}{T} -> \url{U} and wrap bare URLs, so a
    # long arXiv/DOI/proceedings link wraps instead of running into the margin.
    tex = tex.replace("\\usepackage{newtxmath}", "\\usepackage{newtxmath}\n\\usepackage{xurl}")
    def _bib_urls(mo):
        seg = re.sub(r"\\href\{([^}]*)\}\{[^}]*\}", r"\\url{\1}", mo.group(0))
        seg = re.sub(r"(?<![{/])(https?://[^\s{}]+)", r"\\url{\1}", seg)
        return seg
    tex = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", _bib_urls, tex, flags=re.S)

    # promote the two WIDE figures (fig1 two-panel, fig4 bars) to full-width figure*
    def promote(mobj):
        blk = mobj.group(0)
        if "fig1_rho" in blk or "fig4_regret" in blk:
            blk = blk.replace(r"\begin{figure}[tbp]", r"\begin{figure*}[t]").replace(r"\end{figure}", r"\end{figure*}")
        return blk
    tex = re.sub(r"\\begin\{figure\}\[tbp\].*?\\end\{figure\}", promote, tex, flags=re.S)

    # frontmatter grafting: abstract, real author block (Aperstein corresponding),
    # journal, and a tighter top margin above the title.
    tex = tex.replace("\\begin{abstract}\n\n\\end{abstract}",
                      "\\begin{abstract}\n" + abstract + "\n\\end{abstract}")
    tex = tex.replace(
        "\\author{Anonymous Authors}\n\\address{Anonymous Affiliations}",
        "\\author[hit]{Alexander Apartsin}\n"
        "\\author[afeka]{Yehudit Aperstein\\corref{cor1}}\n"
        "\\ead{apersteiny@afeka.ac.il}\n"
        "\\cortext[cor1]{Corresponding author}\n"
        "\\address[hit]{School of Computer Science, Faculty of Sciences, Holon Institute of Technology (HIT), Holon, Israel}\n"
        "\\address[afeka]{Intelligent Systems, Afeka Academic College of Engineering, Tel-Aviv, Israel}")
    tex = tex.replace("__JOURNAL__", "Neurocomputing")
    tex = tex.replace("\\title{Anomaly Detector Model Selection by Normal Manifold Separability}",
                      "\\title{\\vspace*{-2\\baselineskip}Anomaly Detector Model Selection by Normal Manifold Separability}")

    io.open(main_p, "w", encoding="utf-8").write(tex)
    print(f"grafted: abstract {len(abstract)}c, figure* = {tex.count(chr(92)+'begin{figure*}')}, "
          f"bibitems = {tex.count(chr(92)+'bibitem')}")

    sh([PY, os.path.join(SKILL, "scripts", "compile_local.py"), "--in-dir", "_tex", "--auto-patch"])
    shutil.copy(os.path.join(ROOT, "_tex", "main.pdf"), os.path.join(ROOT, "docs", "adrank-2col-latex.pdf"))
    shutil.copy(main_p, os.path.join(ROOT, "paper", "latex", "main.tex"))
    print("done -> docs/adrank-2col-latex.pdf")


if __name__ == "__main__":
    main()
