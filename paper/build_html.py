"""Build the ADRank paper HTML page in the SynSmith house style.

Usage:
  python paper/build_html.py                     # docs/index.html from paper/adrank.md
"""
import argparse
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parents[1]

STYLE = """
:root{
  --fg:#000000;--fg-soft:#000000;--muted:#000000;--accent:#000000;
  --bg:#ffffff;--bg-soft:#fafafa;--rule:#d1d4d8;--line:#d1d4d8;
  --code-bg:#f4f5f7;--code-fg:#000000;
  --font-body:"Charter","Iowan Old Style","Source Serif Pro",Georgia,"Times New Roman","Liberation Serif",serif;
  --font-title:"Charter","Source Serif Pro",Georgia,"Times New Roman",serif;
  --font-mono:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
}
*{box-sizing:border-box}
html{background:var(--bg);scroll-behavior:smooth}
body{font-family:var(--font-body);color:var(--fg);background:var(--bg);
  max-width:720px;margin:0 auto;padding:3rem 1.3rem 4rem;
  line-height:1.55;font-size:11pt;font-feature-settings:"lnum","kern"}
h1{font-family:var(--font-title);font-size:19pt;font-weight:600;line-height:1.22;
  margin:0 0 .8rem;letter-spacing:-.005em;color:var(--fg);text-align:center}
.authors{text-align:center;font-family:var(--font-body);font-size:12pt;color:var(--fg);margin:.6rem 0 .25rem}
.affil{text-align:center;font-size:10pt;color:var(--fg-soft);font-style:italic;margin:.1rem 0 .4rem}
.availability{text-align:center;font-size:9pt;color:var(--muted);font-family:var(--font-mono);margin:.3rem 0 1.8rem;letter-spacing:.01em}
h2{font-family:var(--font-title);font-size:13pt;font-weight:700;margin:2rem 0 .6rem;
  padding-top:.8rem;border-top:none;letter-spacing:-.005em;color:var(--fg)}
h3{font-family:var(--font-title);font-size:11.5pt;font-weight:700;margin:1.3rem 0 .45rem;color:var(--fg)}
h4{font-size:10.5pt;font-weight:700;margin:1rem 0 .3rem;color:var(--fg)}
p{margin:0;text-align:justify;text-justify:inter-word;hyphens:auto;line-height:1.55;text-indent:1.4em}
h1+p,h2+p,h3+p,h4+p,figure+p,table+p,.abstract p,.references p,li,.authors,.affil,.availability{text-indent:0}
p+p{margin-top:.55rem}
li{text-align:justify;text-justify:inter-word;hyphens:auto}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
code{font-family:var(--font-mono);font-size:.9em;background:var(--code-bg);color:var(--code-fg);padding:.08em .28em;border-radius:2px}
pre{background:var(--code-bg);color:var(--code-fg);border:1px solid var(--rule);border-radius:3px;
  padding:.7rem .9rem;overflow-x:auto;font-size:9pt;line-height:1.45;margin:.9rem 0}
pre code{background:transparent;padding:0;font-size:inherit}
table{border-collapse:collapse;width:100%;margin:1rem 0 .3rem;font-size:9.5pt;
  font-feature-settings:"tnum";border-top:1.5px solid var(--fg);border-bottom:1.5px solid var(--fg)}
th,td{padding:.4rem .55rem;text-align:left;border-bottom:.5px solid var(--rule);vertical-align:top}
th{font-weight:700;color:var(--fg);background:var(--bg);border-bottom:1px solid var(--fg);border-top:none}
blockquote{border-left:2px solid var(--accent);margin:1rem 0;padding:.2rem 1rem;color:var(--fg-soft)}
.abstract{margin:1.5rem 0 2rem;padding:0 1.2rem}
.abstract h2{font-family:var(--font-body);font-size:10.5pt;font-weight:700;text-align:center;
  margin:0 0 .5rem;padding:0;border:none;color:var(--fg);font-variant:small-caps;letter-spacing:0}
.abstract p{margin:.55rem 0;font-size:10pt;text-align:justify;hyphens:auto;color:var(--fg);text-indent:0}
hr{border:none;border-top:.5px solid var(--rule);margin:2rem 0}
figure{margin:1.4rem auto;text-align:center;max-width:100%}
img{max-width:100%;height:auto;display:block;margin:1.4rem auto .35rem;border:.5px solid var(--rule);border-radius:0}
figcaption{font-size:9.5pt;color:var(--fg-soft);text-align:justify;text-justify:inter-word;hyphens:auto;margin:.2rem 0 1rem;padding:0 .4rem}
.references{font-size:9pt;line-height:1.5;color:var(--fg);counter-reset:ref;margin-top:.3rem}
.references p{padding-left:1.7rem;text-indent:-1.7rem;margin:0 0 .55rem;text-align:left;overflow-wrap:anywhere;font-feature-settings:"tnum"}
.references p::before{counter-increment:ref;content:counter(ref) ". ";color:var(--muted)}
.references a{overflow-wrap:anywhere}
.docxlinks{position:fixed;top:1rem;right:1rem;display:flex;flex-direction:column;gap:.4rem;align-items:flex-end;z-index:10}
.docxlink{font-family:var(--font-body);font-size:10pt;font-weight:600;padding:.4rem .85rem;
  border:1px solid var(--accent);color:var(--accent);background:var(--bg);text-decoration:none;
  border-radius:3px;letter-spacing:.01em;white-space:nowrap;line-height:1;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.docxlink:hover{background:var(--accent);color:var(--bg);text-decoration:none;box-shadow:0 2px 6px rgba(0,0,0,.12)}
@media print{.docxlinks{display:none}}
@media (max-width:900px){.docxlinks{position:static;flex-direction:row;justify-content:flex-end;margin:0 0 1rem}}
footer{margin-top:3rem;padding-top:.9rem;border-top:.5px solid var(--rule);font-size:9pt;color:var(--muted);text-align:center;font-family:var(--font-body)}
@media print{
  body{max-width:none;font-size:10pt;padding:0;color:#000}
  @page{margin:2cm}
  h2,h3{break-after:avoid}
  img,table,figure{break-inside:avoid;page-break-inside:avoid}
  pre,blockquote{break-inside:avoid}
  .references p{break-inside:avoid}
  a{color:#000;text-decoration:none}
}
"""

KATEX = """<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"
  integrity="sha384-nB0miv6/jRmo5UMMR1wu3Gz6NLsoTkbqJghGIsx//Rlm+ZU03BU6SQNC66uf4l5+" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"
  integrity="sha384-7zkQWkzuo3B5mTepMUcHkMB5jZaolc2xDwL6VFqjFALcbeS9Ggm/Yr2r3Dy4lfFg" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  integrity="sha384-43gviWU0YVjaDtb/GhzOouOXtZMP/7XUzwPTstBeZFe/+rCMvRwr4yROQP43s0Xk" crossorigin="anonymous"
  onload="renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false},{left:'\\\\(',right:'\\\\)',display:false},{left:'\\\\[',right:'\\\\]',display:true}]});"></script>"""


DOCXLINKS = """<div class="docxlinks">
<a class="docxlink" href="adrank.pdf" download>&#8595; PDF (1-col)</a>
<a class="docxlink" href="adrank-2col.pdf" download>&#8595; PDF (2-col)</a>
<a class="docxlink" href="adrank-2col-latex.pdf" download>&#8595; PDF (2-col, LaTeX)</a>
<a class="docxlink" href="adrank.docx" download>&#8595; Word (1-col)</a>
<a class="docxlink" href="adrank-2col.docx" download>&#8595; Word (2-col)</a>
</div>
"""

import re as _re

def _protect_math(md):
    """Replace $$...$$ and $...$ spans with placeholder tokens so the markdown
    processor cannot mangle underscores/asterisks inside math. Returns (text, store)."""
    store = []
    def stash(m):
        store.append(m.group(0))
        return f"\x00MATH{len(store)-1}\x00"
    # display first (greedy-safe, non-nested), then inline (not $$-adjacent)
    md = _re.sub(r"\$\$.+?\$\$", stash, md, flags=_re.DOTALL)
    md = _re.sub(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", stash, md, flags=_re.DOTALL)
    return md, store

def _restore_math(html, store):
    for i, raw in enumerate(store):
        html = html.replace(f"\x00MATH{i}\x00", raw)
    return html


def build(src, out, title):
    md = Path(src).read_text(encoding="utf-8")
    md, math_store = _protect_math(md)
    html = markdown.markdown(md, extensions=["tables", "fenced_code", "toc", "sane_lists",
                                             "attr_list", "md_in_html"])
    html = _restore_math(html, math_store)
    page = (f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f'<title>{title}</title>\n{KATEX}\n<style>{STYLE}</style>\n</head>\n<body>\n'
            f'{DOCXLINKS}{html}\n<footer>ADRank v1 · {title}</footer>\n</body>\n</html>\n')
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(page, encoding="utf-8")
    (outp.parent / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Wrote {outp} ({len(html)} chars)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "paper/adrank.md"))
    ap.add_argument("--out", default=str(ROOT / "docs/index.html"))
    ap.add_argument("--title", default="Ranking Anomaly Detectors from Normal Data Alone")
    a = ap.parse_args()
    build(a.src, a.out, a.title)


if __name__ == "__main__":
    main()
