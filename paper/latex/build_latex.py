# -*- coding: utf-8 -*-
import re, json, io, os, shutil
TEX = r'E:\Projects\Submitted\ADRank\_tex'
SKILL = r'C:\Users\apart\.claude\skills\html2tex'
body = io.open(os.path.join(TEX, 'body.tex'), encoding='utf-8').read()

# 1. abstract (between \section{Abstract} and \section{Introduction})
m = re.search(r'\\section\{Abstract\}\\label\{abstract\}\s*(.*?)\s*\\section\{Introduction\}', body, re.S)
abstract = m.group(1).strip()

# 2. body from Introduction up to (not incl.) References section
start = body.index(r'\section{Introduction}')
end = body.index(r'\section{References}')
body_main = body[start:end].rstrip()

# 3. promote ONLY the wide figures (fig1 two-panel, fig4 bars) to full-width figure*
def promote(mobj):
    blk = mobj.group(0)   # one complete, non-nested figure block
    if 'fig1_rho' in blk or 'fig4_regret' in blk:
        blk = blk.replace(r'\begin{figure}[tbp]', r'\begin{figure*}[t]').replace(r'\end{figure}', r'\end{figure*}')
    return blk
body_main = re.sub(r'\\begin\{figure\}\[tbp\].*?\\end\{figure\}', promote, body_main, flags=re.S)

# 4. strip any leaked download-badge hrefs and stray unicode
body_main = re.sub(r'\\href\{adrank[^}]*\}\{[^}]*\}', '', body_main)
body_main = body_main.replace('\u2208', r'$\in$').replace('\u2193', '')

# 5. bibliography from the 22 refs (HTML order = citation numbering)
refs = json.load(io.open(r'E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\scratchpad\refs.json', encoding='utf-8'))
UNI = {'ü': r'\"u', 'ö': r'\"o', 'ä': r'\"a', 'ß': r'\ss{}', 'é': r"\'e", 'è': r'\`e',
       'á': r"\'a", 'í': r"\'i", 'ó': r"\'o", 'ú': r"\'u", 'ñ': r'\~n', 'ç': r'\c{c}',
       'ý': r"\'y", 'ø': r'\o{}', 'č': r'\v{c}', 'š': r'\v{s}', 'ž': r'\v{z}',
       '\u2013': '--', '\u2014': '---', '\u2019': "'", '\u2018': "'", '\u00d7': r'$\times$'}
def esc(s):
    s = s.replace('\\', r'\textbackslash{}').replace('&', r'\&').replace('%', r'\%')
    s = s.replace('#', r'\#').replace('_', r'\_').replace('~', r'\textasciitilde{}')
    for k, v in UNI.items():
        s = s.replace(k, v)
    return s
biblio = ['\\begin{thebibliography}{99}\n\\setlength{\\itemsep}{2pt}']
for i, r in enumerate(refs, 1):
    biblio.append(f'\\bibitem{{ref{i}}} {esc(r)}')
biblio.append('\\end{thebibliography}')
biblio = '\n'.join(biblio)

# 6. fix stray unicode in abstract too
abstract = abstract.replace('\u2208', r'$\in$')

# 7. main.tex
main = r'''\documentclass[10pt,twocolumn]{article}
\usepackage[letterpaper,margin=0.72in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{mathptmx}
\input{html2tex_compat.tex}
\usepackage[colorlinks=true,linkcolor=black,citecolor=black,urlcolor=black]{hyperref}
\setlength{\columnsep}{0.28in}
\setlength{\parindent}{1.2em}
\title{\vspace{-2.2em}\bfseries\large Ranking Anomaly Detectors from Normal Data Alone\vspace{-0.4em}}
\author{Alexander Apartsin\textsuperscript{1} \and Yehudit Aperstein\textsuperscript{2}\\[3pt]
\normalsize\textsuperscript{1}School of Computer Science, Faculty of Sciences, Holon Institute of Technology (HIT), Holon, Israel\\
\normalsize\textsuperscript{2}Intelligent Systems, Afeka Academic College of Engineering, Tel-Aviv, Israel}
\date{}
\begin{document}
\twocolumn[
\begin{@twocolumnfalse}
\maketitle
\begin{abstract}
\noindent ''' + abstract + r'''
\end{abstract}
\vspace{1.4em}
\end{@twocolumnfalse}
]
''' + body_main + '\n\n' + biblio + '\n\n\\end{document}\n'

io.open(os.path.join(TEX, 'main.tex'), 'w', encoding='utf-8').write(main)
shutil.copy(os.path.join(SKILL, 'templates', '_common', 'html2tex_compat.tex'),
            os.path.join(TEX, 'html2tex_compat.tex'))
print('wrote main.tex (%d chars), abstract %d chars, %d refs, figures promoted' % (len(main), len(abstract), len(refs)))
print('figure* (full-width) count:', main.count(r'\begin{figure*}'))
