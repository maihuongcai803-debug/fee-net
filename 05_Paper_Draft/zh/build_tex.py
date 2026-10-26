# -*- coding: utf-8 -*-
"""Render the Chinese manuscript as a strict IEEEtran document (compile with XeLaTeX)."""
import io
import os
import re
import sys

sys.path.insert(0, "05_Paper_Draft/zh")
import content as C  # noqa: E402
import meta_cn as M  # noqa: E402
from refs_gbt import REFS_GBT  # noqa: E402

OUT = "05_Paper_Draft/Manuscript_FEE_Net_2026_zh.tex"

# ---------------------------------------------------------------- numbering
order, seen = [], set()
for blk in C.BLOCKS:
    texts = []
    if blk[0] in ("para", "subsection", "section"):
        texts = [blk[1]]
    elif blk[0] == "enum":
        texts = list(blk[1])
    elif blk[0] == "table":
        texts = [blk[1]] + [c for r in blk[3] for c in r]
    elif blk[0] == "figure":
        texts = [blk[2]]
    for t in texts:
        for m in re.finditer(r"\[\[([^\]]+)\]\]", t):
            for k in (x.strip() for x in m.group(1).split(",")):
                if k not in seen:
                    seen.add(k)
                    order.append(k)

SYMB = [
    ("ℝ", r"$\mathbb{R}$"), ("∈", r"$\in$"), ("⊙", r"$\odot$"), ("‖", r"$\|$"),
    ("≤", r"$\le$"), ("≥", r"$\ge$"), ("×", r"$\times$"), ("→", r"$\to$"),
    ("≈", r"$\approx$"), ("±", r"$\pm$"), ("Σ", r"$\Sigma$"), ("σ", r"$\sigma$"),
    ("τ", r"$\tau$"), ("λ", r"$\lambda$"), ("α", r"$\alpha$"), ("δ", r"$\delta$"),
    ("ŷ", r"$\hat{y}$"), ("γ", r"$\gamma$"), ("−", "-"),
]
TABREF = {"表Ⅰ": "tab:partitions", "表Ⅱ": "tab:main", "表Ⅲ": "tab:ablation",
          "表Ⅳ": "tab:triage", "表Ⅴ": "tab:literature"}
FIGREF = {"图1": "fig:qual", "图2": "fig:pareto"}


def esc(t):
    # protect the DSL braces, escape only what LaTeX needs
    t = t.replace("\\", r"\textbackslash{}")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("$", r"\$")):
        t = t.replace(a, b)
    return t


def texify(t):
    t = esc(t)
    for a, b in SYMB:
        t = t.replace(a, b)
    t = re.sub(r"\[\[([^\]]+)\]\]", lambda m: r"\cite{" + m.group(1).replace(" ", "") + "}", t)
    # inline sub/superscripts (content holds no nesting)
    for _ in range(3):
        t = re.sub(r"_\{([^{}]*)\}", r"\\textsubscript{\1}", t)
        t = re.sub(r"\^\{([^{}]*)\}", r"\\textsuperscript{\1}", t)
    for k, v in TABREF.items():
        t = t.replace(k, "表\\ref{" + v + "}")
    for k, v in FIGREF.items():
        t = t.replace(k, "图\\ref{" + v + "}")
    # Unicode Roman numerals are absent from Latin Modern; fall back to ASCII
    for a, b in (("Ⅶ", "VII"), ("Ⅵ", "VI"), ("Ⅴ", "V"), ("Ⅳ", "IV"),
                 ("Ⅲ", "III"), ("Ⅱ", "II"), ("Ⅰ", "I")):
        t = t.replace(a, b)
    # any remaining bare underscore/caret (e.g. requires_grad=False) must be escaped
    t = t.replace("_", r"\_").replace("^", r"\textasciicircum{}")
    return t


def strip_label(t, kind=None):
    return re.sub(r"^(表[ⅠⅡⅢⅣⅤ]|图\d+)\s*", "", t)



# 过宽公式：缩小字号并断行，避免压入右栏
EQ_OVERRIDE = {
    "5": (r"\begin{aligned}"
          r"\mathbf{a}=\sigma\!\Big(&\mathbf{W}_2\,\mathrm{ReLU}(\mathbf{W}_1\,\mathrm{GAP}(\mathbf{F}_{cat}))\\"
          r"&+\mathbf{W}_2\,\mathrm{ReLU}(\mathbf{W}_1\,\mathrm{GMP}(\mathbf{F}_{cat}))\Big)"
          r"\end{aligned}"),
    "7": (r"\begin{aligned}"
          r"\mathbf{F}_{dir}=\mathrm{ReLU}\Big(&\mathcal{K}_{1\times3}(\mathbf{F}_{freq})"
          r"+\mathcal{K}_{3\times1}(\mathbf{F}_{freq})\\"
          r"&+\mathcal{K}_{3\times3}^{diag}(\mathbf{F}_{freq})\Big)"
          r"\end{aligned}"),
    "8": (r"\begin{aligned}"
          r"\mathbf{F}_{pure}=\mathbf{F}_{dir}\odot\Big[&\sigma\big(\mathrm{Conv}_{1\times1}"
          r"(\mathrm{ReLU}(\mathrm{BN}(\mathrm{Conv}(\mathbf{F}_{dec}))))\big)\\"
          r"&\odot\mathcal{U}(\sigma(\mathbf{P}_{coarse}))\Big]"
          r"\end{aligned}"),
    "9": (r"\begin{aligned}"
          r"\mathcal{L}=&\lambda_{bce}\mathcal{L}_{BCE}+\lambda_{dice}\mathcal{L}_{Dice}"
          r"+\lambda_{b}\mathcal{L}_{BoundIoU}\\"
          r"&+\lambda_{e}\mathcal{L}_{edge}+\lambda_{ds}\mathcal{L}_{ds}"
          r"\end{aligned}"),
    "10": (r"\begin{aligned}"
           r"\mathbf{z}=[&\mathrm{GAP}(\mathbf{e}_1)\|\mathrm{GMP}(\mathbf{e}_1)\\"
           r"&\|\mathrm{GAP}(\mathbf{F}_{hf}^{1})\|\mathrm{GMP}(\mathbf{F}_{hf}^{1})]\in\mathbb{R}^{4C}"
           r"\end{aligned}"),
}


PREAMBLE = r"""%% !TEX program = xelatex
%% 中文 IEEE 格式稿。编译：xelatex Manuscript_FEE_Net_2026_zh.tex （跑两遍以生成交叉引用）
\documentclass[conference]{IEEEtran}
\usepackage{xeCJK}
\setmainfont{Times New Roman}
\setsansfont{Arial}
\setCJKmainfont[BoldFont=SimHei, ItalicFont=SimSun]{SimSun}
\setCJKsansfont[BoldFont=SimHei]{SimHei}
\setCJKmonofont{SimSun}
\newCJKfontfamily\heiti[BoldFont=SimHei]{SimHei}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{url}
\usepackage{xcolor}
\usepackage{indentfirst}
\setlength{\parindent}{2em}

\renewcommand{\abstractname}{摘要}
\renewcommand{\IEEEkeywordsname}{关键词}
\renewcommand{\refname}{参考文献}
\renewcommand{\thesection}{\arabic{section}}
\renewcommand{\thesubsection}{\thesection.\arabic{subsection}}
\renewcommand{\tablename}{表}
\renewcommand{\figurename}{图}

% --- 会议版面压缩（EI 会议通常 6 页上限）---
\setlength{\textfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\floatsep}{8pt plus 2pt minus 2pt}
\setlength{\intextsep}{8pt plus 2pt minus 2pt}
\setlength{\abovedisplayskip}{4pt plus 1pt minus 1pt}
\setlength{\belowdisplayskip}{4pt plus 1pt minus 1pt}
\renewcommand{\topfraction}{0.92}
\renewcommand{\bottomfraction}{0.70}
\renewcommand{\textfraction}{0.08}
\renewcommand{\floatpagefraction}{0.85}
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{4}

\begin{document}

\title{__TITLE__}

\author{\IEEEauthorblockN{__AUTHORS__}
\IEEEauthorblockA{__AFFIL__}}

\maketitle

\begin{abstract}
\normalfont
__ABSTRACT__
\end{abstract}

\begin{IEEEkeywords}
__KEYWORDS__
\end{IEEEkeywords}
"""

parts = [PREAMBLE
         .replace("__TITLE__", texify(C.TITLE))
         .replace("__AUTHORS__", texify(C.AUTHORS))
         .replace("__AFFIL__", "\\\\".join(texify(x) for x in C.AFFIL_LINES))
         .replace("__RUNNING__", texify(C.RUNNING))
         .replace("__ABSTRACT__", texify(C.ABSTRACT))
         .replace("__KEYWORDS__", texify(C.KEYWORDS))]

tab_i = 0
fig_i = 0
for blk in C.BLOCKS:
    kind = blk[0]
    if kind == "section":
        parts.append("\n\\section{%s}\n" % texify(re.sub(r"^\d+\s+", "", blk[1])))
    elif kind == "subsection":
        parts.append("\n\\subsection{%s}\n" % texify(re.sub(r"^\d\.\d+\s+", "", blk[1])))
    elif kind == "para":
        parts.append("\n" + texify(blk[1]) + "\n")
    elif kind == "enum":
        parts.append("\n\\begin{enumerate}\n")
        for it in blk[1]:
            parts.append("  \\item %s\n" % texify(it))
        parts.append("\\end{enumerate}\n")
    elif kind == "eq":
        tex, variants, nums = blk[1], blk[2], blk[3]
        if len(nums) == 1:
            body = EQ_OVERRIDE.get(nums[0], tex)
            small = "\\small\n" if nums[0] in EQ_OVERRIDE else ""
            parts.append("\n\\begin{equation}\n%s%s\n\\end{equation}\n" % (small, body))
        else:
            inner = tex.strip()
            if inner.startswith("\\begin{aligned}"):
                inner = inner[len("\\begin{aligned}"):]
                if inner.rstrip().endswith("\\end{aligned}"):
                    inner = inner.rstrip()[:-len("\\end{aligned}")]
            parts.append("\n\\begin{align}\n%s\n\\end{align}\n" % inner.strip())
    elif kind == "table":
        cap, headers, rows, widths = blk[1], blk[2], blk[3], blk[4]
        tab_i += 1
        wide = len(headers) > 6
        env = "table*" if wide else "table"
        colspec = "".join("c" if i else "l" for i in range(len(headers)))
        # first column left-aligned, rest centred
        colspec = "l" + "c" * (len(headers) - 1)
        body = ["\\begin{%s}[t]" % env, "\\centering",
                "\\footnotesize" if wide else "",
                "\\caption{%s}" % texify(strip_label(cap, True)),
                "\\label{%s}" % ("tab:partitions" if tab_i == 1 else
                                 "tab:main" if tab_i == 2 else
                                 "tab:ablation" if tab_i == 3 else
                                 "tab:triage" if tab_i == 4 else "tab:literature"),
                ("\\resizebox{\\textwidth}{!}{%" if wide else "\\resizebox{\\columnwidth}{!}{%"),
                "\\begin{tabular}{%s}" % colspec, "\\toprule"]
        body.append(" & ".join("\\textbf{%s}" % texify(h) for h in headers) + " \\\\")
        body.append("\\midrule")
        for r in rows:
            body.append(" & ".join(texify(c) for c in r) + " \\\\")
        body.append("\\bottomrule")
        body.append("\\end{tabular}}")
        body.append("\\end{%s}" % env)
        parts.append("\n" + "\n".join(body) + "\n")
    elif kind == "figure":
        fig_i += 1
        path, cap, width = blk[1], blk[2], blk[3]
        wide = width > 9.0
        env = "figure*" if wide else "figure"
        wspec = "0.98\\textwidth" if wide else "0.96\\columnwidth"
        parts.append(("\n\\begin{%s}[t]\n\\centering\n"
                      "\\includegraphics[width=%s]{%s}\n"
                      "\\caption{%s}\n\\label{%s}\n\\end{%s}\n")
                     % (env, wspec, path, (texify(strip_label(cap)) + "\\\\ " + texify(M.CAPTION_EN.get("图%d" % fig_i, ""))),
                        "fig:qual" if fig_i == 1 else "fig:pareto", env))

parts.append("\n\\begin{thebibliography}{99}\n")
for k in order:
    parts.append("\\bibitem{%s}\n%s\n\n" % (k, esc(REFS_GBT[k])))
parts.append("\\end{thebibliography}\n\n\\end{document}\n")

io.open(OUT, "w", encoding="utf-8", newline="\n").write("".join(parts))
print("wrote", OUT, os.path.getsize(OUT), "bytes | refs:", len(order),
      "| tables:", tab_i, "| figures:", fig_i)
