# -*- coding: utf-8 -*-
"""Render the manuscript to the OFFICIAL Chinese conference template.

All layout constants below were read out of
  papers/templates/CPCC_2026_全文中文模板.docx
(the 2026 Chinese MS-Word template published by the Chinese Process Control
Conference, an EI-indexed conference), not assumed:

  page           A4 21.0 x 29.7 cm
  margins        top 3.0 cm (first section) / 2.0 cm (body), bottom 2.2 cm,
                 left = right = 1.7 cm      -> 版芯 17.6 cm
  sections       §0 one column (title/abstract block)
                 §1 two columns, gap 0.7 cm  -> column 8.45 cm
  title          18 pt 黑体, centred, 18 pt before / 12 pt after
  authors        12 pt 仿宋_GB2312, centred
  affiliation     9 pt 宋体, centred
  abstract/keywords  9 pt, label 黑体, left indent 0.7 cm, single spacing
  English title  18 pt Times New Roman bold, centred
  body           10.5 pt 宋体 / Times New Roman, justified, first line 0.74 cm,
                 single spacing, 2 pt after
  heading 1      12 pt 黑体, flush left, arabic numbering 1 / 2 / …
  heading 2      10.5 pt 黑体, flush left
  captions       9 pt 宋体, centred, 6 pt before and after;
                 table caption above the table, figure caption below
  table rules    top and bottom DOUBLE, inside single, no left/right outer rule
  references     heading 12 pt 黑体 flush left; entries 9 pt
"""
import os
import re
import sys

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

sys.path.insert(0, "05_Paper_Draft/zh")
import content as C            # noqa: E402
import meta_cn as M            # noqa: E402
from refs_gbt import REFS_GBT  # noqa: E402

OUT = "05_Paper_Draft/FEE-Net_中文稿_官方模板版.docx"

SONG, HEI, FANG, TNR = "宋体", "黑体", "仿宋_GB2312", "Times New Roman"

# ---- constants read from the official template -------------------------------
PAGE_W, PAGE_H = 21.0, 29.7
M_TOP_FIRST, M_TOP_BODY = 3.0, 2.0
M_BOTTOM, M_LR = 2.2, 1.7
COL_GAP = 0.7
USABLE = PAGE_W - 2 * M_LR            # 17.6 cm
COL_W = (USABLE - COL_GAP) / 2        # 8.45 cm

REPO_URL = "https://github.com/maihuongcai803-debug/fee-net"

# ---------------------------------------------------------------- citations
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
NUM = {k: i + 1 for i, k in enumerate(order)}


def cite(m):
    nums = sorted(NUM[k] for k in (x.strip() for x in m.group(1).split(",")))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if j == i else
                   (f"{nums[i]},{nums[j]}" if j == i + 1 else f"{nums[i]}-{nums[j]}"))
        i = j + 1
    return "[" + ",".join(out) + "]"


def resolve(t):
    return re.sub(r"\[\[([^\]]+)\]\]", cite, t)


EQ = re.compile(r"_\{([^}]*)\}|\^\{([^}]*)\}")


def font(run, size, bold=False, italic=False, latin=TNR, east=SONG):
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)
    run.font.name = latin
    rPr = run._element.get_or_add_rPr()
    rF = rPr.find(qn("w:rFonts"))
    if rF is None:
        rF = OxmlElement("w:rFonts")
        rPr.append(rF)
    for a in ("w:ascii", "w:hAnsi"):
        rF.set(qn(a), latin)
    rF.set(qn("w:eastAsia"), east)


def add_inline(p, text, size, bold=False, italic=False, east=SONG, latin=TNR):
    text = resolve(text)
    pos = 0
    for m in EQ.finditer(text):
        if m.start() > pos:
            font(p.add_run(text[pos:m.start()]), size, bold, italic, latin, east)
        r = p.add_run(m.group(1) if m.group(1) is not None else (m.group(2) or ""))
        font(r, size, bold, italic, latin, east)
        r.font.subscript = m.group(1) is not None
        r.font.superscript = m.group(1) is None
        pos = m.end()
    if pos < len(text):
        font(p.add_run(text[pos:]), size, bold, italic, latin, east)


def para(doc, text="", *, size=10.5, bold=False, italic=False, align="justify",
         indent=True, before=0, after=2, east=SONG, latin=TNR, left=0.0, line=1.0):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = {"justify": WD_ALIGN_PARAGRAPH.JUSTIFY, "center": WD_ALIGN_PARAGRAPH.CENTER,
                    "left": WD_ALIGN_PARAGRAPH.LEFT}[align]
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    pf.line_spacing = line
    if indent:
        pf.first_line_indent = Cm(0.74)      # 首行缩进两字符
    if left:
        pf.left_indent = Cm(left)
    if text:
        add_inline(p, text, size, bold, italic, east, latin)
    return p


# ---------------------------------------------------------------- math typesetting
# GB/T 3102.11：变量（单个拉丁字母、希腊小写字母）斜体；算子名、下标标签、数字正体。
GREEK_LOWER = set("αβγδεζηθικλμνξοπρστυφχψω")
UPRIGHT_SYMBOLS = set("ℝℕℤ⊙‖∈×·−+−=/()[]{},;:<>≤≥≠≈±∝∑∏∫√∞→←↑↓")


def _math_runs(p, s, size):
    """把一段数学文本按"变量斜体 / 算子与数字正体"逐词写入。"""
    for tok in re.findall(r"[A-Za-z]+|[^A-Za-z]+", s):
        italic = False
        if tok.isalpha():
            if tok.isascii():
                italic = len(tok) == 1          # 单字母变量斜体；ReLU/GAP/Conv 等算子正体
            else:
                italic = all(ch in GREEK_LOWER for ch in tok)  # 希腊小写变量斜体
        if italic:
            r = p.add_run(tok)
            font(r, size, italic=True)
        else:
            r = p.add_run(tok)
            font(r, size, italic=False)


def add_math(p, text, size):
    """公式体：处理 _{...} / ^{...}，其余部分按数学规范区分斜体与正体。"""
    text = resolve(text)
    pos = 0
    for m in EQ.finditer(text):
        if m.start() > pos:
            _math_runs(p, text[pos:m.start()], size)
        base = m.group(1) if m.group(1) is not None else (m.group(2) or "")
        r = p.add_run(base)
        font(r, size, italic=False)             # 下标/上标标签（LL、cat、bce 等）正体
        r.font.subscript = m.group(1) is not None
        r.font.superscript = m.group(1) is None
        pos = m.end()
    if pos < len(text):
        _math_runs(p, text[pos:], size)


def set_cols(section, num, space_cm=COL_GAP):
    sectPr = section._sectPr
    cols = sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sectPr.append(cols)
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), str(int(space_cm / 2.54 * 1440)))


def cell_border(cell, edge, val, sz):
    tcPr = cell._tc.get_or_add_tcPr()
    bs = tcPr.find(qn("w:tcBorders"))
    if bs is None:
        bs = OxmlElement("w:tcBorders")
        tcPr.append(bs)
    el = bs.find(qn("w:" + edge))
    if el is None:
        el = OxmlElement("w:" + edge)
        bs.append(el)
    el.set(qn("w:val"), val)
    el.set(qn("w:sz"), str(sz))
    el.set(qn("w:color"), "000000")


def _tw(cm):
    """cm -> twips (1 inch = 1440 twips)."""
    return int(round(cm / 2.54 * 1440))


def _best_per_col(headers, rows):
    """按表头箭头（↑/↓）判定各列最优值，返回 {(row, col): True}。

    没有箭头的列（如参数量、配置名）不参与排名，不加粗。
    """
    marks = {}
    for c, h in enumerate(headers):
        direction = 1 if "↑" in h else (-1 if "↓" in h else 0)
        if not direction:
            continue
        vals = []
        for r, row in enumerate(rows):
            if c >= len(row):
                continue
            try:
                vals.append((float(str(row[c]).replace("−", "-").strip()), r))
            except ValueError:
                continue
        if not vals:
            continue
        best = max(vals) if direction > 0 else min(vals)
        marks[(best[1], c)] = True
    return marks


def official_table(doc, headers, rows, total_cm, widths=None):
    """标准三线表：顶线 1.5 pt、表头下线 0.75 pt、底线 1.5 pt，无竖线、无数据行横线。

    宽度用固定布局（w:tblLayout=fixed）+ 显式 tblW/gridCol 锁死，
    并把单元格左右边距压到 0.08 cm，避免 Word 按内容撑宽后越出栏宽。
    """
    n_h = len(headers)
    t = doc.add_table(rows=1 + len(rows), cols=n_h)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False                       # 关掉自动适应，否则 Word 会重新分配列宽
    tblPr = t._tbl.tblPr

    # --- 固定布局 + 显式总宽
    for tag in ("w:tblLayout", "w:tblW"):
        old = tblPr.find(qn(tag))
        if old is not None:
            tblPr.remove(old)
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.append(layout)
    tblW = OxmlElement("w:tblW")
    tblW.set(qn("w:type"), "dxa")
    tblW.set(qn("w:w"), str(_tw(total_cm)))
    tblPr.append(tblW)

    # --- 单元格边距压缩（默认左右各 0.19 cm 是越界的隐形来源）
    old_mar = tblPr.find(qn("w:tblCellMar"))
    if old_mar is not None:
        tblPr.remove(old_mar)
    mar = OxmlElement("w:tblCellMar")
    for side, tw in (("top", 0), ("bottom", 0), ("left", 45), ("right", 45)):
        el = OxmlElement("w:" + side)
        el.set(qn("w:w"), str(tw))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tblPr.append(mar)

    # --- 三线表边框
    old_b = tblPr.find(qn("w:tblBorders"))
    if old_b is not None:
        tblPr.remove(old_b)
    borders = OxmlElement("w:tblBorders")
    for edge, val, sz in (("top", "single", 12), ("bottom", "single", 12),
                          ("left", "none", 0), ("right", "none", 0),
                          ("insideH", "none", 0), ("insideV", "none", 0)):
        el = OxmlElement("w:" + edge)
        el.set(qn("w:val"), val)
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)
    tblPr.append(borders)

    # --- 列宽（同时写 gridCol，Word 以 gridCol 为准）
    if widths and len(widths) == n_h:
        scale = total_cm / sum(widths)
        cols_cm = [w * scale for w in widths]
    else:
        cols_cm = [total_cm / n_h] * n_h
    grid = t._tbl.find(qn("w:tblGrid"))
    if grid is not None:
        for gc, w in zip(grid.findall(qn("w:gridCol")), cols_cm):
            gc.set(qn("w:w"), str(_tw(w)))

    best = _best_per_col(headers, rows)
    for r_i, row in enumerate([headers] + rows):
        for c_i, val in enumerate(row):
            cell = t.cell(r_i, c_i)
            cell.width = Cm(cols_cm[c_i])
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.0
            add_inline(p, val, 9, bold=(r_i == 0 or (r_i - 1, c_i) in best))
            tcPr = cell._tc.get_or_add_tcPr()
            va = OxmlElement("w:vAlign")
            va.set(qn("w:val"), "center")
            tcPr.append(va)

    # 表头底线 0.75 pt
    for cell in t.rows[0].cells:
        cell_border(cell, "bottom", "single", 6)
    for r_i, row in enumerate(t.rows):
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        if r_i < len(t.rows) - 1:
            for cell in row.cells:
                for pp in cell.paragraphs:
                    pp.paragraph_format.keep_with_next = True
    return t


def split_caption(text):
    """把"表N 短标题。长说明"拆成 (短标题, 说明)。无句号则整段作标题。"""
    head, sep, rest = text.partition("。")
    if not sep:
        return text.rstrip("。"), ""
    return head.strip(), rest.strip()


def caption_title(doc, text, above=True):
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = above
    p.paragraph_format.line_spacing = 1.0
    add_inline(p, text, 9, bold=True, east=HEI)


def caption_note(doc, text):
    """题注分离后的"注："段，排在图/表下方，9 pt、小号、左对齐。"""
    if not text:
        return
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.space_before = Pt(0)
    pf.space_after = Pt(6)
    pf.line_spacing = 1.0
    add_inline(p, "注：" + text, 8, east=SONG)


# ---------------------------------------------------------------- document
doc = Document()
st = doc.styles["Normal"]
st.font.name = TNR
st.font.size = Pt(10.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), SONG)

sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(PAGE_W), Cm(PAGE_H)
sec.top_margin, sec.bottom_margin = Cm(M_TOP_FIRST), Cm(M_BOTTOM)
sec.left_margin = sec.right_margin = Cm(M_LR)
set_cols(sec, 1)

# ---------------- §0 front matter (single column) ----------------
# 间距整体压紧，保证"英文 Key Words / Code and data"收回第 1 页，避免跨页孤行
para(doc, C.TITLE, size=18, align="center", indent=False, before=10, after=8, east=HEI)
para(doc, C.AUTHORS, size=12, align="center", indent=False, after=4, east=FANG)
for line in C.AFFIL_LINES:
    para(doc, line, size=9, align="center", indent=False, after=0, east=SONG, left=0)

para(doc, f"摘　要：{C.ABSTRACT}", size=9, indent=False, before=6, after=2, left=0.7)
para(doc, f"关键词：{C.KEYWORDS}", size=9, indent=False, after=2, left=0.7)
para(doc, (f"中图分类号：{M.CLC}　　文献标识码：{M.DOC_CODE}　　"
           f"文章编号：（由编辑部填写）"), size=9, indent=False, after=2, left=0.7)
para(doc, M.FUND, size=7.5, indent=False, after=0, left=0.7)
para(doc, M.BIO, size=7.5, indent=False, after=0, left=0.7)
para(doc, f"代码与数据（开源仓库）：{REPO_URL}", size=7.5, indent=False, after=11, left=0.7)

para(doc, M.TITLE_EN, size=17, align="center", indent=False, before=8, after=5, latin=TNR)
para(doc, M.AUTHORS_EN, size=11, align="center", indent=False, after=4, latin=TNR)
for line in M.AFFIL_EN:
    para(doc, line, size=9, align="center", indent=False, after=0, latin=TNR)
para(doc, f"Abstract: {M.ABSTRACT_EN}", size=9, indent=False, before=5, after=2, left=0.7, latin=TNR)
para(doc, f"Key Words: {M.KEYWORDS_EN}", size=9, indent=False, after=2, left=0.7, latin=TNR)
para(doc, f"Code and data: {REPO_URL}", size=8, indent=False, after=10, left=0.7, latin=TNR)

# ---------------- §1 body (two columns) ----------------
body = doc.add_section(WD_SECTION.CONTINUOUS)
body.top_margin, body.bottom_margin = Cm(M_TOP_BODY), Cm(M_BOTTOM)
body.left_margin = body.right_margin = Cm(M_LR)
set_cols(body, 2)


def switch(num):
    s = doc.add_section(WD_SECTION.CONTINUOUS)
    s.top_margin, s.bottom_margin = Cm(M_TOP_BODY), Cm(M_BOTTOM)
    s.left_margin = s.right_margin = Cm(M_LR)
    set_cols(s, num)


WIDE_TABLES = {2, 3, 5}
tab_i = fig_i = 0
for blk in C.BLOCKS:
    kind = blk[0]
    if kind == "section":
        m = re.match(r"^(\d)\s+(.*)$", blk[1])
        p = doc.add_paragraph()
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        add_inline(p, f"{m.group(1)}　{m.group(2)}" if m else blk[1], 12, bold=True, east=HEI)
    elif kind == "subsection":
        m = re.match(r"^(\d\.\d)\s+(.*)$", blk[1])
        p = doc.add_paragraph()
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        add_inline(p, f"{m.group(1)}　{m.group(2)}" if m else blk[1], 10.5, bold=True, east=HEI)
    elif kind == "para":
        para(doc, blk[1])
    elif kind == "enum":
        for i, item in enumerate(blk[1], 1):
            p = doc.add_paragraph()
            pf = p.paragraph_format
            pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            pf.first_line_indent = Cm(0.74)
            pf.space_after = Pt(2)
            add_inline(p, f"（{i}）{item}", 10.5)
    elif kind == "eq":
        # 公式居中、编号右对齐：用居中 tab + 右对齐 tab，避免长公式折行后编号悬空
        for v, n in zip(blk[2], blk[3]):
            p = doc.add_paragraph()
            pf = p.paragraph_format
            pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
            pf.space_before = Pt(3)
            pf.space_after = Pt(3)
            pf.line_spacing = 1.0
            pf.left_indent = Cm(0)
            pf.first_line_indent = Cm(0)
            pf.tab_stops.add_tab_stop(Cm(COL_W / 2), WD_TAB_ALIGNMENT.CENTER)
            pf.tab_stops.add_tab_stop(Cm(COL_W), WD_TAB_ALIGNMENT.RIGHT)
            font(p.add_run("\t"), 10.5)
            add_math(p, v, 10.5)
            font(p.add_run(f"\t（{n}）"), 10.5)
    elif kind == "table":
        tab_i += 1
        wide = tab_i in WIDE_TABLES
        if wide:
            switch(1)
        width = USABLE if wide else COL_W
        title, note = split_caption(re.sub(r"^(表\d+)\s*", "", blk[1]))
        caption_title(doc, f"表{tab_i}　{title}", above=True)
        official_table(doc, blk[2], blk[3], width, blk[4] if len(blk) > 4 else None)
        caption_note(doc, note)
        if wide:
            switch(2)
    elif kind == "figure":
        fig_i += 1
        wide = blk[3] > 9.0
        if wide:
            switch(1)
        img_w = min(USABLE, blk[3]) if wide else COL_W   # 跨栏图按声明宽度居中，不强行拉满版芯
        p = doc.add_paragraph()
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture("05_Paper_Draft/" + blk[1], width=Cm(img_w))
        title, note = split_caption(re.sub(r"^(图\d+)\s*", "", blk[2]))
        caption_title(doc, f"图{fig_i}　{title}", above=False)
        caption_note(doc, note)
        if wide:
            switch(2)

# ---------------- references ----------------
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(6)
p.paragraph_format.space_after = Pt(6)
p.paragraph_format.keep_with_next = True
add_inline(p, "参考文献", 12, bold=True, east=HEI)
for k in order:
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT   # 左对齐：避免长 DOI/URL 在两端对齐下把词距拉开
    pf.left_indent = Cm(0.7)
    pf.first_line_indent = Cm(-0.7)
    pf.space_after = Pt(1)
    pf.line_spacing = 1.0
    add_inline(p, f"[{NUM[k]}]　{REFS_GBT[k]}", 9)

# 所有分节符设为"连续"，否则首页题名块之后会强制换页
for _s in doc.sections:
    try:
        _s.start_type = WD_SECTION.CONTINUOUS
    except Exception:
        pass

# ---------------- 页脚页码 ----------------
# 分节沿用同一页脚，只有第一节真正写入 PAGE 域
for _i, _s in enumerate(doc.sections):
    _ft = _s.footer
    _ft.is_linked_to_previous = _i > 0
    if _i > 0:
        continue
    _p = _ft.paragraphs[0] if _ft.paragraphs else _ft.add_paragraph()
    _p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _p.paragraph_format.space_before = Pt(0)
    _p.paragraph_format.space_after = Pt(0)
    _fld = OxmlElement("w:fldSimple")
    _fld.set(qn("w:instr"), " PAGE ")
    _r = OxmlElement("w:r")
    _rPr = OxmlElement("w:rPr")
    _sz = OxmlElement("w:sz")
    _sz.set(qn("w:val"), "18")          # 9 pt
    _rPr.append(_sz)
    _r.append(_rPr)
    _t = OxmlElement("w:t")
    _t.text = "1"
    _r.append(_t)
    _fld.append(_r)
    _p._p.append(_fld)

try:
    doc.save(OUT)
except PermissionError:
    alt = OUT.replace(".docx", "_new.docx")
    doc.save(alt)
    OUT = alt
    print("[warn] 目标被占用，另存为", alt)
print("wrote", OUT, "| refs", len(order), "| tables", tab_i, "| figures", fig_i)
