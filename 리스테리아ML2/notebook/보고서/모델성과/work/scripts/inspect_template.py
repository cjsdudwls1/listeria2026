# -*- coding: utf-8 -*-
"""양식 docx 구조 보기 (읽기 전용)."""
import sys
from docx import Document
from docx.shared import Emu

path = sys.argv[1]
doc = Document(path)
sec = doc.sections[0]
print("SECTIONS", len(doc.sections))
for i, s in enumerate(doc.sections):
    print(i, "page", s.page_width, s.page_height, "orient", s.orientation,
          "margins L R T B", s.left_margin, s.right_margin, s.top_margin, s.bottom_margin)

body = doc.element.body
ti = 0
pi = 0
for child in body.iterchildren():
    tag = child.tag.split('}')[1]
    if tag == 'p':
        from docx.text.paragraph import Paragraph
        p = Paragraph(child, doc)
        runs = [(r.text, r.bold, r.font.size, r.font.name) for r in p.runs]
        print(f"P{pi} style={p.style.name!r} align={p.alignment} text={p.text!r}")
        if runs:
            print("    runs:", runs[:6])
        pi += 1
    elif tag == 'tbl':
        from docx.table import Table
        t = Table(child, doc)
        print(f"T{ti} style={t.style.name if t.style else None!r} rows={len(t.rows)} cols={len(t.columns)}")
        for ri, row in enumerate(t.rows[:4]):
            print("   ", [c.text for c in row.cells])
        if len(t.rows) > 4:
            print("    ...")
            print("   ", [c.text for c in t.rows[-1].cells])
        ti += 1
    else:
        print("OTHER", tag)
