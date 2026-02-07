#!/usr/bin/env python3
"""Build PDF document from a minimal Markdown source."""

from __future__ import annotations

import argparse
import html
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer


@dataclass
class Block:
    kind: str
    text: str
    level: int = 0


def parse_markdown(text: str) -> list[Block]:
    blocks: list[Block] = []
    lines = text.splitlines()

    in_code = False
    code_lines: list[str] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        paragraph = " ".join(line.strip() for line in paragraph_lines).strip()
        if paragraph:
            blocks.append(Block(kind="paragraph", text=paragraph))
        paragraph_lines = []

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        if line.strip().startswith("```"):
            if in_code:
                blocks.append(Block(kind="code", text="\n".join(code_lines)))
                code_lines = []
                in_code = False
            else:
                flush_paragraph()
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not line.strip():
            flush_paragraph()
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if heading_match:
            flush_paragraph()
            marks, title = heading_match.groups()
            blocks.append(
                Block(
                    kind="heading",
                    level=min(len(marks), 6),
                    text=title.strip(),
                )
            )
            continue

        bullet_match = re.match(r"^\s*[-*]\s+(.*)$", line)
        ordered_match = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if bullet_match:
            flush_paragraph()
            blocks.append(Block(kind="bullet", text=bullet_match.group(1).strip()))
            continue
        if ordered_match:
            flush_paragraph()
            blocks.append(Block(kind="bullet", text=ordered_match.group(1).strip()))
            continue

        paragraph_lines.append(line)

    flush_paragraph()
    if in_code:
        blocks.append(Block(kind="code", text="\n".join(code_lines)))
    return blocks


def _inline_markdown_to_rl(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(
        r"`([^`]+)`",
        r"<font name='Courier'>\1</font>",
        escaped,
    )
    return escaped


def build_story(blocks: list[Block]):
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        spaceAfter=8,
    )
    heading_styles = {
        1: ParagraphStyle(
            "H1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#1F2937"),
            spaceBefore=8,
            spaceAfter=10,
        ),
        2: ParagraphStyle(
            "H2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#111827"),
            spaceBefore=8,
            spaceAfter=8,
        ),
        3: ParagraphStyle(
            "H3",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#111827"),
            spaceBefore=6,
            spaceAfter=6,
        ),
    }
    code_style = ParagraphStyle(
        "Code",
        fontName="Courier",
        fontSize=9,
        leading=12,
        backColor=colors.HexColor("#F3F4F6"),
        borderPadding=6,
        leftIndent=8,
        rightIndent=8,
        spaceBefore=6,
        spaceAfter=8,
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body,
        leftIndent=12,
        firstLineIndent=0,
    )

    story = []
    for block in blocks:
        if block.kind == "heading":
            style = heading_styles.get(block.level, heading_styles[3])
            story.append(Paragraph(_inline_markdown_to_rl(block.text), style))
            continue
        if block.kind == "code":
            code_text = block.text.rstrip("\n")
            if not code_text:
                code_text = " "
            story.append(Preformatted(code_text, style=code_style))
            continue
        if block.kind == "bullet":
            story.append(
                Paragraph(
                    f"&bull; {_inline_markdown_to_rl(block.text)}",
                    bullet_style,
                )
            )
            continue
        if block.kind == "paragraph":
            story.append(Paragraph(_inline_markdown_to_rl(block.text), body))
            continue
    story.append(Spacer(1, 2))
    return story


def _on_page(canvas, doc, generated_at: str, title: str) -> None:
    canvas.saveState()
    canvas.setTitle(title)
    canvas.setAuthor("Codex")
    canvas.setSubject("Documentazione tecnica sistema VLM + YOLO + AI2-THOR")
    canvas.setFont("Helvetica", 8)
    footer_left = f"Generato: {generated_at}"
    footer_right = f"Pagina {doc.page}"
    canvas.drawString(doc.leftMargin, 14, footer_left)
    canvas.drawRightString(A4[0] - doc.rightMargin, 14, footer_right)
    canvas.restoreState()


def build_pdf(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input Markdown non trovato: {input_path}")

    markdown = input_path.read_text(encoding="utf-8")
    blocks = parse_markdown(markdown)
    story = build_story(blocks)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = "Documentazione Sistema VLM YOLO AI2THOR"

    doc = SimpleDocTemplate(
        output_path.as_posix(),
        pagesize=A4,
        leftMargin=42,
        rightMargin=42,
        topMargin=42,
        bottomMargin=28,
        title=title,
        author="Codex",
        subject="Documentazione tecnica",
    )
    doc.build(
        story,
        onFirstPage=lambda c, d: _on_page(c, d, generated_at, title),
        onLaterPages=lambda c, d: _on_page(c, d, generated_at, title),
    )


def _default_paths(script_path: Path) -> tuple[Path, Path]:
    docs_dir = script_path.resolve().parent
    md_path = docs_dir / "Documentazione_Sistema_VLM_YOLO_AI2THOR.md"
    tag = datetime.now().strftime("%Y%m%d")
    pdf_name = f"Documentazione_Sistema_VLM_YOLO_AI2THOR_{tag}.pdf"
    pdf_path = docs_dir / pdf_name
    return md_path, pdf_path


def main() -> int:
    script_path = Path(__file__)
    default_in, default_out = _default_paths(script_path)

    parser = argparse.ArgumentParser(
        description="Genera un PDF A4 da Markdown (supporto minimale heading/paragrafi/code)."
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        default=default_in.as_posix(),
        help="Percorso file Markdown sorgente.",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        default=default_out.as_posix(),
        help="Percorso file PDF di output.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    build_pdf(input_path, output_path)
    print(f"[OK] PDF generato: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
