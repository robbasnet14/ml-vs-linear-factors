"""Build paper/ml-vs-linear-factors.pdf from PAPER.md.

Usage:
    pip install -r paper/requirements.txt
    DYLD_LIBRARY_PATH=/opt/homebrew/lib python paper/build_pdf.py

WeasyPrint needs libpango (a native library, not a Python package) to lay
out text. On macOS: `brew install pango`, then the DYLD_LIBRARY_PATH above
so the Python process can find it outside the default search path. On
Linux: `apt-get install libpango-1.0-0` (or your distro's equivalent) is
usually enough without any extra env var.

Markdown -> HTML (python-markdown, with the `tables` extension so GFM
pipe tables become real <table> elements) -> PDF (WeasyPrint, which lays
out HTML/CSS properly, including table borders and column widths -- the
part pandoc-without-a-PDF-engine and manual reportlab flowables both make
needlessly hard for a ~500-line paper with a dozen tables).
"""
from pathlib import Path

import markdown
from weasyprint import HTML

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "PAPER.md"
OUTPUT = ROOT / "paper" / "ml-vs-linear-factors.pdf"

CSS = """
@page {
    size: Letter;
    margin: 1in 1in 1.1in 1in;
    @bottom-center { content: counter(page); font-family: Georgia, serif; font-size: 9pt; color: #666; }
}
body {
    font-family: Georgia, "Times New Roman", serif;
    font-size: 10.5pt;
    line-height: 1.45;
    color: #1a1a1a;
}
h1 {
    font-size: 17pt;
    line-height: 1.3;
    margin-top: 0;
    margin-bottom: 0.3em;
}
h1 + p {
    /* author/date line right under the title */
    margin-top: 0;
    color: #444;
    font-size: 10pt;
}
h2 {
    font-size: 13pt;
    margin-top: 1.4em;
    margin-bottom: 0.5em;
    border-bottom: 1px solid #ccc;
    padding-bottom: 0.15em;
    page-break-after: avoid;
}
h3 {
    font-size: 11.5pt;
    margin-top: 1.2em;
    margin-bottom: 0.4em;
    page-break-after: avoid;
}
p, ul, ol { margin-top: 0.5em; margin-bottom: 0.5em; }
strong { font-weight: 700; }
em { font-style: italic; }
code {
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 9pt;
    background: #f2f2f2;
    padding: 0.05em 0.3em;
    border-radius: 2px;
}
pre {
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 8.5pt;
    background: #f2f2f2;
    padding: 0.6em 0.8em;
    border-radius: 3px;
    white-space: pre-wrap;
    page-break-inside: avoid;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.8em 0 1em 0;
    font-size: 9pt;
    page-break-inside: avoid;
}
th, td {
    border: 1px solid #999;
    padding: 3px 7px;
    text-align: left;
}
th {
    background: #e8e8e8;
    font-weight: 700;
}
tr:nth-child(even) td { background: #f7f7f7; }
blockquote {
    border-left: 3px solid #999;
    margin: 0.6em 0;
    padding: 0.1em 1em;
    color: #333;
}
a { color: #1a4d8f; text-decoration: none; }
hr { border: none; border-top: 1px solid #ccc; margin: 1.5em 0; }
img { max-width: 100%; }
"""


def main():
    text = SOURCE.read_text()
    html_body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "smarty"],
    )
    full_html = f"<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>{html_body}</body></html>"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    from weasyprint import CSS as WeasyCSS
    HTML(string=full_html, base_url=str(ROOT)).write_pdf(str(OUTPUT), stylesheets=[WeasyCSS(string=CSS)])

    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
