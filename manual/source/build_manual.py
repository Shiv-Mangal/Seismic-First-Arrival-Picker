"""Build the user manual: manual/manual.html (in the picker) and the PDF (Chrome print), one source."""
import os, re, subprocess, datetime
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))          # manual/source
SRC = os.path.join(HERE, 'manual_body.html')
MAN = os.path.dirname(HERE)                                # manual/ (html, pdf, images)
IMG = os.path.join(MAN, 'images')
body = open(SRC, encoding='utf-8').read()


def natural_widths(html):
    """data-w = the image's real pixel width (the help panel scales down to fit)."""
    def repl(m):
        tag = m.group(0)
        src = re.search(r'src="([^"]+)"', tag).group(1)
        p = os.path.join(MAN, src)
        if os.path.isfile(p):
            w = Image.open(p).width
            tag = re.sub(r'data-w="\d+"', f'data-w="{w}"', tag)
        else:
            print('MISSING IMAGE', src)
        return tag
    return re.sub(r'<img\b[^>]*>', repl, html)


body = natural_widths(body)
toc = [(m.group(1), re.sub('<[^>]+>', '', m.group(2))) for m in re.finditer(r'<h2 id="([^"]+)">(.*?)</h2>', body)]

# ---------------- screen version (QTextBrowser: CSS 2.1 subset) ----------------
SCREEN_CSS = """
body { font-family: 'Segoe UI', 'DejaVu Sans', Arial, sans-serif; font-size: 10.5pt; color: #222; }
h1 { color: #1b4f72; font-size: 20pt; margin-bottom: 2px; }
p.sub { color: #555; font-size: 11pt; margin-top: 0; }
h2 { color: #1b4f72; font-size: 15pt; margin-top: 26px; border-bottom: 1px solid #aab7b8; }
h3 { color: #21618c; font-size: 12pt; margin-top: 16px; }
code { font-family: Consolas, 'DejaVu Sans Mono', monospace; color: #7b241c; }
pre { font-family: Consolas, 'DejaVu Sans Mono', monospace; background-color: #f4f6f7; padding: 6px; }
p.fig { margin-top: 10px; margin-bottom: 14px; }
span.cap { color: #555; font-size: 9.5pt; }
table.grid { border-collapse: collapse; margin-top: 6px; margin-bottom: 10px; }
table.grid th { background-color: #d6eaf8; text-align: left; padding: 4px; }
table.grid td { padding: 4px; vertical-align: top; }
table.note { background-color: #fef9e7; margin-top: 8px; margin-bottom: 10px; }
a { color: #1f618d; }
"""
contents = '<h3 id="contents">Contents</h3><ol>' + ''.join(f'<li><a href="#{a}">{re.sub(r"^\d+\.\s*", "", t)}</a></li>' for a, t in toc) + '</ol>'
screen_body = body.replace('<h2 id="workflow">', contents + '\n<h2 id="workflow">', 1)
screen_body = screen_body.replace('<table class="grid">', '<table class="grid" border="1" cellspacing="0" cellpadding="4" width="100%">')
screen_body = screen_body.replace('<table class="note">', '<table class="note" width="100%" cellpadding="8">')
screen = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>First-Arrival Picker - User Manual</title>
<style>{SCREEN_CSS}</style></head><body>
{screen_body}
<p class="sub">Built {datetime.date.today():%d %B %Y}. The same text is in First_Arrival_Picker_User_Manual.pdf.</p>
</body></html>"""
open(os.path.join(MAN, 'manual.html'), 'w', encoding='utf-8').write(screen)

# ---------------- print version (Chrome -> PDF) ----------------
PRINT_CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm;
        @bottom-center { content: counter(page); font: 9pt 'DejaVu Sans', sans-serif; color: #777; }
        @top-right { content: 'First-Arrival Picker - User Manual'; font: 8pt 'DejaVu Sans', sans-serif; color: #999; } }
@page :first { @bottom-center { content: none; } @top-right { content: none; } }
html { font-family: 'Inter', 'DejaVu Sans', 'Liberation Sans', Arial, sans-serif; font-size: 10pt; color: #1f2933; line-height: 1.42; }
body { margin: 0; }
h1#top { display: none; }
p.sub:first-of-type { display: none; }
.cover { height: 250mm; display: flex; flex-direction: column; justify-content: center; page-break-after: always; }
.cover .kicker { color: #2e86c1; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; font-size: 10pt; }
.cover h1 { font-size: 30pt; color: #1b4f72; margin: 8px 0 6px 0; line-height: 1.1; }
.cover .lead { font-size: 13pt; color: #444; max-width: 150mm; }
.cover img { width: 100%; margin-top: 18mm; }
.cover .meta { margin-top: 14mm; color: #666; font-size: 10pt; }
nav.toc { page-break-after: always; }
nav.toc h2 { border: none; margin-top: 0; }
nav.toc ol { padding-left: 0; list-style: none; counter-reset: none; }
nav.toc li { display: flex; padding: 5px 0; border-bottom: 1px dotted #c8ccd0; font-size: 11pt; }
nav.toc a { color: #1f2933; text-decoration: none; flex: 1; }
nav.toc span.pg { color: #2e86c1; min-width: 12mm; text-align: right; }
nav.toc p.intro { margin-top: 10mm; color: #333; }
h2 { page-break-before: always; color: #1b4f72; font-size: 18pt; border-bottom: 2px solid #2e86c1; padding-bottom: 4px; margin-top: 0; }
h3 { color: #21618c; font-size: 12pt; margin-top: 14px; page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }
code { font-family: 'DejaVu Sans Mono', Consolas, monospace; font-size: 8.8pt; color: #7b241c; background: #f6f2f0; padding: 0 2px; border-radius: 2px; }
pre { font-family: 'DejaVu Sans Mono', monospace; font-size: 9pt; background: #f4f6f7; border-left: 3px solid #2e86c1; padding: 6px 10px; }
p.fig { text-align: center; page-break-inside: avoid; margin: 10px 0 14px 0; }
p.fig img { max-width: 100%; max-height: 135mm; border: 1px solid #d5d8dc; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }
span.cap { display: block; text-align: left; color: #555; font-size: 8.8pt; margin-top: 5px; }
table.grid { width: 100%; border-collapse: collapse; margin: 6px 0 12px 0; font-size: 9.2pt; page-break-inside: auto; }
table.grid th { background: #d6eaf8; text-align: left; padding: 5px 6px; border: 1px solid #aed6f1; }
table.grid td { padding: 4px 6px; border: 1px solid #d5d8dc; vertical-align: top; }
table.grid tr { page-break-inside: avoid; }
table.note { width: 100%; background: #fef9e7; border-left: 4px solid #f1c40f; margin: 8px 0 12px 0; }
table.note td { padding: 8px 10px; }
a { color: #1f618d; text-decoration: none; }
"""
intro_m = re.search(r'<p>This manual explains.*?</p>', body, re.S)
intro = intro_m.group(0) if intro_m else ''
body_print = body.replace(intro, '', 1)
cover = f"""<section class="cover">
<div class="kicker">User manual · version 22</div>
<h1>First-Arrival Picker<br>for Refraction Tomography</h1>
<div class="lead">Manual and automatic first-break picking of SEG-Y shot gathers, with preprocessing,
filtering, reciprocity QC and export of S&nbsp;G&nbsp;T travel times.</div>
<img src="images/workflow.png">
<div class="meta">{datetime.date.today():%B %Y} · figures made with the picker on a synthetic survey</div>
</section>"""
pdf = os.path.join(MAN, 'First_Arrival_Picker_User_Manual.pdf')
pp = os.path.join(MAN, '_print.html')


def render(pages):
    toc_html = '<nav class="toc"><h2 style="page-break-before:avoid">Contents</h2><ol>' + ''.join(
        f'<li><a href="#{a}">{t}</a><span class="pg">{pages.get(a, "")}</span></li>' for a, t in toc) + \
        '</ol>' + intro.replace('<p>', '<p class="intro">', 1) + '</nav>'
    printed = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>First-Arrival Picker - User Manual</title>
<style>{PRINT_CSS}</style></head><body>{cover}{toc_html}{body_print}</body></html>"""
    open(pp, 'w', encoding='utf-8').write(printed)
    r = subprocess.run(['google-chrome', '--headless=new', '--no-sandbox', '--disable-gpu', '--no-pdf-header-footer',
                        f'--print-to-pdf={pdf}', '--virtual-time-budget=4000', 'file://' + pp],
                       capture_output=True, text=True, timeout=180)
    os.remove(pp)
    if r.returncode:
        print(r.stderr[-400:])


from pypdf import PdfReader, PdfWriter
render({a: '00' for a, _ in toc})                       # same width as the final numbers
rd = PdfReader(pdf)
texts = [pg.extract_text() or '' for pg in rd.pages]
pages = {}
for a, t in toc:
    for i in range(2, len(texts)):
        norm = lambda z: re.sub(r'[^a-z0-9]', '', z.lower())
        head = norm(' '.join(texts[i].split())[:200])
        if norm(t) in head:
            pages[a] = i + 1
            break
print('toc pages', pages)
render(pages)
rd = PdfReader(pdf); wr = PdfWriter(clone_from=rd)
wr.add_metadata({'/Title': 'First-Arrival Picker for Refraction Tomography - User Manual',
                 '/Subject': 'User manual, version 22', '/Creator': 'first_arrival_picker'})
with open(pdf, 'wb') as fh:
    wr.write(fh)
print('pages', len(rd.pages), 'size', os.path.getsize(pdf) // 1024, 'kB')
