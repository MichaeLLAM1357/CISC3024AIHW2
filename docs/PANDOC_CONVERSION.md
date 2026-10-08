# Converting the report with Pandoc

The report source is `CISC3024_AI2_Report.md`. It is deliberately kept in
portable Markdown — standard pipe tables, `![caption](figures/x.png)` images,
`$...$` / `$$...$$` LaTeX maths, one heading hierarchy — so that Pandoc converts
it cleanly to Word and PDF.

> **These commands are standard Pandoc usage.** They were **not** executed on the
> machine that produced this repository, because Pandoc is not installed there
> (verified: `pandoc`, `soffice`, `wkhtmltopdf` all absent). The `.docx` and
> `.pdf` in this repository were produced by an equivalent in-house pipeline
> instead — see "If you cannot install Pandoc" below.

## 1. Prerequisites

```bash
pandoc --version            # need >= 2.11 for --mathml / OMML handling
```

For **PDF** you additionally need a LaTeX engine:

```bash
# Windows: install MiKTeX or TeX Live, then
xelatex --version
```

For **DOCX** nothing else is needed — Word opens the file directly.

## 2. Markdown → Word (.docx)

Run from the repository root, so `figures/` resolves:

```bash
pandoc CISC3024_AI2_Report.md \
  -o CISC3024_AI2_Report.docx \
  --from=gfm+tex_math_dollars \
  --toc --toc-depth=3 \
  --standalone
```

What each part does:

| Flag | Why |
|---|---|
| `--from=gfm+tex_math_dollars` | GitHub-flavoured Markdown (pipe tables, fenced code) **plus** `$...$` maths. Without `tex_math_dollars` your formulas arrive as literal `$`. |
| `--toc --toc-depth=3` | Generates a table of contents from the `#`/`##`/`###` headings. Word shows a "update field" prompt on open — accept it to populate page numbers. |
| `--standalone` | Emits a complete document rather than a fragment. |

### A note on `--mathml`

You asked for `--mathml`. **For `.docx` you should leave it off.**

* Pandoc's **default** for the Word writer is **OMML** — Office Math Markup, the
  native format Word's equation editor uses. Equations become real, editable
  Word equations.
* `--mathml` targets the HTML-family writers (`html`, `epub`, `docbook`,
  `openxml`). Passing it for `.docx` emits MathML, which Word renders less
  faithfully than OMML.

So: **use `--mathml` for HTML/EPUB output, and omit it for `.docx`.** If your
supervisor specifically wants MathML in the Word file, add `--mathml` and check
the rendering in Word before submitting.

### Matching the reference formatting

To inherit a stylesheet (fonts, heading sizes, margins) from an existing Word
file, pass it as a reference document:

```bash
pandoc CISC3024_AI2_Report.md \
  -o CISC3024_AI2_Report.docx \
  --from=gfm+tex_math_dollars --toc --toc-depth=3 \
  --reference-doc=CISC3024_AI1_Report.docx      # Assignment #1's file
```

Pandoc copies the reference document's styles — body font, `Heading 1`–`Heading 6`,
`Title`, page size and margins — instead of its own defaults. This is the
supported way to make two reports look like a matched pair.

## 3. Markdown → PDF

### Option A — LaTeX engine (best typography)

```bash
pandoc CISC3024_AI2_Report.md \
  -o CISC3024_AI2_Report.pdf \
  --from=gfm+tex_math_dollars \
  --toc --toc-depth=3 \
  --pdf-engine=xelatex \
  -V mainfont="Times New Roman" \
  -V monofont="Courier New" \
  -V geometry:a4paper \
  -V geometry:margin=2.5cm \
  -V colorlinks=true -V linkcolor=blue
```

`xelatex` (not `pdflatex`) is required to use the system's Times New Roman.
Without a LaTeX engine this step cannot run.

### Option B — no LaTeX installed

```bash
pandoc CISC3024_AI2_Report.md -o report.html --standalone --toc \
  --from=gfm+tex_math_dollars --mathml
```

then print `report.html` to PDF from a browser (Edge/Chrome → Print →
Save as PDF → A4 → margins "Default" → **uncheck** Headers and footers).
This is exactly how the PDF in this repository was made.

## 4. Word → PDF, and checking the layout

1. Open `CISC3024_AI2_Report.docx` in Word.
2. **File → Save As → PDF** (or File → Export → Create PDF/XPS).
   In the dialog choose *Options → Document* and tick
   **PDF/A compliant** only if your school asks for it.
3. Before exporting, fix the three things that actually go wrong:

| Symptom | Cause | Fix |
|---|---|---|
| A figure sits at the bottom of a page with its caption on the next | caption paragraph is not tied to the image | select the image **and** its caption → Paragraph → **Keep with next**; also set the image paragraph's Line spacing to *Single* |
| A table splits across pages mid-row | row breaking allowed | select the table → Table Properties → Row → untick **Allow row to break across pages**; repeat the header row with **Repeat as header row** |
| A display equation breaks across a line | equation too wide | reduce the equation's font size by one step, or set the paragraph to *Don't hyphenate* and add a manual line break inside the equation |

Also run **Review → Word Count** and check the page count matches what you
expect, and scroll once through for a stray `[PENDING]` or a raw `$`.

If LibreOffice is available, the whole conversion can be scripted:

```bash
soffice --headless --convert-to pdf --outdir . CISC3024_AI2_Report.docx
```

## 5. If you cannot install Pandoc

This repository already contains a working `.docx` and `.pdf` built from the
same Markdown by an in-house pipeline (Markdown → HTML → Word → Chromium print
to PDF), with the page geometry of Assignment #1 (A4, 2.54 cm vertical /
3.17 cm horizontal margins). Nothing needs to be installed to use them.

## 6. Suggested submission filenames

UMMoodle usually states a pattern; if it does not, these are unambiguous:

```
CISC3024_AI2_LAM_KA_WAI_UC325629.pdf          ← preferred
CISC3024_AI2_LAM_KA_WAI_UC325629.docx
CISC3024_AI2_UC325629_LAM_KA_WAI.pdf          ← alternative ordering
```

Rules worth following whatever you pick:

* keep the student ID **in the filename** — it is what the grader sorts on;
* use the exact extension you actually produced (do not rename a `.docx` to `.pdf`);
* avoid spaces and Chinese characters in the filename;
* if you submit only one file, submit the **PDF** — its layout is fixed and it
  cannot reflow on the grader's machine.
