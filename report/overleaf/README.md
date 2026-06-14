# Rapport LaTeX (Overleaf)

## Upload

1. Create a new project on [Overleaf](https://www.overleaf.com).
2. Upload **all files** in this folder (`main.tex`, `sections/*.tex`, `figures/*`).
3. Set the main document to `main.tex`.
4. Click **Recompile** to generate `main.pdf`.

## Logo

Replace `figures/university_logo.png` with your university logo, or change the path in `main.tex`:

```latex
\newcommand{\UniversityLogoPath}{figures/university_logo.png}
```

## Local compile (optional)

```bash
pdflatex main.tex
pdflatex main.tex   # run twice for TOC
```
