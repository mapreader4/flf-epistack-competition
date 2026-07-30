# Canonical report source

`main.tex` + `references.bib` here are the source that builds **`report.pdf` at the
repository root** — the version the competition submission links to.

Build:

```sh
cd report/final && latexmk -pdf main.tex && cp main.pdf ../../report.pdf
```

**Why this directory exists.** Previously `report.pdf` was newer than every `.tex` in
`report/`, so there was no source in the repo that reproduced it — editing
`report/main-5.tex` and rebuilding would have silently regressed the paper. The older
`report/main*.tex` files are superseded drafts; **do not build from them.**
