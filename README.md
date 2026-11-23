# Seagrass Meadows & AI — Presentation

🌊 **Live presentation (deployed PDF):**
👉 [https://pglez82.github.io/seagrass_ia_presentation/presentation.pdf](https://pglez82.github.io/seagrass_ia_presentation/presentation.pdf)

## About

*"Artificial Intelligence Applied to the Mapping and Monitoring of Seagrass Meadows"*

- **Authors:** Pablo González (AIC, Universidad de Oviedo) · Fernando García-González (IEO-CSIC)

The talk reviews the state of the art in AI techniques applied to the mapping and
monitoring of seagrass meadows: from classical pixel-by-pixel classification to
modern deep learning methods that exploit spatial context, and emerging lines
based on foundation models and self-supervised learning. It also discusses the
main challenge for operational deployment: dataset/distribution shift in dynamic
marine ecosystems.

## Repository structure

| File / folder       | Description                                        |
| ------------------- | -------------------------------------------------- |
| `presentation.tex`  | Beamer presentation source (main file)             |
| `figures/`          | Figures used in the slides                         |
| `slides_preview/`   | PNG previews of the rendered slides                |
| `abstract.md`       | Talk abstract                                      |
| `generate_figures.py` | Script to regenerate scientific figures |
| `.github/workflows/pages.yml` | CI that compiles and deploys the PDF   |

## Building locally

Requires a LaTeX distribution with `latexmk` (e.g. TeX Live):

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error presentation.tex
```

The output is `presentation.pdf`.

## Deployment

Pushing to `master` (or triggering manually via the Actions tab) runs the
GitHub Actions workflow, which compiles the PDF and deploys it to GitHub Pages:

- Site: https://pglez82.github.io/seagrass_ia_presentation/
- PDF:  https://pglez82.github.io/seagrass_ia_presentation/presentation.pdf
