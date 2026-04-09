<a id="readme-top"></a>

<p align="center">
  <img src="docs/images/logo.png" alt="AICA-Bench logo" width="144" />
</p>

<h1 align="center">AICA-Bench</h1>

<p align="center">
  <strong>Holistically Examining the Capabilities of VLMs in Affective Image Content Analysis</strong>
</p>

<p align="center">
  Dong She*, Xianrong Yao*, Liqun Chen, Jinghe Yu, Yang Gao, Zhanpeng Jin&dagger;
  <br />
  * Equal contribution. &dagger; Corresponding author.
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.05900">
    <img src="https://img.shields.io/badge/arXiv-2604.05900-b31b1b?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv" />
  </a>
  <a href="https://scut-hai.github.io/AICA-Bench/">
    <img src="https://img.shields.io/badge/Project%20Page-Live-2ea44f?style=flat-square&logo=githubpages&logoColor=white" alt="Project Page" />
  </a>
  <a href="https://github.com/SCUT-HAI/AICA-Bench">
    <img src="https://img.shields.io/badge/Code-GitHub-24292f?style=flat-square&logo=github&logoColor=white" alt="Code" />
  </a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="License: MIT" />
  </a>
  <img src="https://img.shields.io/badge/Python-%3E%3D3.8-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python >=3.8" />
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.05900"><strong>Paper</strong></a>
  |
  <a href="https://scut-hai.github.io/AICA-Bench/"><strong>Project Page</strong></a>
  |
  <a href="#installation"><strong>Installation</strong></a>
  |
  <a href="#usage"><strong>Usage</strong></a>
</p>

> AICA-Bench is a unified benchmark for affective image content analysis with vision-language models, covering Emotion Understanding (EU), Emotion Reasoning (ER), and Emotion-guided Content Generation (EGCG). It spans 9 affective datasets, 18,124 benchmark instructions, and standardized zero-shot evaluation across 23 VLMs.

## Repository Structure

- `src/`, `benchmark/`, `examples/`, `test/`: core open-source benchmark code and examples
- `docs/`: GitHub Pages source and static assets
- `docs/latex/`: local-only reference materials used to prepare the Pages content; this directory is ignored and is not intended to be pushed to GitHub

## Project Page

The GitHub Pages site is stored under `docs/`.

- Entry page: `docs/index.html`
- Static assets: `docs/images/` and `docs/static/`
- Local-only reference materials: `docs/latex/` (ignored by Git)

For GitHub Pages deployment, this repository publishes the `docs/` directory through GitHub Actions in `.github/workflows/static.yml`.
In the GitHub repository settings, set `Pages -> Build and deployment -> Source` to `GitHub Actions` so the workflow deploys `docs/index.html` correctly.

---

## Installation

### For Users

Install the minimal runtime environment:

```bash
# Install in editable mode (recommended for CLI use)
pip install -e .

# Or traditional method
pip install -r requirements.txt
```

### For Developers

To contribute or extend this project, follow the development setup below:

```bash
# 1. Create and activate a virtual environment (recommended)
conda create -n aica-vlm
conda activate aica-vlm

# 2. Install core and dev dependencies
pip install -r requirements.txt -r requirements-dev.txt

# 3. Set up pre-commit hooks
pre-commit install
```

Run pre-commit on all files:

```bash
pre-commit run --all-files
```

## Usage

Once installed, use the CLI tool `aica-vlm` to run dataset construction and instruction generation.

Tip:
The `benchmark_datasets` path refers to a local directory you create and define. It can be renamed or customized according to your own dataset structure.
Currently, the datasets used in `benchmark_datasets` (for example, EmoSet and FI) are based on third-party emotional image datasets. Due to licensing restrictions, we cannot re-release these datasets publicly on GitHub.
If you wish to use them, please contact the original dataset authors directly or follow the download instructions in their official repositories.
If you would like access to our instruction set for dialogue generation, please contact us directly.

### Build Dataset

```bash
aica-vlm build-dataset run benchmark_datasets/example.yaml --mode random
```

- `mode`: `random` (default) or `balanced`

### Build Instruction

```bash
# For Basic instruction generation
aica-vlm build-instruction run benchmark_datasets/example.yaml

# For Chain of Thought (CoT) generation
aica-vlm build-instruction run-cot benchmark_datasets/example_CoT.yaml
```

### Run Evaluation or Benchmark

```bash
aica-vlm benchmark benchmark_datasets/example.yaml
```
