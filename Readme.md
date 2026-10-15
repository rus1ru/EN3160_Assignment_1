# EN3160 Assignment 1 — Intensity Transformations and Neighborhood Filtering

**Name:** Rasanjana W.P.G.R.A.
**Index Number:** 230536E
**Course:** EN3160 — Image Processing and Machine Vision
**Instructor:** Ranga Rodrigo

## Overview

This repository contains the Jupyter Notebook implementation for EN3160 Assignment 1,
covering intensity transformations, histogram-based enhancement, and neighborhood
(spatial) filtering using OpenCV and NumPy.

## Contents

| Question | Topic |
|---|---|
| Q1 | Piecewise-linear intensity transformation (contrast enhancement) |
| Q2 | Accentuating white matter / gray matter in a brain proton-density image |
| Q3 | Gamma correction on the L plane in L\*a\*b\* color space |
| Q4 | Vibrance enhancement via a Gaussian-shaped saturation transform |
| Q5 | Histogram equalization (custom implementation) |
| Q6 | Histogram equalization restricted to the image foreground |
| Q7 | Sobel filtering — via `filter2D`, from scratch (2D), and as a separable 1D convolution |
| Q8 | Image zooming — nearest-neighbor and bilinear interpolation, evaluated with normalized SSD |
| Q9 | Foreground/background segmentation with GrabCut + background blurring |
| Q10 | Bilateral filtering — OpenCV vs. a from-scratch implementation, compared with MSE/MAE/max error |

## Repository Structure

- `230536E_a01.ipynb`: Assignment 1 solutions; uses `images/`.
- `230536E_a02.ipynb`: Assignment 2 notebook (currently work in progress); assets in `images_02/`.
- `230536E_a03.ipynb`: valid empty Assignment 3 notebook, ready for cells.
- `en3160_assignment_0*.pdf`: assignment briefs.

Run notebooks from this repository root so relative image paths resolve.

## Google Colab (default for GPU)

1. Push this repository to GitHub, then open the notebook from its GitHub URL in [Google Colab](https://colab.research.google.com/) (GitHub tab). For example, after pushing, use `https://colab.research.google.com/github/rus1ru/EN3160_Assignment_1/blob/<branch>/230536E_a01.ipynb`; replace `<branch>` with the branch containing your latest work.
2. Select **Runtime → Change runtime type → GPU**. GPU availability and session limits depend on Colab. Use **Runtime → Run all** only after the data setup below.
3. The hosted runtime does not contain local `images/`, `images_02/`, or datasets. Run a setup cell **before** the assignment cells:

   ```python
   !git clone https://github.com/rus1ru/EN3160_Assignment_1.git /content/EN3160_Assignment_1
   %cd /content/EN3160_Assignment_1
   ```

   If you use a branch not yet merged into the remote default branch, check out that branch after cloning. Uncommitted changes on your laptop are not available in Colab. For any dataset not committed to the repository (such as CIFAR-10 or the hymenoptera dataset), download or mount it in the Colab runtime.
4. Colab supplies its own Python packages; check them in a cell before installing anything:

   ```python
   import torch, torchvision, cv2, numpy, matplotlib
   print(torch.__version__, torchvision.__version__, torch.cuda.is_available())
   if torch.cuda.is_available():
       print(torch.cuda.get_device_name(0))
   ```

   If `torch.cuda.is_available()` is false, choose a GPU runtime and reconnect. Colab sessions and temporary files can be lost; save notebook changes to GitHub or Drive.

The repository's `uv.lock` is for local VS Code CPU work, **not** for installing PyTorch in Colab. It does not control Colab's preinstalled GPU libraries. Assignment 3's notebook is currently empty; Assignment 2's notebook is not yet the completed Assignment 2 solution.

## VS Code + uv setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and the VS Code **Python** and **Jupyter** extensions. From this repository root:

```bash
uv sync --locked
```

Open this folder in VS Code and open `230536E_a01.ipynb` (or another nonempty notebook).
Click **Select Kernel** → **Python Environments** → `.venv/bin/python`.
The workspace settings point VS Code to this interpreter and use this folder as the notebook working directory.
The lockfile specifies Python 3.11, NumPy, OpenCV, Matplotlib, ipykernel, and **CPU-only** PyTorch/torchvision. Local `uv sync` does not download CUDA packages. For GPU runs, use Colab or the optional shared local environment below (once verified).
If the kernel is not listed, reload the VS Code window, then select `.venv/bin/python` again.
`230536E_a03.ipynb` currently has no assignment solutions, but opens as a valid notebook.

VS Code runs notebook kernels directly; no separate JupyterLab server is needed.

## Optional shared local CUDA PyTorch (setup in progress)

A reusable **uv** virtual environment is being installed at `~/.local/share/uv/venvs/pytorch-cu121` with Python 3.11, PyTorch 2.4.1 + CUDA 12.1 and torchvision 0.19.1 + CUDA 12.1. This is **not yet verified**: the dependency download is still running, so do not assume that GPU training works locally. Miniconda is not needed.

After installation succeeds, check it from a terminal:

```bash
~/.local/share/uv/venvs/pytorch-cu121/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA GPU")'
```

To use this shared Python for notebooks in VS Code, it must also contain `ipykernel` and the notebook's other imports. Once they have been installed and verified, choose **Select Kernel → Python Environments → Enter interpreter path** and enter `~/.local/share/uv/venvs/pytorch-cu121/bin/python` (expand `~` if necessary). This kernel is shared between projects: every project uses the same PyTorch version and changes to shared packages affect all users of it. Your project-specific `.venv` remains separate and CPU-only.

**Important limitation:** `uv venv --python /path/to/another/venv/bin/python --system-site-packages` does *not* inherit packages installed inside that other venv. Likewise, `uv sync --locked` in this project selects its CPU-only PyTorch; it will not switch to the shared GPU build. Do not install CUDA PyTorch into each project's `.venv` unless you deliberately want separate copies. We will update these instructions when shared-environment setup and verification are complete.

## Submission

A PDF export of the notebook (`230536E_a01.pdf`) is submitted separately as required
by the assignment guidelines, and includes representative results, parameter choices,
and discussion/interpretation for each question.
