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
The lockfile specifies Python 3.11, NumPy, OpenCV, Matplotlib, ipykernel, and **CPU-only** PyTorch/torchvision. Local `uv sync` does not download CUDA packages. For GPU runs, use Colab or select the shared local CUDA interpreter below.
If the kernel is not listed, reload the VS Code window, then select `.venv/bin/python` again.
`230536E_a03.ipynb` currently has no assignment solutions, but opens as a valid notebook.

VS Code runs notebook kernels directly; no separate JupyterLab server is needed.

## Shared local CUDA PyTorch (VS Code GPU option)

A reusable uv environment is installed at `~/.local/share/uv/venvs/pytorch-cu121` (Python 3.11, PyTorch 2.4.1+cu121, torchvision 0.19.1+cu121, ipykernel, Matplotlib, and OpenCV). It was tested on the local RTX 4050: `torch.cuda.is_available()` was true and a CUDA tensor operation succeeded. **Miniconda is not required.** Check it any time with:

```bash
~/.local/share/uv/venvs/pytorch-cu121/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA GPU")'
```

For **this project's notebooks**, open the folder in VS Code, open a `.ipynb`, click **Select Kernel → Python Environments → Enter interpreter path**, and enter the full path `/home/rus1ru/.local/share/uv/venvs/pytorch-cu121/bin/python`. Run from the repo root so `images/` resolves. In a PyTorch notebook, place tensors and models on the GPU explicitly; having a CUDA kernel alone does not move them:

```python
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
# model = model.to(DEVICE); inputs = inputs.to(DEVICE); labels = labels.to(DEVICE)
```

For **another project** needing different packages, use that project's own uv `.venv` and dependencies. The simplest option is to select the shared interpreter above when only its packages are needed. If you need both project-local packages and the shared GPU packages in one kernel, create a `.pth` file in the project's `.venv/lib/python3.11/site-packages/` containing this one line:

```text
/home/rus1ru/.local/share/uv/venvs/pytorch-cu121/lib/python3.11/site-packages
```

Install project-specific packages into the local `.venv` with `uv pip install --python .venv/bin/python ...`, then select `.venv/bin/python` in VS Code. **Do not install `torch` or `torchvision` locally** in that project: local packages take precedence over the `.pth` path, and a local CPU build would hide the shared GPU build. Verify with `./.venv/bin/python -c 'import torch; print(torch.__file__, torch.cuda.is_available())'`. This shared-path option requires matching Python 3.11 and Linux, and is not a fully isolated/reproducible uv lock; changing the shared environment affects every project using it. A future `uv sync` can remove unmanaged `.pth` files or install locally declared PyTorch dependencies, so recheck the import path and GPU availability after syncing. A normal `uv sync` for *this* repository installs its CPU-only PyTorch and will therefore keep the CPU kernel; use the shared interpreter for GPU work instead. `uv venv --system-site-packages` does not expose packages from another venv.

**Storage:** the shared CUDA environment uses several GB and the filesystem had about 2.3 GB free at verification. Avoid another CUDA installation without checking free space. Colab remains an alternative for GPU runs.

## Submission

A PDF export of the notebook (`230536E_a01.pdf`) is submitted separately as required
by the assignment guidelines, and includes representative results, parameter choices,
and discussion/interpretation for each question.
