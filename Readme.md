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

```
.
├── assignment_01.ipynb      # Main notebook with all solutions
├── images/                  # Input images used by each question (q1.jpeg ... q9.jpeg,
│                             # shells.tif, im01.png, im01small.png, im02.png, im02small.png)
└── README.md
```

> **Note:** The `images/` folder must sit alongside the notebook for the relative
> paths (`images/q1.jpeg`, etc.) used throughout the notebook to resolve correctly.

## Requirements

- Python 3.11+
- `opencv-python`
- `numpy`
- `matplotlib`

Install with:

```bash
pip install opencv-python numpy matplotlib
```

## Running the Notebook

1. Clone this repository:
   ```bash
   git clone https://github.com/rus1ru/EN3160_Assignment_1.git
   cd EN3160_Assignment_1
   ```
2. Install dependencies (see above).
3. Launch Jupyter and run all cells top to bottom:
   ```bash
   jupyter notebook assignment_01.ipynb
   ```

## Submission

A PDF export of the notebook (`230536E_a01.pdf`) is submitted separately as required
by the assignment guidelines, and includes representative results, parameter choices,
and discussion/interpretation for each question.
