# Numerical Methods for Machine Learning — From First Principles

A six-week, graduate-level course on the **numerical methods that underpin modern machine
learning**, delivered as self-contained, fully executed Jupyter notebooks. Every algorithm
is derived from first principles and implemented in **pure NumPy** — no SciPy,
scikit-learn, autograd, JAX, or deep-learning frameworks appear anywhere. We build the
tools, including a complete reverse-mode automatic-differentiation engine, ourselves.

The course is organized around a single discipline that separates code that merely runs
from code that can be trusted at scale:

> **Understand the error. Respect the conditioning. Choose a stable algorithm. Verify every gradient.**

Because nothing is delegated to a numerical library, every result is validated against an
independent reference: a closed-form analytic expression, a finite-difference gradient
check, or NumPy's own linear algebra used strictly as an external oracle — never as part of
the implementation.

---

## Why this course

Most machine-learning education treats the numerical layer as a black box: call
`loss.backward()`, call an optimizer, trust the linear-algebra routines. This course opens
the box. It rebuilds the softmax that doesn't overflow, the SVD behind PCA, the
backpropagation engine that computes gradients, the Adam optimizer that trains the network,
and the Cholesky solve that powers Gaussian-process inference — all from scratch, all in
NumPy, all gradient-checked. The result is the kind of understanding that lets you debug a
training run that silently diverges or a covariance matrix that mysteriously becomes
non-positive-definite.

---

## Curriculum

| Week | Topic | Key ideas | Notebook |
|------|-------|-----------|----------|
| 1 | **Numerical Foundations for ML** | floating point, machine epsilon, catastrophic cancellation, stable softmax & log-sum-exp, stable cross-entropy, vectorization, finite-difference gradient checking | [`01_numerical_foundations_for_ml.ipynb`](notebooks/01_numerical_foundations_for_ml.ipynb) |
| 2 | **Linear Algebra for ML** | LU / Cholesky / Householder QR, power iteration, the QR algorithm, SVD from scratch, PCA, least squares three ways and why conditioning decides accuracy | [`02_linear_algebra_for_ml.ipynb`](notebooks/02_linear_algebra_for_ml.ipynb) |
| 3 | **Automatic Differentiation** | the chain rule and the two modes, forward-mode dual numbers, a complete reverse-mode (backprop) engine on a computational graph, gradient checking, end-to-end learning | [`03_automatic_differentiation.ipynb`](notebooks/03_automatic_differentiation.ipynb) |
| 4 | **Unconstrained Optimization** | gradient descent and its conditioning-dependent rate, heavy-ball momentum, backtracking line search, Newton's method, BFGS | [`04_unconstrained_optimization.ipynb`](notebooks/04_unconstrained_optimization.ipynb) |
| 5 | **Stochastic Optimization & a Neural Net from Scratch** | mini-batch gradients and variance, SGD / AdaGrad / RMSProp / Adam, learning-rate effects, a from-scratch MLP with gradient-checked backprop trained on two-moons | [`05_stochastic_optimization_mlp.ipynb`](notebooks/05_stochastic_optimization_mlp.ipynb) |
| 6 | **Probabilistic Numerics** | Monte Carlo and the √N law, importance sampling, the reparameterization trick, Gaussian-process regression via a stable Cholesky solve, ill-conditioned kernels and jitter | [`06_probabilistic_numerics_gp.ipynb`](notebooks/06_probabilistic_numerics_gp.ipynb) |

Each week is a lecture plus a lab, designed for roughly one focused sitting plus exercises.

---

## What makes it rigorous

- **Pure NumPy, no shortcuts.** The reverse-mode autodiff engine, the SVD, the optimizers,
  and the GP solver are all implemented from primitives. Libraries appear only to *check*
  the from-scratch results.
- **Every gradient is verified.** The autodiff engine (Week 3) and the hand-written MLP
  backpropagation (Week 5) are validated against centered finite differences to roughly
  $10^{-7}$–$10^{-9}$ relative error before being used to train anything.
- **Conditioning as a through-line.** The condition-number / backward-error framework from
  Week 1 recurs in the least-squares accuracy comparison (Week 2), the optimizer
  convergence rates (Week 4), and the kernel-matrix jitter analysis (Week 6).
- **Executed and reproducible.** Notebooks are committed with outputs and figures already
  computed, so they read as a static textbook, and a test suite re-validates the distilled
  implementations.

---

## Getting started

```bash
git clone https://github.com/HAYDARKILIC/numerical_methods_ml.git
cd numerical_methods_ml
python -m venv .venv && source .venv/bin/activate    # optional
pip install -r requirements.txt
jupyter lab                                          # or: jupyter notebook
```

Open any notebook in `notebooks/` and run it top to bottom, or read them as-is — every cell
is already executed with its output and figures embedded.

To re-validate the reusable implementations:

```bash
python tests/test_nmml.py        # plain Python, no pytest needed
# or, if you have pytest:
python -m pytest tests/ -q
```

### Requirements

Python 3.10+ with `numpy`, `matplotlib`, and `jupyterlab` (see
[`requirements.txt`](requirements.txt)). Deliberately minimal — the point of the course is
that nothing else is needed.

---

## Repository layout

```
numerical_methods_ml/
├── notebooks/      # the six weekly notebooks (executed, with outputs)
├── src/
│   └── nmml.py     # reusable from-scratch implementations distilled from the notebooks
├── tests/
│   └── test_nmml.py  # analytic / finite-difference validation of every routine
├── figures/        # (reserved) exported figures
├── data/           # (reserved) datasets
├── requirements.txt
├── LICENSE
└── README.md
```

The `src/nmml.py` module exposes the autograd `Value` node, the `SGD`/`AdaGrad`/`RMSProp`/`Adam`
optimizers, the linear-algebra factorizations and `svd_from_scratch`/`pca`, the `bfgs`
optimizer, the stable `softmax`/`logsumexp`/`cross_entropy` primitives, and `gp_predict` —
all importable directly:

```python
from src.nmml import Value, Adam, svd_from_scratch, gp_predict
```

---

## How to use this repo

- **As a student:** read each derivation, attempt the implementation yourself, then run the
  cell and study the figure and the gradient check. Do the exercises — they extend each
  topic toward research-grade variants (Nesterov acceleration, L-BFGS, randomized SVD,
  the VAE objective, conjugate-gradient GPs).
- **As an instructor:** the six notebooks map onto a one-module graduate course; the
  exercises serve as homework.
- **As a practitioner:** the `src/nmml.py` routines are compact, dependency-light reference
  implementations for when you need to recall exactly how backprop, Adam, or a stable GP
  solve actually works.

---

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

---

*Part of an ongoing series of from-first-principles, visualization-driven course
repositories spanning probability, statistics, time-series, data mining, deep learning,
generative AI, reinforcement learning, and PDE-based image processing.*
