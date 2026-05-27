"""Smoke tests for the from-scratch implementations in src/nmml.py.

These tests validate every routine against an analytic identity, a finite-difference
gradient, or NumPy's reference linear algebra used strictly as an external oracle. Run with:

    python -m pytest tests/ -q          # if pytest is installed
    python tests/test_nmml.py           # plain-Python fallback (no pytest required)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.nmml import (  # noqa: E402
    Value, Adam, SGD, AdaGrad, RMSProp,
    svd_from_scratch, pca, lstsq_svd, lu_solve, cholesky, chol_solve,
    qr_householder, power_iteration, qr_algorithm, bfgs,
    softmax, logsumexp, cross_entropy, numerical_gradient, gp_predict,
)

RNG = np.random.default_rng(0)


def test_autograd_matches_finite_differences():
    xs = [Value(v) for v in [0.5, -1.2, 0.3]]
    out = (xs[0] * xs[1]).tanh() + xs[2].exp() / (1 + xs[0] ** 2)
    out.backward()
    g_auto = np.array([x.grad for x in xs])
    f = lambda t: np.tanh(t[0] * t[1]) + np.exp(t[2]) / (1 + t[0] ** 2)
    g_num = numerical_gradient(f, np.array([0.5, -1.2, 0.3]))
    assert np.allclose(g_auto, g_num, atol=1e-6)


def test_lu_solve_matches_reference():
    A = RNG.standard_normal((6, 6)); b = RNG.standard_normal(6)
    assert np.allclose(lu_solve(A, b), np.linalg.solve(A, b))


def test_cholesky_factorization():
    M = RNG.standard_normal((5, 5)); S = M @ M.T + 3 * np.eye(5)
    L = cholesky(S)
    assert np.allclose(L @ L.T, S)
    b = RNG.standard_normal(5)
    assert np.allclose(chol_solve(L, b), np.linalg.solve(S, b))


def test_qr_orthogonality():
    Q, R = qr_householder(RNG.standard_normal((8, 4)))
    assert np.allclose(Q.T @ Q, np.eye(4), atol=1e-10)
    assert np.allclose(Q @ R, qr_householder(Q @ R)[0] @ qr_householder(Q @ R)[1], atol=1e-8)


def test_svd_reconstruction_and_singular_values():
    A = RNG.standard_normal((7, 4))
    U, s, Vt = svd_from_scratch(A)
    assert np.allclose(U @ np.diag(s) @ Vt, A, atol=1e-6)
    assert np.allclose(np.sort(s), np.sort(np.linalg.svd(A, compute_uv=False)), atol=1e-6)


def test_pca_variance_explained_sums_to_one():
    X = RNG.standard_normal((100, 5)) * np.array([5, 4, 1, 1, 1])
    _, _, _, ve = pca(X, k=5)
    assert abs(ve.sum() - 1.0) < 1e-8
    assert ve[0] >= ve[1]  # sorted descending


def test_lstsq_svd_on_illconditioned_design():
    base = RNG.standard_normal((60, 1))
    X = base + 1e-5 * RNG.standard_normal((60, 4))
    w = RNG.standard_normal(4)
    y = X @ w + 1e-3 * RNG.standard_normal(60)
    assert np.linalg.norm(X @ lstsq_svd(X, y) - y) < 0.1


def test_power_iteration_dominant_eigenpair():
    M = RNG.standard_normal((5, 5)); S = M + M.T
    lam, v = power_iteration(S, iters=5000, tol=1e-14)
    assert np.linalg.norm(S @ v - lam * v) < 1e-6
    ref = np.linalg.eigvalsh(S)
    assert np.isclose(abs(lam), np.max(np.abs(ref)), atol=1e-5)


def test_qr_algorithm_spectrum():
    M = RNG.standard_normal((4, 4)); S = M + M.T
    evals, _ = qr_algorithm(S, iters=800)
    assert np.allclose(np.sort(evals), np.sort(np.linalg.eigvalsh(S)), atol=1e-6)


def test_bfgs_minimizes_rosenbrock():
    f = lambda v: (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2
    g = lambda v: np.array([-2 * (1 - v[0]) - 400 * v[0] * (v[1] - v[0] ** 2),
                            200 * (v[1] - v[0] ** 2)])
    x = bfgs(f, g, np.array([-1.2, 1.0]))
    assert f(x) < 1e-6


def test_softmax_and_logsumexp_stability():
    z = np.array([1000.0, 1001.0, 1002.0])
    p = softmax(z)
    assert np.isclose(p.sum(), 1.0) and np.all(np.isfinite(p))
    assert np.isfinite(logsumexp(z))


def test_cross_entropy_shift_invariance():
    Z = RNG.standard_normal((5, 4)); y = RNG.integers(0, 4, 5)
    assert np.allclose(cross_entropy(Z, y), cross_entropy(Z + 500.0, y))


def test_adam_minimizes_quadratic():
    opt = Adam(lr=0.1); theta = np.ones(4)
    for _ in range(200):
        theta = opt.step(theta, 2 * theta)   # grad of ||theta||^2
    assert np.linalg.norm(theta) < 1e-2


def test_all_optimizers_decrease_loss():
    A = RNG.standard_normal((20, 4)); b = RNG.standard_normal(20)
    grad = lambda w: A.T @ (A @ w - b) / 20
    loss = lambda w: 0.5 * np.mean((A @ w - b) ** 2)
    for Opt, kw in [(SGD, dict(lr=0.05, momentum=0.9)), (AdaGrad, dict(lr=0.3)),
                    (RMSProp, dict(lr=0.05)), (Adam, dict(lr=0.05))]:
        opt = Opt(**kw); w = np.zeros(4); l0 = loss(w)
        for _ in range(300):
            w = opt.step(w, grad(w))
        assert loss(w) < l0


def test_gp_interpolates_and_reports_uncertainty():
    Xtr = np.sort(RNG.uniform(-2, 2, 12))[:, None]
    ytr = np.sin(3 * Xtr.ravel())
    mean, var = gp_predict(Xtr, ytr, Xtr, length=0.6, noise=1e-6)
    assert np.allclose(mean, ytr, atol=0.05)        # near-interpolation at train points
    assert np.all(var >= 0)                          # valid variances


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
    return passed == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
