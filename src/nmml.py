"""nmml — Numerical Methods for Machine Learning, distilled reusable implementations.

These are the from-scratch, pure-NumPy routines developed and explained across the six
weekly notebooks, collected so they can be imported and reused. For derivations, figures,
and gradient-check validation, see the corresponding notebook in ``notebooks/``.

    >>> from src.nmml import Value, Adam, svd_from_scratch, gp_predict

The module deliberately depends on NumPy alone — no SciPy, scikit-learn, or autodiff
frameworks — in keeping with the course's from-first-principles philosophy.
"""
from __future__ import annotations
import numpy as np

__all__ = [
    # week 1
    "logsumexp", "softmax", "cross_entropy", "numerical_gradient",
    # week 2
    "lu_solve", "cholesky", "chol_solve", "qr_householder",
    "power_iteration", "qr_algorithm", "svd_from_scratch", "pca", "lstsq_svd",
    # week 3
    "Value",
    # week 4
    "backtracking_line_search", "gradient_descent", "bfgs",
    # week 5
    "SGD", "AdaGrad", "RMSProp", "Adam",
    # week 6
    "rbf_kernel", "gp_predict",
]


# ============================================================ Week 1: numerics
def logsumexp(z, axis=-1):
    c = np.max(z, axis=axis, keepdims=True)
    return (c + np.log(np.sum(np.exp(z - c), axis=axis, keepdims=True))).squeeze(axis)


def softmax(z):
    c = np.max(z, axis=-1, keepdims=True)
    e = np.exp(z - c)
    return e / np.sum(e, axis=-1, keepdims=True)


def cross_entropy(logits, y):
    idx = np.arange(len(y))
    return -logits[idx, y] + logsumexp(logits, axis=-1)


def numerical_gradient(f, theta, h=1e-5):
    theta = np.asarray(theta, float)
    g = np.zeros_like(theta)
    it = np.nditer(theta, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = theta[i]
        theta[i] = old + h; fp = f(theta)
        theta[i] = old - h; fm = f(theta)
        theta[i] = old
        g[i] = (fp - fm) / (2 * h)
        it.iternext()
    return g


# ============================================================ Week 2: linear algebra
def _fwd(L, b):
    n = len(b); y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]
    return y


def _back(U, y):
    n = len(y); x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def lu_solve(A, b):
    A = np.array(A, float); n = A.shape[0]
    L = np.eye(n); P = np.arange(n)
    for k in range(n - 1):
        p = k + np.argmax(np.abs(A[k:, k]))
        if p != k:
            A[[k, p]] = A[[p, k]]; P[[k, p]] = P[[p, k]]; L[[k, p], :k] = L[[p, k], :k]
        for i in range(k + 1, n):
            L[i, k] = A[i, k] / A[k, k]
            A[i, k:] -= L[i, k] * A[k, k:]
    b = np.asarray(b, float)
    return _back(np.triu(A), _fwd(L, b[P]))


def cholesky(A):
    A = np.array(A, float); n = A.shape[0]; L = np.zeros((n, n))
    for j in range(n):
        s = A[j, j] - L[j, :j] @ L[j, :j]
        if s <= 0:
            raise np.linalg.LinAlgError("matrix not positive definite")
        L[j, j] = np.sqrt(s)
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - L[i, :j] @ L[j, :j]) / L[j, j]
    return L


def chol_solve(L, b):
    n = len(b); y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - L[i + 1:, i] @ x[i + 1:]) / L[i, i]
    return x


def qr_householder(A):
    A = np.array(A, float); m, n = A.shape; Q = np.eye(m)
    for k in range(n):
        x = A[k:, k].copy()
        e = np.zeros_like(x); e[0] = np.copysign(np.linalg.norm(x), -x[0])
        v = x - e; nv = np.linalg.norm(v)
        if nv == 0:
            continue
        v = v / nv
        A[k:, k:] -= 2.0 * np.outer(v, v @ A[k:, k:])
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ v, v)
    return Q[:, :n], np.triu(A[:n])


def power_iteration(A, iters=1000, tol=1e-12, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(A.shape[0]); v /= np.linalg.norm(v)
    lam_old = 0.0
    for _ in range(iters):
        w = A @ v; v = w / np.linalg.norm(w)
        lam = v @ A @ v
        if abs(lam - lam_old) < tol:
            break
        lam_old = lam
    return lam, v


def qr_algorithm(A, iters=500):
    A = np.array(A, float); n = A.shape[0]; V = np.eye(n)
    for _ in range(iters):
        Q, R = qr_householder(A)
        A = R @ Q; V = V @ Q
    return np.diag(A), V


def svd_from_scratch(A, iters=800):
    A = np.array(A, float)
    evals, V = qr_algorithm(A.T @ A, iters=iters)
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 0, None); V = V[:, order]
    sigma = np.sqrt(evals)
    U = np.zeros((A.shape[0], len(sigma)))
    for i, s in enumerate(sigma):
        U[:, i] = (A @ V[:, i]) / s if s > 1e-12 else 0.0
    return U, sigma, V.T


def pca(X, k):
    X = np.array(X, float); mean = X.mean(0); Xc = X - mean
    U, s, Vt = svd_from_scratch(Xc)
    comps = Vt[:k]
    return mean, comps, Xc @ comps.T, (s ** 2) / (s ** 2).sum()


def lstsq_svd(X, y, rcond=1e-12):
    U, s, Vt = svd_from_scratch(X)
    s_inv = np.where(s > rcond * s.max(), 1.0 / s, 0.0)
    return Vt.T @ (s_inv * (U.T @ y))


# ============================================================ Week 3: autograd
class Value:
    """Scalar node in a reverse-mode automatic-differentiation graph."""

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, p):
        assert isinstance(p, (int, float))
        out = Value(self.data ** p, (self,), f"**{p}")
        def _backward():
            self.grad += p * self.data ** (p - 1) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(np.exp(self.data), (self,), "exp")
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(np.log(self.data), (self,), "log")
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, o):
        return self + (-o if isinstance(o, Value) else Value(-o))

    def __rsub__(self, o):
        return (Value(o) if not isinstance(o, Value) else o) + (-self)

    def __radd__(self, o):
        return self + o

    def __rmul__(self, o):
        return self * o

    def __truediv__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        return self * o ** -1

    def __rtruediv__(self, o):
        return (o if isinstance(o, Value) else Value(o)) * self ** -1

    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"


# ============================================================ Week 4: optimization
def backtracking_line_search(f, grad, x, p, alpha0=1.0, c1=1e-4, rho=0.5, maxit=50):
    fx = f(x); gp = grad(x) @ p; alpha = alpha0
    for _ in range(maxit):
        if f(x + alpha * p) <= fx + c1 * alpha * gp:
            return alpha
        alpha *= rho
    return alpha


def gradient_descent(grad, x0, alpha, iters=500, tol=1e-10):
    x = np.array(x0, float)
    for _ in range(iters):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        x = x - alpha * g
    return x


def bfgs(f, grad, x0, iters=200, tol=1e-10):
    x = np.array(x0, float); n = x.size
    H = np.eye(n); I = np.eye(n); g = grad(x)
    for _ in range(iters):
        if np.linalg.norm(g) < tol:
            break
        p = -H @ g
        a = backtracking_line_search(f, grad, x, p, alpha0=1.0)
        x_new = x + a * p; g_new = grad(x_new)
        s = x_new - x; y = g_new - g; sy = s @ y
        if sy > 1e-12:
            rho = 1.0 / sy
            V = I - rho * np.outer(s, y)
            H = V @ H @ V.T + rho * np.outer(s, s)
        x, g = x_new, g_new
    return x


# ============================================================ Week 5: stochastic optimizers
class SGD:
    def __init__(self, lr=0.1, momentum=0.0):
        self.lr, self.mu, self.v = lr, momentum, None

    def step(self, theta, g):
        if self.v is None:
            self.v = np.zeros_like(theta)
        self.v = self.mu * self.v + g
        return theta - self.lr * self.v


class AdaGrad:
    def __init__(self, lr=0.1, eps=1e-8):
        self.lr, self.eps, self.G = lr, eps, None

    def step(self, theta, g):
        if self.G is None:
            self.G = np.zeros_like(theta)
        self.G += g * g
        return theta - self.lr * g / (np.sqrt(self.G) + self.eps)


class RMSProp:
    def __init__(self, lr=0.01, beta=0.9, eps=1e-8):
        self.lr, self.beta, self.eps, self.s = lr, beta, eps, None

    def step(self, theta, g):
        if self.s is None:
            self.s = np.zeros_like(theta)
        self.s = self.beta * self.s + (1 - self.beta) * g * g
        return theta - self.lr * g / (np.sqrt(self.s) + self.eps)


class Adam:
    def __init__(self, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = self.v = None; self.t = 0

    def step(self, theta, g):
        if self.m is None:
            self.m = np.zeros_like(theta); self.v = np.zeros_like(theta)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ============================================================ Week 6: probabilistic
def rbf_kernel(X1, X2, length=1.0, var=1.0):
    s1 = np.sum(X1 ** 2, 1)[:, None]; s2 = np.sum(X2 ** 2, 1)[None, :]
    d2 = np.maximum(s1 - 2 * X1 @ X2.T + s2, 0.0)
    return var * np.exp(-0.5 * d2 / length ** 2)


def gp_predict(Xtr, ytr, Xte, length=1.0, var=1.0, noise=1e-2, jitter=1e-8):
    K = rbf_kernel(Xtr, Xtr, length, var) + (noise + jitter) * np.eye(len(Xtr))
    L = cholesky(K)
    alpha = chol_solve(L, ytr)
    Ks = rbf_kernel(Xtr, Xte, length, var)
    mean = Ks.T @ alpha
    v = np.array([chol_solve(L, Ks[:, j]) for j in range(Xte.shape[0])]).T
    cov = rbf_kernel(Xte, Xte, length, var) - Ks.T @ v
    return mean, np.clip(np.diag(cov), 0, None)
