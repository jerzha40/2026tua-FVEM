"""Reproducible 1D quadratic finite-volume-element convergence comparison.

ILLUSTRATIVE model, not a verified transcription of the user's homework:
   -u'' = pi^2 sin(pi*x),  0<x<1,  u(0)=u(1)=0,
exact solution u(x)=sin(pi*x).

Quadratic Lagrange trial functions use nodes 0, 1/2, 1 on each cell.
Piecewise constant test functions use dual control volumes bounded by two
interface points per cell. Both control volumes and coefficients are formed
according to flux conservation. No numerical Gaussian quadrature is used
as a substitution for the scheme's dual interfaces. We use accurate quadrature
of the load and 10-point Gaussian integration to measure the actual errors.
"""

import csv
import math
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.linalg import solve


def shape(t):
    return np.array(
        [
            (2 * t - 1) * (t - 1),
            4 * t * (1 - t),
            (2 * t - 1) * t,
        ]
    )


def shape_grad(t, h):
    return (
        np.array(
            [
                4 * t - 3,
                4 - 8 * t,
                4 * t - 1,
            ]
        )
        / h
    )


def exact(x):
    return np.sin(np.pi * x)


def exact_prime(x):
    return np.pi * np.cos(np.pi * x)


def forcing(x):
    return (np.pi**2) * np.sin(np.pi * x)


def solve_fvem(n, alpha):
    h = 1.0 / n
    cuts = np.array([(k + t) * h for k in range(n) for t in (alpha, 1 - alpha)])

    # 2n+1 primal nodes (including boundary nodes); 2n-1 interior degrees of freedom.

    A = np.zeros((2 * n - 1, 2 * n - 1))
    F = np.zeros(2 * n - 1)

    for eq in range(2 * n - 1):
        left, right = cuts[eq : eq + 2]
        for sign, x in ((+1.0, left), (-1.0, right)):
            k = min(int(x * n), n - 1)
            local_coordinate = x / h - k
            for local_id, grad in enumerate(shape_grad(local_coordinate, h)):
                global_id = 2 * k + local_id
                if 0 < global_id < 2 * n:
                    A[eq, global_id - 1] += sign * grad
        F[eq] = quad(forcing, left, right, epsabs=2e-13)[0]
    coefficients = np.zeros(2 * n + 1)
    coefficients[1:-1] = solve(A, F)
    return coefficients


def measure(n, coefficients):
    h = 1.0 / n
    t, w = leggauss(10)
    t = (t + 1.0) / 2.0
    w = w / 2.0
    err_0_sq = 0.0
    err_grad_sq = 0.0
    for k in range(n):
        three_coefficients = coefficients[2 * k : 2 * k + 3]
        x = (k + t) * h
        numerical = np.array([shape(tt) @ three_coefficients for tt in t])
        numerical_prime = np.array([shape_grad(tt, h) @ three_coefficients for tt in t])
        err_0_sq += h * np.dot(w, (exact(x) - numerical) ** 2)
        err_grad_sq += h * np.dot(w, (exact_prime(x) - numerical_prime) ** 2)
    return math.sqrt(err_0_sq + err_grad_sq), math.sqrt(err_0_sq)


def main():
    sizes = (8, 16, 32, 64, 128)
    schemes = {
        "ordinary_midpoints": 1.0 / 4.0,
        "gauss_dual_points": (1.0 - 1.0 / math.sqrt(3.0)) / 2.0,
    }
    records = []
    for name, alpha in schemes.items():
        print("\nScheme:", name, "| alpha:", round(alpha, 9))
        print(
            f"{'n':>5} {'H1 error':>13} {'H1 order':>11} {'L2 error':>13} {'L2 order':>11}"
        )
        last = None
        for n in sizes:
            uh = solve_fvem(n, alpha)
            h1, l2 = measure(n, uh)
            h1_order = None if last is None else math.log2(last[0] / h1)
            l2_order = None if last is None else math.log2(last[1] / l2)
            print(
                f"{n:5d} {h1:13.6e} {h1_order if h1_order is not None else float('nan'):11.4f} {l2:13.6e} {l2_order if l2_order is not None else float('nan'):11.4f}"
            )
            records.append(
                {
                    "scheme": name,
                    "n": n,
                    "h1_error": h1,
                    "h1_order": h1_order,
                    "l2_error": l2,
                    "l2_order": l2_order,
                }
            )
            last = (h1, l2)
    path = Path(__file__).with_name("fvem_gauss_comparison_demo.csv")
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print("\nCSV:", path)


if __name__ == "__main__":
    main()
