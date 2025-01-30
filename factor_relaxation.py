"""
[1] Halmos, P., Liu, X., Gold, J., & Raphael, B. (2024). Low-Rank Optimal Transport through Factor Relaxation with Latent Coupling.
In Proceedings of the Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024).
"""

# to do
# change numpy to get backend

import numpy as np


import ot


def initialize_couplings(
    a: np.array, b: np.array, r: int, seed=42
) -> tuple[np.array, np.array, np.array]:
    n = a.shape[0]
    m = b.shape[0]
    r = g_Q.shape[0]

    C_Q = np.random.uniform(size=(n, r), seed=seed)
    C_R = np.random.uniform(size=(m, r), seed=seed)
    C_T = np.random.uniform(size=(r, r), seed=seed)

    K_Q = np.exp(C_Q)
    K_R = np.exp(C_R)
    K_T = np.exp(C_T)

    g_Q, g_R = np.full(r, 1 / r), np.full(r, 1 / r)  # Shape (r,) and (r,)

    Q = ot.sinkhorn(a, g_Q, K_Q)
    R = ot.sinkhorn(b, g_R, K_R)
    T = ot.sinkhorn(
        Q.T @ np.ones(n, dtype=float), R.T @ np.ones(m, dtype=float), g_R, K_T
    )

    return Q, R, T


def compute_gradient_Q(C, Q, R, X, g_Q):
    n = Q.shape[0]

    term1 = C @ R @ X.T
    term2 = np.ones(n, dtype=float) @ np.diag(np.diag(1 / g_Q) @ Q.T @ term1)
    grad_Q = term1 - term2

    return grad_Q


def compute_gradient_R(C, Q, R, X, g_R):
    m = R.shape[0]

    term1 = C.T @ Q @ X
    term2 = np.ones(m, dtype=float) @ np.diag(term1.T @ R @ np.diag(1 / g_R))
    grad_R = term1 - term2

    return grad_R


def compute_gradient_T(Q, R, C, g_Q, g_R):
    return np.diag(1 / g_Q) @ Q.T @ C @ R @ np.diag(1 / g_R)


def compute_distance(Q_new, R_new, T_new, Q, R, T):
    return (
        np.sum((Q_new - Q) ** 2) + np.sum((R_new - R) ** 2) + np.sum((T_new - T) ** 2)
    )


def solve_semi_relaxed_projection(K, gamma, tau, a, b, delta, max_iter=100):
    n, r = K.shape
    u = np.ones(n, dtype=float)
    v = np.ones(r, dtype=float)
    for _ in max_iter:
        u_prime = u
        v_prime = v
        u = a / (K @ v)
        v = np.pow((b / (K.T @ u)), tau / (tau + 1 / gamma))
        if (
            np.max(
                np.max(np.abs(np.log(u) - np.log(u_prime))),
                np.max(np.abs(np.log(v) - np.log(v_prime))),
            )
            < delta * gamma
        ):
            return np.diag(u) @ K @ np.diag(v)


def solve_balanced_FRLC(
    C: np.array,
    r: int,
    a: np.array,
    b: np.array,
    tau: float,
    gamma: float,
    delta: float,
    epsilon: float,
    max_iter: int,
) -> np.array:
    """
    Solves the low-rank balanced optimal transport problem using Factor Relaxation
    with Latent Coupling (FRLC).

    Parameters
    ----------
    C : np.array
        Cost matrix of shape (n, m), where n and m are the number of source
        and target points, respectively.
    r : int
        Rank constraint for the transport plan P.
    a : np.array
        Probability distribution vector of size n representing the source marginals.
        Should sum to 1.
    b : np.array
        Probability distribution vector of size m representing the target marginals.
        Should sum to 1.
    tau : float
        Regularization parameter controlling the relaxation of the inner marginals.
    gamma : float
        Step size (learning rate) for the coordinate mirror descent algorithm.
    delta : float
        Lower bound threshold for numerical stability in semi-relaxed projections.
    epsilon : float
        Stopping threshold for the mirror descent optimization.
    max_iter : int
        Maximum number of iterations for the mirror descent optimization.

    Returns
    -------
    P : np.array
        The computed low-rank optimal transport plan of shape (n, m).
    """

    n, m = C.shape
    ones_n, ones_m = (
        np.ones(n, dtype=float),
        np.ones(m, dtype=float),
    )  # Shape (n,) and (m,)
    Q, R, T = initialize_couplings(a, b, r)  # Shape (n,r), (m,r), (r,r)
    X = np.diag(1 / (Q.T @ ones_n)) @ T @ np.diag(1 / (R.T @ ones_m))  # Shape (r,r)
    g_Q = np.dot(Q_new.T, ones_n)
    g_R = np.dot(R_new.T, ones_m)

    for _ in range(max_iter):
        grad_Q = compute_gradient_Q(C, Q, R, X, g_Q)  # Shape (n,r)
        grad_R = compute_gradient_R(C, Q, R, X, g_R)  # Shape (m,r)

        gamma_k = gamma / np.max(
            np.linalg.norm(grad_Q, ord=np.inf), np.linalg.norm(grad_R, ord=np.inf)
        )  # l-inf normalization of Scetbon & Cuturi 2021

        K_Q = Q * np.exp(-gamma_k * grad_Q)  #  Shape (n,r)
        K_R = R * np.exp(-gamma_k * grad_R)  #  Shape (m,r)

        Q_new = solve_semi_relaxed_projection(
            K_Q, gamma_k, tau, a, np.dot(Q.T, ones_n), delta
        )  # Shape (n, r)

        R_new = solve_semi_relaxed_projection(
            K_R, gamma_k, tau, b, np.dot(R.T, ones_m), delta
        )  # Shape (m, r)

        g_Q = np.dot(Q_new.T, ones_n)
        g_R = np.dot(R_new.T, ones_m)

        grad_T = compute_gradient_T(Q_new, R_new, C, g_Q, g_R)  # Shape (r, r)

        gamma_T = gamma / np.linalg.norm(grad_T, ord=np.inf)

        K_T = T * np.exp(-gamma_T * grad_T)  # Shape (r,r)

        T_new = ot.sinkhorn(g_R, g_Q, K_T, reg=delta)  # Shape (r, r)

        X_new = np.diag(1 / g_Q) @ T_new @ np.diag(1 / g_R)  # Shape (r, r)

        if compute_distance(Q_new, R_new, T_new, Q, R, T) < gamma_k * gamma_k * epsilon:
            return Q_new @ X_new @ R_new.T  # Shape (n, m)

        Q, R, T, X = Q_new, R_new, T_new, X_new
