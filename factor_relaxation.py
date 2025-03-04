"""
[1] Halmos, P., Liu, X., Gold, J., & Raphael, B. (2024). Low-Rank Optimal Transport through Factor Relaxation with Latent Coupling.
In Proceedings of the Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024).
"""

# to do
# change numpy to get backend

import numpy as np


import ot


def initialize_couplings(
    a: np.array, b: np.array, r: int, reg_init=1, seed=42
) -> tuple[np.array, np.array, np.array]:
    n = a.shape[0]
    m = b.shape[0]

    np.random.seed(seed)
    C_Q = np.random.uniform(size=(n, r))
    C_R = np.random.uniform(size=(m, r))
    C_T = np.random.uniform(size=(r, r))

    g_Q, g_R = np.full(r, 1 / r), np.full(r, 1 / r)  # Shape (r,) and (r,)

    Q = ot.sinkhorn(a, g_Q, C_Q, reg_init)
    R = ot.sinkhorn(b, g_R, C_R, reg_init)
    T = ot.sinkhorn(
        Q.T @ np.ones(n, dtype=float), R.T @ np.ones(m, dtype=float), C_T, reg_init
    )

    return Q, R, T


def compute_gradient_Q(C, Q, R, X, g_Q):
    n = Q.shape[0]

    term1 = (C @ R) @ X.T  # parantheses for faster multiplication order
    term2 = np.ones((n, 1), dtype=float) @ (
        np.diag((term1.T @ Q) @ np.diag(1 / g_Q)).reshape(1, -1)
    )
    grad_Q = term1 - term2

    return grad_Q


def compute_gradient_R(C, Q, R, X, g_R):
    m = R.shape[0]

    term1 = (C.T @ Q) @ X
    term2 = np.ones((m, 1), dtype=float) @ (
        np.diag(np.diag(1 / g_R) @ (R.T @ term1)).reshape(1, -1)
    )
    grad_R = term1 - term2

    return grad_R


def compute_gradient_T(Q, R, C, g_Q, g_R):
    return np.diag(1 / g_Q) @ Q.T @ C @ R @ np.diag(1 / g_R)


def compute_distance(Q_new, R_new, T_new, Q, R, T):
    return (
        np.sum((Q_new - Q) ** 2) + np.sum((R_new - R) ** 2) + np.sum((T_new - T) ** 2)
    )


"""
# the bug is here (numerical stability)
def solve_semi_relaxed_projection(K, gamma, tau, a, b, delta, max_iter=100):
    n, r = K.shape
    u = np.ones(n, dtype=float)
    v = np.ones(r, dtype=float)
    for _ in range(max_iter):
        u_prime = u
        v_prime = v
        u = a / (K @ v)
        v = np.power((b / (K.T @ u)), tau / (tau + 1 / gamma))
        if (
            max(
                np.max(np.abs(np.log(u) - np.log(u_prime))),
                np.max(np.abs(np.log(v) - np.log(v_prime))),
            )
            < delta * gamma
        ):
            return np.diag(u) @ K @ np.diag(v)
"""


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
    ---------- numItermax=10000
 numItermax=10000
 numItermax=10000
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
    g_Q = Q.T @ ones_n
    g_R = R.T @ ones_m

    for it in range(max_iter):
        grad_Q = compute_gradient_Q(C, Q, R, X, g_Q)  # Shape (n,r)
        grad_R = compute_gradient_R(C, Q, R, X, g_R)  # Shape (m,r)

        gamma_k = gamma / max(
            np.max(np.abs(grad_Q)), np.max(np.abs(grad_R))
        )  # l-inf normalization

        Q_new = ot.sinkhorn_unbalanced(
            a=a, b=g_Q, M=grad_Q, reg=1 / gamma_k, c=Q, reg_m=[float("inf"), tau]
        )

        R_new = ot.sinkhorn_unbalanced(
            a=b, b=g_R, M=grad_R, reg=1 / gamma_k, c=R, reg_m=[float("inf"), tau]
        )

        g_Q = Q_new.T @ ones_n
        g_R = R_new.T @ ones_m

        grad_T = compute_gradient_T(Q_new, R_new, C, g_Q, g_R)  # Shape (r, r)

        gamma_T = gamma / np.max(np.abs(grad_T))

        T_new = ot.sinkhorn_unbalanced(M=grad_T, a=g_R,b=g_Q,reg=1/gamma_T,c=T,reg_m=[float("inf"),float("inf")])  # Shape (r, r)

        X_new = np.diag(1 / g_Q) @ T_new @ np.diag(1 / g_R)  # Shape (r, r)

        if compute_distance(Q_new, R_new, T_new, Q, R, T) < gamma_k * gamma_k * epsilon:
            return Q_new @ X_new @ R_new.T  # Shape (n, m)

        Q, R, T, X = Q_new, R_new, T_new, X_new
