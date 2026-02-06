import torch
import numpy as np

__all__ = ["compute_ub_eigenval"]

def compute_max_eigenval(H, M=32, device=None, verbose=False):
    """
    Upper bound on the maximum eigenvalue of symmetric H,
    using the same Lanczos logic as compute_lb_ub() in LayerHessians/hessian/hessian.py.
    Returns only ub (no lb) for speed when only max eigenvalue bound is needed.

    Returns:
        ub: upper bound on max eigenvalue (max_eig <= ub)
    """
    assert H.dim() == 2 and H.shape[0] == H.shape[1]
    n = H.shape[0]
    device = device or H.device
    M = min(M or 32, n)  # smaller default M for speed when only ub is needed
    dtype = H.dtype
    H = H.to(device)

    v = torch.randn(n, device=device, dtype=dtype)
    v = v / torch.linalg.norm(v)

    alp = torch.zeros(M, device=device, dtype=dtype)
    bet = torch.zeros(M, device=device, dtype=dtype)
    v_prev = None

    for j in range(M):
        if verbose:
            print("Iteration: [{}/{}]".format(j + 1, M))
            sys.stdout.flush()
        v_next = H @ v
        if j > 0:
            v_next = v_next - bet[j - 1] * v_prev
        alp[j] = v_next.dot(v)
        v_next = v_next - alp[j] * v
        bet[j] = torch.linalg.norm(v_next)
        v_next = v_next / (bet[j] + 1e-14)
        v_prev = v
        v = v_next

    # n = alp.size(0)
    # B = torch.diag(alp) + torch.diag(bet[:-1], diagonal=1) + torch.diag(bet[:-1], diagonal=-1)
    
    # # Compute eigenvalues and eigenvectors
    # ritz_val, S = torch.linalg.eigh(B)
    
    # # Get the maximum eigenvalue and corresponding eigenvector component
    # theta_k = ritz_val[-1]
    # s_k = bet[-1] * S[-1, -1]
    # ub = theta_k + torch.abs(s_k)
    
    # return ub.item()

    alp_np = alp.cpu().numpy()
    bet_np = bet.cpu().numpy()
    B = np.diag(alp_np) + np.diag(bet_np[:-1], k=1) + np.diag(bet_np[:-1], k=-1)
    ritz_val, S = np.linalg.eigh(B)

    # only ub (max eigenvalue bound)
    theta_k = ritz_val[-1]
    s_k = float(bet_np[-1]) * float(S[-1, -1])
    ub = theta_k + abs(s_k)
    return ub