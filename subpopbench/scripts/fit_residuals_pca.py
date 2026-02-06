import argparse
from pathlib import Path
import json
import torch

def ridge_fit(C: torch.Tensor, F: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Solve W = argmin ||F - C W||^2 + lam ||W||^2.

    This preserves the math (same lam), but is more numerically stable:
      - compute in float64 on CPU
      - solve SPD system with Cholesky
    """
    if lam <= 0:
        raise ValueError(f"ridge_lam must be > 0 for a well-posed ridge system, got {lam}")

    # Move to CPU float64 for stable linear algebra
    C = C.detach().to(device="cpu", dtype=torch.float64)
    F = F.detach().to(device="cpu", dtype=torch.float64)

    K = C.shape[1]
    A = C.T @ C + float(lam) * torch.eye(K, dtype=torch.float64)
    B = C.T @ F

    # Cholesky-based solve (same solution as solve(A,B), but stable)
    L = torch.linalg.cholesky(A)
    W = torch.cholesky_solve(B, L)

    return W.to(dtype=torch.float32)


def pca_fit_transform(X: torch.Tensor, k: int):
    """
    PCA via SVD on centered data.
    Returns:
      Z = (X - mu) @ V_k   [N, k]
      mu: [d]
      V_k: [d, k]
    """
    X = X.float()
    mu = X.mean(dim=0, keepdim=True)
    Xc = X - mu
    # economy SVD
    # Xc = U S V^T ; principal directions are columns of V
    U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    V = Vt.T
    Vk = V[:, :k].contiguous()
    Z = (Xc @ Vk).contiguous()
    return Z, mu.squeeze(0), Vk

def apply_pca(X: torch.Tensor, mu: torch.Tensor, Vk: torch.Tensor) -> torch.Tensor:
    Xc = X.float() - mu.view(1, -1)
    return (Xc @ Vk).contiguous()

def load_concepts(cache_path: str, use_mask: torch.Tensor):
    cache = torch.load(cache_path, map_location="cpu")
    if isinstance(cache, dict) and "concepts" in cache:
        cache = cache["concepts"]
    C = cache.float()
    if use_mask is not None:
        C = C[:, use_mask]
    return C.contiguous()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept_meta", required=True)
    ap.add_argument("--concept_va", required=True)
    ap.add_argument("--concept_te", required=True)
    ap.add_argument("--feat_va", required=True)
    ap.add_argument("--feat_te", required=True)
    ap.add_argument("--block_channels", default="", help="comma-separated channels to block (optional)")
    ap.add_argument("--ridge_lam", type=float, default=1e-2)
    ap.add_argument("--pca_k", type=int, default=64)
    ap.add_argument("--out_va", required=True)
    ap.add_argument("--out_te", required=True)
    ap.add_argument("--out_meta", required=True)
    args = ap.parse_args()

    with open(args.concept_meta, "r") as f:
        meta = json.load(f)

    channels = meta.get("channel", None)
    use_in_training = meta.get("use_in_training", None)

    K_full = len(channels) if channels is not None else None
    block = set([c.strip() for c in args.block_channels.split(",") if c.strip()])

    if K_full is None:
        # if no channels, just use everything
        use_mask = None
    else:
        channel_mask = torch.tensor([c not in block for c in channels], dtype=torch.bool)
        if use_in_training is None:
            use_mask = channel_mask
        else:
            use_mask = torch.as_tensor(use_in_training, dtype=torch.bool) & channel_mask

    # Load concepts (va/te) with same mask as CR would use
    C_va = load_concepts(args.concept_va, use_mask)
    C_te = load_concepts(args.concept_te, use_mask)

    # Load features (va/te)
    F_va = torch.load(args.feat_va, map_location="cpu")
    F_va = F_va["feats"] if isinstance(F_va, dict) and "feats" in F_va else F_va
    F_te = torch.load(args.feat_te, map_location="cpu")
    F_te = F_te["feats"] if isinstance(F_te, dict) and "feats" in F_te else F_te
    F_va = F_va.float().contiguous()
    F_te = F_te.float().contiguous()

    assert C_va.shape[0] == F_va.shape[0], f"va N mismatch: C {C_va.shape} vs F {F_va.shape}"
    assert C_te.shape[0] == F_te.shape[0], f"te N mismatch: C {C_te.shape} vs F {F_te.shape}"
    print(f"C_va {tuple(C_va.shape)}  F_va {tuple(F_va.shape)}")

    # Fit ridge on VA (since stage2 trains on VA) – consistent & avoids leakage
    W = ridge_fit(C_va, F_va, lam=args.ridge_lam)  # [K_used, d]

    # Residuals
    R_va = (F_va - C_va @ W).contiguous()
    R_te = (F_te - C_te @ W).contiguous()

    # PCA compress on residuals (fit on VA)
    Z_va, mu, Vk = pca_fit_transform(R_va, k=args.pca_k)
    Z_te = apply_pca(R_te, mu, Vk)

    out_va = Path(args.out_va); out_va.parent.mkdir(parents=True, exist_ok=True)
    out_te = Path(args.out_te); out_te.parent.mkdir(parents=True, exist_ok=True)

    torch.save({"resid": Z_va}, out_va)
    torch.save({"resid": Z_te}, out_te)

    out_meta = Path(args.out_meta); out_meta.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "ridge_lam": args.ridge_lam,
        "pca_k": args.pca_k,
        "blocked_channels": sorted(list(block)),
        "use_mask_sum": int(use_mask.sum().item()) if use_mask is not None else None,
        "W": W,      # optional: keep for reproducibility
        "mu": mu,
        "Vk": Vk
    }, out_meta)

    print(f"Saved resid va: {out_va}  shape={tuple(Z_va.shape)}")
    print(f"Saved resid te: {out_te}  shape={tuple(Z_te.shape)}")
    print(f"Saved resid meta: {out_meta}")

if __name__ == "__main__":
    main()
