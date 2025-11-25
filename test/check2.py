import numpy as np

def build_J(D, eri):
    # J_mn = Σ_rs D_rs (mn|rs)
    return np.tensordot(D, eri, axes=([0, 1], [2, 3]))

def build_K(D, eri, mode="STD"):
    # K_mn = Σ_rs D_rs (mr|ns)
    if mode == "STD":
        eri_perm = eri.transpose(0, 2, 1, 3)  # (mr|ns)
        einsum_expr = "rs,mrns->mn"
    elif mode == "ALT1":
        eri_perm = eri.transpose(0, 3, 2, 1)  # (ms|rn)
        einsum_expr = "rs,msrn->mn"
    elif mode == "ALT2":
        eri_perm = eri.transpose(0, 1, 3, 2)  # (mn|sr)
        einsum_expr = "rs,mnsr->mn"
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return np.einsum(einsum_expr, D, eri_perm, optimize=True)

# ==== テスト用データ ====
nbf = 3  # basis functions
np.random.seed(42)
D = np.random.rand(nbf, nbf)
D = 0.5 * (D + D.T)  # 対称化
eri = np.random.rand(nbf, nbf, nbf, nbf)

# ==== JとKの構築 ====
J = build_J(D, eri)
K_std = build_K(D, eri, mode="STD")
K_alt1 = build_K(D, eri, mode="ALT1")
K_alt2 = build_K(D, eri, mode="ALT2")

# ==== 診断 ====
norm_diff_std = np.linalg.norm(J - K_std)
norm_diff_alt1 = np.linalg.norm(J - K_alt1)
norm_diff_alt2 = np.linalg.norm(J - K_alt2)

EJ = np.einsum("mn,mn->", D, J)
EK_std = np.einsum("mn,mn->", D, K_std)
R_KJ_std = EK_std / EJ if EJ != 0 else np.nan

print("=== JK Diagnostic ===")
print(f"||J-K|| (STD)  = {norm_diff_std:.6e}")
print(f"||J-K|| (ALT1) = {norm_diff_alt1:.6e}")
print(f"||J-K|| (ALT2) = {norm_diff_alt2:.6e}")
print(f"ΣD·J = {EJ:.6f}, ΣD·K(STD) = {EK_std:.6f}, R_KJ(STD) = {R_KJ_std:.6f}")