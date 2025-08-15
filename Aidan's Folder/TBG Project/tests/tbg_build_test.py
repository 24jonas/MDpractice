import numpy as np
from geometry.tbg_builder import build_tbg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def cart_to_frac(R, H):
    return np.dot(R, np.linalg.inv(H).T)

if __name__ == "__main__":
    a, d = 2.46, 3.35
    nx, ny = 10, 10
    theta = 21.79

    R, H, meta = build_tbg(nx=nx, ny=ny, a=a, d=d, theta_deg=theta, registry="AB")

    print(f"Bottom atoms: {meta['n_bottom']}, Top atoms: {meta['n_top']}")
    print(f"Total atoms: {len(R)}")
    print(f"Cell matrix:\n{H}")

    # z-plane check
    z = R[:, 2]
    zmin, zmax = z.min(), z.max()
    print(f"Z min: {zmin:.6f}, expected {-0.5*d:.6f}")
    print(f"Z max: {zmax:.6f}, expected {+0.5*d:.6f}")

    # Wrap check
    S = cart_to_frac(R, H)
    eps = 1e-12
    in_bounds = (S[:, 0] >= -eps) & (S[:, 0] < 1.0 + eps) & \
                (S[:, 1] >= -eps) & (S[:, 1] < 1.0 + eps)
    print(f"All atoms inside cell: {np.all(in_bounds)}")

    # Area & vacuum
    area = np.linalg.norm(np.cross(H[:, 0], H[:, 1]))
    print(f"In-plane area: {area:.6f}")
    print(f"Vacuum height: {H[2, 2]:.6f}")
    
    # Plot
    colors = ['blue' if z_val < 0 else 'orange' for z_val in z]
    plt.figure(figsize=(6, 6))
    plt.scatter(R[:, 0], R[:, 1], s=5, c=colors)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"TBG Quick View\nθ={theta}°, Bottom={meta['n_bottom']}, Top={meta['n_top']}")
    plt.xlabel("x [Å]")
    plt.ylabel("y [Å]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("test_tbg_quickview(1).png", dpi=200)
