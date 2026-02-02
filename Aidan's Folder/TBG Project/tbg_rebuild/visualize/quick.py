# tbg_rebuild/visualize/quick.py
from __future__ import annotations

from pathlib import Path
import numpy as np


def _downsample_idx(N: int, max_points: int, seed: int = 0) -> np.ndarray:
    if N <= max_points:
        return np.arange(N, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(N, size=max_points, replace=False))


def save_3d_views_by_layer(
    positions: np.ndarray,
    layer_id: np.ndarray,
    outdir: Path,
    prefix: str,
    *,
    max_points: int = 50_000,
) -> Path:
    """
    Always emits a 3D view (Agg backend) colored by layer.
    Deterministic downsampling for very large N.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as e:
        raise RuntimeError(f"matplotlib required for views but unavailable: {e}") from e

    R = np.asarray(positions, float)
    L = np.asarray(layer_id, int)
    idx = _downsample_idx(len(R), max_points, seed=0)
    R, L = R[idx], L[idx]

    x, y, z = R[:, 0], R[:, 1], R[:, 2]
    layers = np.unique(L)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for ll in layers:
        m = (L == ll)
        ax.scatter(x[m], y[m], z[m], s=2, label=f"layer {int(ll)}")
    ax.set_title(f"{prefix}: 3D (by layer)  N={len(R)}")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    ax.legend(loc="best", markerscale=4)
    fig.tight_layout()

    out = outdir / f"{prefix}_3d_layer.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out
