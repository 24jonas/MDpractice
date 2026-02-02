######## Deprecated ########

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np

StrainModel = Literal["directed_small", "random_small"]


@dataclass(frozen=True, slots=True)
class StrainPlan:
    """
    (Deprecated) A minimal "affine in-plane strain" plan.

    Fields:
      - model: which simple strain generator produced this plan
      - F2: 2x2 deformation gradient applied to xy coordinates
      - info: small, human-readable dictionary describing the parameters used

    Notes:
      - This module is kept for reference/back-compatibility with older scripts.
      - The current codebase prefers the non-deprecated strain utilities in
        `tbg_rebuild.strain.*`, but the math here is still valid for small strains.
    """
    model: StrainModel
    F2: np.ndarray                 # (2,2) deformation applied to xy
    info: dict                     # human-readable parameters


def directed_small_strain(
    eps: float,
    theta_deg: float,
    *,
    poisson: float = 0.165,
) -> StrainPlan:
    """
    (Deprecated) Small uniaxial in-plane strain.

    Constructs a small-strain tensor E for uniaxial extension/compression:
      - principal extension axis is at angle `theta_deg` (degrees) in the xy plane
      - extension magnitude is `eps` along that axis
      - transverse response is set by an effective Poisson ratio `poisson`

    Model:
      1) Build a local-frame diagonal strain:
           D = diag(eps, -poisson*eps)
      2) Rotate into the global xy frame:
           E = R D R^T
      3) Convert small-strain tensor to a small deformation gradient:
           F = I + E

    Parameters:
      eps:
        Typical small value, e.g. 0.005 for 0.5% strain.
      theta_deg:
        Direction (deg) of the principal extension axis in-plane.
      poisson:
        Effective in-plane Poisson ratio (graphene is often quoted ~0.165).

    Returns:
      StrainPlan with F2 = deformation gradient and info = parameter summary.

    Notes:
      - Intended for "small" strain (|eps| << 1). For large strain, this linearized
        relationship is not appropriate.
      - This function does not enforce symmetry/physical constraints beyond the
        simple Poisson coupling; it is a convenience generator.
    """
    th = np.deg2rad(float(theta_deg))
    c, s = float(np.cos(th)), float(np.sin(th))
    R = np.array([[c, -s], [s, c]], dtype=float)
    D = np.diag([float(eps), float(-poisson * eps)])
    E = R @ D @ R.T
    F = np.eye(2, dtype=float) + E

    return StrainPlan(
        model="directed_small",
        F2=F,
        info={"eps": float(eps), "theta_deg": float(theta_deg), "poisson": float(poisson)},
    )


def random_small_strain(
    eps_max: float,
    *,
    shear_max: Optional[float] = None,
    seed: int = 0,
) -> StrainPlan:
    """
    (Deprecated) Random small in-plane strain with bounded components.

    Draws independent small-strain components uniformly at random:
      - ex ~ U(-eps_max, eps_max)
      - ey ~ U(-eps_max, eps_max)
      - gxy ~ U(-shear_max, shear_max)

    Here `gxy` is the *engineering* shear strain, related to the small-strain tensor
    component by:
      E_xy = gxy / 2

    Small-strain tensor:
        E = [[ex,   gxy/2],
             [gxy/2, ey  ]]

    Small deformation gradient:
        F = I + E

    Parameters:
      eps_max:
        Max magnitude for normal strain components (typical small value, e.g. 0.005).
      shear_max:
        Max magnitude for engineering shear. Defaults to eps_max if not provided.
      seed:
        RNG seed for reproducibility.

    Returns:
      StrainPlan with F2 and info including the drawn components.

    Notes:
      - This is a simple synthetic perturbation model, useful for exercising the
        strain pipeline and validators.
      - Intended only for small strain (components much less than 1).
    """
    rng = np.random.default_rng(int(seed))
    ex = float(rng.uniform(-eps_max, eps_max))
    ey = float(rng.uniform(-eps_max, eps_max))
    if shear_max is None:
        shear_max = eps_max
    gxy = float(rng.uniform(-float(shear_max), float(shear_max)))

    E = np.array([[ex, 0.5 * gxy], [0.5 * gxy, ey]], dtype=float)
    F = np.eye(2, dtype=float) + E

    return StrainPlan(
        model="random_small",
        F2=F,
        info={"eps_max": float(eps_max), "ex": ex, "ey": ey, "gxy": gxy, "seed": int(seed)},
    )
