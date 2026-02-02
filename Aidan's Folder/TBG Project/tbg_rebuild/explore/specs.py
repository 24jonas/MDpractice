from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Any, Dict, List, Literal, Tuple, Optional
from datetime import datetime


# -----------------------------
# Type aliases used in the spec
# -----------------------------
AngleUnits = Literal["deg", "rad"]

# How a registry (in-plane shift) is interpreted. v1 keeps this explicit but does not yet use it
# for stacking beyond default (0, 0).
RegistryType = Literal["frac_supercell", "frac_primitive", "cart_angstrom"]

# How in-plane periodicity is defined:
# - tiled: explicit n1/n2 repeats of the primitive cell
# - supercell_core: commensurate supercell planned by supercell-core
PeriodicityMode = Literal["tiled", "supercell_core"]

# Strain models supported in Part I (used only for the "apply strain then validate" stage).
StrainModel = Literal["none", "directed_small", "random_small"]


@dataclass(frozen=True, slots=True)
class LayerSpec:
    """
    Per-layer inputs describing material identity and (future) registry/twist parameters.

    Notes (current behavior):
    - material_id is used everywhere.
    - twist_angle/twist_units and registry_* are kept for forward-compatibility; Part I builders
      currently implement AA stacking (tiled) and solver-planned commensurate cells (supercell_core).
    """
    material_id: str
    twist_angle: float = 0.0
    twist_units: AngleUnits = "deg"

    # Registry is an in-plane shift between layers. v1 defaults to no shift (0,0).
    # Kept explicit so the spec format doesn't need to change later.
    registry_type: RegistryType = "frac_supercell"
    registry_value: Tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True, slots=True)
class CapsPolicy:
    """
    Basic constraints gate used before building atoms.

    - max_atoms: hard skip if the estimated atom count exceeds this value.
    - downsample_views_above: visualization policy only (not a skip condition).
    - max_cell_len: optional hard skip if |a1| or |a2| exceeds this value (Å).
    """
    max_atoms: int = 200_000
    downsample_views_above: int = 25_000
    max_cell_len: Optional[float] = None


@dataclass(frozen=True, slots=True)
class OutputPolicy:
    """
    Output controls for Part I pipelines.

    make_views:
      If True, write simple diagnostic plots (3D layer-colored point cloud).
    downsample_views_above:
      If atom count exceeds this threshold, visualization code may downsample points.
    """
    make_views: bool = True
    downsample_views_above: int = 25_000


@dataclass(frozen=True, slots=True)
class GeometrySpec:
    """
    Geometric parameters shared across builders/stages.

    repeats:
      (n1, n2) repeat counts in primitive a1/a2 directions. Used ONLY for tiled mode.
    vacuum:
      Cell z-length (Å). In Part I we treat systems as slabs (pbc in z is False).
    pbc:
      Periodic boundary conditions; default (True, True, False) for a slab.
    interlayer_distance:
      Separation between layer centers (Å) for bilayers/multilayers.
    z_center:
      Center of the stack along z (Å). Layers are placed symmetrically around this.
    """
    repeats: Tuple[int, int]                    # (n1, n2) used ONLY for tiled mode
    vacuum: float = 24.0                        # Å, cell z-length
    pbc: Tuple[bool, bool, bool] = (True, True, False)
    interlayer_distance: float = 3.35           # Å, used for bilayers/multilayers
    z_center: float = 0.0                       # Å, center of stack


@dataclass(frozen=True, slots=True)
class PeriodicitySpec:
    """
    Periodicity and solver configuration.

    mode:
      Selects which domain planner / builder path is taken.
    caps:
      Resource gates applied before building atoms.
    solver:
      Mode-specific solver settings. For supercell_core mode, this includes requested theta,
      search window, max_el, strain tolerance, etc.
    """
    mode: PeriodicityMode = "tiled"
    caps: CapsPolicy = field(default_factory=CapsPolicy)

    solver: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StrainSpec:
    """
    Optional artificial strain stage configuration (Part I testing utility).

    enabled:
      If True, the strain stage runs after "built" and before validation.
    model:
      One of the supported small-strain generators.
    params:
      Model parameters (eps/theta/poisson or eps_max/shear_max/seed).
    """
    enabled: bool = False
    model: StrainModel = "none"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Provenance:
    """
    Lightweight provenance metadata for run bookkeeping.
    """
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    git_commit: str = ""
    notes: str = ""


@dataclass(frozen=True, slots=True)
class ConfigSpec:
    """
    Top-level configuration object passed through planning, building, validation, and (optionally) strain.

    spec_version:
      A simple version string for the config schema (not the codebase).
    job_name:
      Unique identifier used to name output folders.
    layers:
      One entry per layer in the stack.
    geometry / periodicity / strain / output_policy:
      Grouped settings used by Part I pipelines.
    provenance:
      Optional human notes and timestamp to help trace runs.
    """
    spec_version: str
    job_name: str
    layers: List[LayerSpec]
    geometry: GeometrySpec
    periodicity: PeriodicitySpec = field(default_factory=PeriodicitySpec)
    strain: StrainSpec = field(default_factory=StrainSpec)
    output_policy: OutputPolicy = field(default_factory=OutputPolicy)
    provenance: Provenance = field(default_factory=Provenance)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataclass tree to a plain-JSON-serializable dict."""
        return asdict(self)


# -----------------------------
# Spec constructors (inputs)
# -----------------------------
def make_spec_monolayer(
    *,
    job_name: str,
    material_id: str,
    repeats: Tuple[int, int],
    vacuum: float = 24.0,
    pbc: Tuple[bool, bool, bool] = (True, True, False),
    caps: CapsPolicy = CapsPolicy(),
    notes: str = "",
) -> ConfigSpec:
    """
    Convenience constructor for a tiled monolayer job.

    This is the simplest entry point:
      - periodicity.mode="tiled"
      - one layer
      - repeats controls the in-plane supercell
    """
    return ConfigSpec(
        spec_version="0.1",
        job_name=job_name,
        layers=[LayerSpec(material_id=material_id)],
        geometry=GeometrySpec(repeats=repeats, vacuum=vacuum, pbc=pbc),
        periodicity=PeriodicitySpec(mode="tiled", caps=caps),
        strain=StrainSpec(enabled=False, model="none"),
        output_policy=OutputPolicy(make_views=True, downsample_views_above=caps.downsample_views_above),
        provenance=Provenance(notes=notes),
    )


def make_spec_bilayer(
    *,
    job_name: str,
    material_id: str,
    repeats: Tuple[int, int],
    interlayer_distance: float = 3.35,
    vacuum: float = 24.0,
    pbc: Tuple[bool, bool, bool] = (True, True, False),
    caps: CapsPolicy = CapsPolicy(),
    notes: str = "",
) -> ConfigSpec:
    """
    Convenience constructor for a tiled AA bilayer job (Part I baseline).

    Current behavior:
      - periodicity.mode="tiled"
      - two identical layers
      - twist/registry are currently unused (kept for future extensions)
    """
    layers = [
        LayerSpec(material_id=material_id, twist_angle=0.0),
        LayerSpec(material_id=material_id, twist_angle=0.0),
    ]
    return ConfigSpec(
        spec_version="0.1",
        job_name=job_name,
        layers=layers,
        geometry=GeometrySpec(
            repeats=repeats,
            vacuum=vacuum,
            pbc=pbc,
            interlayer_distance=interlayer_distance,
            z_center=0.0,
        ),
        periodicity=PeriodicitySpec(mode="tiled", caps=caps),
        strain=StrainSpec(enabled=False, model="none"),
        output_policy=OutputPolicy(make_views=True, downsample_views_above=caps.downsample_views_above),
        provenance=Provenance(notes=notes),
    )


def make_spec_tbg_supercell_core(
    *,
    job_name: str,
    material_id: str,
    theta_deg: float,
    vacuum: float = 24.0,
    interlayer_distance: float = 3.35,
    max_el: int = 20,
    theta_window_deg: float = 0.2,
    theta_step_deg: float = 0.01,
    strain_tol: float = 5e-4,
    caps: CapsPolicy = CapsPolicy(),
    notes: str = "",
) -> ConfigSpec:
    """
    Convenience constructor for solver-assisted commensurate bilayers (supercell-core mode).

    Key idea:
      supercell-core searches for a commensurate supercell near the requested theta,
      and may apply a small strain if required to enforce commensurability.

    Implementation notes:
      - geometry.repeats is set to (1,1) and ignored by this mode.
      - twist_angle fields are not used; theta is passed via periodicity.solver.
      - artificial strain stage is disabled (we only apply solver-provided strain tensors).
    """
    layers = [
        LayerSpec(material_id=material_id, twist_angle=0.0),
        LayerSpec(material_id=material_id, twist_angle=0.0),
    ]
    return ConfigSpec(
        spec_version="0.2",
        job_name=job_name,
        layers=layers,
        geometry=GeometrySpec(
            repeats=(1, 1),  # not used in solver mode
            vacuum=vacuum,
            interlayer_distance=interlayer_distance,
        ),
        periodicity=PeriodicitySpec(
            mode="supercell_core",
            caps=caps,
            solver={
                "method": "opt",
                "max_el": int(max_el),
                "theta_deg": float(theta_deg),
                "theta_window_deg": float(theta_window_deg),
                "theta_step_deg": float(theta_step_deg),
                "strain_tol": float(strain_tol),
                "flag_if_corrected": True,
            },
        ),
        strain=StrainSpec(enabled=False, model="none"),  # don’t use artificial strain here
        output_policy=OutputPolicy(make_views=True, downsample_views_above=caps.downsample_views_above),
        provenance=Provenance(notes=notes),
    )


def with_directed_strain(
    spec: ConfigSpec,
    *,
    eps: float,
    theta_deg: float,
    poisson: float = 0.165,
) -> ConfigSpec:
    """
    Return a copy of `spec` with the directed small-strain stage enabled.

    This is a testing utility to exercise the strain pipeline on a built structure.
    It is intended for tiled jobs (not supercell-core commensurate jobs).
    """
    return replace(
        spec,
        strain=StrainSpec(
            enabled=True,
            model="directed_small",
            params={"eps": float(eps), "theta_deg": float(theta_deg), "poisson": float(poisson)},
        ),
    )


def with_random_strain(
    spec: ConfigSpec,
    *,
    eps_max: float,
    shear_max: float | None = None,
    seed: int = 0,
) -> ConfigSpec:
    """
    Return a copy of `spec` with the random small-strain stage enabled.

    eps_max controls the maximum normal strain magnitude.
    shear_max (optional) controls maximum shear magnitude.
    seed makes the random draw reproducible.
    """
    params = {"eps_max": float(eps_max), "seed": int(seed)}
    if shear_max is not None:
        params["shear_max"] = float(shear_max)

    return replace(
        spec,
        strain=StrainSpec(enabled=True, model="random_small", params=params),
    )
