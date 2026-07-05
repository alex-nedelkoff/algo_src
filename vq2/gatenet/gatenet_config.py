"""COR-132 T7 (Pass A) — the unified, typed, YAML-backed GateNet config tree.

ONE top-level :class:`GateNetConfig` composed of typed nested dataclasses, one
per YAML block the GateNet pipeline reads today. Every nested default is
BYTE-IDENTICAL to the ``cfg.get("block", {}).get(key, default)`` fallback the
current consumer uses — sourced from the ``.get(...)`` call sites, NOT from any
YAML (YAMLs set values; they are not the defaults).

Provenance of each default (call site / module constant) is documented inline
per field so the verifier can check it against the source.

Idioms mirror :mod:`perception.gate_dataset.config.GateDatasetConfig` (read it):

  * ``to_dict``/``from_dict`` and ``to_yaml``/``from_yaml`` round-trip as a fixed
    point, including tuple-typed fields (YAML emits tuples as lists; ``from_dict``
    coerces them back);
  * ``from_dict`` filters each block to its dataclass's known fields (so existing
    YAMLs carrying extra/legacy keys still load — e.g. ``optimizer.type`` which
    the trainer never reads), exactly like ``GateDatasetConfig.from_dict``;
  * PyYAML is imported LAZILY inside ``to_yaml``/``from_yaml`` only, so this
    module stays a light import (numpy-only via the reused tracker dataclasses;
    no torch, no yaml at import time).

The ``dataset`` field REUSES :class:`GateDatasetConfig` verbatim (COR-131); the
``tracker`` field REUSES :class:`TrackerConfig` (which itself nests
:class:`BranchArbiterConfig`) verbatim from :mod:`perception.decode` — those are
module-default knobs, deliberately YAML-unplumbed this pass (no COR-76 consumer
story yet); consolidating them here makes them ENUMERABLE in the unified tree.
"""
from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Reused verbatim (do NOT redefine).
from perception.gate_dataset.config import GateDatasetConfig
from perception.decode.track_state import TrackerConfig
from perception.decode.branch_persist import BranchArbiterConfig  # noqa: F401  (re-export)


# --------------------------------------------------------------------------- #
# Nested config blocks — defaults pinned to the current consumer's ``.get``
# fallback (NOT the YAML value).
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """Mirrors ``build_model``'s ``m.get(key, default)`` reads in
    ``train_gatenet.py`` (lines 110-119). Defaults are byte-identical."""

    backbone: str = "b2"                  # train_gatenet.py:110 m.get("backbone","b2")
    dense_heads: bool = True              # :111 m.get("dense_heads", True)
    num_seg_classes: int = 4              # :112 m.get("num_seg_classes", 4)
    enable_seg: bool = False              # :113 m.get("enable_seg", False)
    enable_visibility: bool = True        # :114 m.get("enable_visibility", True)
    enable_covariance: bool = False       # :115 m.get("enable_covariance", False)
    enable_direct_pose: bool = False      # :116 m.get("enable_direct_pose", False)
    enable_association_heads: bool = False  # :117 m.get("enable_association_heads", False)
    pretrained: bool = True               # :118 m.get("pretrained", True)
    head_mid_channels: int = 64           # :119 m.get("head_mid_channels", 64)


@dataclass
class OptimizerConfig:
    """``optimizer`` block. Trainer reads only ``lr``/``weight_decay``
    (train_gatenet.py:1523-1524). ``type`` is present in every YAML but the
    trainer never reads it (AdamW is hardcoded) — kept here as a passthrough so
    existing YAMLs round-trip; default mirrors the YAMLs' uniform "adamw"."""

    type: str = "adamw"                   # YAML-present, trainer-unread (passthrough)
    lr: float = 6e-5                      # :1523 opt_cfg.get("lr", 6e-5)
    weight_decay: float = 0.01            # :1524 opt_cfg.get("weight_decay", 0.01)


@dataclass
class LRScheduleConfig:
    """``lr_schedule`` block. NO code consumer today (declarative-only in the
    YAMLs — the LR schedule is not yet implemented in the trainer). Defaults are
    therefore sourced from the YAMLs themselves (the only existing source); they
    are uniform across every config that declares the block."""

    warmup_fraction: float = 0.05        # YAML lr_schedule.warmup_fraction
    warmup_start_lr: float = 1e-7        # YAML lr_schedule.warmup_start_lr
    min_lr: float = 1e-7                 # YAML lr_schedule.min_lr


@dataclass
class TrainingConfig:
    """``training`` block. Field-by-field from the ``tcfg.get(key, default)`` /
    ``cfg.get("training",{}).get(...)`` call sites in ``train_gatenet.py``.

    Note on the YAML-only knobs (``save_every``, ``accum_steps``): the trainer
    does NOT ``.get`` these in the call sites surveyed, but every YAML carries
    them; they are kept as passthroughs (default = the YAMLs' uniform value) so
    existing configs round-trip. ``stage``/``init_from``/``max_steps`` have no
    universal numeric default in a ``.get`` (stage defaults "mean"; init_from is
    optional -> None; max_steps -> the 2000 fallback)."""

    stage: str = "mean"                  # :144 cfg.get("training",{}).get("stage","mean")
    init_from: Optional[str] = None      # :1443/:1471/:1495 tcfg.get("init_from") -> None
    max_steps: int = 2000                # :1409 tcfg.get("max_steps", 2000)
    epochs: int = 1                      # :1415 tcfg.get("epochs", 1)
    batch_size: int = 16                 # :1404 tcfg.get("batch_size", 16)
    num_workers: int = 16                # :1162 tcfg.get("num_workers", 16)
    persistent_workers: bool = True      # :1163 tcfg.get("persistent_workers", True)
    prefetch_factor: int = 4             # :1164 tcfg.get("prefetch_factor", 4)
    pin_memory: bool = True              # :1165 tcfg.get("pin_memory", True)
    accum_steps: int = 1                 # YAML-present, trainer-unread (passthrough)
    grad_clip_max_norm: float = 1.0      # :1526 tcfg.get("grad_clip_max_norm", 1.0)
    precision: str = "float32"           # :1549 tcfg.get("precision", "float32")
    log_every: int = 25                  # :1552 tcfg.get("log_every", 25)
    save_every: int = 5                  # YAML-present, trainer-unread (passthrough)
    warmstart_steps: int = 300           # :1392 tcfg.get("warmstart_steps", 300)


@dataclass
class LossWeights:
    """``loss_weights`` block. EVERY term defaults to the trainer's ``weights.get(
    name, 1.0)`` fallback = 1.0 (train_gatenet.py:623-624,639-642,856,1042-1043;
    multi_instance.py:510-511,540-545,705). The union of term names across all
    stages (mean / sigma / direct / multi) is enumerated here; a missing term in
    a YAML falls back to 1.0 exactly as ``.get`` does.

    IMPORTANT: every default is 1.0 — NOT the YAML value. E.g. the direct
    configs set ``direct_t: 0.25`` but the trainer's fallback is
    ``weights.get("direct_t", 1.0)`` (:1042), so the dataclass default is 1.0."""

    center_heatmap_focal: float = 1.0    # :640 weights.get(...,1.0)
    corner_heatmap_focal: float = 1.0    # :639
    center_offset_l1: float = 1.0        # :624
    corner_offset_l1: float = 1.0        # :623
    visibility_bce: float = 1.0          # :642
    corner_delta_l1: float = 1.0         # multi_instance.py:544 weights.get(...,1.0)
    log_size_l1: float = 1.0             # multi_instance.py:545
    corner_nll: float = 1.0              # :856 weights.get("corner_nll",1.0)
    direct_t: float = 1.0                # :1042 weights.get("direct_t",1.0) (YAML 0.25)
    direct_rot: float = 1.0              # :1043 weights.get("direct_rot",1.0)


@dataclass
class CornerLossConfig:
    """``corner_loss`` block. Trainer reads ``cl.get("metric","l1")`` (:1379).
    ``edge_on_band_flip_min`` is present in every YAML but has NO code consumer
    (the band-flip behavior is baked into ``symmin_corner_loss``); kept as a
    passthrough so existing YAMLs round-trip (default = the YAMLs' uniform True).

    NOTE: the retired ``warmstart_fixed_k_epochs`` knob is deliberately ABSENT —
    the trainer rejects it loudly (:1387); modelling it here would resurrect a
    dead knob (lessons.md "no silently-dead knobs")."""

    metric: str = "l1"                   # :1379 cl.get("metric", "l1")
    edge_on_band_flip_min: bool = True   # YAML-present, no code consumer (passthrough)


@dataclass
class CovLossConfig:
    """``cov_loss`` block. Trainer reads ``cov_cfg.get("sigma_floor_px", 0.25)``
    (:1371)."""

    sigma_floor_px: float = 0.25         # :1371 cov_cfg.get("sigma_floor_px", 0.25)


@dataclass
class MultiInstanceConfig:
    """``multi_instance`` block. From ``_resolve_multi``'s reads
    (train_gatenet.py:170,178-179)."""

    enabled: bool = False                # :170 mi.get("enabled", False)
    t_lo_px: float = 14.0                # :178 mi.get("t_lo_px", 14.0)
    t_hi_px: float = 22.0                # :179 mi.get("t_hi_px", 22.0)


@dataclass
class DecodeConfig:
    """``decode`` block (inference-side association knobs). The TRAINER never
    reads this block; the canonical consumers are
    :func:`perception.decode.associate.decode_associate` / ``solve_instance``
    and :func:`perception.eval.sequence_metrics` (which ``dk.get(...)`` it).
    Defaults are byte-identical to those consumers' signature defaults / module
    constants (the YAMLs match these exactly)."""

    center_score_min: float = 0.25       # associate.decode_associate(center_score_min=0.25)
    corner_score_min: float = 0.10       # associate.decode_associate(corner_score_min=0.10)
    snap_radius_frac: float = 0.15       # associate.SNAP_RADIUS_FRAC_DEFAULT
    snap_radius_floor_px: float = 12.0   # associate.SNAP_RADIUS_FLOOR_PX_DEFAULT
    s_min_px: float = 20.0               # associate.S_MIN_PX_DEFAULT
    s_max_px: float = 700.0              # associate.S_MAX_PX_DEFAULT
    snap_miss_sigma_frac: float = 0.05   # associate.SNAP_MISS_SIGMA_FRAC_DEFAULT
    max_instances: int = 8               # associate.MAX_INSTANCES_DEFAULT
    min_corners_for_pnp: int = 4         # associate.MIN_CORNERS_FOR_PNP_DEFAULT
    truncation_sigma_inflation: float = 10.0  # associate.TRUNCATION_SIGMA_INFLATION_DEFAULT
    two_branch: bool = True              # associate.solve_instance(two_branch=True)
    mirror_band_px: float = 3.0          # perception.geometry.weighted_pnp.MIRROR_BAND_PX_DEFAULT


@dataclass
class WandbConfig:
    """``wandb`` block. Declarative-only (the trainer does not ``.get`` it in the
    surveyed call sites). Defaults mirror the YAMLs' uniform values; ``tags`` is
    a tuple (YAML round-trips it as a list -> ``from_dict`` coerces it back)."""

    project: str = "corvidx-cor132-gatenet"  # YAML wandb.project
    entity: Optional[str] = None             # YAML wandb.entity (null)
    tags: Tuple[str, ...] = ()               # YAML wandb.tags (default empty -> tuple)


# --------------------------------------------------------------------------- #
# Generic nested-dataclass (de)serialization helpers — follow the
# GateDatasetConfig idioms (filter-to-known-fields + tuple coercion), made
# recursive so a nested dataclass field dispatches to its own from_dict.
# --------------------------------------------------------------------------- #
def _tuple_typed_field_names(cls: type) -> Tuple[str, ...]:
    """Names of ``cls``'s fields whose declared type is a tuple (so YAML lists
    must be coerced back). Recognizes ``Tuple[...]`` annotations (typing) and the
    rare bare-``tuple`` default."""
    out: List[str] = []
    for f in fields(cls):
        ann = f.type
        # ``from __future__ import annotations`` stringifies the annotations.
        if isinstance(ann, str):
            if ann.startswith("Tuple") or ann == "tuple":
                out.append(f.name)
        elif ann is tuple or getattr(ann, "__origin__", None) is tuple:
            out.append(f.name)
    return tuple(out)


def _from_dict_generic(cls: type, d: Dict[str, Any]):
    """Build a ``cls`` instance from a raw dict: keep only known fields, recurse
    into nested dataclass fields, and coerce tuple-typed fields from lists.

    Mirrors ``GateDatasetConfig.from_dict`` (filter-to-known + tuple coercion)
    but generic over any dataclass and recursive over nested dataclass fields."""
    if hasattr(cls, "from_dict") and cls is not GateNetConfig:
        # Reused dataclasses (GateDatasetConfig) bring their own from_dict —
        # honor it so their tuple-coercion / field-filter rules stay canonical.
        return cls.from_dict(d)  # type: ignore[attr-defined]
    field_by_name = {f.name: f for f in fields(cls)}
    kwargs: Dict[str, Any] = {}
    for name, f in field_by_name.items():
        if name not in d:
            continue
        raw = d[name]
        # Resolve the (possibly stringified) declared type to a runtime class.
        ftype = _resolve_type(f.type)
        if is_dataclass(ftype) and isinstance(raw, dict):
            kwargs[name] = _from_dict_generic(ftype, raw)
        else:
            kwargs[name] = raw
    # Tuple coercion (YAML round-trips tuples as lists).
    for tname in _tuple_typed_field_names(cls):
        if tname in kwargs and isinstance(kwargs[tname], list):
            kwargs[tname] = tuple(kwargs[tname])
    return cls(**kwargs)


# Map of the nested-dataclass field types we may need to resolve from a
# stringified annotation (``from __future__ import annotations`` is active).
_NESTED_TYPES = {
    "ModelConfig": ModelConfig,
    "GateDatasetConfig": GateDatasetConfig,
    "OptimizerConfig": OptimizerConfig,
    "LRScheduleConfig": LRScheduleConfig,
    "TrainingConfig": TrainingConfig,
    "LossWeights": LossWeights,
    "CornerLossConfig": CornerLossConfig,
    "CovLossConfig": CovLossConfig,
    "DecodeConfig": DecodeConfig,
    "MultiInstanceConfig": MultiInstanceConfig,
    "TrackerConfig": TrackerConfig,
    "BranchArbiterConfig": BranchArbiterConfig,
    "WandbConfig": WandbConfig,
}


def _resolve_type(ann: Any) -> Any:
    """Resolve a field annotation to a runtime type for dataclass dispatch.
    Handles the stringified-annotation case (PEP 563) for our nested types."""
    if isinstance(ann, str):
        return _NESTED_TYPES.get(ann, ann)
    return ann


# --------------------------------------------------------------------------- #
# Top-level config
# --------------------------------------------------------------------------- #
@dataclass
class GateNetConfig:
    """The unified GateNet config tree (Pass A: typed home, not yet consumed).

    Nested blocks mirror the existing YAML layout one-to-one; scalar top-level
    keys (``output_dir``) match the trainer's direct ``cfg.get`` reads
    (train_gatenet.py:1838 ``cfg.get("output_dir")``)."""

    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: GateDatasetConfig = field(default_factory=GateDatasetConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_schedule: LRScheduleConfig = field(default_factory=LRScheduleConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)
    corner_loss: CornerLossConfig = field(default_factory=CornerLossConfig)
    cov_loss: CovLossConfig = field(default_factory=CovLossConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    multi_instance: MultiInstanceConfig = field(default_factory=MultiInstanceConfig)
    # Module-default tracker tree (YAML-unplumbed this pass; reused verbatim).
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    # Scalar top-level key the trainer reads directly off ``cfg``.
    output_dir: Optional[str] = None     # train_gatenet.py:1838 cfg.get("output_dir")

    # --- serialization (mirrors GateDatasetConfig) ------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GateNetConfig":
        """Build from a raw dict shaped like the existing YAMLs (nested blocks).

        Each block is dispatched to the right nested ``from_dict`` and filtered
        to that dataclass's known fields (so a YAML with extra/legacy keys —
        e.g. ``optimizer.type`` — still loads). Unknown TOP-LEVEL keys are
        ignored, matching ``GateDatasetConfig.from_dict``'s tolerant contract."""
        field_by_name = {f.name: f for f in fields(cls)}
        kwargs: Dict[str, Any] = {}
        for name, f in field_by_name.items():
            if name not in d:
                continue
            raw = d[name]
            ftype = _resolve_type(f.type)
            if is_dataclass(ftype) and isinstance(raw, dict):
                kwargs[name] = _from_dict_generic(ftype, raw)
            else:
                kwargs[name] = raw
        return cls(**kwargs)

    def to_yaml(self, path: Union[str, Path]) -> Path:
        """Write the tree to ``path`` as YAML (tuples emitted as lists). Returns
        the written path. ``from_yaml(to_yaml(cfg)) == cfg``."""
        import yaml  # lazy: keep PyYAML off this module's import-time deps
        path = Path(path)
        payload = _to_yaml_payload(self.to_dict())
        path.write_text(yaml.safe_dump(payload, sort_keys=True))
        return path

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "GateNetConfig":
        """Load a tree from YAML. Inverse of :meth:`to_yaml` (a fixed point with
        it), including tuple-typed fields (YAML lists coerced back to tuples)."""
        import yaml  # lazy
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)


def _to_yaml_payload(obj: Any) -> Any:
    """Recursively render tuples as plain lists for ``yaml.safe_dump`` (the
    inverse coercion lives in the per-dataclass ``from_dict`` tuple step)."""
    if isinstance(obj, dict):
        return {k: _to_yaml_payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_yaml_payload(v) for v in obj]
    return obj
