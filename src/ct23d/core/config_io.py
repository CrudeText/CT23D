from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .models import (
    IntensityBin,
    MeshingConfig,
    PreprocessConfig,
    ProjectConfig,
    ColorRGB,
)


def _path_from_yaml(value: Any, base_dir: Path) -> Optional[Path]:
    """
    Convert a YAML path value (string or None) to a Path relative to base_dir.
    """
    if value is None:
        return None
    s = str(value)
    p = Path(s)
    if not p.is_absolute():
        p = base_dir / p
    return p


def _path_to_yaml(path: Optional[Path], base_dir: Path) -> Optional[str]:
    """
    Convert a Path to a relative string for YAML, relative to base_dir.
    """
    if path is None:
        return None
    try:
        rel = path.relative_to(base_dir)
    except ValueError:
        # If not under base_dir, fall back to normal relative path
        rel = path
    return str(rel)


def load_project_config(path: Path) -> ProjectConfig:
    """
    Load a ProjectConfig from a YAML file.

    Paths inside the YAML are interpreted as relative to the YAML file
    location.
    """
    path = Path(path)
    base_dir = path.parent

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # --- PreprocessConfig ---
    pre_d: Dict[str, Any] = data.get("preprocess", {}) or {}
    pre_cfg = PreprocessConfig(
        input_dir=_path_from_yaml(pre_d.get("input_dir"), base_dir),
        processed_dir=_path_from_yaml(pre_d.get("processed_dir"), base_dir),
        use_cache=pre_d.get("use_cache", True),
        grayscale_tolerance=pre_d.get("grayscale_tolerance", 1),
        saturation_threshold=float(pre_d.get("saturation_threshold", 0.08)),
        remove_bed=pre_d.get("remove_bed", True),
    )

    # --- MeshingConfig ---
    mesh_d: Dict[str, Any] = data.get("meshing", {}) or {}
    spacing_list = mesh_d.get("spacing", [1.6, 1.0, 1.0])
    if len(spacing_list) != 3:
        raise ValueError(
            f"meshing.spacing must be a list of 3 numbers, got {spacing_list!r}"
        )
    spacing = tuple(float(x) for x in spacing_list)  # type: ignore[assignment]

    mesh_cfg = MeshingConfig(
        spacing=spacing,  # type: ignore[arg-type]
        non_black_threshold=int(mesh_d.get("non_black_threshold", 15)),
        min_component_size=int(mesh_d.get("min_component_size", 500)),
        smoothing_sigma=float(mesh_d.get("smoothing_sigma", 1.0)),
        n_bins=int(mesh_d.get("n_bins", 6)),
        min_intensity=int(mesh_d.get("min_intensity", 1)),
        max_intensity=int(mesh_d.get("max_intensity", 255)),
        use_custom_bins=bool(mesh_d.get("use_custom_bins", False)),
        output_dir=_path_from_yaml(mesh_d.get("output_dir"), base_dir),
        output_prefix=str(mesh_d.get("output_prefix", "ct_layer")),
        adaptive_component_filtering=bool(
            mesh_d.get("adaptive_component_filtering", True)
        ),
        skip_empty_bins=bool(mesh_d.get("skip_empty_bins", True)),
    )

    # --- Bins ---
    bins_list: List[IntensityBin] = []
    for b_d in data.get("bins", []) or []:
        color_val = b_d.get("color")
        color: Optional[ColorRGB]
        if color_val is None:
            color = None
        else:
            if len(color_val) != 3:
                raise ValueError(
                    f"bin.color must have length 3, got {color_val!r}"
                )
            color = (float(color_val[0]), float(color_val[1]), float(color_val[2]))

        bins_list.append(
            IntensityBin(
                index=int(b_d.get("index", len(bins_list))),
                low=int(b_d["low"]),
                high=int(b_d["high"]),
                name=b_d.get("name"),
                color=color,
                enabled=bool(b_d.get("enabled", True)),
            )
        )

    project = ProjectConfig(
        name=str(data.get("name", "CT23D Project")),
        preprocess=pre_cfg,
        meshing=mesh_cfg,
        bins=bins_list,
        config_path=path,
    )

    return project


def save_project_config(cfg: ProjectConfig, path: Path) -> None:
    """
    Save a ProjectConfig to YAML.

    Paths are stored as strings relative to the YAML file location.
    """
    path = Path(path)
    base_dir = path.parent
    base_dir.mkdir(parents=True, exist_ok=True)

    # --- Preprocess ---
    pre = cfg.preprocess
    pre_d: Dict[str, Any] = {
        "input_dir": _path_to_yaml(pre.input_dir, base_dir),
        "processed_dir": _path_to_yaml(pre.processed_dir, base_dir),
        "use_cache": pre.use_cache,
        "grayscale_tolerance": pre.grayscale_tolerance,
        "saturation_threshold": pre.saturation_threshold,
        "remove_bed": pre.remove_bed,
    }

    # --- Meshing ---
    mesh = cfg.meshing
    mesh_d: Dict[str, Any] = {
        "spacing": list(mesh.spacing),
        "non_black_threshold": mesh.non_black_threshold,
        "min_component_size": mesh.min_component_size,
        "smoothing_sigma": mesh.smoothing_sigma,
        "n_bins": mesh.n_bins,
        "min_intensity": mesh.min_intensity,
        "max_intensity": mesh.max_intensity,
        "use_custom_bins": mesh.use_custom_bins,
        "output_dir": _path_to_yaml(mesh.output_dir, base_dir),
        "output_prefix": mesh.output_prefix,
        "adaptive_component_filtering": mesh.adaptive_component_filtering,
        "skip_empty_bins": mesh.skip_empty_bins,
    }

    # --- Bins ---
    bins_yaml: List[Dict[str, Any]] = []
    for b in cfg.bins:
        color = list(b.color) if b.color is not None else None
        bins_yaml.append(
            {
                "index": b.index,
                "low": b.low,
                "high": b.high,
                "name": b.name,
                "color": color,
                "enabled": b.enabled,
            }
        )

    data: Dict[str, Any] = {
        "name": cfg.name,
        "preprocess": pre_d,
        "meshing": mesh_d,
        "bins": bins_yaml,
    }

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            default_flow_style=False,
        )
