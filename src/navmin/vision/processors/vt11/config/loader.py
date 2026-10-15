from __future__ import annotations

from pathlib import Path
import yaml


DEFAULT_CONFIG = Path(__file__).with_name("default.yaml")


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    validate_config(cfg)
    return cfg


def load_profile_config(profile_name: str, path: str | Path | None = None) -> dict:
    """Load and resolve one VT-11 observation profile for non-GUI embedding."""
    from navmin.vision.processors.vt11.profiles import effective_config
    return effective_config(load_config(path), profile_name)


def validate_config(cfg: dict) -> None:
    required_top = {"project", "video", "acquire", "identity", "tracking", "adaptation", "evaluation", "ui", "observation", "profiles"}
    missing = required_top - set(cfg)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")

    _validate_evaluation_section(cfg)
    _validate_hotfix2_section(cfg)
    _validate_thermal_section(cfg)
    _validate_profiles_section(cfg)

    acquire = cfg["acquire"]
    if acquire["roi_radius_px"] <= 0:
        raise ValueError("roi_radius_px must be > 0")
    if acquire["initial_frames"] < 1:
        raise ValueError("initial_frames must be >= 1")
    if acquire["top_k_candidates"] < 2:
        raise ValueError("top_k_candidates must be >= 2")

    detector = acquire.get("candidate_detector", {})
    required_detector = {
        "blur_sigma_px", "threshold_sigma", "min_response", "min_area_px",
        "max_area_px", "morph_close_px", "spatial_sigma_px",
        "spatial_weight", "detector_weight",
    }
    missing_detector = required_detector - set(detector)
    if missing_detector:
        raise ValueError(f"Missing candidate_detector settings: {sorted(missing_detector)}")
    if detector["blur_sigma_px"] <= 0 or detector["threshold_sigma"] <= 0:
        raise ValueError("candidate detector blur/threshold sigma must be > 0")
    if detector["min_area_px"] < 1 or detector["max_area_px"] < detector["min_area_px"]:
        raise ValueError("candidate detector area bounds are invalid")
    if detector["spatial_sigma_px"] <= 0:
        raise ValueError("candidate detector spatial_sigma_px must be > 0")
    if detector["spatial_weight"] < 0 or detector["detector_weight"] < 0:
        raise ValueError("candidate detector ranking weights must be >= 0")
    if detector["spatial_weight"] + detector["detector_weight"] <= 0:
        raise ValueError("candidate detector needs a positive ranking weight")

    scoring = acquire.get("scoring", {})
    required_scoring = {
        "mode", "combined_spatial_weight", "confidence_gap_low",
        "confidence_gap_high", "custom_weights",
    }
    missing_scoring = required_scoring - set(scoring)
    if missing_scoring:
        raise ValueError(f"Missing acquire.scoring settings: {sorted(missing_scoring)}")
    allowed_modes = {"nearest", "appearance", "edge", "projection", "contrast", "geometry", "combined", "custom"}
    if scoring["mode"] not in allowed_modes:
        raise ValueError("invalid acquire scoring mode")
    if not 0 <= scoring["combined_spatial_weight"] <= 1:
        raise ValueError("combined_spatial_weight must be in [0,1]")
    if scoring["confidence_gap_low"] < 0 or scoring["confidence_gap_high"] < scoring["confidence_gap_low"]:
        raise ValueError("confidence gap thresholds are invalid")
    custom = scoring["custom_weights"]
    required_custom = {"spatial", "appearance", "edge", "projection", "contrast", "geometry"}
    if required_custom - set(custom):
        raise ValueError("custom scoring weights are incomplete")
    if any(value < 0 for value in custom.values()) or sum(custom.values()) <= 0:
        raise ValueError("custom scoring weights are invalid")

    initial_train = acquire.get("initial_train", {})
    required_train = {
        "reference_bank_size", "identity_weight", "motion_weight", "size_weight",
        "motion_sigma_px", "bank_update_min_score", "confirm_min_valid_fraction",
        "confirm_min_median_score", "confirm_min_median_gap",
        "observe_min_valid_fraction", "observe_min_median_score",
    }
    missing_train = required_train - set(initial_train)
    if missing_train:
        raise ValueError(f"Missing acquire.initial_train settings: {sorted(missing_train)}")
    if initial_train["reference_bank_size"] < 1 or initial_train["motion_sigma_px"] <= 0:
        raise ValueError("initial_train bank size/motion sigma invalid")
    train_weights = [initial_train["identity_weight"], initial_train["motion_weight"], initial_train["size_weight"]]
    if any(v < 0 for v in train_weights) or sum(train_weights) <= 0:
        raise ValueError("initial_train weights invalid")
    for name in (
        "bank_update_min_score", "confirm_min_valid_fraction", "confirm_min_median_score",
        "confirm_min_median_gap", "observe_min_valid_fraction", "observe_min_median_score",
    ):
        if not 0 <= initial_train[name] <= 1:
            raise ValueError(f"initial_train {name} must be in [0,1]")

    weights = cfg["identity"]["weights"]
    for name, value in weights.items():
        if value < 0:
            raise ValueError(f"identity weight {name} must be >= 0")
    if sum(weights.values()) <= 0:
        raise ValueError("at least one identity weight must be > 0")


def _validate_evaluation_section(cfg: dict) -> None:
    ev = cfg.get("evaluation", {})
    for name in ("ground_truth_tolerance_px", "acquisition_tolerance_px"):
        if float(ev.get(name, 0)) <= 0:
            raise ValueError(f"evaluation.{name} must be > 0")
    if int(ev.get("experiment_click_directions", 0)) < 1:
        raise ValueError("evaluation.experiment_click_directions must be >= 1")
    codec = str(ev.get("annotated_video_codec", ""))
    if len(codec) != 4:
        raise ValueError("evaluation.annotated_video_codec must be four characters")


def _validate_hotfix2_section(cfg: dict) -> None:
    h = cfg.get("hotfix2", {})
    if not h:
        return
    if not str(h.get("output_root", "")).strip():
        raise ValueError("hotfix2.output_root must not be empty")
    for name in ("pre_lost_frames", "post_lost_frames"):
        if int(h.get(name, -1)) < 0:
            raise ValueError(f"hotfix2.{name} must be >= 0")


def _validate_thermal_section(cfg: dict) -> None:
    th = cfg.get("thermal", {})
    if not th:
        return
    if str(th.get("video_mode", "auto")) not in {"auto", "visible", "thermal"}:
        raise ValueError("thermal.video_mode invalid")
    if "hud_suppression_enabled" in th and not isinstance(th.get("hud_suppression_enabled"), bool):
        raise ValueError("thermal.hud_suppression_enabled invalid")
    if str(th.get("reticle_mode", "auto")) not in {"off", "auto", "manual"}:
        raise ValueError("thermal.reticle_mode invalid")
    if not 0.0 < float(th.get("auto_channel_corr_min", 0.0)) <= 1.0:
        raise ValueError("thermal.auto_channel_corr_min invalid")
    if float(th.get("auto_channel_spread_p90_max", 0.0)) <= 0:
        raise ValueError("thermal.auto_channel_spread_p90_max invalid")
    if int(th.get("manual_search_radius_px", 0)) < 5:
        raise ValueError("thermal.manual_search_radius_px invalid")
    if not 0.0 <= float(th.get("static_overlay_penalty_weight", -1.0)) <= 1.0:
        raise ValueError("thermal.static_overlay_penalty_weight invalid")
    if int(th.get("hud_soft_dilate_px", 0)) < 0:
        raise ValueError("thermal.hud_soft_dilate_px invalid")
    for name in ("hud_negative_penalty_weight", "hud_hard_overlap_ratio", "hud_soft_overlap_ratio"):
        if not 0.0 <= float(th.get(name, 0.0)) <= 1.0:
            raise ValueError(f"thermal.{name} invalid")
    weights = cfg.get("identity", {}).get("thermal_weights", {})
    needed = {"appearance", "edge", "projection", "thermal_contrast", "hot_area"}
    if needed - set(weights):
        raise ValueError("identity.thermal_weights incomplete")
    if any(float(v) < 0 for v in weights.values()) or sum(float(v) for v in weights.values()) <= 0:
        raise ValueError("identity.thermal_weights invalid")


def _validate_profiles_section(cfg: dict) -> None:
    obs = cfg.get("observation", {})
    if str(obs.get("profile", "auto")) not in {"auto", "day", "thermal"}:
        raise ValueError("observation.profile invalid")
    profiles = cfg.get("profiles", {})
    for name in ("day", "thermal"):
        spec = profiles.get(name)
        if not isinstance(spec, dict):
            raise ValueError(f"profiles.{name} missing")
        if str(spec.get("video_mode", "")) not in {"visible", "thermal"}:
            raise ValueError(f"profiles.{name}.video_mode invalid")
        overrides = spec.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"profiles.{name}.overrides invalid")
        weights = overrides.get("identity", {}).get("weights", {})
        supported = {"appearance", "edge", "projection", "geometry", "color", "thermal_contrast", "hot_area",
                     "peak_prominence", "hot_core_compactness", "radial_hotspot_profile", "energy_concentration", "thermal_2d_ratio"}
        positive = 0.0
        for key, value in weights.items():
            fv = float(value)
            if fv < 0:
                raise ValueError(f"profiles.{name} identity weight {key} must be >=0")
            if key in supported:
                positive += fv
        if positive <= 0:
            raise ValueError(f"profiles.{name} needs positive supported identity weights")
        ema = overrides.get("adaptation", {}).get("ema", {})
        for key in ("shape_alpha_multiplier", "thermal_alpha_multiplier", "color_alpha_multiplier"):
            if key in ema and float(ema[key]) <= 0:
                raise ValueError(f"profiles.{name}.{key} must be >0")
