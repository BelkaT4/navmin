from __future__ import annotations

import cv2
import numpy as np

from navmin.vision.processors.vt11.acquisition.candidate_detector import Candidate, CandidateDetectorConfig, detect_local_candidates
from navmin.vision.processors.vt11.acquisition.roi import RoiRect, make_operator_selection
from navmin.vision.processors.vt11.identity.descriptor import descriptor_from_candidate, descriptor_similarity
from navmin.vision.processors.vt11.thermal.processing import (
    ThermalProcessingConfig,
    build_thermal_profile,
    infer_video_mode,
    preprocess_frame,
    runtime_detector_config,
)


def _thermal_frame(target_xy=(120, 120), *, w=320, h=240):
    y = np.linspace(55, 115, h, dtype=np.float32)[:, None]
    gray = np.repeat(y, w, axis=1)
    gray += 2.0 * np.sin(np.linspace(0, 5, w, dtype=np.float32))[None, :]
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    cx, cy = w // 2, h // 2
    cv2.line(gray, (cx - 16, cy), (cx + 16, cy), 8, 1)
    cv2.line(gray, (cx, cy - 16), (cx, cy + 16), 8, 1)
    tx, ty = map(int, target_xy)
    cv2.circle(gray, (tx, ty), 5, 225, -1)
    # Thermal source encoded as monochrome with a fixed BGR offset/tint.
    b = np.clip(gray.astype(np.int16) + 12, 0, 255).astype(np.uint8)
    g = gray
    r = np.clip(gray.astype(np.int16) + 1, 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])


def _candidate(cx, cy, r=6, area=81):
    return Candidate(
        candidate_id=1,
        bbox=RoiRect(int(cx-r), int(cy-r), 2*r+1, 2*r+1),
        center_x=float(cx), center_y=float(cy), area_px=int(area),
        mean_response=20.0, peak_response=80.0,
        distance_to_click_px=0.0, spatial_prior=1.0,
        detector_strength=1.0, rank_score=1.0, size_prior=1.0,
    )


def test_auto_mode_recognizes_monochrome_thermal_encoding():
    frames = [_thermal_frame((100+i*3, 100)) for i in range(4)]
    cfg = ThermalProcessingConfig(video_mode="auto")
    assert infer_video_mode(frames, cfg) == "thermal"


def test_auto_reticle_mask_finds_static_cross_not_moving_target():
    frames = [_thermal_frame((80+i*12, 165)) for i in range(6)]
    target = _candidate(80, 165)
    profile = build_thermal_profile(frames, config=ThermalProcessingConfig(video_mode="thermal", hud_suppression_enabled=True), initial_candidate=target)
    assert profile.is_thermal
    assert profile.masked_pixels > 20
    assert profile.static_centers
    cx, cy = min(profile.static_centers, key=lambda p: abs(p[0]-160)+abs(p[1]-120))
    assert abs(cx - 160) < 5
    assert abs(cy - 120) < 5
    # The first moving target is far from the learned static mask.
    assert profile.reticle_mask[165, 80] == 0


def test_thermal_hot_blob_detector_ignores_dark_reticle_and_keeps_bright_target():
    training = [_thermal_frame((80+i*8, 175)) for i in range(5)]
    profile = build_thermal_profile(training, config=ThermalProcessingConfig(video_mode="thermal", hud_suppression_enabled=True), initial_candidate=_candidate(80,175))
    frame = _thermal_frame((150, 145))
    analysis = preprocess_frame(frame, profile)
    base = CandidateDetectorConfig(
        min_area_px=5, max_area_px=1000, threshold_sigma=2.0, min_response=3.0,
        adaptive_area_enabled=True, adaptive_min_area_ratio=0.20, adaptive_max_area_ratio=4.0,
    )
    det_cfg = runtime_detector_config(base, profile)
    selection = make_operator_selection(frame_index=10, click_xy=(150,145), radius_px=65, frame_width=320, frame_height=240)
    result = detect_local_candidates(analysis, selection, top_k=5, config=det_cfg, expected_area_px=90.0)
    assert result.candidates
    best = result.candidates[0]
    assert np.hypot(best.center_x - 150, best.center_y - 145) < 6.0
    # The dark reticle center must not outrank the bright target.
    assert np.hypot(best.center_x - 160, best.center_y - 120) > 10.0


def test_thermal_descriptor_rejects_dark_reticle_against_bright_target():
    frame = _thermal_frame((120, 170))
    tdesc = descriptor_from_candidate(frame, _candidate(120,170))
    rdesc = descriptor_from_candidate(frame, _candidate(160,120, r=7, area=50))
    weights = {
        "appearance": 0.10941503,
        "projection": 0.09329633,
        "edge": 0.08464987,
        "thermal_contrast": 0.25757899,
        "hot_area": 0.45505978,
    }
    same = descriptor_similarity(tdesc, tdesc, identity_weights=weights)
    false = descriptor_similarity(tdesc, rdesc, identity_weights=weights)
    assert same.total > 0.99
    assert false.thermal_contrast < same.thermal_contrast
    assert false.hot_area < same.hot_area
    assert false.total < 0.80


def test_visible_mode_keeps_legacy_detector_method():
    base = CandidateDetectorConfig(method="local_contrast_components")
    visible_profile = build_thermal_profile(
        [_thermal_frame((80,160))],
        config=ThermalProcessingConfig(video_mode="visible", reticle_mode="off"),
        initial_candidate=_candidate(80,160),
    )
    assert runtime_detector_config(base, visible_profile).method == "local_contrast_components"


def test_end_to_end_thermal_target_crosses_reticle_without_static_idswitch():
    from navmin.vision.processors.vt11.adaptation.ema import EMAConfig
    from navmin.vision.processors.vt11.acquisition.initial_train import InitialTrainConfig, run_initial_train
    from navmin.vision.processors.vt11.config.loader import load_config
    from navmin.vision.processors.vt11.tracking.reacquisition import ReacquisitionConfig
    from navmin.vision.processors.vt11.tracking.session import TrackingSessionConfig, run_tracking_session
    from navmin.vision.processors.vt11.tracking.tracker import TrackingConfig

    frames = [_thermal_frame((80 + i * 5, 120)) for i in range(30)]

    def reader(i):
        return i, frames[i]

    first = _candidate(80, 120, r=6, area=81)
    detector_cfg = CandidateDetectorConfig(
        min_area_px=5, max_area_px=1000, threshold_sigma=2.0, min_response=3.0,
        adaptive_area_enabled=True, adaptive_min_area_ratio=0.15, adaptive_max_area_ratio=4.0,
    )
    cfg = load_config()
    train = run_initial_train(
        frame_reader=reader, frame_count=len(frames), frame_width=320, frame_height=240,
        start_frame=0, initial_candidate=first, roi_radius_px=70, detector_config=detector_cfg,
        config=InitialTrainConfig(
            initial_frames=7, top_k=5, reference_bank_size=5,
            bank_update_min_score=0.50, confirm_min_median_score=0.40, confirm_min_median_gap=0.01,
        ),
        identity_weights=cfg["identity"]["weights"],
        thermal_identity_weights=cfg["identity"]["thermal_weights"],
        video_mode="auto", thermal_config=ThermalProcessingConfig(video_mode="auto", reticle_mode="auto"),
    )
    assert train.decision == "CONFIRMED"
    assert train.video_mode == "thermal"
    selected = [r for r in train.frames if r.selected is not None]
    session = run_tracking_session(
        frame_reader=reader, frame_count=len(frames), frame_width=320, frame_height=240,
        start_frame=train.frames[-1].frame_index + 1, reference_bank=train.reference_bank,
        initial_history=[r.selected.center for r in selected],
        initial_history_frames=[r.frame_index for r in selected],
        detector_config=detector_cfg,
        tracking_config=TrackingConfig(
            top_k=5, search_radius_px=60, min_search_radius_px=40, max_search_radius_px=100,
            min_track_score=0.40, min_identity_similarity=0.35, min_confidence_gap=0.0,
            lost_after_misses=3,
        ),
        reacquisition_config=ReacquisitionConfig(
            top_k=5, initial_radius_px=80, max_radius_px=140,
            min_score=0.40, min_identity_similarity=0.35, min_confidence_gap=0.0,
            confirm_frames=2, confirm_window_frames=3,
        ),
        session_config=TrackingSessionConfig(max_reacq_cycles=3),
        identity_weights=cfg["identity"]["thermal_weights"], adaptation_config=EMAConfig(mode="off"),
        thermal_profile=train.thermal_profile,
    )
    assert session.status == "COMPLETED"
    assert session.trajectory
    last = session.trajectory[-1]
    assert last[0] == 29
    assert abs(last[1] - 225.0) < 4.0
    assert abs(last[2] - 120.0) < 4.0
    # Reticle is fixed at (160,120); final track must remain on the moving target.
    assert abs(last[1] - 160.0) > 40.0
