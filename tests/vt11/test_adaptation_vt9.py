import cv2
import numpy as np
import pytest

from navmin.vision.processors.vt11.acquisition.candidate_detector import CandidateDetectorConfig, detect_local_candidates
from navmin.vision.processors.vt11.acquisition.initial_train import InitialTrainConfig, run_initial_train
from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection
from navmin.vision.processors.vt11.adaptation.ema import EMAConfig, EMAReferenceAdapter, ema_blend_descriptor
from navmin.vision.processors.vt11.identity.descriptor import TargetDescriptor
from navmin.vision.processors.vt11.identity.reference_bank import ReferenceBank
from navmin.vision.processors.vt11.tracking.reacquisition import ReacquisitionConfig
from navmin.vision.processors.vt11.tracking.session import TrackingSessionConfig, run_tracking_session
from navmin.vision.processors.vt11.tracking.tracker import TrackingConfig


def _descriptor(value: float) -> TargetDescriptor:
    a = np.full((4, 4), value, np.float32)
    p = np.full(4, value, np.float32)
    g = np.asarray([value, value], np.float32)
    return TargetDescriptor(a, p, p.copy(), a.copy(), g, 10.0 + value, (1, 2, 3, 4))


def test_vt9_ema_blend_is_numeric_and_uses_current_bbox():
    old = _descriptor(0.0)
    new = _descriptor(1.0)
    out = ema_blend_descriptor(old, new, 0.25)
    assert float(out.appearance.mean()) == pytest.approx(0.25)
    assert float(out.projection_x.mean()) == pytest.approx(0.25)
    assert out.size_px == pytest.approx(10.25)
    assert out.bbox == new.bbox


def test_vt9_reference_bank_keeps_base_references_untouched():
    base = _descriptor(0.0)
    bank = ReferenceBank(max_size=2, descriptors=[base])
    before = bank.descriptors[0].appearance.copy()
    bank.set_adaptive(_descriptor(0.5))
    assert bank.base_count == 1
    assert bank.effective_count == 2
    assert np.array_equal(bank.descriptors[0].appearance, before)
    cloned = bank.clone(include_adaptive=False)
    assert cloned.base_count == 1
    assert cloned.adaptive_descriptor is None


def test_vt9_adapter_blocks_low_confidence_and_updates_after_warmup():
    bank = ReferenceBank(max_size=2, descriptors=[_descriptor(0.0)])
    adapter = EMAReferenceAdapter(
        bank,
        EMAConfig(
            mode="ema", alpha=0.2, min_confidence=0.8,
            min_identity_similarity=0.7, min_confidence_gap=0.05,
            max_prediction_error_px=20.0, warmup_accepted_frames=1,
            update_interval_frames=1,
        ),
    )
    e0 = adapter.consider(
        frame_index=1, descriptor=_descriptor(0.2), confidence=0.4,
        identity_similarity=0.9, confidence_gap=0.2, prediction_error_px=2.0,
    )
    assert not e0.updated
    assert "CONFIDENCE" in e0.reason
    e1 = adapter.consider(
        frame_index=2, descriptor=_descriptor(0.2), confidence=0.95,
        identity_similarity=0.9, confidence_gap=0.2, prediction_error_px=2.0,
    )
    assert not e1.updated and e1.reason == "WARMUP"
    e2 = adapter.consider(
        frame_index=3, descriptor=_descriptor(0.3), confidence=0.95,
        identity_similarity=0.9, confidence_gap=0.2, prediction_error_px=2.0,
    )
    assert e2.updated
    assert adapter.update_count == 1
    assert bank.adaptive_descriptor is not None


def _detector_cfg():
    return CandidateDetectorConfig(
        threshold_sigma=1.8,
        min_response=3.0,
        min_area_px=2,
        max_area_px=1200,
        spatial_sigma_px=35.0,
    )


def _draw_target(frame, k):
    x = 38 + 3 * k
    level = min(235, 185 + 2 * k)  # gradual appearance drift
    cv2.rectangle(frame, (x, 53), (x + 9, 61), (level, level, level), -1)
    cv2.rectangle(frame, (x + 7, 55), (x + 12, 58), (145, 145, 145), -1)


def _frames(count=26, invisible=range(10, 14)):
    invisible = set(invisible)
    frames = []
    for k in range(count):
        frame = np.full((120, 180, 3), 28, np.uint8)
        if k not in invisible:
            _draw_target(frame, k)
        frames.append(frame)
    return frames


def _train(frames):
    sel = make_operator_selection(frame_index=0, click_xy=(43, 57), radius_px=65, frame_width=180, frame_height=120)
    detected = detect_local_candidates(frames[0], sel, top_k=5, config=_detector_cfg())
    assert detected.candidates

    def reader(index):
        return index, frames[index].copy()

    result = run_initial_train(
        frame_reader=reader,
        frame_count=len(frames), frame_width=180, frame_height=120,
        start_frame=0, initial_candidate=detected.candidates[0], roi_radius_px=65,
        detector_config=_detector_cfg(), config=InitialTrainConfig(initial_frames=7, top_k=5),
    )
    assert result.decision == "CONFIRMED"
    return result, reader


def _run_session(frames, adapt):
    train, reader = _train(frames)
    accepted = [r for r in train.frames if r.selected is not None]
    base_count = train.reference_bank.base_count
    session = run_tracking_session(
        frame_reader=reader, frame_count=len(frames), frame_width=180, frame_height=120,
        start_frame=7, reference_bank=train.reference_bank,
        initial_history=[r.selected.center for r in accepted],
        initial_history_frames=[r.frame_index for r in accepted],
        detector_config=_detector_cfg(),
        tracking_config=TrackingConfig(search_radius_px=70, max_search_radius_px=140, min_confidence_gap=0.0, lost_after_misses=2),
        reacquisition_config=ReacquisitionConfig(
            initial_radius_px=95, max_radius_px=180, radius_growth_px_per_frame=15,
            confirm_frames=3, max_frames=14, min_confidence_gap=0.0,
            max_confirmation_error_px=90,
        ),
        session_config=TrackingSessionConfig(max_reacq_cycles=3),
        adaptation_config=adapt,
    )
    return train, session, base_count


def test_vt9_fixed_mode_is_reproducible_and_does_not_mutate_train_bank():
    frames = _frames()
    train, session, base_count = _run_session(frames, EMAConfig(mode="off"))
    assert session.adaptation_mode == "off"
    assert session.adaptation_updates == 0
    assert session.final_reference_bank is not None
    assert session.final_reference_bank.adaptive_descriptor is None
    assert train.reference_bank.base_count == base_count
    assert train.reference_bank.adaptive_descriptor is None


def test_vt9_ema_updates_only_on_track_not_reacq_and_preserves_base_bank():
    frames = _frames()
    cfg = EMAConfig(
        mode="ema", alpha=0.08, min_confidence=0.0,
        min_identity_similarity=0.0, min_confidence_gap=0.0,
        max_prediction_error_px=200.0, warmup_accepted_frames=0,
        update_interval_frames=1,
    )
    train, session, base_count = _run_session(frames, cfg)
    assert session.status == "COMPLETED"
    assert session.lost_events >= 1
    assert session.reacq_successes >= 1
    assert session.adaptation_updates > 0
    assert session.final_reference_bank is not None
    assert session.final_reference_bank.adaptive_descriptor is not None
    assert session.final_reference_bank.base_count == base_count
    assert train.reference_bank.adaptive_descriptor is None

    reacq_frames = {f.frame_index for r in session.reacquisitions for f in r.frames}
    updated_frames = {e.frame_index for e in session.adaptation_events if e.updated}
    assert updated_frames
    assert updated_frames.isdisjoint(reacq_frames)


def test_vt9_config_validation():
    with pytest.raises(ValueError):
        EMAConfig(mode="bad").validate()
    with pytest.raises(ValueError):
        EMAConfig(alpha=0.0).validate()
    with pytest.raises(ValueError):
        EMAConfig(update_interval_frames=0).validate()
