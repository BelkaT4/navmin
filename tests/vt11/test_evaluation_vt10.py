import csv
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from navmin.vision.processors.vt11.acquisition.candidate_detector import CandidateDetectorConfig
from navmin.vision.processors.vt11.evaluation.annotated_video import export_annotated_video
from navmin.vision.processors.vt11.evaluation.experiments import run_acquisition_experiment
from navmin.vision.processors.vt11.evaluation.ground_truth import load_ground_truth_csv
from navmin.vision.processors.vt11.evaluation.logger import export_session_csv
from navmin.vision.processors.vt11.evaluation.metrics import PredictionPoint, evaluate_predictions
from navmin.vision.processors.vt11.identity.scorer import AcquireScoringConfig


def _write_gt(path: Path):
    with path.open('w', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['frame','target_id','x','y','visible','is_target'])
        for f in range(5):
            w.writerow([f,'A',20+f,20,1,1])
            w.writerow([f,'B',70,20,1,0])


def test_gt_loader_and_wrong_id_metrics(tmp_path):
    p = tmp_path / 'gt.csv'
    _write_gt(p)
    gt = load_ground_truth_csv(p)
    preds = (
        PredictionPoint(0,20,20),  # correct
        PredictionPoint(1,21,20),  # correct
        # frame 2 lost
        PredictionPoint(3,70,20),  # wrong id starts
        PredictionPoint(4,70,20),  # same wrong-id episode
    )
    s = evaluate_predictions(gt, preds, tolerance_px=5)
    assert s.evaluated_visible_frames == 5
    assert s.correct_frames == 2
    assert s.lost_frames == 1
    assert s.wrong_id_frames == 2
    assert s.id_switches == 1
    assert s.p_correct == pytest.approx(0.4)
    assert s.p_lost == pytest.approx(0.2)
    assert s.p_wrong_id == pytest.approx(0.4)


def test_gt_rejects_two_selected_targets_same_frame(tmp_path):
    p = tmp_path / 'bad.csv'
    p.write_text('frame,target_id,x,y,visible,is_target\n0,A,1,1,1,1\n0,B,2,2,1,1\n', encoding='utf-8')
    with pytest.raises(ValueError):
        load_ground_truth_csv(p)


def test_acquisition_grid_runs_modes_and_click_errors():
    frame = np.full((100, 120, 3), 25, np.uint8)
    cv2.rectangle(frame, (48, 45), (57, 54), (230,230,230), -1)
    result = run_acquisition_experiment(
        frame_bgr=frame, frame_index=0, gt_center=(52.5,49.5), roi_radius_px=45,
        frame_width=120, frame_height=100,
        detector_config=CandidateDetectorConfig(threshold_sigma=1.5, min_response=2, min_area_px=2, max_area_px=500),
        top_k=5, click_errors_px=[0,10], directions=4,
        modes=['nearest','contrast','combined'], scoring_template=AcquireScoringConfig(), success_tolerance_px=10,
    )
    assert len(result.trials) == (1+4)*3
    assert {t.mode for t in result.trials} == {'nearest','contrast','combined'}
    assert 0 <= result.success_rate <= 1
    assert any(t.success for t in result.trials if t.click_error_px == 0)


def _candidate(x, y):
    bbox = SimpleNamespace(x=int(x-2), y=int(y-2), width=5, height=5)
    return SimpleNamespace(center_x=float(x), center_y=float(y), center=(float(x),float(y)), bbox=bbox)


def _track_item(fi, x, y, ms=1.25):
    return SimpleNamespace(
        frame_index=fi, selected=_candidate(x,y), predicted_xy=(x-1,y), prediction_error_px=1.0,
        search_radius_px=40, selected_score=.8, identity_similarity=.8, motion_score=.9,
        size_score=.9, confidence_gap=.2, confidence=.85, candidate_count=2, status='TRACK',
        diagnostic_reason='ACCEPTED', adaptation_reason='ADAPT_OFF', adaptation_update_count=0,
        processing_time_ms=ms,
    )


def test_session_csv_contains_runtime(tmp_path):
    items = (_track_item(0,20,20,1.5), _track_item(1,21,20,2.5))
    session = SimpleNamespace(all_track_frames=items, reacquisitions=())
    p = export_session_csv(session, tmp_path/'session.csv', fps=25)
    text = p.read_text(encoding='utf-8-sig')
    assert 'processing_time_ms' in text
    assert '1.500000' in text
    assert '0.040000000' in text


def test_annotated_video_export(tmp_path):
    frames = []
    for k in range(3):
        f = np.zeros((64,96,3), np.uint8)
        frames.append(f)
    def reader(i):
        return i, frames[i].copy()
    session = SimpleNamespace(all_track_frames=(_track_item(1,30,30),), reacquisitions=())
    p = export_annotated_video(frame_reader=reader, frame_count=3, width=96, height=64, fps=10, session=session, path=tmp_path/'ann.mp4')
    assert p.exists() and p.stat().st_size > 0
    cap = cv2.VideoCapture(str(p))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    cap.release()
