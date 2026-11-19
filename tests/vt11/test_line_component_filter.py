from dataclasses import replace

import cv2
import numpy as np

from navmin.vision.processors.vt11.acquisition.candidate_detector import CandidateDetectorConfig, _suppress_lines, detect_local_candidates
from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection


def test_diagonal_wire_is_removed_without_discarding_compact_target():
    frame = np.full((180, 240, 3), 170, np.uint8)
    cv2.line(frame, (20, 55), (218, 115), (35, 35, 35), 2)
    cv2.ellipse(frame, (113, 113), (12, 7), 0, 0, 360, (40, 40, 40), -1)
    selection = make_operator_selection(
        frame_index=0, click_xy=(120, 90), radius_px=110, frame_width=240, frame_height=180,
    )
    cfg = CandidateDetectorConfig(max_area_px=5000)
    baseline = detect_local_candidates(frame, selection, config=cfg)
    filtered = detect_local_candidates(frame, selection, config=replace(cfg, reject_line_components=True))
    assert any(candidate.bbox.width > 150 for candidate in baseline.candidates)
    assert filtered.diagnostics.rejected_line_components >= 1
    assert all(candidate.bbox.width < 100 for candidate in filtered.candidates)
    assert any(abs(candidate.center_x - 113) < 3 and abs(candidate.center_y - 113) < 3
               for candidate in filtered.candidates)


def test_wire_overlap_can_leave_a_compact_target_core():
    mask = np.zeros((180, 240), np.uint8)
    cv2.line(mask, (10, 90), (227, 90), 1, 2)
    cv2.ellipse(mask, (120, 90), (14, 9), 0, 0, 360, 1, -1)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    cleaned, restored, rejected = _suppress_lines(labels, stats, expected_area=400)
    assert rejected == 1
    assert np.array_equal(cleaned, restored)
    assert restored[90, 120] == 1
    assert restored[90, 30] == 0
    assert cv2.boundingRect(restored)[2] < 40
