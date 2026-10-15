import pytest

from navmin.vision.processors.vt11.acquisition.roi import clamp_point, make_operator_selection, make_roi


def test_centered_roi_has_nominal_odd_side():
    roi = make_roi((500.0, 300.0), 70, 1024, 768)
    assert (roi.x, roi.y, roi.width, roi.height) == (430, 230, 141, 141)
    assert roi.center == pytest.approx((500.0, 300.0))


def test_roi_is_clipped_at_frame_edges_without_shifting_click_prior():
    roi = make_roi((10.0, 20.0), 70, 1024, 768)
    assert roi.x == 0
    assert roi.y == 0
    assert roi.x2 == 81
    assert roi.y2 == 91
    assert roi.width == 81
    assert roi.height == 91


def test_click_is_clamped_to_source_frame():
    assert clamp_point(-4, 900, 1024, 768) == (0.0, 767.0)


def test_operator_selection_carries_source_frame_and_roi():
    s = make_operator_selection(12, (253.5, 157.25), 70, 496, 368)
    assert s.frame_index == 12
    assert s.click_x == pytest.approx(253.5)
    assert s.click_y == pytest.approx(157.25)
    assert s.roi_radius_px == 70
    assert s.roi.width == 141
    assert s.roi.height == 141


def test_invalid_roi_radius_rejected():
    with pytest.raises(ValueError):
        make_roi((10, 10), 0, 100, 100)
