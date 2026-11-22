from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from navmin.vision.processors.vt11.identity.descriptor import (
    TargetDescriptor,
    _cached_blur_shapes,
    _unit_vector,
    descriptor_similarity,
)

WEIGHTS = dict(appearance=0.50, edge=0.24, projection=0.09, geometry=0.07, color=0.10)
BLUR_WEIGHTS = dict(WEIGHTS, _visible_blur_robust=1.0)


def _shape() -> np.ndarray:
    image = np.zeros((16, 16), np.float32)
    image[6:10, 6:10] = 1
    for y, x in ((4, 3), (9, 10), (4, 10), (9, 3)):
        image[y : y + 3, x : x + 3] = 0.7
    return image


def _descriptor(image: np.ndarray) -> TargetDescriptor:
    image = (image - image.mean()) / (image.std() + 0.03)
    edges = cv2.magnitude(
        cv2.Sobel(image, -1, 1, 0),
        cv2.Sobel(image, -1, 0, 1),
    )
    edges = (edges - edges.mean()) / (edges.std() + 0.03)
    return TargetDescriptor(
        appearance=image,
        projection_x=_unit_vector(image.sum(axis=0)),
        projection_y=_unit_vector(image.sum(axis=1)),
        edge_map=edges,
        geometry=np.array([0.7, 0.5], np.float32),
        size_px=80,
        bbox=(0, 0, 16, 16),
        color_hist=np.array([1, 0], np.float32),
    )


@pytest.mark.parametrize(
    "direction", ("horizontal", "vertical", "diagonal", "antidiagonal")
)
def test_recognizes_directional_blur_without_promoting_a_wire(
    direction: str,
) -> None:
    image = _shape()
    kernel = np.zeros((5, 5), np.float32)
    if direction == "horizontal":
        kernel[2, :] = 0.2
    elif direction == "vertical":
        kernel[:, 2] = 0.2
    elif direction == "diagonal":
        np.fill_diagonal(kernel, 0.2)
    else:
        kernel[:] = np.fliplr(np.eye(5, dtype=np.float32)) * 0.2

    reference = _descriptor(image)
    candidate = _descriptor(cv2.filter2D(image, -1, kernel))
    original = descriptor_similarity(reference, candidate, identity_weights=WEIGHTS)
    robust = descriptor_similarity(reference, candidate, identity_weights=BLUR_WEIGHTS)

    wire = np.zeros_like(image)
    wire[7:9, :] = 1
    distractor = descriptor_similarity(
        reference,
        _descriptor(wire),
        identity_weights=BLUR_WEIGHTS,
    )
    assert robust.total > original.total + 0.15
    assert robust.total > 0.8
    assert distractor.total < 0.45
    assert robust.geometry == original.geometry
    assert robust.color == original.color
    assert robust.blur_variant == f"5-{direction}"


def test_blur_is_opt_in_and_preserves_sharp_reference() -> None:
    reference = _descriptor(_shape())
    before = reference.appearance.copy()
    assert (
        descriptor_similarity(
            reference,
            reference,
            identity_weights=BLUR_WEIGHTS,
        ).total
        > 0.999
    )
    candidate = _descriptor(cv2.GaussianBlur(_shape(), (5, 5), 1))
    plain = descriptor_similarity(reference, candidate, identity_weights=WEIGHTS)
    disabled = descriptor_similarity(
        reference,
        candidate,
        identity_weights=dict(WEIGHTS, _visible_blur_robust=0),
    )
    assert plain == disabled
    assert plain.blur_variant is None
    robust = descriptor_similarity(
        reference, candidate, identity_weights=BLUR_WEIGHTS
    )
    assert robust.blur_variant is not None
    np.testing.assert_array_equal(reference.appearance, before)


def test_custom_weights_and_changed_reference_do_not_reuse_wrong_scores() -> None:
    reference = _descriptor(_shape())
    candidate = _descriptor(cv2.GaussianBlur(_shape(), (5, 5), 1))
    original = descriptor_similarity(reference, candidate, identity_weights=BLUR_WEIGHTS)
    reference.appearance[:] *= -1
    changed = descriptor_similarity(reference, candidate, identity_weights=BLUR_WEIGHTS)
    assert changed.total < original.total

    candidate = replace(candidate, color_hist=np.array([0, 1], np.float32))
    color_only = dict(color=1.0, _visible_blur_robust=1.0)
    assert (
        descriptor_similarity(reference, candidate, identity_weights=color_only).total
        == 0
    )


def test_blur_cache_is_bounded_to_reference_limit() -> None:
    assert _cached_blur_shapes.cache_parameters()["maxsize"] == 192
