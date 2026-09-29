"""Tolerancia a huecos en el camino dinámico (Bloque 2, ADR 0021).

Los cinco casos del plan viven también en los golden (`gap_cases`); aquí van
las propiedades que un caso suelto no deja ver.
"""

from __future__ import annotations

import pytest

from lsm.gaps import (
    GapFilled,
    GapPolicy,
    GapRejected,
    GapRejection,
    can_bridge,
    fill_gaps,
    interpolate_frames,
)
from lsm.synthetic import canonical_hand, moving_sequence
from lsm.types import FrameStream, InvalidFrame, InvalidReason, RawFrame

HUECO = InvalidFrame(reason=InvalidReason.NO_HAND)
POLITICA = GapPolicy(max_gap_frames=5, max_fraction=0.25)


def _mano_que_viaja(n: int = 10) -> tuple[RawFrame, ...]:
    return moving_sequence(
        canonical_hand(), tuple((10.0 * i, 0.0) for i in range(n))
    ).frames


def test_la_interpolacion_es_lineal_en_los_landmarks_crudos() -> None:
    a, _, b = _mano_que_viaja(3)

    (medio,) = interpolate_frames(a, b, 1)

    for pa, pm, pb in zip(a.landmarks, medio.landmarks, b.landmarks, strict=True):
        assert pm.x == pytest.approx((pa.x + pb.x) / 2)
        assert pm.y == pytest.approx((pa.y + pb.y) / 2)
        assert pm.z == pytest.approx((pa.z + pb.z) / 2)


def test_un_frame_inventado_no_es_mas_fiable_que_los_reales() -> None:
    from dataclasses import replace

    a, b = _mano_que_viaja(2)
    a = replace(a, detection_score=0.9, handedness_score=0.7)
    b = replace(b, detection_score=0.6, handedness_score=0.95)

    (relleno,) = interpolate_frames(a, b, 1)

    assert relleno.detection_score == 0.6
    assert relleno.handedness_score == 0.7
    assert relleno.detected_handedness is None  # nadie lo detectó


def test_no_se_interpola_entre_resoluciones_distintas() -> None:
    from dataclasses import replace

    a, b = _mano_que_viaja(2)

    assert can_bridge(a, b)
    assert not can_bridge(a, replace(b, width=640, height=480))


def test_un_flujo_sin_frames_validos_se_rechaza() -> None:
    salida = fill_gaps((HUECO, HUECO), POLITICA)

    assert isinstance(salida, GapRejected)
    assert salida.reason is GapRejection.NO_VALID_FRAMES


def test_un_flujo_sin_huecos_sale_igual() -> None:
    frames = _mano_que_viaja()

    salida = fill_gaps(frames, POLITICA)

    assert isinstance(salida, GapFilled)
    assert salida.sequence.frames == frames
    assert (salida.interpolated, salida.trimmed_start, salida.trimmed_end) == (0, 0, 0)


def test_el_primer_hueco_que_falla_decide_el_rechazo() -> None:
    frames = _mano_que_viaja(12)
    flujo: FrameStream = (
        *frames[:2],
        *([HUECO] * 6),  # largo: decide este
        *frames[8:9],
        HUECO,
        *frames[10:],
    )

    salida = fill_gaps(flujo, POLITICA)

    assert isinstance(salida, GapRejected)
    assert salida.reason is GapRejection.GAP_TOO_LONG
    assert salida.frame_index == 2
