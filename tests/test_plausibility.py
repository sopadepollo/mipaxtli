"""El filtro de plausibilidad (§0.4, ADR 0027), sin cámara.

Un cuadro físicamente imposible no se corrige: se invalida y lo resuelve el
relleno de huecos. Aquí se comprueba cada una de las tres comprobaciones, que la
referencia solo aprende de cuadros aceptados, y que se reinicia cuando deja de
describir a la mano que hay delante.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from lsm.config import Config
from lsm.landmark_stats import BONES, bone_lengths, canonical_points
from lsm.plausibility import (
    ImplausibleKind,
    PlausibilityFilter,
    PlausibilityParams,
    filter_stream,
)
from lsm.segmentation import (
    FrameThresholds,
    WindowDynamic,
    run_segmentation,
)
from lsm.synthetic import canonical_hand, to_frame, translated
from lsm.types import (
    FrameSlot,
    InvalidFrame,
    InvalidReason,
    Landmark,
    Prediction,
    RawFrame,
    Sequence,
    WindowOrigin,
)

PARAMS = PlausibilityParams(
    enabled=True,
    bone_max_deviation=0.45,
    bone_reference_frames=5,
    max_mcp_dorsal_deg=None,
    max_palm_speed_per_s=30.0,
    reset_after_ms=150.0,
)


def _mano(dx: float = 0.0) -> RawFrame:
    return to_frame(
        translated(canonical_hand(), 400.0 + dx, 360.0), width=1280, height=720
    )


def _dedo_largo(frame: RawFrame, factor: float) -> RawFrame:
    """La punta del índice alejada de su DIP: un hueso que cambió de largo."""
    puntos = list(frame.landmarks)
    dip, punta = puntos[7], puntos[8]
    puntos[8] = Landmark(
        x=dip.x + (punta.x - dip.x) * factor,
        y=dip.y + (punta.y - dip.y) * factor,
        z=dip.z + (punta.z - dip.z) * factor,
    )
    return replace(frame, landmarks=tuple(puntos))


def test_la_mano_sintetica_tiene_20_huesos_de_largo_positivo() -> None:
    largos = bone_lengths(canonical_points(_mano()), depth=True)

    assert largos is not None
    assert len(largos) == len(BONES) == 20
    assert min(largos) > 0.0


def test_un_hueso_que_cambia_de_largo_se_invalida() -> None:
    filtro = PlausibilityFilter(PARAMS)
    for i in range(5):
        assert filtro.check(_mano(), 33.0 * i) is None

    assert filtro.check(_dedo_largo(_mano(), 6.0), 200.0) is ImplausibleKind.BONE


def test_sin_referencia_llena_no_se_juzgan_los_huesos() -> None:
    filtro = PlausibilityFilter(PARAMS)
    filtro.check(_mano(), 0.0)

    assert filtro.check(_dedo_largo(_mano(), 6.0), 33.0) is None


def test_un_salto_de_la_palma_se_invalida_y_uno_lento_no() -> None:
    filtro = PlausibilityFilter(PARAMS)
    filtro.check(_mano(), 0.0)

    # 600 px en 33 ms: decenas de palmas por segundo.
    assert filtro.check(_mano(600.0), 33.0) is ImplausibleKind.JUMP
    # El mismo desplazamiento en 2 s es un movimiento normal.
    assert filtro.check(_mano(600.0), 2000.0) is None


def test_un_cuadro_invalido_no_mueve_la_referencia() -> None:
    """El salto se mide contra el último cuadro ACEPTADO: tras un cuadro
    teletransportado, volver a la posición de antes no es otro salto."""
    filtro = PlausibilityFilter(PARAMS)
    filtro.check(_mano(), 0.0)
    assert filtro.check(_mano(600.0), 33.0) is ImplausibleKind.JUMP

    assert filtro.check(_mano(2.0), 66.0) is None


def test_una_racha_de_rechazos_larga_reinicia_la_referencia() -> None:
    """Si la mano de delante ya no se parece a la referencia durante
    `reset_after_ms`, la referencia es la que está mal: se empieza otra."""
    filtro = PlausibilityFilter(PARAMS)
    filtro.check(_mano(), 0.0)
    lejos = _mano(600.0)

    assert filtro.check(lejos, 33.0) is ImplausibleKind.JUMP
    assert filtro.check(lejos, 100.0) is ImplausibleKind.JUMP
    assert filtro.check(lejos, 233.0) is None
    assert filtro.check(_mano(602.0), 266.0) is None


def test_desactivado_no_invalida_nada() -> None:
    filtro = PlausibilityFilter(replace(PARAMS, enabled=False))
    filtro.check(_mano(), 0.0)

    assert filtro.check(_mano(600.0), 33.0) is None


def test_el_flujo_convierte_el_cuadro_imposible_en_un_hueco_con_su_marca() -> None:
    flujo: tuple[FrameSlot, ...] = (
        replace(_mano(), timestamp_ms=10.0),
        replace(_mano(600.0), timestamp_ms=43.0),
        replace(_mano(1.0), timestamp_ms=76.0),
    )

    salida = list(filter_stream(flujo, PARAMS, 30.0))

    assert salida[0] == flujo[0]
    assert salida[2] == flujo[2]
    hueco = salida[1]
    assert isinstance(hueco, InvalidFrame)
    assert hueco.reason is InvalidReason.IMPLAUSIBLE
    assert hueco.detail == ImplausibleKind.JUMP.value
    assert hueco.timestamp_ms == 43.0


def test_los_valores_de_fabrica_salen_de_la_medicion() -> None:
    """ADR 0026/0027: 0.45 palmas, 15 cuadros, 30 palmas/s, sin articulación,
    reinicio a los 100 ms."""
    p = PlausibilityParams.from_config(Config())

    assert (p.bone_max_deviation, p.bone_reference_frames) == (0.45, 15)
    assert p.max_palm_speed_per_s == 30.0
    assert p.max_mcp_dorsal_deg is None
    assert p.reset_after_ms == 100.0


def test_en_vivo_un_cuadro_imposible_a_mitad_de_trazo_se_rellena_y_se_cuenta() -> None:
    """La máquina lo trata como un hueco corto: el trazo sigue entero y el
    evento dice cuántos rellenados sustituyen a cuadros imposibles."""
    config = Config()
    umbrales = FrameThresholds.from_config(config, 30.0)
    quieta = [_mano()] * 20
    trazo = [_mano(10.0 * i) for i in range(1, 31)]
    trazo[20] = _mano(10.0 * 21 + 500.0)  # un cuadro teletransportado
    reposo = [trazo[-1]] * (umbrales.motion_confirm_low_frames + 5)

    def ninguno(_w: Sequence, _o: WindowOrigin) -> Prediction:
        return Prediction.unknown()

    eventos = list(run_segmentation([*quieta, *trazo, *reposo], config, ninguno))
    trazos = [e for e in eventos if isinstance(e, WindowDynamic)]

    assert len(trazos) == 1
    assert trazos[0].implausible_frames == 1
    assert trazos[0].interpolated_frames == 1


@pytest.mark.parametrize("fps", [15.0, 30.0])
def test_sin_marcas_el_tiempo_es_el_del_indice(fps: float) -> None:
    """Sin marca de tiempo, 600 px en un cuadro sigue siendo un salto a
    cualquier tasa razonable."""
    salida = list(filter_stream((_mano(), _mano(600.0)), PARAMS, fps))

    assert isinstance(salida[1], InvalidFrame)
