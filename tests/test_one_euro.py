"""El filtro One Euro (§4, ADR 0028), sin cámara."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from lsm.config import Config
from lsm.one_euro import (
    OneEuroFilter,
    OneEuroParams,
    filter_sequence,
    smoothing_factor,
)
from lsm.synthetic import canonical_hand, still_sequence, to_frame, translated
from lsm.types import RawFrame, Sequence

ACTIVO = OneEuroParams(enabled=True, min_cutoff=1.0, beta=0.0, d_cutoff=1.0)


def _mano(dx: float = 0.0) -> RawFrame:
    return to_frame(
        translated(canonical_hand(), 400.0 + dx, 360.0), width=1280, height=720
    )


def test_apagado_es_la_identidad() -> None:
    filtro = OneEuroFilter(replace(ACTIVO, enabled=False))
    filtro.step(_mano(), 0.0)

    assert filtro.step(_mano(50.0), 33.0) == _mano(50.0)


def test_el_primer_frame_sale_tal_cual() -> None:
    assert OneEuroFilter(ACTIVO).step(_mano(), 0.0) == _mano()


def test_una_mano_quieta_no_se_mueve() -> None:
    frames = still_sequence(canonical_hand(), length=10)
    tiempos = [33.0 * i for i in range(10)]

    salida = filter_sequence(frames, tiempos, ACTIVO)

    for filtrado, crudo in zip(salida.frames, frames.frames, strict=True):
        for a, b in zip(filtrado.landmarks, crudo.landmarks, strict=True):
            assert (a.x, a.y, a.z) == pytest.approx((b.x, b.y, b.z), abs=1e-15)


def test_con_beta_cero_es_un_pasa_bajos_con_la_formula_del_contrato() -> None:
    """x̂_1 = α·x_1 + (1 − α)·x̂_0, con α = r / (r + 1), r = 2π·f·Δ."""
    a, b = _mano(), _mano(40.0)
    salida = OneEuroFilter(ACTIVO)
    salida.step(a, 0.0)
    filtrado = salida.step(b, 50.0)

    alfa = smoothing_factor(1.0, 0.05)
    assert alfa == pytest.approx(2 * math.pi * 0.05 / (2 * math.pi * 0.05 + 1))
    esperado = alfa * b.landmarks[0].x + (1 - alfa) * a.landmarks[0].x
    assert filtrado.landmarks[0].x == pytest.approx(esperado, abs=1e-15)


def test_beta_abre_el_filtro_cuando_la_mano_se_mueve() -> None:
    """Con velocidad, el corte sube: el frame filtrado queda más cerca del crudo."""
    trazo = [_mano(30.0 * i) for i in range(8)]
    tiempos = [33.0 * i for i in range(8)]
    lento = filter_sequence(Sequence(frames=tuple(trazo)), tiempos, ACTIVO)
    rapido = filter_sequence(
        Sequence(frames=tuple(trazo)), tiempos, replace(ACTIVO, beta=5.0)
    )

    crudo = trazo[-1].landmarks[0].x
    assert abs(rapido.frames[-1].landmarks[0].x - crudo) < abs(
        lento.frames[-1].landmarks[0].x - crudo
    )


def test_usa_el_tiempo_real_entre_cuadros() -> None:
    """El mismo salto filtrado más fuerte si llega a 10 ms que a 100 ms."""
    a, b = _mano(), _mano(40.0)
    corto = OneEuroFilter(ACTIVO)
    corto.step(a, 0.0)
    largo = OneEuroFilter(ACTIVO)
    largo.step(a, 0.0)

    x_corto = corto.step(b, 10.0).landmarks[0].x
    x_largo = largo.step(b, 100.0).landmarks[0].x

    assert abs(x_corto - a.landmarks[0].x) < abs(x_largo - a.landmarks[0].x)


def test_el_tiempo_que_no_avanza_es_un_error() -> None:
    filtro = OneEuroFilter(ACTIVO)
    filtro.step(_mano(), 10.0)

    with pytest.raises(ValueError, match="no avanza"):
        filtro.step(_mano(5.0), 10.0)


def test_reiniciar_olvida_la_historia() -> None:
    filtro = OneEuroFilter(ACTIVO)
    filtro.step(_mano(), 0.0)
    filtro.reset()

    assert filtro.step(_mano(300.0), 33.0) == _mano(300.0)


def test_el_filtro_no_toca_los_metadatos_del_frame() -> None:
    filtro = OneEuroFilter(ACTIVO)
    filtro.step(replace(_mano(), timestamp_ms=0.0), 0.0)

    salida = filtro.step(replace(_mano(20.0), timestamp_ms=33.0), 33.0)

    assert salida.timestamp_ms == 33.0
    assert (salida.width, salida.height) == (1280, 720)


def test_de_fabrica_esta_apagado_hasta_elegir_parametros() -> None:
    """ADR 0028: ningún punto del barrido cumplió temblor y retraso a la vez."""
    assert OneEuroParams.from_config(Config()).enabled is False


def test_la_demo_puede_encenderlo_con_otros_parametros() -> None:
    from lsm.cli.demo import con_one_euro

    config = con_one_euro(Config(), "on", "0.5,1,2")

    assert config.smoothing.enabled is True
    assert (config.smoothing.min_cutoff, config.smoothing.beta) == (0.5, 1.0)
    assert config.smoothing.d_cutoff == 2.0
    assert con_one_euro(Config(), None, None) == Config()


def test_diagnosticar_puede_cambiar_los_umbrales_de_mediapipe() -> None:
    """Paso 5 (ADR 0030): cada sesión del barrido con sus umbrales, que quedan
    en los metadatos del reporte."""
    from lsm.cli.demo import con_umbrales_de_mediapipe

    config = con_umbrales_de_mediapipe(Config(), 0.3, 0.2)

    assert config.hands.min_hand_presence_confidence == 0.3
    assert config.hands.min_tracking_confidence == 0.2
    assert con_umbrales_de_mediapipe(Config(), None, None) == Config()
