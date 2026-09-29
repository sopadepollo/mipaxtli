"""La marca de tiempo por cuadro (ADR 0018), sin cámara.

Lo que se comprueba: el tiempo de un cuadro sale de un solo sitio
(`lsm.timing`), viaja con el cuadro por disco y por las copias que hace la
tubería, y un flujo que mezcla o desordena marcas no se lee.
"""

from __future__ import annotations

import json
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import pytest

from lsm.gaps import GapFilled, GapPolicy, fill_gaps, interpolate_frames
from lsm.io.hands import dump_frame_stream, frames_from_json, load_frame_stream
from lsm.synthetic import canonical_hand, still_sequence, to_frame, translated
from lsm.timing import TimestampError, check_timestamps, frame_times_ms
from lsm.types import FrameSlot, InvalidFrame, InvalidReason, RawFrame


def _con_marcas(
    frames: tuple[FrameSlot, ...], inicio: float = 1000.0
) -> tuple[FrameSlot, ...]:
    return tuple(
        replace(f, timestamp_ms=inicio + 34.0 * i) for i, f in enumerate(frames)
    )


def test_sin_marcas_el_tiempo_es_el_del_indice_a_la_tasa_dada() -> None:
    frames = still_sequence(canonical_hand(), length=4).frames

    assert frame_times_ms(frames, 25.0) == (0.0, 40.0, 80.0, 120.0)


def test_con_marcas_el_tiempo_es_el_de_las_marcas() -> None:
    frames = _con_marcas(still_sequence(canonical_hand(), length=3).frames)

    assert frame_times_ms(frames, 30.0) == (1000.0, 1034.0, 1068.0)


def test_un_flujo_que_mezcla_marcas_se_rechaza() -> None:
    frames = still_sequence(canonical_hand(), length=3).frames
    mezclado = (replace(frames[0], timestamp_ms=5.0), *frames[1:])

    with pytest.raises(TimestampError, match="mezcla"):
        check_timestamps(mezclado)


def test_un_flujo_que_retrocede_o_repite_se_rechaza() -> None:
    frames = still_sequence(canonical_hand(), length=3).frames
    repetido = (
        replace(frames[0], timestamp_ms=5.0),
        replace(frames[1], timestamp_ms=5.0),
        replace(frames[2], timestamp_ms=9.0),
    )

    with pytest.raises(TimestampError, match="cuadro 1"):
        check_timestamps(repetido)


def test_la_marca_sobrevive_el_viaje_a_disco_huecos_incluidos(tmp_path: Path) -> None:
    mano = to_frame(canonical_hand(), width=1280, height=720)
    flujo: tuple[FrameSlot, ...] = (
        replace(mano, timestamp_ms=100.0),
        InvalidFrame(reason=InvalidReason.NO_HAND, timestamp_ms=133.5),
        replace(mano, timestamp_ms=167.25),
    )
    ruta = tmp_path / "flujo.json"

    dump_frame_stream(flujo, ruta)

    assert load_frame_stream(ruta) == flujo
    assert json.loads(ruta.read_text(encoding="utf-8"))["schema_version"] == 2


def test_un_fixture_de_la_version_1_se_lee_sin_marcas(tmp_path: Path) -> None:
    mano = to_frame(canonical_hand(), width=1280, height=720)
    ruta = tmp_path / "viejo.json"
    dump_frame_stream((mano, mano), ruta)
    payload = json.loads(ruta.read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    ruta.write_text(json.dumps(payload), encoding="utf-8")

    leido = load_frame_stream(ruta)

    assert all(slot.timestamp_ms is None for slot in leido)


def test_un_archivo_con_marcas_desordenadas_no_se_lee() -> None:
    mano = to_frame(canonical_hand(), width=1280, height=720)
    entradas = [
        {**_json(mano), "t_ms": 50.0},
        {**_json(mano), "t_ms": 40.0},
    ]

    with pytest.raises(TimestampError):
        frames_from_json(entradas)


def _json(frame: RawFrame) -> dict[str, object]:
    from lsm.io.hands import frames_to_json

    return dict(frames_to_json((frame,))[0])


def test_un_frame_rellenado_ocupa_el_instante_del_hueco() -> None:
    """El cuadro inventado reemplaza al hueco que la cámara entregó: toma su
    marca, no una interpolada. La geometría sigue siendo la de la v7."""
    a = replace(to_frame(canonical_hand(), width=1280, height=720), timestamp_ms=0.0)
    b = replace(
        to_frame(translated(canonical_hand(), 30.0, 0.0), width=1280, height=720),
        timestamp_ms=100.0,
    )

    rellenos = interpolate_frames(a, b, 2, (30.0, 70.0))

    assert [f.timestamp_ms for f in rellenos] == [30.0, 70.0]
    sin_marcas = interpolate_frames(
        replace(a, timestamp_ms=None), replace(b, timestamp_ms=None), 2
    )
    assert [f.landmarks for f in rellenos] == [f.landmarks for f in sin_marcas]


def test_fill_gaps_conserva_las_marcas_del_flujo() -> None:
    frames = _con_marcas(still_sequence(canonical_hand(), length=5).frames)
    hueco = InvalidFrame(
        reason=InvalidReason.NO_HAND, timestamp_ms=frames[2].timestamp_ms
    )
    flujo = (*frames[:2], hueco, *frames[3:])

    reconstruida = fill_gaps(flujo, GapPolicy(max_gap_frames=3, max_fraction=0.5))

    assert isinstance(reconstruida, GapFilled)
    assert [f.timestamp_ms for f in reconstruida.sequence.frames] == [
        f.timestamp_ms for f in frames
    ]


def test_el_reposo_del_replay_avanza_el_reloj() -> None:
    """`evaluation.replay_sample` fabrica el reposo repitiendo el último frame:
    con marcas, cada repetido llega 1000/fps ms después, o el flujo repetiría
    marca."""
    from datetime import UTC, datetime

    from lsm.config import Config
    from lsm.evaluation import replay_sample
    from lsm.types import (
        Distance,
        Handedness,
        LightDirection,
        LightLevel,
        Prediction,
        Sample,
        SampleKind,
        Sequence,
    )

    frames = _con_marcas(still_sequence(canonical_hand(), length=6).frames)
    vistos: list[tuple[float | None, ...]] = []

    def clasificar(ventana: Sequence, _origen: object) -> Prediction:
        vistos.append(tuple(f.timestamp_ms for f in ventana.frames))
        return Prediction.unknown()

    muestra = Sample(
        sequence=Sequence(frames=tuple(f for f in frames if isinstance(f, RawFrame))),
        label="A",
        signer_id="s01",
        session_id="x",
        timestamp=datetime(2026, 9, 29, tzinfo=UTC),
        handedness=Handedness.RIGHT,
        light_level=LightLevel.INDOOR,
        light_direction=LightDirection.FRONTAL,
        distance=Distance.MEDIUM,
        mean_luminance=0.4,
        mean_scale_px=90.0,
        kind=SampleKind.STATIC,
    )

    replay_sample(muestra, Config(), clasificar, rest_after=10)

    assert vistos
    for marcas in vistos:
        assert all(
            b is not None and a is not None and b > a for a, b in pairwise(marcas)
        )
