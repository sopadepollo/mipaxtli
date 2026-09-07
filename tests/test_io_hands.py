"""La frontera con MediaPipe, sin MediaPipe.

Toda la suite tiene que correr en una máquina sin webcam y sin MediaPipe
instalado. Este archivo verifica dos cosas: que el detector falso reproduzca
secuencias grabadas fielmente —huecos incluidos—, y que el paquete siga sin
importar MediaPipe en ningún módulo.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from lsm.io.hands import (
    FIXTURE_SCHEMA_VERSION,
    FakeHandDetector,
    HandDetector,
    dump_frame_stream,
    load_frame_stream,
)
from lsm.synthetic import arc_offsets, canonical_hand, moving_sequence
from lsm.types import FrameStream, Handedness, InvalidFrame, InvalidReason, RawFrame

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sequences"


def a_stream() -> FrameStream:
    sequence = moving_sequence(canonical_hand(), arc_offsets(5))
    return (
        sequence.frames[0],
        sequence.frames[1],
        InvalidFrame(reason=InvalidReason.NO_HAND, detail="la mano salió del encuadre"),
        *sequence.frames[2:],
    )


def test_el_detector_falso_satisface_el_protocolo() -> None:
    assert isinstance(FakeHandDetector(slots=a_stream()), HandDetector)


def test_reproduce_los_frames_en_orden() -> None:
    stream = a_stream()
    detector = FakeHandDetector(slots=stream)

    assert tuple(detector.stream()) == stream


def test_detect_ignora_la_imagen_y_avanza_la_grabacion() -> None:
    """El doble no mira la imagen: reproduce lo grabado. Es lo que permite
    ejercitar la tubería completa sin cámara."""
    detector = FakeHandDetector(slots=a_stream())

    first = detector.detect(image=None)
    second = detector.detect(image=None)

    assert first is not second
    assert isinstance(first, RawFrame)


def test_al_agotarse_la_grabacion_devuelve_frames_invalidos() -> None:
    """Se acabó el video: no hay mano. Marcador explícito, no una excepción ni
    un `None`, para que el consumidor no necesite un caso especial."""
    detector = FakeHandDetector(slots=(a_stream()[0],))

    detector.detect(image=None)
    exhausted = detector.detect(image=None)

    assert isinstance(exhausted, InvalidFrame)
    assert exhausted.reason is InvalidReason.NO_HAND


def test_el_hueco_se_conserva_al_ida_y_vuelta_por_disco(tmp_path: Path) -> None:
    """Si el hueco se perdiera al guardar, una secuencia interrumpida se
    convertiría en una continua y el dataset mentiría."""
    stream = a_stream()
    path = tmp_path / "seq.json"

    dump_frame_stream(stream, path)
    restored = load_frame_stream(path)

    assert len(restored) == len(stream)
    assert isinstance(restored[2], InvalidFrame)
    assert restored[2].reason is InvalidReason.NO_HAND
    assert restored == stream


def test_los_landmarks_sobreviven_el_ida_y_vuelta_sin_perder_precision(
    tmp_path: Path,
) -> None:
    stream = a_stream()
    path = tmp_path / "seq.json"

    dump_frame_stream(stream, path)
    restored = load_frame_stream(path)

    original = stream[0]
    recovered = restored[0]
    assert isinstance(original, RawFrame)
    assert isinstance(recovered, RawFrame)
    for before, after in zip(original.landmarks, recovered.landmarks, strict=True):
        assert after.x == before.x
        assert after.y == before.y
        assert after.z == before.z


def test_un_fixture_de_otra_version_se_rechaza(tmp_path: Path) -> None:
    path = tmp_path / "seq.json"
    dump_frame_stream(a_stream(), path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = FIXTURE_SCHEMA_VERSION + 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        load_frame_stream(path)


def test_el_fixture_de_ejemplo_del_repositorio_se_carga() -> None:
    stream = load_frame_stream(FIXTURES / "ejemplo_trazo_j.json")

    assert len(stream) >= 8
    assert any(isinstance(slot, InvalidFrame) for slot in stream)
    assert all(
        slot.handedness is Handedness.RIGHT
        for slot in stream
        if isinstance(slot, RawFrame)
    )


def test_desde_un_fixture_se_construye_el_detector() -> None:
    detector = FakeHandDetector.from_fixture(FIXTURES / "ejemplo_trazo_j.json")

    assert len(tuple(detector.stream())) >= 8


def test_ningun_modulo_del_paquete_importa_mediapipe() -> None:
    """`CLAUDE.md` §3: MediaPipe solo puede vivir en `io/hands.py`, y en Fase 0 ni
    siquiera ahí. Importar el paquete entero no debe arrastrarlo."""
    for module in (
        "lsm.classifiers.base",
        "lsm.classifiers.dummy",
        "lsm.config",
        "lsm.features",
        "lsm.io.hands",
        "lsm.segmentation",
        "lsm.synthetic",
        "lsm.types",
    ):
        importlib.import_module(module)

    assert "mediapipe" not in sys.modules
    assert "cv2" not in sys.modules
