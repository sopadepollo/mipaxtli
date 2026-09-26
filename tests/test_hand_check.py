"""La mano declarada (ADR 0017): quién decide el espejo y cuándo se avisa.

La etiqueta de MediaPipe cambia de opinión a mitad de una J; desde
`FEATURE_SPEC_VERSION` 2 no decide nada. Estos tests fijan las tres piezas: el
detector pone la mano declarada, el dataset la aplica a todos los frames, y el
aviso «¿cambiaste de mano?» solo salta ante una contradicción sostenida con la
mano quieta — y nunca cambia la mano por su cuenta.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lsm.cli.demo import main as demo_main
from lsm.config import Config
from lsm.features import SequenceFeatures, extract_sequence_features
from lsm.hand_check import HandMismatchWatcher
from lsm.io.dataset import (
    HandSource,
    SampleMetadata,
    StoredSample,
    read_sample,
    write_sample,
)
from lsm.io.hands import MediaPipeHandDetector
from lsm.synthetic import canonical_hand, to_frame, translated
from lsm.types import (
    Distance,
    FrameStream,
    Handedness,
    InvalidFrame,
    InvalidReason,
    LightDirection,
    LightLevel,
    RawFrame,
    SampleKind,
)

CONFIG = Config()
FPS = 30.0


# Dobles con la forma de `HandLandmarkerResult` (ver `test_io_hands.py`).
@dataclass
class _Categoria:
    category_name: str
    score: float


@dataclass
class _Punto:
    x: float
    y: float
    z: float


@dataclass
class _Resultado:
    handedness: list[list[_Categoria]]
    hand_landmarks: list[list[_Punto]]


def _detector(declared: Handedness) -> MediaPipeHandDetector:
    return MediaPipeHandDetector(
        model_path=Path("data/models/hand_landmarker.task"),
        min_detection_score=0.5,
        declared_hand=declared,
    )


def _puntos() -> list[_Punto]:
    return [_Punto(x=0.5 + 0.01 * i, y=0.5 - 0.01 * i, z=0.0) for i in range(21)]


def muestra(frames: FrameStream | None = None, **cambios: object) -> StoredSample:
    base: dict[str, object] = {
        "label": "A",
        "signer_id": "s01",
        "session_id": "sesion-a",
        "timestamp": datetime(2026, 9, 26, 10, 0, tzinfo=UTC),
        "handedness": Handedness.RIGHT,
        "light_level": LightLevel.INDOOR,
        "light_direction": LightDirection.FRONTAL,
        "distance": Distance.MEDIUM,
        "mean_luminance": 0.4,
        "mean_scale_px": 100.0,
        "kind": SampleKind.STATIC,
        "dispersion": 0.01,
        "arc_length": 0.0,
        "handedness_swapped": False,
    }
    base.update(cambios)
    if frames is None:
        frames = tuple(mano(400.0, Handedness.RIGHT) for _ in range(8))
    return StoredSample(metadata=SampleMetadata(**base), frames=frames)  # type: ignore[arg-type]


def mano(x: float, detectada: Handedness, score: float = 0.97) -> RawFrame:
    frame = to_frame(translated(canonical_hand(), x, 400.0), width=1280, height=720)
    return replace(
        frame,
        handedness=Handedness.RIGHT,
        detected_handedness=detectada,
        handedness_score=score,
    )


# --------------------------------------------------------------------------- #
# El detector pone la mano declarada
# --------------------------------------------------------------------------- #


def test_con_mano_declarada_la_etiqueta_de_mediapipe_solo_se_guarda() -> None:
    resultado = _Resultado(
        handedness=[[_Categoria("Left", 0.9)]], hand_landmarks=[_puntos()]
    )

    slot = _detector(Handedness.RIGHT)._to_slot(resultado, width=640, height=480)

    assert isinstance(slot, RawFrame)
    assert slot.handedness is Handedness.RIGHT
    assert slot.detected_handedness is Handedness.LEFT


def test_un_cambio_de_etiqueta_ya_no_espeja_la_mano() -> None:
    """El defecto del ADR 0017, en un test: dos cuadros idénticos con etiquetas
    distintas daban una velocidad absurda porque el paso 2 espejaba uno de los
    dos. Con la mano declarada, la misma mano da velocidad cero."""
    from lsm.types import Sequence

    a = mano(400.0, Handedness.RIGHT)
    b = mano(400.0, Handedness.LEFT)

    features = extract_sequence_features(Sequence(frames=(a, b)), CONFIG)

    assert isinstance(features, SequenceFeatures)
    assert features.velocities == (0.0,)


# --------------------------------------------------------------------------- #
# El dataset
# --------------------------------------------------------------------------- #


def test_una_muestra_nueva_declara_su_mano(tmp_path: Path) -> None:
    recuperada = read_sample(write_sample(tmp_path, muestra()))

    assert recuperada.metadata.handedness_source is HandSource.DECLARED


def test_una_muestra_v2_se_lee_con_la_mano_que_dijo_mediapipe(tmp_path: Path) -> None:
    """El dataset de la Fase 1 no declaró la mano. Su `handedness` es la etiqueta
    de MediaPipe, que la captura exigía igual en todos los frames."""
    ruta = write_sample(tmp_path, muestra())
    payload = json.loads(ruta.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    del payload["handedness_source"]
    for frame in payload["frames"]:
        frame.pop("detected_handedness", None)
    ruta.write_text(json.dumps(payload), encoding="utf-8")

    recuperada = read_sample(ruta)

    assert recuperada.metadata.handedness_source is HandSource.DETECTED
    primero = recuperada.frames[0]
    assert isinstance(primero, RawFrame)
    assert primero.detected_handedness is primero.handedness


def test_la_mano_de_la_muestra_gobierna_a_todos_sus_frames() -> None:
    """Aunque un frame guardado diga otra cosa, al cargar manda la muestra."""
    frames = tuple(
        replace(mano(400.0, Handedness.LEFT), handedness=Handedness.LEFT)
        if i == 3
        else mano(400.0, Handedness.RIGHT)
        for i in range(8)
    )

    sample = muestra(frames=frames, handedness=Handedness.RIGHT).to_sample()

    assert {f.handedness for f in sample.sequence.frames} == {Handedness.RIGHT}
    assert sample.sequence.frames[3].detected_handedness is Handedness.LEFT


# --------------------------------------------------------------------------- #
# El aviso
# --------------------------------------------------------------------------- #


def vigia() -> HandMismatchWatcher:
    return HandMismatchWatcher(config=CONFIG, declared=Handedness.RIGHT, fps=FPS)


def test_una_contradiccion_sostenida_con_la_mano_quieta_avisa() -> None:
    v = vigia()
    avisos = [
        v.observe(mano(400.0, Handedness.LEFT)) for _ in range(v.frames_needed + 2)
    ]

    assert not avisos[0]
    assert avisos[-1]


def test_los_parpadeos_de_un_trazo_no_avisan() -> None:
    """Es exactamente el caso de la J: la etiqueta cambia mientras la mano se
    mueve. No es un cambio de mano y no se pregunta nada."""
    v = vigia()
    avisos = [
        v.observe(mano(400.0 + 20.0 * i, Handedness.LEFT))
        for i in range(3 * v.frames_needed)
    ]

    assert not any(avisos)


def test_con_score_bajo_no_avisa() -> None:
    v = vigia()
    umbral = CONFIG.hands.mismatch_min_score
    avisos = [
        v.observe(mano(400.0, Handedness.LEFT, score=umbral - 0.05))
        for _ in range(3 * v.frames_needed)
    ]

    assert not any(avisos)


def test_una_perdida_de_mano_reinicia_la_cuenta() -> None:
    v = vigia()
    for _ in range(v.frames_needed - 1):
        v.observe(mano(400.0, Handedness.LEFT))
    v.observe(InvalidFrame(reason=InvalidReason.NO_HAND))

    assert not v.observe(mano(400.0, Handedness.LEFT))


# --------------------------------------------------------------------------- #
# La demo exige la mano
# --------------------------------------------------------------------------- #


def test_la_demo_en_vivo_sin_mano_se_niega(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import lsm.cli.demo as demo
    from lsm.classifiers.registry import ClassifierRegistry

    monkeypatch.setattr(demo, "cargar_registro", lambda *_a: ClassifierRegistry())

    codigo = demo_main(["--config", "config.yaml"])

    assert codigo == 2
    assert "--mano" in capsys.readouterr().out
