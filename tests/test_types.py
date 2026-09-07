"""Los tipos base tienen que hacer imposible lo que la arquitectura prohíbe.

En particular: que un frame inválido se cuele como `None`, que una `Sequence`
llegue vacía al clasificador, o que una `Sample` viaje sin los metadatos que
necesita la validación leave-one-signer-out.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from lsm.types import (
    NUM_FEATURES,
    NUM_LANDMARKS,
    UNKNOWN_LABEL,
    Distance,
    FeatureVector,
    Handedness,
    InvalidFrame,
    InvalidReason,
    Landmark,
    LandmarkIndex,
    Lighting,
    Prediction,
    RawFrame,
    Sample,
    Sequence,
    TrajectoryChannel,
)


def _landmarks() -> tuple[Landmark, ...]:
    return tuple(Landmark(0.5, 0.5, 0.0) for _ in range(NUM_LANDMARKS))


def _frame(width: int = 1280, height: int = 720) -> RawFrame:
    return RawFrame(
        landmarks=_landmarks(),
        width=width,
        height=height,
        handedness=Handedness.RIGHT,
        handedness_score=0.99,
        detection_score=0.95,
    )


def test_los_indices_de_landmark_siguen_la_numeracion_de_mediapipe() -> None:
    assert LandmarkIndex.WRIST == 0
    assert LandmarkIndex.MIDDLE_MCP == 9
    assert LandmarkIndex.PINKY_TIP == 20
    assert len(LandmarkIndex) == NUM_LANDMARKS


def test_un_frame_con_veinte_landmarks_no_se_construye() -> None:
    with pytest.raises(ValueError, match="21 landmarks"):
        RawFrame(
            landmarks=_landmarks()[:-1],
            width=1280,
            height=720,
            handedness=Handedness.RIGHT,
            handedness_score=1.0,
            detection_score=1.0,
        )


def test_el_aspect_ratio_sale_del_frame_y_no_se_asume() -> None:
    assert _frame(1280, 720).aspect_ratio == pytest.approx(16 / 9)
    assert _frame(720, 720).aspect_ratio == 1.0


def test_el_frame_invalido_es_un_marcador_con_motivo_no_un_none() -> None:
    invalid = InvalidFrame(reason=InvalidReason.NO_HAND)

    assert invalid.reason is InvalidReason.NO_HAND
    assert invalid != None  # noqa: E711 — justamente ese es el punto


def test_una_secuencia_vacia_no_se_construye() -> None:
    with pytest.raises(ValueError, match="al menos un frame"):
        Sequence(frames=())


def test_la_secuencia_reporta_forma_t_21_3() -> None:
    sequence = Sequence(frames=(_frame(), _frame(), _frame()))

    assert sequence.shape == (3, NUM_LANDMARKS, 3)
    assert len(sequence) == 3


def test_recortar_una_secuencia_devuelve_una_secuencia() -> None:
    sequence = Sequence(frames=(_frame(), _frame(), _frame()))

    window = sequence.window(1, 3)

    assert isinstance(window, Sequence)
    assert len(window) == 2


def test_el_vector_de_features_exige_42_componentes() -> None:
    FeatureVector(values=tuple(0.0 for _ in range(NUM_FEATURES)), spec_version=1)

    with pytest.raises(ValueError, match="42 componentes"):
        FeatureVector(values=(0.0, 1.0), spec_version=1)


def test_unknown_es_una_prediccion_de_primera_clase() -> None:
    prediction = Prediction.unknown(confidence=0.31)

    assert prediction.label == UNKNOWN_LABEL
    assert prediction.is_unknown
    assert not Prediction(label="A", confidence=0.9).is_unknown


def test_la_confianza_vive_en_cero_uno() -> None:
    with pytest.raises(ValueError, match="confianza"):
        Prediction(label="A", confidence=1.2)


def test_el_canal_de_trayectoria_exige_escala_positiva() -> None:
    TrajectoryChannel(points=((0.0, 0.0), (0.1, 0.2)), mean_scale=0.08)

    with pytest.raises(ValueError, match="escala media"):
        TrajectoryChannel(points=((0.0, 0.0),), mean_scale=0.0)


def test_una_muestra_sin_signer_id_no_se_construye() -> None:
    with pytest.raises(ValueError, match="leave-one-signer-out"):
        Sample(
            sequence=Sequence(frames=(_frame(),)),
            label="A",
            signer_id="",
            session_id="s1",
            timestamp=datetime.now(UTC),
            handedness=Handedness.RIGHT,
            lighting=Lighting.INDOOR,
            distance=Distance.MEDIUM,
        )


def test_una_muestra_exige_timestamp_con_zona_horaria() -> None:
    with pytest.raises(ValueError, match="zona horaria"):
        Sample(
            sequence=Sequence(frames=(_frame(),)),
            label="A",
            signer_id="signer_01",
            session_id="s1",
            timestamp=datetime(2026, 3, 10, 12, 0, 0),  # noqa: DTZ001
            handedness=Handedness.RIGHT,
            lighting=Lighting.INDOOR,
            distance=Distance.MEDIUM,
        )
