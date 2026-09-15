"""El dataset sintético: manos distintas y deterministas, sin cámara.

Existe porque la Fase 2 tiene que ser ejecutable antes de que haya grabaciones.
Lo que estos tests protegen no es la verosimilitud —no la tiene, y el reporte lo
grita— sino las dos propiedades de las que depende que sirva de algo:
**determinismo bit a bit** y **clases que de verdad se separan**. Un generador
que devolviera manos parecidas daría una matriz de confusión ilegible y nadie
sabría si el fallo está en el clasificador o en los datos.
"""

from __future__ import annotations

from lsm.config import Config
from lsm.features import SequenceFeatures, extract_sequence_features
from lsm.synthetic import class_hand, synthetic_samples
from lsm.types import NUM_FEATURES


def shape_of(sample_sequence: object) -> tuple[float, ...]:
    outcome = extract_sequence_features(sample_sequence, Config())  # type: ignore[arg-type]
    assert isinstance(outcome, SequenceFeatures)
    return outcome.static.shape.values


def test_dos_generaciones_dan_exactamente_los_mismos_bits() -> None:
    """`make eval` tiene que ser reproducible; un generador con `random` global
    no lo sería."""
    primera = synthetic_samples(("A", "B"), signers=2, sessions=1, repetitions=2)
    segunda = synthetic_samples(("A", "B"), signers=2, sessions=1, repetitions=2)

    assert [shape_of(s.sequence) for s in primera] == [
        shape_of(s.sequence) for s in segunda
    ]


def test_cada_clase_tiene_una_configuracion_de_mano_distinta() -> None:
    manos = [class_hand(ordinal) for ordinal in range(22)]

    assert len(set(manos)) == 22


def test_el_generador_respeta_firmantes_sesiones_y_repeticiones() -> None:
    muestras = synthetic_samples(("A", "B", "C"), signers=3, sessions=2, repetitions=4)

    assert len(muestras) == 3 * 3 * 2 * 4
    assert len({s.signer_id for s in muestras}) == 3
    assert len({s.session_id for s in muestras}) == 3 * 2
    assert {s.label for s in muestras} == {"A", "B", "C"}


def test_las_muestras_no_son_frames_congelados() -> None:
    """Una muestra sin jitter daría σ = 0 y `max_dispersion` no podría calibrarse
    contra nada."""
    muestra = synthetic_samples(("A",), signers=1, sessions=1, repetitions=1)[0]
    outcome = extract_sequence_features(muestra.sequence, Config())
    assert isinstance(outcome, SequenceFeatures)

    assert outcome.static.dispersion > 0.0
    assert len(outcome.static.shape.values) == NUM_FEATURES


def test_dos_firmantes_ejecutan_la_misma_letra_de_forma_distinta() -> None:
    """Sin variación por persona, leave-one-signer-out no mediría nada: el fold
    de prueba sería una copia del de entrenamiento."""
    muestras = synthetic_samples(("A",), signers=2, sessions=1, repetitions=1)

    assert shape_of(muestras[0].sequence) != shape_of(muestras[1].sequence)
