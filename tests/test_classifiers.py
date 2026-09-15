"""El contrato común de los clasificadores.

En Fase 0 no hay ni `static_knn` ni `dynamic_dtw`: hay una interfaz y un doble de
pruebas. Lo que sí tiene que estar cerrado desde ahora es el formato de export,
porque de él depende que el modelo entrenado en Python corra igual en el
navegador, y que un modelo viejo se **rechace** en vez de dar predicciones
silenciosamente malas.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from lsm.classifiers.base import (
    SCHEMA_VERSION,
    Classifier,
    IncompatibleModelError,
    check_export_compatibility,
)
from lsm.classifiers.dummy import DummyClassifier
from lsm.features import FEATURE_SPEC_VERSION
from lsm.synthetic import canonical_hand, still_sequence, translated
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


def a_sequence() -> Sequence:
    return still_sequence(translated(canonical_hand(), 640.0, 400.0), length=5)


def a_sample(label: str) -> Sample:
    return Sample(
        sequence=a_sequence(),
        label=label,
        signer_id="signer_01",
        session_id="session_01",
        timestamp=datetime(2026, 3, 10, 12, 0, tzinfo=UTC),
        handedness=Handedness.RIGHT,
        light_level=LightLevel.INDOOR,
        light_direction=LightDirection.FRONTAL,
        distance=Distance.MEDIUM,
        mean_luminance=0.42,
        mean_scale_px=100.2,
        kind=SampleKind.STATIC,
    )


def test_el_doble_de_pruebas_satisface_el_protocolo() -> None:
    assert isinstance(DummyClassifier(), Classifier)


def test_predict_recibe_una_secuencia_y_devuelve_confianza() -> None:
    classifier = DummyClassifier(responses=(Prediction(label="A", confidence=0.91),))

    prediction = classifier.predict(a_sequence())

    assert prediction.label == "A"
    assert prediction.confidence == 0.91


def test_predict_puede_devolver_unknown() -> None:
    """Un clasificador de 27 clases siempre devuelve una de las 27. UNKNOWN es lo
    que impide que una mano rascándose la nariz escriba una letra."""
    classifier = DummyClassifier()

    assert classifier.predict(a_sequence()).is_unknown


def test_las_respuestas_se_consumen_en_orden_y_la_ultima_se_repite() -> None:
    classifier = DummyClassifier(
        responses=(
            Prediction(label="A", confidence=0.9),
            Prediction(label="B", confidence=0.8),
        )
    )
    sequence = a_sequence()

    labels = [classifier.predict(sequence).label for _ in range(4)]

    assert labels == ["A", "B", "B", "B"]


def test_fit_recoge_las_etiquetas_del_dataset() -> None:
    classifier = DummyClassifier()

    classifier.fit([a_sample("B"), a_sample("A"), a_sample("B")])

    assert classifier.export()["labels"] == ["A", "B"]


def test_el_export_lleva_los_siete_campos_del_contrato() -> None:
    classifier = DummyClassifier()
    classifier.fit([a_sample("A")])

    payload = classifier.export()

    assert set(payload) == {
        "schema_version",
        "feature_spec_version",
        "handedness_convention",
        "classifier",
        "labels",
        "params",
        "data",
    }
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["feature_spec_version"] == FEATURE_SPEC_VERSION
    assert payload["classifier"] == "dummy"


def test_el_export_es_serializable_a_json_sin_ayuda() -> None:
    """Lo va a leer JavaScript: nada de tuplas, enums ni numpy."""
    classifier = DummyClassifier()
    classifier.fit([a_sample("A"), a_sample("E")])

    text = json.dumps(classifier.export())

    assert json.loads(text)["labels"] == ["A", "E"]


def test_un_modelo_de_otra_version_del_spec_se_rechaza_al_cargarse() -> None:
    """CLAUDE.md §4: si cambia la normalización, los modelos viejos se rechazan.
    Ejecutarlos daría predicciones malas sin ningún síntoma visible."""
    payload = DummyClassifier().export()
    payload["feature_spec_version"] = FEATURE_SPEC_VERSION + 1

    with pytest.raises(IncompatibleModelError, match="feature_spec_version"):
        check_export_compatibility(payload)


def test_un_modelo_con_otro_schema_version_se_rechaza() -> None:
    payload = DummyClassifier().export()
    payload["schema_version"] = SCHEMA_VERSION + 1

    with pytest.raises(IncompatibleModelError, match="schema_version"):
        check_export_compatibility(payload)


def test_un_export_al_que_le_falta_un_campo_se_rechaza() -> None:
    payload = DummyClassifier().export()
    del payload["labels"]

    with pytest.raises(IncompatibleModelError, match="labels"):
        check_export_compatibility(payload)


def test_un_export_valido_pasa_la_verificacion() -> None:
    check_export_compatibility(DummyClassifier().export())
