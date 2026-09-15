"""El clasificador estático de la Fase 2 (`ARQUITECTURA.md` §4.3).

Es un vecino más cercano por **centroides**: promedia los frames de la ventana con
la agregación del `feature-spec.md` §2 y compara contra un vector por clase. Se
eligió por lo mismo que el DTW en la Fase 5 —dataset pequeño, modelo que tiene que
correr en un navegador móvil, decisiones que se puedan depurar mirando números— y
no porque sea lo más preciso posible.

Lo que estos tests fijan, además de que acierte, son las **tres puertas de
rechazo**. Un clasificador de 22 clases siempre devuelve una de las 22, y sin ellas
escribe una letra cada vez que alguien se rasca la nariz (`ARQUITECTURA.md` §4.4).
"""

from __future__ import annotations

import json

import pytest

from lsm.classifiers.base import Classifier, check_export_compatibility
from lsm.classifiers.static_knn import CLASSIFIER_NAME, StaticKnnClassifier
from lsm.config import Config
from lsm.synthetic import class_hand, still_sequence, synthetic_samples, translated
from lsm.types import NUM_FEATURES, UNKNOWN_LABEL, Sequence

ETIQUETAS = ("A", "B", "C", "NONE")


def un_dataset() -> list[object]:
    return list(synthetic_samples(ETIQUETAS, signers=3, sessions=2, repetitions=3))


def una_secuencia_de(ordinal: int) -> Sequence:
    return still_sequence(translated(class_hand(ordinal), 640.0, 400.0), length=8)


def entrenado(config: Config | None = None) -> StaticKnnClassifier:
    classifier = StaticKnnClassifier(config=config or Config())
    classifier.fit(un_dataset())  # type: ignore[arg-type]
    return classifier


def test_satisface_el_protocol_de_clasificador() -> None:
    assert isinstance(StaticKnnClassifier(config=Config()), Classifier)


def test_reconoce_la_clase_con_la_que_se_entreno() -> None:
    classifier = entrenado()

    prediccion = classifier.predict(una_secuencia_de(0))

    assert prediccion.label == "A"
    assert prediccion.confidence > 0.0


def test_una_mano_lejos_de_todo_centroide_se_rechaza_por_distancia() -> None:
    """`max_distance` es la puerta que tapa lo que no se parece a ninguna letra.

    El umbral que hace fallar la puerta **depende de la métrica**, y por eso no es
    un número redondo: con la coseno del `config.yaml` calibrado la distancia de
    esta muestra a su centroide es 3.6e-4, tres órdenes de magnitud menor que con
    la euclidiana (0.16). El 0.001 que servía antes ya no rechaza nada. Ver
    `docs/adr/0011-calibracion-de-la-fase-2.md`.
    """
    estricto = Config.model_validate({"static_knn": {"max_distance": 0.0001}})
    classifier = entrenado(estricto)

    assert classifier.predict(una_secuencia_de(0)).label == UNKNOWN_LABEL


def test_un_empate_entre_dos_clases_se_rechaza_por_margen() -> None:
    """`min_margin` es la puerta que tapa las confundibles del §4.8: dos centroides
    igual de cerca no son una letra, son una duda."""
    exigente = Config.model_validate({"static_knn": {"min_margin": 0.99}})
    classifier = entrenado(exigente)

    assert classifier.predict(una_secuencia_de(0)).label == UNKNOWN_LABEL


def test_una_ventana_inestable_se_rechaza_sin_mirar_los_centroides() -> None:
    """σ por encima de `quality.max_dispersion`: la mano se movía o el detector
    saltó, y clasificar eso produce basura (`feature-spec.md` §2).

    La ventana tiene que venir del generador con jitter y no de `still_sequence`:
    una secuencia de frames idénticos tiene σ = 0 exactamente y no hay umbral
    positivo que la rechace.
    """
    temblorosa = synthetic_samples(("A",), signers=1, sessions=1, repetitions=1)[
        0
    ].sequence
    intolerante = Config.model_validate({"quality": {"max_dispersion": 1e-9}})
    classifier = entrenado(intolerante)

    assert classifier.predict(temblorosa).label == UNKNOWN_LABEL


def test_la_confianza_es_el_margen_y_vive_en_el_intervalo_unitario() -> None:
    classifier = entrenado()

    for ordinal in range(len(ETIQUETAS)):
        confianza = classifier.predict(una_secuencia_de(ordinal)).confidence
        assert 0.0 <= confianza <= 1.0


def test_la_metrica_coseno_tambien_clasifica() -> None:
    coseno = Config.model_validate({"static_knn": {"metric": "cosine"}})
    classifier = entrenado(coseno)

    assert classifier.predict(una_secuencia_de(1)).label == "B"


def test_el_export_lleva_los_siete_campos_y_pasa_la_verificacion() -> None:
    payload = entrenado().export()

    assert set(payload) == {
        "schema_version",
        "feature_spec_version",
        "handedness_convention",
        "classifier",
        "labels",
        "params",
        "data",
    }
    assert payload["classifier"] == CLASSIFIER_NAME
    check_export_compatibility(payload)


def test_el_export_es_json_puro_y_trae_un_centroide_de_42_componentes() -> None:
    """Lo va a leer JavaScript sin dependencias: nada de tuplas ni de numpy."""
    payload = json.loads(json.dumps(entrenado().export()))

    centroides = payload["data"]["centroids"]
    assert sorted(centroides) == sorted(ETIQUETAS)
    for vector in centroides.values():
        assert len(vector) == NUM_FEATURES
        assert all(isinstance(componente, float) for componente in vector)


def test_el_export_lleva_los_umbrales_que_la_reimplementacion_necesita() -> None:
    """Sin ellos, el JavaScript no puede reproducir la decisión: acertaría en la
    etiqueta y no sabría cuándo callarse."""
    params = entrenado().export()["params"]

    assert params["metric"] == Config().static_knn.metric.value
    assert params["max_distance"] == Config().static_knn.max_distance
    assert params["min_margin"] == Config().static_knn.min_margin
    assert params["max_dispersion"] == Config().quality.max_dispersion


def test_un_modelo_recargado_desde_su_json_predice_lo_mismo() -> None:
    """Es la prueba de que el export está completo. Si al recargarlo hiciera falta
    algo que no viaja en el archivo, la app web tampoco lo tendría."""
    original = entrenado()
    recargado = StaticKnnClassifier.from_export(original.export())

    for ordinal in range(len(ETIQUETAS)):
        sequence = una_secuencia_de(ordinal)
        assert original.predict(sequence) == recargado.predict(sequence)


def test_recargar_un_modelo_de_otra_version_del_spec_falla() -> None:
    from lsm.classifiers.base import IncompatibleModelError
    from lsm.features import FEATURE_SPEC_VERSION

    payload = entrenado().export()
    payload["feature_spec_version"] = FEATURE_SPEC_VERSION + 1

    with pytest.raises(IncompatibleModelError):
        StaticKnnClassifier.from_export(payload)


def test_predecir_sin_entrenar_devuelve_unknown() -> None:
    """No lanza: la máquina de estados llama a esto treinta veces por segundo y un
    modelo vacío es una configuración posible, no un error de programación."""
    assert StaticKnnClassifier(config=Config()).predict(una_secuencia_de(0)).is_unknown
