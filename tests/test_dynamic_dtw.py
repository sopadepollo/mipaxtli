"""El clasificador dinámico: DTW contra plantillas (`feature-spec.md` §3.4).

Dos bloques. Primero la distancia, con secuencias escritas a mano cuyo costo se
puede calcular de cabeza: es lo que `dtw.ts` tendrá que reproducir, y un error
aquí no se ve en ninguna matriz de confusión. Después el clasificador sobre el
corpus sintético dinámico: plantillas, puerta de distancia, bloqueo de LL/RR y
export.
"""

from __future__ import annotations

import json
import math

import pytest

from lsm.classifiers.base import Classifier, IncompatibleModelError
from lsm.classifiers.dynamic_dtw import (
    CLASSIFIER_NAME,
    DYNAMIC_ROW_WIDTH,
    DynamicDtwClassifier,
    Rows,
    dtw_distance,
    dynamic_rows,
    label_distances,
    local_distance,
    medoid,
)
from lsm.config import Config
from lsm.features import RESAMPLE_LENGTH
from lsm.preprocessing import preprocessing_record
from lsm.synthetic import (
    class_hand,
    still_sequence,
    synthetic_dynamic_samples,
    translated,
)
from lsm.types import Sample
from lsm.vocabulary import DIRECTION_PENDING_LABELS, Label

T = RESAMPLE_LENGTH


def filas(valores: list[float]) -> Rows:
    """Una secuencia de 24 filas de una sola componente relevante."""
    assert len(valores) == T
    return tuple((v,) + (0.0,) * (DYNAMIC_ROW_WIDTH - 1) for v in valores)


def rampa(desde: int) -> Rows:
    """Un escalón de 0 a 1 que empieza en el índice `desde`."""
    return filas([1.0 if i >= desde else 0.0 for i in range(T)])


# --------------------------------------------------------------------------- #
# La distancia
# --------------------------------------------------------------------------- #


def test_una_secuencia_esta_a_distancia_cero_de_si_misma() -> None:
    a = rampa(10)

    assert dtw_distance(a, a, band_radius=6) == 0.0


def test_la_distancia_local_es_euclidiana() -> None:
    a = (3.0, 0.0) + (0.0,) * (DYNAMIC_ROW_WIDTH - 2)
    b = (0.0, 4.0) + (0.0,) * (DYNAMIC_ROW_WIDTH - 2)

    assert local_distance(a, b) == 5.0


def test_el_costo_se_normaliza_por_la_longitud_del_camino() -> None:
    """Dos secuencias a distancia local constante 1: sea cual sea el camino, el
    costo por paso es 1. Sin normalizar saldría la longitud del camino."""
    ceros = filas([0.0] * T)
    unos = filas([1.0] * T)

    assert dtw_distance(ceros, unos, band_radius=6) == 1.0


def test_dtw_absorbe_un_retraso_que_cabe_en_la_banda() -> None:
    """El mismo escalón, tres frames más tarde. La alineación diagonal paga
    tres frames de desacuerdo; DTW los absorbe estirando el tiempo."""
    temprano = rampa(10)
    tarde = rampa(13)

    assert dtw_distance(temprano, tarde, band_radius=6) == 0.0
    assert dtw_distance(temprano, tarde, band_radius=0) == pytest.approx(3.0 / T)


def test_un_retraso_mayor_que_la_banda_no_se_absorbe_entero() -> None:
    """Es el riesgo del glosario para la Z: un trazo que ocupa toda la ventana
    puede necesitar desplazamientos más largos que `band_radius`, y lo que
    excede la banda se paga como distancia."""
    temprano = rampa(4)
    tarde = rampa(14)

    corta = dtw_distance(temprano, tarde, band_radius=6)
    ancha = dtw_distance(temprano, tarde, band_radius=12)

    assert ancha == 0.0
    assert corta > 0.0


def test_el_desempate_prefiere_la_diagonal() -> None:
    """Con todos los costos iguales a cero el camino elegido es la diagonal: 24
    celdas. Se comprueba a través de la normalización, que es donde el camino
    se nota: un costo constante c solo en la última celda da c / 24."""
    a = filas([0.0] * (T - 1) + [1.0])
    b = filas([0.0] * T)

    assert dtw_distance(a, b, band_radius=6) == pytest.approx(1.0 / T)


def test_secuencias_de_distinto_largo_se_rechazan() -> None:
    with pytest.raises(ValueError, match="distinta longitud"):
        dtw_distance(rampa(3), rampa(3)[:-1], band_radius=6)


def test_el_medoide_es_la_muestra_central() -> None:
    grupo = [rampa(9), rampa(10), rampa(11), rampa(20)]

    assert medoid(grupo, band_radius=0) == 1


def test_con_un_medoide_k_medoids_es_el_medoide_de_siempre() -> None:
    """ADR 0035: con `templates_per_signer = 1` el modelo no cambia."""
    from lsm.classifiers.dynamic_dtw import k_medoids

    grupo = [rampa(9), rampa(10), rampa(11), rampa(20)]

    assert k_medoids(grupo, 1, band_radius=0) == [medoid(grupo, band_radius=0)]


def test_el_segundo_medoide_cubre_lo_que_el_primero_deja_lejos() -> None:
    from lsm.classifiers.dynamic_dtw import k_medoids

    grupo = [rampa(9), rampa(10), rampa(11), rampa(20), rampa(21)]

    elegidos = k_medoids(grupo, 2, band_radius=0)

    assert elegidos[0] == medoid(grupo, band_radius=0)
    assert elegidos[1] in (3, 4)  # uno de los escalones tardíos
    assert k_medoids(grupo, 2, band_radius=0) == elegidos  # determinista
    assert k_medoids(grupo, 9, band_radius=0) == [0, 1, 2, 3, 4]


def test_las_plantillas_por_firmante_salen_de_la_config() -> None:
    muestras = list(MUESTRAS)
    por_grupo = len(muestras) // (len(ETIQUETAS) * 3)

    uno = DynamicDtwClassifier(
        config=Config.model_validate({"dtw": {"templates_per_signer": 1}})
    )
    uno.fit(muestras)
    dos = DynamicDtwClassifier(
        config=Config.model_validate({"dtw": {"templates_per_signer": 2}})
    )
    dos.fit(muestras)
    todas = DynamicDtwClassifier(
        config=Config.model_validate({"dtw": {"templates_per_signer": 0}})
    )
    todas.fit(muestras)

    assert len(uno.templates) == len(ETIQUETAS) * 3
    assert len(dos.templates) == len(ETIQUETAS) * 3 * min(2, por_grupo)
    assert len(todas.templates) == len(muestras)
    # El primero de cada grupo sigue siendo el medoide.
    assert {t.rows for t in uno.templates} <= {t.rows for t in dos.templates}


def test_la_distancia_a_una_letra_es_la_de_su_plantilla_mas_cercana() -> None:
    from lsm.classifiers.dynamic_dtw import Template

    plantillas = [
        Template(label="J", signer_id="a", rows=rampa(5)),
        Template(label="J", signer_id="b", rows=rampa(12)),
        Template(label="Z", signer_id="a", rows=filas([1.0] * T)),
    ]

    distancias = label_distances(rampa(12), plantillas, band_radius=0)

    assert distancias["J"] == 0.0
    assert distancias["Z"] == pytest.approx(12.0 / T)


# --------------------------------------------------------------------------- #
# El clasificador
# --------------------------------------------------------------------------- #

ETIQUETAS = ("J", "Z", "K")
CONFIG = Config()


def _por_firmante(muestras: tuple[Sample, ...], firmante: str) -> list[Sample]:
    return [m for m in muestras if m.signer_id == firmante]


MUESTRAS = synthetic_dynamic_samples(ETIQUETAS, signers=3, repetitions=3)


@pytest.fixture(scope="module")
def entrenado() -> DynamicDtwClassifier:
    """Entrenado con dos firmantes; el tercero queda para probar."""
    classifier = DynamicDtwClassifier(config=CONFIG)
    classifier.fit([m for m in MUESTRAS if m.signer_id != "sint02"])
    return classifier


def test_cumple_el_protocolo() -> None:
    assert isinstance(DynamicDtwClassifier(config=CONFIG), Classifier)


def test_las_plantillas_van_por_letra_y_persona(
    entrenado: DynamicDtwClassifier,
) -> None:
    """`templates_per_signer` por (letra, persona); el corpus sintético tiene 3
    repeticiones por grupo, así que con el valor de fábrica entran todas."""
    claves = [(t.label, t.signer_id) for t in entrenado.templates]
    k = min(CONFIG.dtw.templates_per_signer, 3)

    assert claves == sorted(
        (label, firmante)
        for label in ETIQUETAS
        for firmante in ("sint00", "sint01")
        for _ in range(k)
    )


def test_reconoce_a_una_persona_que_no_vio(entrenado: DynamicDtwClassifier) -> None:
    """Leave-one-signer-out en miniatura, sobre el corpus sintético."""
    for muestra in _por_firmante(MUESTRAS, "sint02"):
        prediccion = entrenado.predict(muestra.sequence)
        assert prediccion.label == muestra.label
        assert prediccion.confidence > 0.5


def test_la_puerta_de_distancia_rechaza_lo_que_queda_lejos_de_todo() -> None:
    """El mecanismo de `max_distance`: DTW siempre tiene una plantilla menos
    lejana, y la puerta es lo que convierte «menos lejana» en UNKNOWN.

    Con un umbral explícito y no con el de fábrica, a propósito: el de fábrica
    (6.0) es una puerta gruesa, y el ADR 0016 midió que la distancia **no**
    separa una mano quieta de un trazo real. A las estáticas las deja fuera la
    segmentación, que no las manda al camino dinámico, no esta puerta.
    """
    estricta = Config.model_validate({"dtw": {"max_distance": 0.5}})
    classifier = DynamicDtwClassifier(config=estricta)
    classifier.fit([m for m in MUESTRAS if m.signer_id != "sint02"])
    quieta = still_sequence(translated(class_hand(0), 600.0, 360.0), length=20)

    assert classifier.predict(quieta).is_unknown
    ranking_sin_puerta = DynamicDtwClassifier(
        config=Config.model_validate({"dtw": {"max_distance": 1000.0}})
    )
    ranking_sin_puerta.fit([m for m in MUESTRAS if m.signer_id != "sint02"])
    assert not ranking_sin_puerta.predict(quieta).is_unknown


def test_un_trazo_demasiado_corto_es_unknown(entrenado: DynamicDtwClassifier) -> None:
    """Menos frames que `dtw.min_source_frames`: el §3.2 no lo remuestrea."""
    corta = MUESTRAS[0].sequence.window(0, CONFIG.dtw.min_source_frames - 1)

    assert entrenado.predict(corta).is_unknown


def test_ll_y_rr_no_tienen_plantilla() -> None:
    """PENDIENTE-HUMANO en el glosario: su dirección canónica no está decidida,
    y el medoide de dos direcciones mezcladas sería el de una de ellas al azar."""
    assert {Label.DOBLE_L, Label.DOBLE_R} == DIRECTION_PENDING_LABELS
    muestras = synthetic_dynamic_samples(("J", "DOBLE_L", "DOBLE_R"), repetitions=2)
    classifier = DynamicDtwClassifier(config=CONFIG)

    classifier.fit(list(muestras))

    assert {t.label for t in classifier.templates} == {"J"}
    assert classifier.blocked == {"DOBLE_L": 6, "DOBLE_R": 6}


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def test_el_export_es_json_y_lleva_lo_que_el_navegador_necesita(
    entrenado: DynamicDtwClassifier,
) -> None:
    payload = json.loads(json.dumps(entrenado.export()))

    assert payload["classifier"] == CLASSIFIER_NAME
    assert payload["labels"] == sorted(ETIQUETAS)
    assert payload["params"] == {
        "band_radius": CONFIG.dtw.band_radius,
        "max_distance": CONFIG.dtw.max_distance,
        "trajectory_weight": CONFIG.features.trajectory_weight,
        "depth_weight": CONFIG.features.depth_weight,
        "min_source_frames": CONFIG.dtw.min_source_frames,
        "resample_length": RESAMPLE_LENGTH,
        "preprocessing": preprocessing_record(CONFIG),
    }
    plantilla = payload["data"]["templates"][0]
    assert len(plantilla["rows"]) == RESAMPLE_LENGTH
    assert len(plantilla["rows"][0]) == DYNAMIC_ROW_WIDTH
    assert payload["data"]["blocked_labels"] == ["DOBLE_L", "DOBLE_R"]


def test_recargar_el_export_da_las_mismas_predicciones(
    entrenado: DynamicDtwClassifier,
) -> None:
    recargado = DynamicDtwClassifier.from_export(entrenado.export())

    for muestra in _por_firmante(MUESTRAS, "sint02"):
        assert recargado.predict(muestra.sequence) == entrenado.predict(
            muestra.sequence
        )


def test_un_modelo_con_plantilla_de_ll_se_rechaza_al_cargar(
    entrenado: DynamicDtwClassifier,
) -> None:
    payload = entrenado.export()
    payload["data"]["templates"][0]["label"] = "DOBLE_L"

    with pytest.raises(ValueError, match="pendiente de decisión humana"):
        DynamicDtwClassifier.from_export(payload)


def test_un_modelo_de_otra_version_de_features_se_rechaza(
    entrenado: DynamicDtwClassifier,
) -> None:
    payload = entrenado.export()
    payload["feature_spec_version"] = 0

    with pytest.raises(IncompatibleModelError):
        DynamicDtwClassifier.from_export(payload)


def test_una_plantilla_mal_formada_se_rechaza(
    entrenado: DynamicDtwClassifier,
) -> None:
    payload = entrenado.export()
    payload["data"]["templates"][0]["rows"] = payload["data"]["templates"][0]["rows"][
        :-1
    ]

    with pytest.raises(ValueError, match="no es"):
        DynamicDtwClassifier.from_export(payload)


def test_el_peso_de_trayectoria_viaja_con_el_modelo() -> None:
    """Las plantillas ya llevan `w_τ` multiplicado: un navegador que construyera
    la entrada con otro peso compararía filas incomparables. Por eso el peso
    viaja en `params` y la recarga lo respeta."""
    config = Config.model_validate(
        {"features": {"trajectory_weight": 9.0, "depth_weight": 3.0}}
    )
    classifier = DynamicDtwClassifier(config=config)
    classifier.fit(list(MUESTRAS))

    recargado = DynamicDtwClassifier.from_export(classifier.export())

    assert recargado.config.features.trajectory_weight == 9.0
    assert recargado.config.features.depth_weight == 3.0
    rows = dynamic_rows(MUESTRAS[0].sequence, recargado.config)
    base = dynamic_rows(MUESTRAS[0].sequence, CONFIG)
    assert rows is not None
    assert base is not None
    # g_t = (f_t, w_τ·τ_t, w_δ·δ_t): la y de τ es la penúltima columna.
    assert math.isclose(
        rows[-1][-2], 9.0 / CONFIG.features.trajectory_weight * base[-1][-2]
    )
    assert math.isclose(rows[-1][-1], 3.0 / CONFIG.features.depth_weight * base[-1][-1])
