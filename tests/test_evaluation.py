"""El protocolo de evaluación de la Fase 2.

Lo que estos tests protegen es lo que hace que un número del reporte signifique
algo:

- **El alcance.** La Fase 2 evalúa las 21 letras estáticas más `NONE`. Meter las
  ocho dinámicas ensucia la matriz de confusión con clases que este clasificador
  no puede resolver por construcción, y una matriz ilegible no se mira.
- **El split.** Leave-one-signer-out, nunca aleatorio (`CLAUDE.md` §6). Un split
  aleatorio de frames mezcla frames de la misma grabación entre train y test y da
  métricas falsamente optimistas.
- **El contraste de hipótesis.** El glosario predijo qué letras se confundirían
  *antes* de tener datos. Comparar esas predicciones con la matriz real es lo que
  convierte la evaluación en una medición y no en un vistazo.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from lsm.classifiers.static_knn import Thresholds
from lsm.config import Config, Metric
from lsm.evaluation import (
    EXCLUDED_LABELS,
    PHASE2_LABELS,
    Axis,
    InsufficientFoldsError,
    Protocol,
    build_folds,
    hypothesis_contrast,
    most_confused,
    observe,
    precompute,
    score,
    sweep,
)
from lsm.synthetic import synthetic_samples
from lsm.types import NEGATIVE_LABEL, UNKNOWN_LABEL, SampleKind
from lsm.vocabulary import DYNAMIC_LABELS, STATIC_LABELS

CONFIG = Config()


def un_dataset(signers: int = 3, sessions: int = 2, repetitions: int = 2):  # type: ignore[no-untyped-def]
    return synthetic_samples(
        PHASE2_LABELS[:6], signers=signers, sessions=sessions, repetitions=repetitions
    )


# --------------------------------------------------------------------------- #
# Alcance
# --------------------------------------------------------------------------- #


def test_el_alcance_son_las_estaticas_mas_la_clase_negativa() -> None:
    assert set(PHASE2_LABELS) == {label.value for label in STATIC_LABELS} | {"NONE"}
    assert len(PHASE2_LABELS) == 22


def test_las_ocho_dinamicas_quedan_explicitamente_excluidas() -> None:
    """Hasta la Fase 5. `static_knn` promedia la secuencia, y el promedio de una
    Z es una mancha sin forma que aterrizará sobre alguna estática."""
    assert {label.value for label in DYNAMIC_LABELS} == EXCLUDED_LABELS
    assert len(EXCLUDED_LABELS) == 8
    assert not set(PHASE2_LABELS) & EXCLUDED_LABELS


def test_observe_descarta_las_dinamicas_y_dice_cuantas_descarto() -> None:
    muestras = synthetic_samples(("A", "J", "Z"), signers=2, sessions=1, repetitions=2)

    dataset = observe(muestras, CONFIG)

    assert {o.label for o in dataset.observations} == {"A"}
    assert dataset.excluded == {"J": 4, "Z": 4}


def test_observe_descarta_la_clase_negativa_grabada_en_movimiento() -> None:
    """`NONE` está en el alcance, pero sus muestras dinámicas no.

    Sale por su `kind`, no por su etiqueta, y esa es toda la diferencia: la clase
    negativa se graba de las dos formas a propósito —mano en reposo, y también
    transiciones y saludos—, pero `static_knn` le da un único centroide y lo
    construye promediando los frames. El promedio de un saludo no es ninguna
    configuración de mano. Ver
    `docs/adr/0010-la-clase-negativa-y-el-modo-de-grabacion.md`.
    """
    estaticas = synthetic_samples(
        ("A", NEGATIVE_LABEL), signers=2, sessions=1, repetitions=2
    )
    dinamicas = tuple(
        replace(muestra, kind=SampleKind.DYNAMIC)
        for muestra in estaticas
        if muestra.label == NEGATIVE_LABEL
    )

    dataset = observe(estaticas + dinamicas, CONFIG)

    assert dataset.dynamic_negatives == len(dinamicas) == 4
    assert sum(1 for o in dataset.observations if o.label == NEGATIVE_LABEL) == 4
    assert dataset.excluded == {}, "no salen por su etiqueta, que sí está en el alcance"


def test_observe_no_descarta_en_silencio_una_etiqueta_desconocida() -> None:
    """Una etiqueta que no es ni estática, ni dinámica, ni NONE es un dataset
    corrupto o un glosario que cambió, y las dos cosas hay que verlas."""
    muestras = synthetic_samples(("A", "PLATANO"), signers=2, sessions=1, repetitions=1)

    with pytest.raises(ValueError, match="PLATANO"):
        observe(muestras, CONFIG)


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #


def test_leave_one_signer_out_deja_un_firmante_entero_fuera() -> None:
    dataset = observe(un_dataset(signers=3), CONFIG)

    folds = build_folds(dataset, Protocol.SIGNER)

    assert len(folds) == 3
    for fold in folds:
        firmantes_train = {dataset.observations[i].signer_id for i in fold.train}
        firmantes_test = {dataset.observations[i].signer_id for i in fold.test}
        assert firmantes_test == {fold.held_out}
        assert fold.held_out not in firmantes_train


def test_ningun_fold_comparte_una_sola_muestra_entre_train_y_test() -> None:
    """Es la propiedad que un split aleatorio de frames rompería."""
    dataset = observe(un_dataset(signers=3), CONFIG)

    for fold in build_folds(dataset, Protocol.SIGNER):
        assert not set(fold.train) & set(fold.test)


def test_con_un_solo_firmante_leave_one_signer_out_se_niega() -> None:
    """No se degrada solo: una métrica por sesión y una por persona no son
    comparables, y alguien acabaría citando la primera como si fuera la segunda."""
    dataset = observe(un_dataset(signers=1, sessions=2), CONFIG)

    with pytest.raises(InsufficientFoldsError, match="firmante"):
        build_folds(dataset, Protocol.SIGNER)


def test_con_un_solo_firmante_el_protocolo_por_sesion_si_corre() -> None:
    dataset = observe(un_dataset(signers=1, sessions=2), CONFIG)

    folds = build_folds(dataset, Protocol.SESSION)

    assert len(folds) == 2


def test_los_folds_son_deterministas() -> None:
    dataset = observe(un_dataset(), CONFIG)

    primera = build_folds(dataset, Protocol.SIGNER)
    segunda = build_folds(dataset, Protocol.SIGNER)

    assert primera == segunda


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #


def test_la_matriz_de_confusion_suma_exactamente_las_muestras_evaluadas() -> None:
    dataset = observe(un_dataset(), CONFIG)
    folds = build_folds(dataset, Protocol.SIGNER)
    distancias = precompute(dataset, folds, Metric.EUCLIDEAN)

    reporte = score(distancias, Thresholds.from_config(CONFIG))

    assert sum(reporte.confusion.values()) == reporte.total
    assert reporte.total == len(dataset.observations)


def test_el_accuracy_por_letra_cubre_todas_las_clases_presentes() -> None:
    dataset = observe(un_dataset(), CONFIG)
    folds = build_folds(dataset, Protocol.SIGNER)
    reporte = score(
        precompute(dataset, folds, Metric.EUCLIDEAN), Thresholds.from_config(CONFIG)
    )

    assert set(reporte.per_label) == set(PHASE2_LABELS[:6])
    for score_de_letra in reporte.per_label.values():
        assert 0.0 <= score_de_letra.accuracy <= 1.0


def test_un_umbral_de_distancia_imposible_manda_todo_a_unknown() -> None:
    dataset = observe(un_dataset(), CONFIG)
    folds = build_folds(dataset, Protocol.SIGNER)
    distancias = precompute(dataset, folds, Metric.EUCLIDEAN)

    reporte = score(
        distancias,
        Thresholds(
            metric=Metric.EUCLIDEAN,
            max_distance=1e-9,
            min_margin=0.5,
            max_dispersion=10.0,
        ),
    )

    assert reporte.accuracy == 0.0
    assert reporte.unknown == reporte.total


def test_los_pares_mas_confundidos_se_reportan_ordenados_y_sin_unknown() -> None:
    """`UNKNOWN` no es una confusión entre dos letras: es un rechazo, y mezclarlo
    con los pares taparía las confusiones reales."""
    confusion = {
        ("A", "B"): 5,
        ("B", "A"): 3,
        ("C", "D"): 2,
        ("A", UNKNOWN_LABEL): 40,
        ("A", "A"): 100,
    }

    pares = most_confused(confusion, limit=5)

    assert pares == ((("A", "B"), 8), (("C", "D"), 2))


# --------------------------------------------------------------------------- #
# Contraste de hipótesis
# --------------------------------------------------------------------------- #


def test_el_contraste_separa_confirmados_refutados_e_imprevistos() -> None:
    predichos = frozenset({("A", "E"), ("M", "N"), ("S", "T")})
    confusion = {
        ("A", "E"): 4,  # predicho y ocurrió
        ("M", "N"): 0,  # predicho y no ocurrió
        ("V", "W"): 3,  # ocurrió sin estar predicho
    }

    contraste = hypothesis_contrast(
        predichos, confusion, evaluated=frozenset({"A", "E", "M", "N", "V", "W"})
    )

    assert contraste.confirmed == (("A", "E"),)
    assert contraste.refuted == (("M", "N"),)
    assert contraste.unforeseen == (("V", "W"),)


def test_un_par_predicho_con_una_letra_dinamica_no_cuenta_como_refutado() -> None:
    """Trece de los veintiséis pares del glosario tocan una dinámica. Contarlos
    como refutados diría que el glosario se equivocó cuando lo que pasa es que la
    Fase 2 no los evaluó."""
    predichos = frozenset({("I", "J"), ("A", "E")})

    contraste = hypothesis_contrast(
        predichos, {("A", "E"): 2}, evaluated=frozenset({"A", "E", "I"})
    )

    assert contraste.not_evaluable == (("I", "J"),)
    assert contraste.refuted == ()


def test_el_contraste_es_simetrico_en_el_par() -> None:
    """El glosario dice que A se confunde con E; la matriz puede registrar E→A."""
    contraste = hypothesis_contrast(
        frozenset({("A", "E")}), {("E", "A"): 3}, evaluated=frozenset({"A", "E"})
    )

    assert contraste.confirmed == (("A", "E"),)


# --------------------------------------------------------------------------- #
# Barrido
# --------------------------------------------------------------------------- #


def test_el_barrido_recorre_la_rejilla_entera_y_elige_el_mejor_punto() -> None:
    muestras = un_dataset()
    ejes = (
        Axis(path="static_knn.max_distance", values=(1e-9, 1.0)),
        Axis(path="static_knn.min_margin", values=(0.5, 0.99)),
    )

    resultado = sweep(muestras, CONFIG, ejes, Protocol.SIGNER)

    assert len(resultado.points) == 4
    assert resultado.best.accuracy >= max(p.accuracy for p in resultado.points) - 1e-12
    assert resultado.best.override("static_knn.max_distance") == 1.0


def test_el_barrido_marca_como_inertes_los_ejes_que_no_mueven_la_metrica() -> None:
    """`features.trajectory_weight` solo pondera el canal de trayectoria, que el
    camino estático no usa. El reporte tiene que decirlo en vez de mostrar una
    columna constante sin explicación."""
    ejes = (
        Axis(path="features.trajectory_weight", values=(0.0, 4.0, 20.0)),
        Axis(path="static_knn.max_distance", values=(1e-9, 1.0)),
    )

    resultado = sweep(un_dataset(), CONFIG, ejes, Protocol.SIGNER)

    assert "features.trajectory_weight" in resultado.inert
    assert "static_knn.max_distance" in resultado.effective


def test_el_barrido_es_reproducible() -> None:
    """El criterio de aceptación de la fase: `make eval` da lo mismo dos veces."""
    muestras = un_dataset()
    ejes = (Axis(path="static_knn.max_distance", values=(0.5, 1.0)),)

    primero = sweep(muestras, CONFIG, ejes, Protocol.SIGNER)
    segundo = sweep(muestras, CONFIG, ejes, Protocol.SIGNER)

    assert primero == segundo


def test_un_eje_que_no_existe_en_la_configuracion_se_rechaza_al_empezar() -> None:
    """Y no a las dos horas de barrido."""
    ejes = (Axis(path="static_knn.max_distancia", values=(1.0,)),)

    with pytest.raises(ValueError, match="max_distancia"):
        sweep(un_dataset(), CONFIG, ejes, Protocol.SIGNER)
