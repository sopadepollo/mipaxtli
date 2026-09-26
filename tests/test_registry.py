"""El registry obedece la ruta que decidió la segmentación (`ARQUITECTURA.md` §4.3).

Lo que se prueba es la ausencia de un segundo criterio: el registry no mira la
ventana para decidir a quién se la da, solo el `WindowOrigin`. Se comprueba con
dobles que anotan si se les llamó.
"""

from __future__ import annotations

import pytest

from lsm.classifiers.dummy import DummyClassifier
from lsm.classifiers.dynamic_dtw import DynamicDtwClassifier
from lsm.classifiers.registry import ClassifierRegistry
from lsm.classifiers.static_knn import StaticKnnClassifier
from lsm.config import Config
from lsm.synthetic import canonical_hand, still_sequence
from lsm.types import Prediction, WindowOrigin

VENTANA = still_sequence(canonical_hand(), length=6)
L_SEGURA = Prediction(label="L", confidence=0.95)
LL_MENOS_SEGURA = Prediction(label="DOBLE_L", confidence=0.80)


def test_una_ventana_estable_va_al_estatico_y_solo_a_el() -> None:
    estatico = DummyClassifier(responses=(L_SEGURA,))
    dinamico = DummyClassifier(responses=(LL_MENOS_SEGURA,))
    registry = ClassifierRegistry(static=estatico, dynamic=dinamico)

    assert registry(VENTANA, WindowOrigin.STABLE) == L_SEGURA
    assert dinamico._calls == 0


def test_un_trazo_va_al_dinamico_aunque_el_estatico_estaria_mas_seguro() -> None:
    """El caso L/LL del glosario: la mano de la LL es la de la L, así que el
    estático diría L con más confianza. Si el registry eligiera por confianza,
    la LL no saldría nunca. Elige por origen."""
    estatico = DummyClassifier(responses=(L_SEGURA,))
    dinamico = DummyClassifier(responses=(LL_MENOS_SEGURA,))
    registry = ClassifierRegistry(static=estatico, dynamic=dinamico)

    assert registry(VENTANA, WindowOrigin.DYNAMIC) == LL_MENOS_SEGURA
    assert estatico._calls == 0


def test_sin_clasificador_para_esa_ruta_sale_unknown() -> None:
    """La demo sin modelo dinámico: los trazos se rechazan en vez de caer al
    estático."""
    estatico = DummyClassifier(responses=(L_SEGURA,))
    registry = ClassifierRegistry(static=estatico)

    assert registry(VENTANA, WindowOrigin.DYNAMIC).is_unknown
    assert estatico._calls == 0


def test_con_las_dos_rutas_gana_la_mayor_confianza() -> None:
    """La red del caso que la máquina de estados no produce."""
    estatico = DummyClassifier(responses=(Prediction(label="I", confidence=0.7),))
    dinamico = DummyClassifier(responses=(Prediction(label="J", confidence=0.9),))
    registry = ClassifierRegistry(static=estatico, dynamic=dinamico)

    ambas = (WindowOrigin.STABLE, WindowOrigin.DYNAMIC)

    assert registry.classify_routes(VENTANA, ambas).label == "J"


def test_con_las_dos_rutas_un_unknown_no_compite() -> None:
    estatico = DummyClassifier(responses=(Prediction(label="I", confidence=0.7),))
    dinamico = DummyClassifier(responses=(Prediction.unknown(confidence=0.99),))
    registry = ClassifierRegistry(static=estatico, dynamic=dinamico)

    resultado = registry.classify_routes(
        VENTANA, (WindowOrigin.DYNAMIC, WindowOrigin.STABLE)
    )

    assert resultado.label == "I"


def test_con_las_dos_rutas_el_empate_lo_gana_el_estatico() -> None:
    igual = Prediction(label="I", confidence=0.8)
    otra = Prediction(label="J", confidence=0.8)
    registry = ClassifierRegistry(
        static=DummyClassifier(responses=(igual,)),
        dynamic=DummyClassifier(responses=(otra,)),
    )

    resultado = registry.classify_routes(
        VENTANA, (WindowOrigin.DYNAMIC, WindowOrigin.STABLE)
    )

    assert resultado == igual


def test_sin_rutas_es_un_error_de_programacion() -> None:
    with pytest.raises(ValueError, match="al menos un origen"):
        ClassifierRegistry().classify_routes(VENTANA, ())


def test_un_modelo_en_la_ranura_equivocada_se_rechaza_al_cargar() -> None:
    config = Config()
    estatico = StaticKnnClassifier(config=config).export()
    dinamico = DynamicDtwClassifier(config=config).export()

    with pytest.raises(ValueError, match="ranura STABLE"):
        ClassifierRegistry.from_exports(static=dinamico)
    with pytest.raises(ValueError, match="ranura DYNAMIC"):
        ClassifierRegistry.from_exports(static=estatico, dynamic=estatico)


def test_desde_los_exports_cada_modelo_queda_en_su_ranura() -> None:
    config = Config()
    registry = ClassifierRegistry.from_exports(
        static=StaticKnnClassifier(config=config).export(),
        dynamic=DynamicDtwClassifier(config=config).export(),
    )

    assert isinstance(registry.static, StaticKnnClassifier)
    assert isinstance(registry.dynamic, DynamicDtwClassifier)
