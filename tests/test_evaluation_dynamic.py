"""La evaluación de la Fase 5: alcance, folds, puerta y reproducción.

Sobre el corpus sintético dinámico, que no se parece a LSM pero es determinista:
lo que se prueba es que la tubería de evaluación mide lo que dice medir.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from lsm.classifiers.dynamic_dtw import DynamicDtwClassifier
from lsm.classifiers.registry import ClassifierRegistry
from lsm.cli.evaluate_dynamic import main
from lsm.config import Config
from lsm.evaluation import (
    PHASE5_LABELS,
    Protocol,
    nearest_distances,
    observe_dynamic,
    precompute_dynamic,
    replay_sample,
    score_dynamic,
    sweep_dynamic,
    without_dynamic_path,
)
from lsm.segmentation import FrameThresholds
from lsm.synthetic import synthetic_dynamic_samples, synthetic_samples
from lsm.types import WindowOrigin

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = Config()
MUESTRAS = synthetic_dynamic_samples(("J", "Z", "DOBLE_L"), repetitions=2)


def test_el_alcance_de_la_fase_5_deja_fuera_ll_y_rr() -> None:
    assert "DOBLE_L" not in PHASE5_LABELS
    assert "DOBLE_R" not in PHASE5_LABELS
    assert set(PHASE5_LABELS) == {"ENIE", "J", "K", "Q", "X", "Z"}


def test_las_bloqueadas_se_cuentan_aparte_de_las_fuera_de_alcance() -> None:
    estaticas = synthetic_samples(("A",), signers=1, sessions=1, repetitions=2)

    dataset = observe_dynamic([*MUESTRAS, *estaticas], CONFIG)

    assert dataset.labels == ("J", "Z")
    assert dataset.blocked == {"DOBLE_L": 6}
    assert dataset.excluded == {"A": 2}


def test_leave_one_signer_out_no_usa_plantillas_del_firmante_de_prueba() -> None:
    """Si el fold usara plantillas de la persona que deja fuera, cada muestra
    tendría su propio medoide a distancia casi cero y el accuracy mentiría."""
    dataset = observe_dynamic(MUESTRAS, CONFIG)

    folds = precompute_dynamic(dataset, Protocol.SIGNER, CONFIG.dtw.band_radius)

    assert [f.held_out for f in folds] == ["sint00", "sint01", "sint02"]
    for fold in folds:
        for muestra in fold.samples:
            assert muestra.ranking is not None
            assert muestra.ranking.nearest > 0.0


def test_la_puerta_abierta_da_el_accuracy_del_vecino_mas_cercano() -> None:
    dataset = observe_dynamic(MUESTRAS, CONFIG)
    folds = precompute_dynamic(dataset, Protocol.SIGNER, CONFIG.dtw.band_radius)

    abierta = score_dynamic(folds, math.inf)
    cerrada = score_dynamic(folds, 0.0)

    assert abierta.unknown == 0
    assert cerrada.unknown == cerrada.total
    assert len(nearest_distances(folds, correct=True)) == abierta.correct


def test_el_barrido_recorre_el_producto_y_reextrae_por_peso() -> None:
    puntos = sweep_dynamic(MUESTRAS, CONFIG, (0.0, 4.0), (2, 6), Protocol.SIGNER)

    assert [(p.trajectory_weight, p.band_radius) for p in puntos] == [
        (0.0, 2),
        (0.0, 6),
        (4.0, 2),
        (4.0, 6),
    ]
    # Con w_τ = 0 el canal de trayectoria no cuenta: las distancias cambian.
    assert puntos[0].correct_nearest != puntos[2].correct_nearest


def test_una_dinamica_reproducida_sale_entera_y_con_su_letra() -> None:
    """La reproducción por la máquina de estados, con umbrales de fábrica."""
    entrenamiento = [m for m in MUESTRAS if m.signer_id != "sint02"]
    dinamico = DynamicDtwClassifier(config=CONFIG)
    dinamico.fit(entrenamiento)
    registry = ClassifierRegistry(dynamic=dinamico)
    reposo = FrameThresholds.from_config(CONFIG, 30.0).motion_confirm_low_frames + 5
    prueba = next(m for m in MUESTRAS if m.signer_id == "sint02" and m.label == "Z")

    salida = replay_sample(prueba, CONFIG, registry, rest_after=reposo)

    assert salida.dynamic_windows == 1
    assert salida.emitted == (("Z", WindowOrigin.DYNAMIC),)


def test_sin_camino_dinamico_ningun_trazo_llega_al_clasificador() -> None:
    reposo = FrameThresholds.from_config(CONFIG, 30.0).motion_confirm_low_frames + 5

    salida = replay_sample(
        MUESTRAS[0],
        without_dynamic_path(CONFIG),
        ClassifierRegistry(),
        rest_after=reposo,
    )

    assert salida.dynamic_windows == 0


def test_el_cli_escribe_el_reporte_con_la_z_aparte(tmp_path: Path) -> None:
    codigo = main(
        [
            "--raiz",
            str(tmp_path / "vacio"),
            "--config",
            str(REPO_ROOT / "config.yaml"),
            "--salida",
            str(tmp_path / "eval"),
            "--rejilla",
            "rapido",
            "--sin-reproduccion",
        ]
    )

    reporte = (tmp_path / "eval" / "reporte-fase5.md").read_text(encoding="utf-8")
    assert codigo == 0
    assert "CORPUS SINTÉTICO" in reporte
    assert "### Z, aparte" in reporte
    assert "## 3. `dtw.max_distance`" in reporte
    assert (tmp_path / "eval" / "resultados-fase5.json").exists()


def test_el_cli_sin_dataset_y_sin_sintetico_falla(tmp_path: Path) -> None:
    codigo = main(
        [
            "--raiz",
            str(tmp_path / "vacio"),
            "--config",
            str(REPO_ROOT / "config.yaml"),
            "--salida",
            str(tmp_path / "eval"),
            "--sin-sintetico",
        ]
    )

    assert codigo == 1


@pytest.mark.parametrize("protocolo", list(Protocol))
def test_los_dos_protocolos_producen_folds(protocolo: Protocol) -> None:
    dataset = observe_dynamic(MUESTRAS, CONFIG)

    assert precompute_dynamic(dataset, protocolo, CONFIG.dtw.band_radius)
