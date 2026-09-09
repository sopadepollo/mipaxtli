"""`make train`: entrena `static_knn` y escribe el modelo exportado.

Lo que se comprueba aquí no es que acierte —eso es `make eval`— sino las tres
promesas del artefacto que produce: que solo contiene clases de la Fase 2, que
puede volver a cargarse desde su propio JSON, y que dice contra qué dataset se
entrenó. La tercera es la que permite, tres semanas después, saber si el modelo
que está en `data/models/` corresponde a las grabaciones que hay en `data/raw/`.
"""

from __future__ import annotations

import json
from pathlib import Path

from lsm.classifiers.static_knn import StaticKnnClassifier
from lsm.cli.train import main
from lsm.evaluation import EXCLUDED_LABELS, PHASE2_LABELS

REPO_ROOT = Path(__file__).resolve().parents[1]


def entrenar(tmp_path: Path, *extra: str) -> Path:
    salida = tmp_path / "modelo.json"
    codigo = main(
        [
            "--raiz",
            str(tmp_path / "vacio"),
            "--config",
            str(REPO_ROOT / "config.yaml"),
            "--salida",
            str(salida),
            "--repeticiones-sinteticas",
            "2",
            *extra,
        ]
    )
    assert codigo == 0
    return salida


def test_escribe_un_modelo_que_se_puede_volver_a_cargar(tmp_path: Path) -> None:
    payload = json.loads(entrenar(tmp_path).read_text(encoding="utf-8"))

    modelo = StaticKnnClassifier.from_export(payload)

    assert sorted(modelo.export()["labels"]) == sorted(PHASE2_LABELS)


def test_el_modelo_no_contiene_ninguna_letra_dinamica(tmp_path: Path) -> None:
    """Las ocho esperan a la Fase 5. Un centroide de una Z sería el promedio de un
    trazo, que no es una configuración de mano."""
    payload = json.loads(entrenar(tmp_path).read_text(encoding="utf-8"))

    assert not set(payload["labels"]) & EXCLUDED_LABELS


def test_el_modelo_registra_contra_que_dataset_se_entreno(tmp_path: Path) -> None:
    payload = json.loads(entrenar(tmp_path).read_text(encoding="utf-8"))

    dataset = payload["data"]["dataset"]
    assert len(dataset["fingerprint"]) == 64
    assert dataset["synthetic"] is True
    assert dataset["sample_count"] > 0
    assert dataset["feature_spec_version"] == payload["feature_spec_version"]


def test_un_corpus_sintetico_se_marca_dentro_del_propio_modelo(tmp_path: Path) -> None:
    """El modelo y el reporte circulan por separado; cualquiera de los dos puede
    acabar citado sin el otro al lado."""
    payload = json.loads(entrenar(tmp_path).read_text(encoding="utf-8"))

    assert "SINTÉTICO" in payload["data"]["dataset"]["warning"]


def test_sin_dataset_y_sin_sintetico_falla_diciendo_que_falta(tmp_path: Path) -> None:
    codigo = main(
        [
            "--raiz",
            str(tmp_path / "vacio"),
            "--config",
            str(REPO_ROOT / "config.yaml"),
            "--salida",
            str(tmp_path / "modelo.json"),
            "--sin-sintetico",
        ]
    )

    assert codigo == 1


def test_dos_entrenamientos_dan_el_mismo_archivo(tmp_path: Path) -> None:
    primero = entrenar(tmp_path).read_bytes()
    segundo = entrenar(tmp_path).read_bytes()

    assert primero == segundo
