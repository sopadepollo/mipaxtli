"""`make eval`: el reporte, el barrido y el contraste de hipótesis.

El criterio de aceptación de la Fase 2 dice que las tres cosas salgan de un solo
comando y sean reproducibles. Estos tests son ese criterio, escrito de forma que
se pueda comprobar sin cámara y sin dataset grabado.

Lo que se verifica, en orden de importancia:

1. **Reproducibilidad byte a byte.** Dos ejecuciones dan el mismo reporte, o el
   archivo no sirve para comparar dos calibraciones.
2. **El alcance.** La matriz solo lleva estáticas y `NONE`. Si se colara una
   dinámica, todas las conclusiones sobre las confundibles quedarían contaminadas.
3. **La trazabilidad.** La calibración dice contra qué dataset se ajustó.
4. **El contraste.** Las cuatro cubetas están, incluida la de los pares que la
   Fase 2 no puede evaluar.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lsm.cli.evaluate import main
from lsm.evaluation import EXCLUDED_LABELS

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def reporte(tmp_path: Path) -> Path:
    salida = tmp_path / "eval"
    codigo = main(
        [
            "--raiz",
            str(tmp_path / "vacio"),
            "--config",
            str(REPO_ROOT / "config.yaml"),
            "--glosario",
            str(REPO_ROOT / "docs" / "glosario-lsm.md"),
            "--salida",
            str(salida),
            "--repeticiones-sinteticas",
            "2",
            "--barrido",
            "minimo",
        ]
    )
    assert codigo == 0
    return salida


def test_un_solo_comando_produce_los_tres_archivos(reporte: Path) -> None:
    assert (reporte / "reporte-fase2.md").is_file()
    assert (reporte / "resultados-fase2.json").is_file()
    assert (reporte / "calibracion-fase2.json").is_file()


def test_el_reporte_trae_todas_las_secciones_obligatorias(reporte: Path) -> None:
    texto = (reporte / "reporte-fase2.md").read_text(encoding="utf-8")

    for seccion in (
        "## Alcance",
        "## Accuracy global",
        "## Accuracy por letra",
        "## Matriz de confusión",
        "## Pares más confundidos",
        "## Contraste de hipótesis",
        "## Barrido de calibración",
        "## Diagnóstico empírico",
    ):
        assert seccion in texto, f"falta la sección {seccion}"


def test_el_contraste_de_hipotesis_lleva_sus_cuatro_cubetas(reporte: Path) -> None:
    texto = (reporte / "reporte-fase2.md").read_text(encoding="utf-8")

    assert "Predichos que sí se confundieron" in texto
    assert "Predichos que NO se confundieron" in texto
    assert "Confundidos sin estar predichos" in texto
    assert "Predichos pero no evaluables en la Fase 2" in texto


def test_la_matriz_solo_contiene_clases_estaticas_y_none(reporte: Path) -> None:
    datos = json.loads((reporte / "resultados-fase2.json").read_text(encoding="utf-8"))

    presentes = {fila["truth"] for fila in datos["confusion"]}
    assert not presentes & EXCLUDED_LABELS


def test_el_reporte_dice_que_protocolo_uso(reporte: Path) -> None:
    texto = (reporte / "reporte-fase2.md").read_text(encoding="utf-8")

    assert "leave-one-signer-out" in texto


def test_un_corpus_sintetico_se_anuncia_en_la_primera_pantalla(reporte: Path) -> None:
    texto = (reporte / "reporte-fase2.md").read_text(encoding="utf-8")

    assert "SINTÉTICO" in texto.split("## ")[0]


def test_la_calibracion_registra_contra_que_dataset_se_ajusto(reporte: Path) -> None:
    """Es el requisito de trazabilidad: unos umbrales sin el dataset contra el que
    se midieron son un número sin unidades."""
    datos = json.loads((reporte / "calibracion-fase2.json").read_text(encoding="utf-8"))

    assert len(datos["dataset"]["fingerprint"]) == 64
    assert datos["dataset"]["synthetic"] is True
    assert datos["protocol"] == "leave-one-signer-out"
    assert "best" in datos
    assert "baseline" in datos


def test_el_barrido_distingue_los_ejes_efectivos_de_los_inertes(reporte: Path) -> None:
    datos = json.loads((reporte / "calibracion-fase2.json").read_text(encoding="utf-8"))

    assert "features.trajectory_weight" in datos["inert"]
    assert "segmentation.velocity_threshold" in datos["inert"]
    assert datos["effective"]


def test_dos_ejecuciones_dan_exactamente_el_mismo_reporte(tmp_path: Path) -> None:
    """El criterio de aceptación de la fase, literal."""

    def correr(destino: str) -> bytes:
        assert (
            main(
                [
                    "--raiz",
                    str(tmp_path / "vacio"),
                    "--config",
                    str(REPO_ROOT / "config.yaml"),
                    "--glosario",
                    str(REPO_ROOT / "docs" / "glosario-lsm.md"),
                    "--salida",
                    str(tmp_path / destino),
                    "--repeticiones-sinteticas",
                    "2",
                    "--barrido",
                    "minimo",
                ]
            )
            == 0
        )
        return (tmp_path / destino / "reporte-fase2.md").read_bytes()

    assert correr("a") == correr("b")


def test_con_un_solo_firmante_se_niega_y_dice_como_seguir(tmp_path: Path) -> None:
    """Sin dos personas, leave-one-signer-out no mide generalización. La
    degradación a leave-one-session-out se pide, no se toma sola."""
    codigo = main(
        [
            "--raiz",
            str(tmp_path / "vacio"),
            "--config",
            str(REPO_ROOT / "config.yaml"),
            "--salida",
            str(tmp_path / "eval"),
            "--repeticiones-sinteticas",
            "2",
            "--firmantes-sinteticos",
            "1",
            "--barrido",
            "ninguno",
        ]
    )

    assert codigo == 1
