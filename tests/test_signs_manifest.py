"""El criterio de la Fase 4: 29 letras con asset, fuente documentada y revisión.

Lee disco a propósito, como los tests del glosario: el entregable de la fase es
un archivo concreto del repositorio y este test es lo que impide cerrarla con
una letra sin dibujar o sin revisar.
"""

from __future__ import annotations

from pathlib import Path

from lsm.config import Config
from lsm.io.signs import (
    DEFAULT_ASSETS_DIR,
    MANIFEST_FILENAME,
    gif_frame_count,
    load_manifest,
)
from lsm.signs import FUENTE_NORMATIVA, manifest_drift
from lsm.vocabulary import DYNAMIC_LABELS, LETTERS, Label

RAIZ_REPO = Path(__file__).resolve().parent.parent
ASSETS = RAIZ_REPO / DEFAULT_ASSETS_DIR


def test_el_manifest_existe_y_carga() -> None:
    manifest = load_manifest(ASSETS / MANIFEST_FILENAME)

    assert manifest.fuente_normativa == FUENTE_NORMATIVA


def test_las_29_letras_estan_y_ninguna_sobra() -> None:
    manifest = load_manifest(ASSETS / MANIFEST_FILENAME)

    assert set(manifest.letras) == set(LETTERS)
    assert Label.NONE not in manifest.letras


def test_el_manifest_dice_lo_mismo_que_el_glosario() -> None:
    assert manifest_drift(load_manifest(ASSETS / MANIFEST_FILENAME)) == []


def test_cada_asset_existe_y_las_dinamicas_se_mueven() -> None:
    manifest = load_manifest(ASSETS / MANIFEST_FILENAME)
    fps = Config().signs.render_fps

    for label, asset in manifest.letras.items():
        archivo = ASSETS / asset.archivo
        assert archivo.is_file(), f"{label}: falta {asset.archivo}"
        frames = gif_frame_count(archivo)
        if label in DYNAMIC_LABELS:
            assert asset.duracion_ms is not None
            assert frames == round(asset.duracion_ms * fps / 1000), label
            assert frames >= 2, f"{label}: una dinámica con un solo frame no se mueve"
        else:
            assert frames == 1, label


def test_cada_asset_remite_a_una_muestra_del_dataset() -> None:
    manifest = load_manifest(ASSETS / MANIFEST_FILENAME)

    for label, asset in manifest.letras.items():
        assert asset.fuente.tipo == "esqueleto_desde_dataset"
        assert asset.fuente.muestra.split("/")[2] == label
        assert asset.fuente.signer_id == asset.fuente.muestra.split("/")[0]


def test_todas_las_revisiones_coinciden_con_la_descripcion() -> None:
    manifest = load_manifest(ASSETS / MANIFEST_FILENAME)

    pendientes = {
        str(label): asset.revision.resultado
        for label, asset in manifest.letras.items()
        if asset.revision.resultado != "coincide"
    }
    assert pendientes == {}, f"revisiones sin cerrar: {pendientes}"


def test_cada_revision_lleva_la_etiqueta_de_honestidad() -> None:
    """La revisión coteja dibujo y descripción; no es validación por persona usuaria."""
    manifest = load_manifest(ASSETS / MANIFEST_FILENAME)

    for label, asset in manifest.letras.items():
        assert "no validación por persona usuaria" in asset.revision.revisor, label
