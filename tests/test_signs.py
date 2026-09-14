"""La dirección texto → señas, sin disco ni ventana.

`lsm.signs` es código puro por la regla 2 de `CLAUDE.md`: recibe manifests ya
cargados, texto y ticks de reloj, y devuelve estados. Todo lo que hay aquí se
ejercita con datos construidos a mano.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from lsm.signs import (
    FUENTE_NORMATIVA,
    MANIFEST_SCHEMA_VERSION,
    AssetReview,
    AssetSource,
    Manifest,
    SignAsset,
    expected_filename,
    manifest_drift,
)
from lsm.types import Handedness
from lsm.vocabulary import LETTERS, Label, spec

HOY = date(2026, 9, 14)


def asset(label: Label, **cambios: object) -> SignAsset:
    """Una entrada válida del manifest, copiada de `vocabulary.py`."""
    letra = spec(label)
    base: dict[str, object] = {
        "letra": letra.display,
        "archivo": expected_filename(label),
        "es_dinamica": letra.es_dinamica,
        "descripcion": letra.descripcion,
        "trayectoria": letra.trayectoria,
        "pagina": letra.pagina,
        "duracion_ms": 2000 if letra.es_dinamica else None,
        "fuente": AssetSource(
            tipo="esqueleto_desde_dataset",
            muestra=f"s01/2026-09-09-manana/{label}/007.json",
            signer_id="s01",
            lateralidad_original=Handedness.RIGHT,
            espejada=False,
        ),
        "revision": AssetReview(fecha=HOY, revisor="tests", resultado="coincide"),
    }
    base.update(cambios)
    return SignAsset.model_validate(base)


def manifiesto(**cambios: SignAsset) -> Manifest:
    letras: dict[Label, SignAsset] = {label: asset(label) for label in LETTERS}
    for key, value in cambios.items():
        letras[Label(key)] = value
    return Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        fuente_normativa=FUENTE_NORMATIVA,
        letras=letras,
    )


# --------------------------------------------------------------------------- #
# Esquema
# --------------------------------------------------------------------------- #


def test_el_nombre_del_archivo_lo_fija_la_etiqueta() -> None:
    assert expected_filename(Label.A) == "A.png"
    assert expected_filename(Label.J) == "J.gif"
    assert expected_filename(Label.DOBLE_L) == "DOBLE_L.gif"


def test_un_manifest_completo_no_deriva_del_glosario() -> None:
    assert manifest_drift(manifiesto()) == []


def test_una_dinamica_con_imagen_fija_se_rechaza() -> None:
    """Las dinámicas necesitan movimiento: GIF, no PNG. Es la regla del spec
    hecha tipo, y falla al cargar en vez de al reproducir."""
    with pytest.raises(ValidationError, match="gif"):
        asset(Label.J, archivo="J.png")


def test_una_estatica_con_gif_se_rechaza() -> None:
    with pytest.raises(ValidationError, match="png"):
        asset(Label.A, archivo="A.gif")


def test_una_dinamica_sin_duracion_se_rechaza() -> None:
    with pytest.raises(ValidationError, match="duracion_ms"):
        asset(Label.J, duracion_ms=None)


def test_una_estatica_con_duracion_se_rechaza() -> None:
    with pytest.raises(ValidationError, match="duracion_ms"):
        asset(Label.A, duracion_ms=1500)


def test_la_muestra_de_origen_tiene_la_forma_del_dataset() -> None:
    with pytest.raises(ValidationError, match="firmante/sesion"):
        AssetSource(
            tipo="esqueleto_desde_dataset",
            muestra="J.json",
            signer_id="s01",
            lateralidad_original=Handedness.RIGHT,
            espejada=False,
        )


def test_el_archivo_tiene_que_llamarse_como_la_etiqueta() -> None:
    with pytest.raises(ValidationError, match=r"A\.png"):
        manifiesto(A=asset(Label.A, archivo="B.png"))


def test_la_muestra_de_origen_tiene_que_ser_de_la_misma_letra() -> None:
    fuente = AssetSource(
        tipo="esqueleto_desde_dataset",
        muestra="s01/2026-09-09-manana/B/001.json",
        signer_id="s01",
        lateralidad_original=Handedness.RIGHT,
        espejada=False,
    )
    with pytest.raises(ValidationError, match="muestra"):
        manifiesto(A=asset(Label.A, fuente=fuente))


def test_la_clase_negativa_no_es_una_letra() -> None:
    letras = {label: asset(label) for label in LETTERS}
    letras[Label.NONE] = asset(Label.A)
    with pytest.raises(ValidationError, match="NONE"):
        Manifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            fuente_normativa=FUENTE_NORMATIVA,
            letras=letras,
        )


def test_otra_version_de_esquema_se_rechaza() -> None:
    with pytest.raises(ValidationError, match="schema_version"):
        Manifest(
            schema_version=MANIFEST_SCHEMA_VERSION + 1,
            fuente_normativa=FUENTE_NORMATIVA,
            letras={label: asset(label) for label in LETTERS},
        )


def test_campos_desconocidos_se_rechazan() -> None:
    with pytest.raises(ValidationError):
        AssetReview.model_validate(
            {"fecha": HOY, "revisor": "x", "resultado": "coincide", "extra": 1}
        )


# --------------------------------------------------------------------------- #
# Deriva contra vocabulary.py
# --------------------------------------------------------------------------- #


def test_una_letra_que_falta_es_deriva() -> None:
    letras = {label: asset(label) for label in LETTERS}
    del letras[Label.Q]
    manifest = Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        fuente_normativa=FUENTE_NORMATIVA,
        letras=letras,
    )

    deriva = manifest_drift(manifest)

    assert len(deriva) == 1
    assert "Q" in deriva[0]
    assert "falta" in deriva[0]


def test_una_descripcion_distinta_es_deriva() -> None:
    deriva = manifest_drift(manifiesto(M=asset(Label.M, descripcion="otra cosa")))

    assert len(deriva) == 1
    assert "M" in deriva[0]
    assert "descripcion" in deriva[0]


def test_una_pagina_distinta_es_deriva() -> None:
    deriva = manifest_drift(manifiesto(Z=asset(Label.Z, pagina=1)))

    assert deriva
    assert "pagina" in deriva[0]


def test_la_forma_de_escribir_la_letra_tambien_se_compara() -> None:
    deriva = manifest_drift(manifiesto(ENIE=asset(Label.ENIE, letra="N")))

    assert deriva
    assert "letra" in deriva[0]
