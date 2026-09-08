"""El glosario, verificado.

Dos clases de test conviven aquí y conviene no confundirlas:

- **Los de transcripción**, sin marca, verifican que `src/lsm/vocabulary.py` diga
  exactamente lo mismo que la tabla de `docs/glosario-lsm.md`. Deben pasar siempre:
  si fallan, alguien editó uno de los dos y olvidó el otro.
- **Los marcados `glosario`**, que validan el *contenido* del documento contra lo
  que el proyecto necesita. Hoy varios están en rojo a propósito: el glosario tiene
  errores reales y huecos sin llenar, y arreglarlos exige consultar las páginas
  15-19 de *Manos con voz*. Son decisiones humanas, no de código.

Que estén en rojo es la señal, no un descuido: **bloquean la Fase 1**. Empezar a
capturar dataset con un glosario incompleto significa grabar veinte repeticiones
por letra de una seña que quizá esté mal.

Para trabajar en el núcleo sin ruido: `pytest -m "not glosario"`.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pytest

from lsm.types import NEGATIVE_LABEL
from lsm.vocabulary import (
    ALPHABET,
    DYNAMIC_LABELS,
    LETTERS,
    NO_TRAJECTORY,
    STATIC_LABELS,
    Label,
    spec,
)

GLOSARIO = Path(__file__).resolve().parents[1] / "docs" / "glosario-lsm.md"

EXPECTED_LETTERS = 29


def parse_glossary_table() -> list[dict[str, str]]:
    """Lee la tabla de letras del glosario en Markdown.

    El documento escapa los guiones bajos (`DOBLE\\_L`), así que hay que
    desescaparlos antes de comparar.
    """
    lines = GLOSARIO.read_text(encoding="utf-8").splitlines()
    rows = [line for line in lines if line.startswith("|")]

    header_index = next(
        index for index, line in enumerate(rows) if line.startswith("|label|")
    )
    header = [
        cell.strip().replace("\\_", "_")
        for cell in rows[header_index].strip("|").split("|")
    ]

    parsed: list[dict[str, str]] = []
    for line in rows[header_index + 2 :]:
        cells = [
            cell.strip().replace("\\_", "_") for cell in line.strip("|").split("|")
        ]
        if len(cells) != len(header):
            break  # empezó la tabla siguiente
        parsed.append(dict(zip(header, cells, strict=True)))
    return parsed


def normaliza(celda: str) -> str:
    """Quita los `<br />` de la tabla y colapsa los espacios.

    Los saltos de línea son de la presentación, no del texto: una celda de una
    tabla Markdown no puede contener saltos reales, así que quien redactó el
    glosario los escribió como HTML. `vocabulary.py` guarda la prosa.
    """
    return re.sub(r"\s+", " ", re.sub(r"<br\s*/?>", " ", celda)).strip()


def parse_confundible(cell: str) -> set[str]:
    return {item.strip() for item in cell.split(",") if item.strip()}


# --------------------------------------------------------------------------- #
# Transcripción: el código y el documento dicen lo mismo
# --------------------------------------------------------------------------- #


def test_el_glosario_tiene_las_29_letras() -> None:
    assert len(parse_glossary_table()) == EXPECTED_LETTERS
    assert len(LETTERS) == EXPECTED_LETTERS
    assert len(ALPHABET) == EXPECTED_LETTERS


def test_el_enum_son_las_29_letras_mas_la_clase_negativa() -> None:
    assert len(Label) == EXPECTED_LETTERS + 1
    assert Label.NONE not in LETTERS
    assert Label.NONE.value == NEGATIVE_LABEL


def test_el_codigo_transcribe_el_documento_sin_desviarse() -> None:
    """Si esto falla, alguien editó el glosario o `vocabulary.py` y no el otro."""
    for row in parse_glossary_table():
        label = Label(row["label"])
        letter = spec(label)

        assert letter.display == row["letra"], label
        assert letter.es_dinamica is (row["es_dinamica"] == "true"), label
        assert letter.trayectoria == normaliza(row["trayectoria"]), label
        assert letter.descripcion == normaliza(
            row["descripción de la configuración"]
        ), label
        assert letter.confundible_con == {
            Label(name) for name in parse_confundible(row["confundible_con"])
        }, label
        assert letter.pagina == int(row["página"]), label


def test_el_orden_del_codigo_es_el_del_documento() -> None:
    assert list(ALPHABET) == [Label(row["label"]) for row in parse_glossary_table()]


def test_estaticas_y_dinamicas_particionan_el_alfabeto() -> None:
    assert set(ALPHABET) == DYNAMIC_LABELS | STATIC_LABELS
    assert not DYNAMIC_LABELS & STATIC_LABELS
    assert DYNAMIC_LABELS  # si esto queda vacío, el proyecto perdió su mitad difícil


# --------------------------------------------------------------------------- #
# Validación del contenido — estos son los que bloquean la Fase 1
# --------------------------------------------------------------------------- #


@pytest.mark.glosario
def test_confundible_con_es_simetrico() -> None:
    """Si X se confunde con Y, Y se confunde con X. La confusión no tiene sentido
    de la marcha.

    Importa porque `confundible_con` es la hipótesis previa contra la que se
    contrastará la matriz de confusión real en la Fase 2. Una relación asimétrica
    hace que el par se revise en una dirección y no en la otra.
    """
    roturas = [
        f"{letra.label} lista a {otra}, pero {otra} no lista a {letra.label}"
        for letra in LETTERS.values()
        for otra in sorted(letra.confundible_con)
        if letra.label not in spec(otra).confundible_con
    ]

    assert not roturas, (
        "Asimetrías en el glosario (verificar contra pp. 15-19):\n"
        + "\n".join(f"  - {rotura}" for rotura in roturas)
    )


@pytest.mark.glosario
def test_toda_letra_describe_su_configuracion_manual() -> None:
    """Sin descripción no se puede: enseñar la seña a quien graba el dataset,
    revisar el glosario con un intérprete, ni redactar el manifest de la dirección
    texto → señas de la Fase 4.

    Se redacta con palabras propias a partir de las páginas 15-19, no se copia.
    """
    faltantes = [
        f"{letter.label} (p. {letter.pagina})"
        for letter in LETTERS.values()
        if not letter.descripcion.strip()
    ]

    assert not faltantes, (
        f"{len(faltantes)} de {EXPECTED_LETTERS} letras sin descripción de la "
        "configuración manual. Transcribir de Manos con voz, pp. 15-19:\n  "
        + ", ".join(faltantes)
    )


@pytest.mark.glosario
def test_la_trayectoria_concuerda_con_es_dinamica() -> None:
    """Una letra con movimiento sin recorrido descrito no se puede reconocer ni
    enseñar; una estática con recorrido es una contradicción en la tabla."""
    problemas = [
        f"{letter.label}: es_dinamica={letter.es_dinamica} pero "
        f"trayectoria={letter.trayectoria!r}"
        for letter in LETTERS.values()
        if letter.es_dinamica != (letter.trayectoria.strip() not in {"", NO_TRAJECTORY})
    ]

    assert not problemas, "\n".join(problemas)


@pytest.mark.glosario
def test_las_etiquetas_son_identificadores_sin_acentos() -> None:
    """Las etiquetas viajan a nombres de archivo, claves JSON y variables de
    TypeScript. Una `Ñ` ahí dentro se rompe en algún punto del camino."""
    problemas = [
        f"{label!r} no es un identificador en mayúsculas y sin acentos"
        for label in Label
        if not (
            label.value.isidentifier()
            and label.value.isupper()
            and label.value == unicodedata.normalize("NFKD", label.value)
            and re.fullmatch(r"[A-Z_]+", label.value)
        )
    ]

    assert not problemas, "\n".join(problemas)
