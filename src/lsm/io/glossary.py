"""Lectura de `docs/glosario-lsm.md`: la tabla de letras y el registro de validación.

El glosario es la referencia normativa **para las personas**; `lsm.vocabulary` lo
es para el código. Tener dos fuentes de verdad es lo mismo que no tener ninguna, de
modo que hace falta algo que compare las dos y grite cuando se separan.

Ese algo vivía dentro de `tests/test_vocabulary.py`. Se movió aquí cuando apareció
un segundo consumidor —el bloqueo de sesión formal de `cli/capture.py`, que
necesita saber si la sección 5 está firmada— porque dos analizadores del mismo
Markdown se desincronizan igual de callados que las dos fuentes que pretendían
vigilar.

**Sobre el Markdown que analiza.** El documento escapa los guiones bajos
(`DOBLE\\_L`) y usa `<br />` para los saltos dentro de una celda, porque una celda
de tabla no puede contener saltos reales. Las dos cosas son presentación y se
deshacen al leer: `vocabulary.py` guarda la prosa.

Este módulo lee disco pero no importa OpenCV ni MediaPipe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from lsm.vocabulary import NO_TRAJECTORY, Label, spec

#: Ruta del glosario relativa a la raíz del repositorio.
DEFAULT_GLOSSARY: Final = Path("docs/glosario-lsm.md")

#: Encabezado de la tabla de letras (§3), ya desescapado.
_LETTER_HEADER: Final = "label"

#: Columnas del registro de validación (§5), ya desescapadas.
_VALIDATION_HEADER: Final = "fecha"


class GlossaryError(RuntimeError):
    """El glosario no tiene la forma que este código sabe leer."""


@dataclass(frozen=True, slots=True)
class ValidationEntry:
    """Una fila del registro de validación de la sección 5.

    `ARQUITECTURA.md` §4.11 y el propio glosario §2 exigen que una persona usuaria
    de LSM o un intérprete revise el documento antes de cerrar la Fase 1. Esto es
    la anotación de que ocurrió.
    """

    fecha: str
    revisor: str
    rol: str
    letras: str
    observaciones: str

    @property
    def is_complete(self) -> bool:
        """Una fila sirve si dice **quién** revisó y **cuándo**.

        `letras` y `observaciones` pueden quedar vacías —"todas, sin objeciones" es
        un resultado legítimo— pero una fila sin fecha ni revisor no es un registro,
        es la plantilla sin llenar.
        """
        return bool(self.fecha.strip()) and bool(self.revisor.strip())


def _unescape(cell: str) -> str:
    """Deshace los escapes de Markdown de una celda."""
    return cell.strip().replace("\\_", "_")


def _collapse(cell: str) -> str:
    """Quita los `<br />` y colapsa los espacios: devuelve la prosa."""
    return re.sub(r"\s+", " ", re.sub(r"<br\s*/?>", " ", cell)).strip()


def _read_table(path: Path, first_column: str) -> list[dict[str, str]]:
    """Lee la primera tabla cuyo encabezado empiece por `first_column`.

    Se corta en cuanto una fila no tiene el mismo número de celdas que el
    encabezado, que es como termina una tabla de Markdown y como empieza la
    siguiente.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [line for line in lines if line.startswith("|")]

    header_index = next(
        (
            index
            for index, line in enumerate(rows)
            if _unescape(line.strip("|").split("|")[0]).lower() == first_column
        ),
        None,
    )
    if header_index is None:
        msg = f"{path}: no se encontró una tabla que empiece por '{first_column}'"
        raise GlossaryError(msg)

    # Las claves se normalizan a minúsculas: la tabla de letras las escribe así y
    # la del registro de validación con inicial mayúscula, y quien consume esto no
    # tiene por qué saber cuál es cuál.
    header = [
        _unescape(cell).lower() for cell in rows[header_index].strip("|").split("|")
    ]

    parsed: list[dict[str, str]] = []
    for line in rows[header_index + 2 :]:
        cells = [_unescape(cell) for cell in line.strip("|").split("|")]
        if len(cells) != len(header):
            break
        parsed.append(dict(zip(header, cells, strict=True)))
    return parsed


def read_letter_table(path: Path = DEFAULT_GLOSSARY) -> list[dict[str, str]]:
    """Las filas de la tabla de letras (§3), con las celdas ya desescapadas."""
    return _read_table(path, _LETTER_HEADER)


def read_validation_log(path: Path = DEFAULT_GLOSSARY) -> list[ValidationEntry]:
    """Las filas del registro de validación (§5).

    Incluye las vacías: la plantilla trae una fila de guiones para que se vea la
    forma, y distinguir "no hay filas" de "hay una fila en blanco" no le importa a
    nadie salvo a esta función.
    """
    return [
        ValidationEntry(
            fecha=row.get("fecha", ""),
            revisor=row.get("revisor", ""),
            rol=row.get("rol", ""),
            letras=row.get("letras revisadas", ""),
            observaciones=row.get("observaciones", ""),
        )
        for row in _read_table(path, _VALIDATION_HEADER)
    ]


def is_validated(path: Path = DEFAULT_GLOSSARY) -> bool:
    """Si alguien revisó el glosario y lo firmó en la sección 5."""
    return any(entry.is_complete for entry in read_validation_log(path))


def parse_confundible(cell: str) -> set[str]:
    return {item.strip() for item in cell.split(",") if item.strip()}


def vocabulary_drift(path: Path = DEFAULT_GLOSSARY) -> list[str]:
    """Todo aquello en que `lsm.vocabulary` y el glosario ya no dicen lo mismo.

    Devuelve **todas** las discrepancias, no la primera. Es deliberado: la deriva
    real que ocurrió en este proyecto afectaba a ocho campos repartidos por seis
    letras, y un test que revienta en la primera obliga a arreglar, volver a
    correr, arreglar, ocho veces.

    Que esto vuelva a pasar sin que nadie se entere es el fallo que hay que
    impedir, así que el test que lo consume **no lleva marca `glosario`**: los
    marcados están en rojo a propósito esperando a una persona, y una regresión
    de verdad escondida entre ellos es invisible.
    """
    problemas: list[str] = []
    filas = read_letter_table(path)

    etiquetas_doc = [row["label"] for row in filas]

    for row in filas:
        nombre = row["label"]
        try:
            label = Label(nombre)
        except ValueError:
            problemas.append(f"{nombre}: está en el glosario pero no en Label")
            continue
        try:
            letra = spec(label)
        except KeyError:
            problemas.append(f"{nombre}: está en el glosario pero no en LETTERS")
            continue

        esperado_dinamica = row["es_dinamica"] == "true"
        comprobaciones: list[tuple[str, object, object]] = [
            ("display", letra.display, row["letra"]),
            ("es_dinamica", letra.es_dinamica, esperado_dinamica),
            ("trayectoria", letra.trayectoria, _collapse(row["trayectoria"])),
            (
                "descripcion",
                letra.descripcion,
                _collapse(row["descripción de la configuración"]),
            ),
            (
                "confundible_con",
                {item.value for item in letra.confundible_con},
                parse_confundible(row["confundible_con"]),
            ),
            ("pagina", letra.pagina, int(row["página"])),
        ]
        for campo, en_codigo, en_documento in comprobaciones:
            if en_codigo != en_documento:
                problemas.append(
                    f"{nombre}.{campo}: vocabulary.py dice {en_codigo!r}, "
                    f"el glosario dice {en_documento!r}"
                )

    faltantes = {label.value for label in Label if label is not Label.NONE} - set(
        etiquetas_doc
    )
    for nombre in sorted(faltantes):
        problemas.append(f"{nombre}: está en Label pero no en la tabla del glosario")

    return problemas


#: `NO_TRAJECTORY` se reexporta porque quien compara la tabla necesita saber con
#: qué se marca "esta letra no traza recorrido", y ese valor lo fija el glosario.
__all__ = [
    "DEFAULT_GLOSSARY",
    "NO_TRAJECTORY",
    "GlossaryError",
    "ValidationEntry",
    "is_validated",
    "parse_confundible",
    "read_letter_table",
    "read_validation_log",
    "vocabulary_drift",
]
