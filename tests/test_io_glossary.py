"""El glosario leído por código: la sincronización y el registro de validación.

Dos cosas distintas se apoyan en este módulo, y las dos han fallado ya en este
proyecto o están a punto de hacerlo:

1. **Que `vocabulary.py` siga diciendo lo mismo que la tabla.** Ya se separaron
   una vez y nadie se enteró en el momento. El test que lo comprueba no lleva
   marca `glosario` **a propósito**: los marcados están en rojo esperando a una
   persona, y una regresión de verdad escondida entre ellos es invisible.

2. **Que la sección 5 esté firmada antes de una sesión formal.** Tres personas por
   veintinueve letras por dos sesiones contra un glosario sin revisar por un
   intérprete es el error caro del proyecto, y no se arregla salvo regrabando.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lsm.io.glossary import (
    DEFAULT_GLOSSARY,
    GlossaryError,
    ValidationEntry,
    is_validated,
    read_letter_table,
    read_validation_log,
    vocabulary_drift,
)
from lsm.vocabulary import ALPHABET

REPO = Path(__file__).resolve().parents[1]
GLOSARIO = REPO / DEFAULT_GLOSSARY


# --------------------------------------------------------------------------- #
# Sincronización con vocabulary.py
# --------------------------------------------------------------------------- #


def test_vocabulary_py_dice_exactamente_lo_que_dice_el_glosario() -> None:
    """**Este test no lleva marca `glosario` y no debe llevarla nunca.**

    La deriva entre el documento y su transcripción ya ocurrió una vez, y lo que
    la hizo pasar desapercibida no fue que no hubiera test: fue que el objetivo
    `make test` ya estaba en rojo esperando a una persona, así que un fallo más
    no llamó la atención. Marcar esto como `glosario` reproduciría exactamente esa
    situación.

    Rompe `make test-nucleo`, que es el que debe estar siempre limpio.
    """
    problemas = vocabulary_drift(GLOSARIO)

    assert not problemas, (
        "src/lsm/vocabulary.py y docs/glosario-lsm.md dejaron de coincidir.\n"
        "Alguien editó uno de los dos y no el otro. El documento manda: "
        "vocabulary.py lo transcribe, no lo corrige.\n"
        + "\n".join(f"  - {problema}" for problema in problemas)
    )


def test_la_deriva_se_informa_entera_y_no_solo_la_primera() -> None:
    """La deriva real afectó a ocho campos en seis letras. Un informe que se corta
    en el primero obliga a arreglar, volver a correr, arreglar, ocho veces."""
    falso = _glosario_con_dos_errores()

    problemas = vocabulary_drift(falso)

    assert len(problemas) >= 2


def test_la_tabla_del_glosario_tiene_las_29_letras() -> None:
    filas = read_letter_table(GLOSARIO)

    assert len(filas) == len(ALPHABET)
    assert [fila["label"] for fila in filas] == [label.value for label in ALPHABET]


def test_las_celdas_llegan_desescapadas() -> None:
    """El documento escribe `DOBLE\\_L` para que Markdown no lo lea como cursiva.
    Ese escape es presentación y no debe llegar al código."""
    etiquetas = {fila["label"] for fila in read_letter_table(GLOSARIO)}

    assert "DOBLE_L" in etiquetas
    assert not any("\\" in etiqueta for etiqueta in etiquetas)


# --------------------------------------------------------------------------- #
# Registro de validación (§5)
# --------------------------------------------------------------------------- #


def test_una_fila_necesita_fecha_y_revisor_para_contar() -> None:
    """`letras` y `observaciones` pueden ir vacías: "todas, sin objeciones" es un
    resultado legítimo. Una fila sin fecha ni revisor no es un registro, es la
    plantilla sin llenar."""
    completa = ValidationEntry(
        fecha="2026-09-10",
        revisor="quien revisó",
        rol="intérprete",
        letras="",
        observaciones="",
    )
    sin_revisor = ValidationEntry(
        fecha="2026-09-10", revisor="", rol="", letras="todas", observaciones="ok"
    )
    vacia = ValidationEntry(fecha="", revisor="", rol="", letras="", observaciones="")

    assert completa.is_complete
    assert not sin_revisor.is_complete
    assert not vacia.is_complete


def test_la_plantilla_vacia_no_cuenta_como_validacion(tmp_path: Path) -> None:
    documento = _escribir_glosario(
        tmp_path,
        validacion=(
            "|Fecha|Revisor|Rol|Letras revisadas|Observaciones|\n|-|-|-|-|-|\n||||||\n"
        ),
    )

    # La fila de guiones de la plantilla no tiene ni celdas: se cae al leer, y
    # aunque sobreviviera tampoco contaría. Las dos cosas dan lo mismo.
    assert not any(entrada.is_complete for entrada in read_validation_log(documento))
    assert not is_validated(documento)


def test_una_revision_firmada_si_cuenta(tmp_path: Path) -> None:
    documento = _escribir_glosario(
        tmp_path,
        validacion=(
            "|Fecha|Revisor|Rol|Letras revisadas|Observaciones|\n"
            "|-|-|-|-|-|\n"
            "|2026-09-12|A. Pérez|intérprete LSM|todas|sin objeciones|\n"
        ),
    )

    assert is_validated(documento)


def test_el_glosario_del_repositorio_declara_su_estado_de_validacion() -> None:
    """No afirma cuál debe ser el estado: afirma que se puede leer.

    Cuál sea la respuesta hoy es una cuestión humana —falta la revisión con una
    persona usuaria de LSM, `PENDIENTE-HUMANO G`—, pero que la pregunta se pueda
    hacer sin reventar es lo que sostiene el bloqueo de la sesión formal.
    """
    assert isinstance(is_validated(GLOSARIO), bool)


def test_un_documento_sin_la_tabla_esperada_falla_claro(tmp_path: Path) -> None:
    documento = tmp_path / "vacio.md"
    documento.write_text("# nada que ver aquí\n", encoding="utf-8")

    with pytest.raises(GlossaryError, match="no se encontró"):
        read_letter_table(documento)


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #


def _escribir_glosario(tmp_path: Path, *, validacion: str) -> Path:
    """Un glosario mínimo con la tabla de letras real y la §5 que se le pase."""
    original = GLOSARIO.read_text(encoding="utf-8")
    corte = original.index("## 5")
    documento = tmp_path / "glosario.md"
    documento.write_text(
        original[:corte] + "## 5. Registro de validación\n\n" + validacion,
        encoding="utf-8",
    )
    return documento


def _glosario_con_dos_errores() -> Path:
    """Un glosario con dos celdas cambiadas, para comprobar que se informan las dos."""
    import tempfile

    original = GLOSARIO.read_text(encoding="utf-8")
    roto = original.replace("|A|A|false|", "|A|A|true|", 1).replace(
        "|B|B|false|", "|B|B|true|", 1
    )
    destino = Path(tempfile.mkdtemp()) / "glosario-roto.md"
    destino.write_text(roto, encoding="utf-8")
    return destino
