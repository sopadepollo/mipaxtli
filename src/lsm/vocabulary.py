"""El alfabeto dactilológico de LSM, como código.

Transcripción de la tabla de `docs/glosario-lsm.md` §3. Ese documento es la
referencia normativa para las personas; este módulo lo es para el código, y
`tests/test_vocabulary.py` verifica que los dos digan lo mismo. Sin esa
verificación habría dos fuentes de verdad, que es lo mismo que no tener ninguna.

**Este módulo transcribe el glosario tal como está.** No corrige ni completa nada
por su cuenta: si la tabla tuviera un hueco o una incoherencia, aquí se reflejaría
igual y los tests marcados `glosario` lo dirían. Rellenarlo desde el código sería
inventar LSM, que es exactamente lo que este proyecto no debe hacer.

Las descripciones se guardan como prosa. Los `<br />` del Markdown son saltos de
línea de la tabla, no parte del texto: la tabla los necesita porque una celda no
puede contener saltos reales.

Advertencia que vale repetir: **LSM no es ASL.** Casi todo el material que circula
en internet como "abecedario en lengua de señas" es estadounidense. Cada fila lleva
la página de la fuente primaria donde se verificó.

Código puro: sin disco, sin cámara, sin MediaPipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from lsm.types import NEGATIVE_LABEL

#: Marca de "esta letra no traza recorrido" en la columna `trayectoria`.
NO_TRAJECTORY: Final = "—"


class Label(StrEnum):
    """Etiquetas del dataset: 29 letras del abecedario más la clase negativa.

    Los nombres son identificadores en mayúsculas y sin acentos, porque viajan a
    nombres de archivo, claves JSON y variables de TypeScript: `Ñ` → `ENIE`,
    `LL` → `DOBLE_L`, `RR` → `DOBLE_R`.
    """

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    H = "H"
    I = "I"  # noqa: E741 — es una letra del alfabeto, no una variable
    J = "J"
    K = "K"
    L = "L"
    DOBLE_L = "DOBLE_L"
    M = "M"
    N = "N"
    ENIE = "ENIE"
    O = "O"  # noqa: E741 — idem
    P = "P"
    Q = "Q"
    R = "R"
    DOBLE_R = "DOBLE_R"
    S = "S"
    T = "T"
    U = "U"
    V = "V"
    W = "W"
    X = "X"
    Y = "Y"
    Z = "Z"

    #: Clase negativa: mano relajada, transiciones entre letras y gestos que no
    #: pertenecen al alfabeto. Sin ella el clasificador asigna una de las 29 letras
    #: aunque la persona no esté firmando (`ARQUITECTURA.md` §4.4).
    NONE = NEGATIVE_LABEL


@dataclass(frozen=True, slots=True)
class LetterSpec:
    """Todo lo que el glosario dice de una letra."""

    label: Label
    #: Cómo se escribe la letra para una persona: `Ñ`, `LL`, `RR`.
    display: str
    #: `True` si la ejecución requiere movimiento. Se marca según las flechas del
    #: diccionario, no según intuición.
    es_dinamica: bool
    #: Descripción de la configuración manual, con palabras propias. **Vacía hoy.**
    descripcion: str
    #: Recorrido, si es dinámica. `NO_TRAJECTORY` si no lo es.
    trayectoria: str
    #: Letras de configuración similar. Alimenta el análisis de la matriz de
    #: confusión de la Fase 2.
    confundible_con: frozenset[Label]
    #: Página de *Manos con voz* donde se verificó.
    pagina: int


def _spec(
    label: Label,
    display: str,
    *,
    dinamica: bool = False,
    descripcion: str = "",
    trayectoria: str = NO_TRAJECTORY,
    confundible: tuple[Label, ...] = (),
    pagina: int,
) -> LetterSpec:
    return LetterSpec(
        label=label,
        display=display,
        es_dinamica=dinamica,
        descripcion=descripcion,
        trayectoria=trayectoria,
        confundible_con=frozenset(confundible),
        pagina=pagina,
    )


_LETTERS: Final = (
    _spec(
        Label.A,
        "A",
        descripcion=(
            "Mano cerrada, se muestran las uñas y se estira el dedo pulgar "
            "hacia un lado. La palma mira al frente"
        ),
        confundible=(Label.E, Label.L, Label.DOBLE_L),
        pagina=15,
    ),
    _spec(
        Label.B,
        "B",
        descripcion=(
            "Dedos índice, medio, anular y meñique se estiran unidos y el "
            "pulgar se dobla dirección a la palma, la cual mira al frente"
        ),
        confundible=(Label.F,),
        pagina=15,
    ),
    _spec(
        Label.C,
        "C",
        descripcion=(
            "Dedos índice, medio, anular y meñique se mantienen unidos y en "
            "posición cóncava; el pulgar también se pone de esa forma. La "
            "palma mira a un lado"
        ),
        confundible=(Label.O,),
        pagina=15,
    ),
    _spec(
        Label.D,
        "D",
        descripcion=(
            "Dedos medio, anular, meñique y pulgar se unen por las puntas y "
            "el dedo índice se estira. La palma mira al frente"
        ),
        confundible=(Label.U, Label.R, Label.DOBLE_R),
        pagina=15,
    ),
    _spec(
        Label.E,
        "E",
        descripcion=(
            "Dedos completamente doblados, se muestran las uñas. La palma "
            "mira al frente"
        ),
        confundible=(Label.A, Label.L, Label.DOBLE_L),
        pagina=15,
    ),
    _spec(
        Label.F,
        "F",
        descripcion=(
            "Mano abierta y los dedos unidos, se dobla el índice hasta que "
            "su parte lateral toque la yema del pulgar. La palma mira a un "
            "lado"
        ),
        confundible=(Label.B,),
        pagina=15,
    ),
    _spec(
        Label.G,
        "G",
        descripcion=(
            "Mano cerrada y los dedos índice y pulgar estirados. La palma mira adentro"
        ),
        confundible=(Label.H,),
        pagina=16,
    ),
    _spec(
        Label.H,
        "H",
        descripcion=(
            "Mano cerrada y los dedos índice y medio estirados y unidos, se "
            "extiende el dedo pulgar señalando hacia arriba. La palma mira "
            "adentro"
        ),
        confundible=(Label.G,),
        pagina=16,
    ),
    _spec(
        Label.I,
        "I",
        descripcion=(
            "Mano cerrada, el dedo meñique se estira señalando hacia "
            "arriba. La palma se pone de lado"
        ),
        confundible=(Label.J, Label.Y),
        pagina=16,
    ),
    _spec(
        Label.J,
        "J",
        dinamica=True,
        descripcion=(
            "Mano cerrada, el dedo meñique bien estirado señalando hacia "
            "arriba y la palma a un lado dibuja una j en el aire"
        ),
        trayectoria="dibuja una j en el aire",
        confundible=(Label.I, Label.Y),
        pagina=16,
    ),
    _spec(
        Label.K,
        "K",
        dinamica=True,
        descripcion=(
            "Se cierra la mano con los dedos índice, medio y pulgar "
            "estirados. La yema del pulgar se pone entre el índice y el "
            "medio. Se mueve la muñeca hacia arriba"
        ),
        trayectoria="se mueve la muneca hacia arriba",
        confundible=(Label.P,),
        pagina=16,
    ),
    _spec(
        Label.L,
        "L",
        descripcion=(
            "Mano cerrada y los dedos índice y pulgar estirados, se forma "
            "una letra l. La palma mira al frente"
        ),
        confundible=(Label.E, Label.A, Label.DOBLE_L),
        pagina=16,
    ),
    _spec(
        Label.DOBLE_L,
        "LL",
        dinamica=True,
        descripcion=(
            "Mano cerrada y los dedos índice y pulgar alargados, se forma "
            "una letra l. La palma mira al frente y se hacen movimientos "
            "horizontales"
        ),
        trayectoria="movimientos horizontales",
        confundible=(Label.E, Label.A, Label.L),
        pagina=16,
    ),
    _spec(
        Label.M,
        "M",
        descripcion=(
            "Mano cerrada, se ponen los dedos índice, medio y anular sobre el pulgar"
        ),
        confundible=(Label.N,),
        pagina=17,
    ),
    _spec(
        Label.N,
        "N",
        descripcion=(
            "Con la mano cerrada, se ponen los dedos índice y medio sobre "
            "el dedo pulgar"
        ),
        confundible=(Label.M, Label.ENIE),
        pagina=17,
    ),
    _spec(
        Label.ENIE,
        "Ñ",
        dinamica=True,
        descripcion=(
            "Mano cerrada, se ponen los dedos índice y medio sobre el "
            "pulgar. Se mueve la muñeca a los lados"
        ),
        trayectoria="rotacion de ida y vuelta",
        confundible=(Label.N, Label.Q, Label.X),
        pagina=17,
    ),
    _spec(
        Label.O,
        "O",
        descripcion=(
            "Con la mano se forma una letra o. Todos los dedos se tocan por las puntas"
        ),
        confundible=(Label.C,),
        pagina=17,
    ),
    _spec(
        Label.P,
        "P",
        descripcion=(
            "Mano cerrada y los dedos índice, medio y pulgar estirados, se "
            "pone la yema del pulgar entre el índice y el medio"
        ),
        confundible=(Label.K,),
        pagina=17,
    ),
    _spec(
        Label.Q,
        "Q",
        dinamica=True,
        descripcion=(
            "Mano cerrada, se ponen los dedos índice y pulgar en posición "
            "de garra. La palma mira hacia abajo, y se mueve la muñeca "
            "hacia los lados"
        ),
        trayectoria="rotacion de ida y vuelta",
        confundible=(Label.ENIE, Label.X),
        pagina=17,
    ),
    _spec(
        Label.R,
        "R",
        descripcion=(
            "Mano cerrada, se estiran y entrelazan los dedos índice y "
            "medio. La palma mira al frente"
        ),
        confundible=(Label.U, Label.D, Label.DOBLE_R),
        pagina=18,
    ),
    _spec(
        Label.DOBLE_R,
        "RR",
        dinamica=True,
        descripcion=(
            "Mano cerrada, se alargan y entrelazan los dedos índice y "
            "medio. La palma mira al frente y hace movimientos horizontales"
        ),
        trayectoria="movimientos horizontales",
        confundible=(Label.U, Label.D, Label.R),
        pagina=18,
    ),
    _spec(
        Label.S,
        "S",
        descripcion=(
            "Mano cerrada, se pone el pulgar sobre los otros dedos. La "
            "palma mira al frente"
        ),
        confundible=(Label.T,),
        pagina=18,
    ),
    _spec(
        Label.T,
        "T",
        descripcion=(
            "Mano cerrada, el pulgar se pone entre el índice y el medio. La "
            "palma mira al frente"
        ),
        confundible=(Label.S,),
        pagina=18,
    ),
    _spec(
        Label.U,
        "U",
        descripcion=(
            "Mano cerrada, se estiran los dedos índice y medio unidos. La "
            "palma mira al frente"
        ),
        confundible=(Label.R, Label.DOBLE_R, Label.D),
        pagina=18,
    ),
    _spec(
        Label.V,
        "V",
        descripcion=(
            "Mano cerrada, se estiran los dedos índice y medio separados. "
            "La palma mira al frente"
        ),
        confundible=(Label.W,),
        pagina=18,
    ),
    _spec(
        Label.W,
        "W",
        descripcion=(
            "Mano cerrada, se estiran los dedos índice, medio y anular "
            "separados. La palma mira al frente"
        ),
        confundible=(Label.V,),
        pagina=18,
    ),
    _spec(
        Label.X,
        "X",
        dinamica=True,
        descripcion=(
            "Mano cerrada, el índice y el pulgar en posición de garra y la "
            "palma dirigida a un lado, se realiza un movimiento al frente y "
            "de regreso"
        ),
        trayectoria="movimiento al frente y de regreso",
        confundible=(Label.Q, Label.ENIE),
        pagina=19,
    ),
    _spec(
        Label.Y,
        "Y",
        descripcion=(
            "Mano cerrada, se estira el meñique y el pulgar. La palma mira hacia dentro"
        ),
        confundible=(Label.I, Label.J),
        pagina=19,
    ),
    _spec(
        Label.Z,
        "Z",
        dinamica=True,
        descripcion=(
            "Mano cerrada, el dedo índice estirado y la palma al frente, se "
            "dibuja una letra z en el aire"
        ),
        trayectoria="dibuja una letra z en el aire",
        pagina=19,
    ),
)

#: Las 29 letras, en el orden del glosario. **No incluye `NONE`**: la clase
#: negativa es una etiqueta del dataset, no una letra del abecedario, y no tiene
#: configuración, trayectoria ni página que verificar.
LETTERS: Final[MappingProxyType[Label, LetterSpec]] = MappingProxyType(
    {letter.label: letter for letter in _LETTERS}
)

#: Orden canónico de las letras, tal como aparecen en el glosario.
ALPHABET: Final = tuple(letter.label for letter in _LETTERS)

#: Letras que requieren movimiento. Deciden qué clasificador las atiende.
DYNAMIC_LABELS: Final = frozenset(
    letter.label for letter in _LETTERS if letter.es_dinamica
)

#: Letras sin movimiento: una configuración sostenida.
STATIC_LABELS: Final = frozenset(
    letter.label for letter in _LETTERS if not letter.es_dinamica
)


def spec(label: Label) -> LetterSpec:
    """Datos del glosario para una letra.

    Levanta `KeyError` con `Label.NONE`, y es lo correcto: preguntar por la
    trayectoria de la clase negativa es un error de programación, no un caso a
    contemplar.
    """
    return LETTERS[label]
