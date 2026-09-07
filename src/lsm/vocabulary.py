"""El alfabeto dactilológico de LSM, como código.

Transcripción de la tabla de `docs/glosario-lsm.md` §3. Ese documento es la
referencia normativa para las personas; este módulo lo es para el código, y
`tests/test_vocabulary.py` verifica que los dos digan lo mismo. Sin esa
verificación habría dos fuentes de verdad, que es lo mismo que no tener ninguna.

**Este módulo transcribe el glosario tal como está, errores incluidos.** Hoy la
tabla tiene relaciones `confundible_con` asimétricas y las 29 descripciones
vacías. No se corrigen aquí: son decisiones que exigen consultar las páginas 15-19
de *Manos con voz*, y falsearlas en código sería peor que dejarlas en rojo. Los
tests marcados `glosario` fallan a propósito hasta que el equipo humano las
resuelva contra la fuente primaria.

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
    _spec(Label.A, "A", confundible=(Label.E,), pagina=15),
    _spec(Label.B, "B", confundible=(Label.F,), pagina=15),
    _spec(Label.C, "C", confundible=(Label.O,), pagina=15),
    _spec(Label.D, "D", confundible=(Label.U, Label.R), pagina=15),
    _spec(Label.E, "E", confundible=(Label.A,), pagina=15),
    _spec(Label.F, "F", confundible=(Label.B,), pagina=15),
    _spec(Label.G, "G", confundible=(Label.H,), pagina=16),
    _spec(Label.H, "H", confundible=(Label.G,), pagina=16),
    _spec(Label.I, "I", confundible=(Label.J,), pagina=16),
    _spec(
        Label.J,
        "J",
        dinamica=True,
        trayectoria="hacia abajo y vuelta en u",
        confundible=(Label.I,),
        pagina=16,
    ),
    _spec(
        Label.K,
        "K",
        dinamica=True,
        trayectoria="izquierda a derecha o derecha a izquierda",
        confundible=(Label.P,),
        pagina=16,
    ),
    _spec(Label.L, "L", confundible=(Label.E,), pagina=16),
    _spec(Label.DOBLE_L, "LL", confundible=(Label.E,), pagina=16),
    _spec(Label.M, "M", confundible=(Label.N,), pagina=17),
    _spec(Label.N, "N", confundible=(Label.M,), pagina=17),
    _spec(
        Label.ENIE,
        "Ñ",
        dinamica=True,
        trayectoria="rotacion de ida y vuelta",
        confundible=(Label.N,),
        pagina=17,
    ),
    _spec(Label.O, "O", confundible=(Label.C,), pagina=17),
    _spec(Label.P, "P", confundible=(Label.K,), pagina=17),
    _spec(
        Label.Q,
        "Q",
        dinamica=True,
        trayectoria="rotacion de ida y vuelta",
        pagina=17,
    ),
    _spec(Label.R, "R", confundible=(Label.U,), pagina=18),
    _spec(Label.DOBLE_R, "RR", pagina=18),
    _spec(Label.S, "S", confundible=(Label.T,), pagina=18),
    _spec(Label.T, "T", confundible=(Label.S,), pagina=18),
    _spec(Label.U, "U", confundible=(Label.R, Label.D), pagina=18),
    _spec(Label.V, "V", confundible=(Label.W,), pagina=18),
    _spec(Label.W, "W", confundible=(Label.V,), pagina=18),
    _spec(
        Label.X,
        "X",
        dinamica=True,
        trayectoria="de un lado al otro horizontalmente, de ida y vuelta",
        pagina=19,
    ),
    _spec(Label.Y, "Y", pagina=19),
    _spec(
        Label.Z,
        "Z",
        dinamica=True,
        trayectoria="se dibuja la letra z",
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
