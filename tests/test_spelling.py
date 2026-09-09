"""El reductor de deletreo, sin camara y sin modelo.

Cada test es una lista de señales y una afirmacion sobre el texto resultante.
Eso es todo lo que hace falta: `spelling.py` es puro por la regla 2 de
`CLAUDE.md`, y esa pureza es lo que permite ejercitar aqui el criterio de la
fase en vez de delante de una webcam.
"""

from __future__ import annotations

from lsm.config import Config
from lsm.spelling import (
    LetterSignal,
    Signal,
    SpellingState,
    render_text,
    render_word,
    step,
)
from lsm.vocabulary import Label

CONFIG = Config()


def aplicar(señales: list[Signal], config: Config = CONFIG) -> SpellingState:
    """Corre una lista de señales desde el estado vacio."""
    state = SpellingState()
    for señal in señales:
        state = step(state, señal, config).state
    return state


def letras(*labels: Label) -> list[Signal]:
    return [LetterSignal(label=label) for label in labels]


def test_cinco_letras_dan_cinco_simbolos() -> None:
    state = aplicar(letras(Label.C, Label.A, Label.S, Label.A, Label.S))

    assert state.word == (Label.C, Label.A, Label.S, Label.A, Label.S)
    assert render_word(state) == "casas"


def test_la_clase_negativa_no_escribe_nada() -> None:
    """`segmentation.py` filtra UNKNOWN y la confianza baja, pero NO filtra NONE:
    con la clase negativa acertando bien, llegaran LetterEmitted con label NONE
    cada vez que la persona baje la mano. Escribirlos seria poner una letra en
    cada transicion, que es justo lo que la clase negativa existe para evitar."""
    state = aplicar(letras(Label.C, Label.NONE, Label.A))

    assert state.word == (Label.C, Label.A)
    assert render_word(state) == "ca"


def test_el_digrafo_es_un_simbolo_y_dos_erres_son_dos() -> None:
    """El buffer guarda simbolos del glosario, no caracteres. Las dos formas se
    leen igual y se deshacen distinto, que es la razon de la decision."""
    una = aplicar(letras(Label.DOBLE_R))
    dos = aplicar(letras(Label.R, Label.R))

    assert render_word(una) == render_word(dos) == "rr"
    assert len(una.word) == 1
    assert len(dos.word) == 2


def test_la_enie_se_escribe_con_su_letra() -> None:
    assert render_word(aplicar(letras(Label.ENIE))) == "ñ"


def test_el_texto_vacio_es_una_cadena_vacia() -> None:
    assert render_text(SpellingState()) == ""
