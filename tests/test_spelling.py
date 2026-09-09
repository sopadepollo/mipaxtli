"""El reductor de deletreo, sin camara y sin modelo.

Cada test es una lista de señales y una afirmacion sobre el texto resultante.
Eso es todo lo que hace falta: `spelling.py` es puro por la regla 2 de
`CLAUDE.md`, y esa pureza es lo que permite ejercitar aqui el criterio de la
fase en vez de delante de una webcam.
"""

from __future__ import annotations

from lsm.config import Config
from lsm.spelling import (
    Backspace,
    CommitText,
    HandAbsent,
    HandPresent,
    LetterSignal,
    NothingToDelete,
    Signal,
    SpaceWritten,
    SpellingState,
    SymbolDeleted,
    TextCommitted,
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


def ausencia(frames: int) -> list[Signal]:
    return [HandAbsent()] * frames


UMBRAL = CONFIG.spelling.space_after_absent_frames


def test_la_ausencia_corta_no_pone_espacio() -> None:
    """Un parpadeo del detector no es una intencion de quien firma."""
    state = aplicar([*letras(Label.C, Label.A), *ausencia(UMBRAL - 1)])

    assert state.finished == ()
    assert render_text(state) == "ca"


def test_al_alcanzar_el_umbral_pone_un_espacio_y_solo_uno() -> None:
    """La mano abajo no es un evento, es un estado que dura: sin cerrojo pondria
    un espacio en cada frame. Es el mismo problema que `pending_repeat` resuelve
    en segmentation.py, un nivel mas arriba."""
    state = aplicar([*letras(Label.C, Label.A), *ausencia(UMBRAL * 3)])

    assert state.finished == ((Label.C, Label.A),)
    assert state.word == ()
    assert render_text(state) == "ca"


def test_la_mano_de_vuelta_libera_el_cerrojo() -> None:
    señales = [
        *letras(Label.C, Label.A),
        *ausencia(UMBRAL),
        HandPresent(),
        *letras(Label.S, Label.A),
        *ausencia(UMBRAL),
    ]

    state = aplicar(señales)

    assert state.finished == ((Label.C, Label.A), (Label.S, Label.A))
    assert render_text(state) == "ca sa"


def test_la_ausencia_sobre_una_palabra_vacia_no_pone_espacio() -> None:
    """Ni al principio ni entre dos ausencias seguidas: espacios sueltos o
    dobles serian texto que nadie seño."""
    state = aplicar([*ausencia(UMBRAL * 2), HandPresent(), *ausencia(UMBRAL * 2)])

    assert state.finished == ()
    assert render_text(state) == ""


def test_el_espacio_emite_su_evento() -> None:
    state = aplicar([*letras(Label.A), *ausencia(UMBRAL - 1)])

    resultado = step(state, HandAbsent(), CONFIG)

    assert isinstance(resultado.event, SpaceWritten)


def test_el_cerrojo_sobrevive_a_una_letra_sin_mano_de_vuelta() -> None:
    """El cerrojo no es redundante con la guarda de palabra vacia.

    Si una letra llega mientras la ausencia sigue en curso —sin HandPresent que
    la libere—, el siguiente HandAbsent encuentra `absent_frames` ya por encima
    del umbral y una palabra con contenido. Sin `space_emitted` la cerraria de
    golpe, tras una sola letra. Con el, no pasa nada hasta que la mano vuelva.
    """
    state = aplicar([*letras(Label.C, Label.A), *ausencia(UMBRAL)])
    assert state.finished == ((Label.C, Label.A),)

    resultado = step(state, LetterSignal(label=Label.S), CONFIG)
    resultado = step(resultado.state, HandAbsent(), CONFIG)

    assert resultado.event is None, "el cerrojo sigue puesto: no se cierra nada"
    assert resultado.state.word == (Label.S,)
    assert resultado.state.finished == ((Label.C, Label.A),)


def test_backspace_borra_un_simbolo() -> None:
    state = aplicar([*letras(Label.C, Label.A, Label.S), Backspace()])

    assert render_word(state) == "ca"


def test_backspace_deshace_exactamente_lo_que_se_seño() -> None:
    """La razon de guardar simbolos: sobre una RR el borrado quita la seña
    entera, y sobre dos R quita una R. Con un buffer de caracteres, lo primero
    dejaria una `r` que nadie ejecuto."""
    digrafo = aplicar([*letras(Label.DOBLE_R), Backspace()])
    dos_erres = aplicar([*letras(Label.R, Label.R), Backspace()])

    assert render_word(digrafo) == ""
    assert render_word(dos_erres) == "r"


def test_backspace_sobre_una_palabra_vacia_no_hace_nada_y_lo_dice() -> None:
    """No recupera la palabra anterior: reabrir algo ya cerrado no vale la
    complejidad en esta fase."""
    state = aplicar([*letras(Label.A), *ausencia(UMBRAL)])

    resultado = step(state, Backspace(), CONFIG)

    assert resultado.state == state
    assert isinstance(resultado.event, NothingToDelete)


def test_backspace_emite_el_simbolo_que_quito() -> None:
    state = aplicar(letras(Label.C, Label.DOBLE_L))

    resultado = step(state, Backspace(), CONFIG)

    assert resultado.event == SymbolDeleted(label=Label.DOBLE_L)


def test_enter_cierra_la_frase_y_deja_el_estado_vacio() -> None:
    señales = [
        *letras(Label.C, Label.A),
        *ausencia(UMBRAL),
        HandPresent(),
        *letras(Label.S, Label.A),
    ]
    state = aplicar(señales)

    resultado = step(state, CommitText(), CONFIG)

    assert resultado.event == TextCommitted(text="ca sa")
    assert resultado.state == SpellingState()


def test_enter_sobre_un_texto_vacio_no_emite_nada() -> None:
    resultado = step(SpellingState(), CommitText(), CONFIG)

    assert resultado.event is None
    assert resultado.state == SpellingState()
