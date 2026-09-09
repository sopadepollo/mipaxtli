"""Acumulación de letras en palabras: el buffer de deletreo.

`ARQUITECTURA.md` §4.2 dejó esto para la Fase 3, y con él la pregunta de qué se
hace con los dígrafos. La respuesta está en el tipo: **el buffer guarda símbolos
del glosario, no caracteres.**

`DOBLE_R` es un símbolo que se lee `rr`; dos `R` seguidas son dos símbolos que
también se leen `rr`. Se ven igual y se deshacen distinto, que es exactamente lo
que se quiere: `BACKSPACE` borra un símbolo, así que deshace lo que se señó. Con
un buffer de caracteres, borrar sobre una `RR` dejaría una `r` que nadie ejecutó.

## Puro por la regla 2 de `CLAUDE.md`

Sin OpenCV, sin MediaPipe, sin disco, sin cámara. Un reductor: `step` recibe el
estado y una señal, y devuelve el estado nuevo. Todo el criterio de esta fase se
ejercita con una lista de señales, que es lo que permite comprobar el criterio de
aceptación —cinco letras, cinco símbolos— en CI y no delante de una webcam.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypeAlias

from lsm.config import Config
from lsm.vocabulary import Label, spec


@dataclass(frozen=True, slots=True)
class LetterSignal:
    """Una letra que la segmentación dio por buena."""

    label: Label


@dataclass(frozen=True, slots=True)
class HandPresent:
    """Un frame con mano. Libera el cerrojo del espacio."""


@dataclass(frozen=True, slots=True)
class HandAbsent:
    """Un frame sin mano."""


@dataclass(frozen=True, slots=True)
class SpaceWritten:
    """Se cerró la palabra en curso y se abrió otra."""


#: Las señales que la demo produce. Se irá ampliando en las tareas siguientes.
Signal: TypeAlias = LetterSignal | HandPresent | HandAbsent


@dataclass(frozen=True, slots=True)
class LetterWritten:
    """Se añadió un símbolo a la palabra en curso."""

    label: Label


#: Qué acaba de pasar. `None` significa que la señal no cambió nada.
SpellingEvent: TypeAlias = LetterWritten | SpaceWritten


@dataclass(frozen=True, slots=True)
class SpellingState:
    """El buffer de deletreo. Inmutable.

    Guarda **símbolos** y no texto: el texto se deriva con `render_word` y
    `render_text`, y así el estado conserva qué se señó de verdad.
    """

    #: La palabra en curso.
    word: tuple[Label, ...] = ()
    #: Palabras ya cerradas, cada una con sus símbolos.
    finished: tuple[tuple[Label, ...], ...] = ()
    #: Frames consecutivos sin mano. Vive en el estado y no en el CLI para que el
    #: criterio del espacio sea puro y se pueda testear con una lista de señales.
    absent_frames: int = 0
    #: Ya se puso espacio por esta ausencia. Sin esto, la mano quieta abajo
    #: escribiría un espacio por frame: la mano abajo no es un evento, es un
    #: estado que dura. Es el mismo cerrojo que `pending_repeat` en
    #: `segmentation.py`, un nivel más arriba.
    space_emitted: bool = False


@dataclass(frozen=True, slots=True)
class StepResult:
    """El estado nuevo y qué pasó al llegar ahí."""

    state: SpellingState
    event: SpellingEvent | None = None


def step(
    state: SpellingState,
    signal: Signal,
    config: Config,
) -> StepResult:
    """Aplica una señal. Nunca muta `state`.

    `config` entra aunque esta versión todavía no lo use: el umbral del espacio
    llega en la tarea siguiente y la firma no debe cambiar entonces.
    """
    match signal:
        case LetterSignal(label=label):
            return _escribir_letra(state, label)
        case HandPresent():
            return StepResult(
                state=replace(state, absent_frames=0, space_emitted=False)
            )
        case HandAbsent():
            return _mano_ausente(state, config)


def _escribir_letra(state: SpellingState, label: Label) -> StepResult:
    """La clase negativa no escribe.

    `segmentation.py` filtra `UNKNOWN` y la confianza por debajo del umbral, pero
    no filtra `NONE`. Con la clase negativa acertando bien, llegará una predicción
    de `NONE` cada vez que quien firma baje la mano o transite entre letras.
    Escribirla sería poner una letra en cada transición, que es justo lo que la
    clase negativa existe para evitar.

    Se descarta aquí y no en `cli/demo.py` porque es un criterio, y los criterios
    viven en código puro; y no en `segmentation.py` porque su contrato está
    versionado y no hay motivo para tocarlo.
    """
    if label is Label.NONE:
        return StepResult(state=state)
    return StepResult(
        state=replace(state, word=(*state.word, label)),
        event=LetterWritten(label=label),
    )


def _mano_ausente(state: SpellingState, config: Config) -> StepResult:
    """Cierra la palabra cuando la ausencia deja de ser un parpadeo."""
    absent = state.absent_frames + 1
    alcanzado = absent >= config.spelling.space_after_absent_frames

    if not alcanzado or state.space_emitted:
        return StepResult(state=replace(state, absent_frames=absent))

    # El cerrojo se pone aunque no haya nada que cerrar: si no, cada frame
    # siguiente volvería a evaluar el umbral sobre una palabra vacía.
    if not state.word:
        return StepResult(
            state=replace(state, absent_frames=absent, space_emitted=True)
        )

    return StepResult(
        state=replace(
            state,
            word=(),
            finished=(*state.finished, state.word),
            absent_frames=absent,
            space_emitted=True,
        ),
        event=SpaceWritten(),
    )


def render_word(state: SpellingState) -> str:
    """La palabra en curso, tal como se escribe para una persona."""
    return _render(state.word)


def render_text(state: SpellingState) -> str:
    """Todo el texto: las palabras cerradas y la que está en curso."""
    palabras = [_render(word) for word in state.finished]
    if state.word:
        palabras.append(_render(state.word))
    return " ".join(palabras)


def _render(word: tuple[Label, ...]) -> str:
    """`LetterSpec.display` existe justo para esto: `Ñ`, `LL`, `RR`."""
    return "".join(spec(label).display.lower() for label in word)
