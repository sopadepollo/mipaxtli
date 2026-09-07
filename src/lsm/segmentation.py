"""Máquina de estados de segmentación (`ARQUITECTURA.md` §4.2).

El video es continuo y las letras son discretas. Sin esta capa, el sistema emite
treinta letras por segundo, o letras basura mientras la mano viaja de una posición
a otra.

    IDLE ──(mano detectada)──────────────> TRACKING
    TRACKING ──(velocidad < umbral × N)──> STABLE
    STABLE ──(confianza > umbral)────────> EMIT
    EMIT ──(cooldown)────────────────────> TRACKING
    TRACKING ──(sin mano por M frames)───> IDLE

**Transiciones que el diagrama no dibuja y que hubo que decidir.** El diagrama de
§4.2 describe el camino feliz; una implementación tiene que cubrir el resto:

- `STABLE → TRACKING` cuando la mano vuelve a moverse. Sin esta arista, una
  ventana que se estabilizó y no llegó a emitir se quedaría estable para siempre.
- `STABLE → IDLE` y `EMIT → IDLE` cuando la mano desaparece. Puede desaparecer en
  cualquier estado, no solo en TRACKING.
- **Qué pasa cuando la clasificación devuelve UNKNOWN o baja confianza.** Es la
  decisión con más consecuencias de las tres. Si un rechazo no cuesta nada, la
  máquina se queda en STABLE reclasificando la misma ventana en cada frame:
  treinta llamadas por segundo al clasificador para volver a rechazarla. Si un
  rechazo cuesta lo mismo que una emisión, una letra que quedó apenas bajo el
  umbral obliga a rehacer la seña completa.

  Se resuelve con **dos cooldowns distintos**: `emit_cooldown_frames` tras emitir
  y `reject_cooldown_frames`, más corto, tras rechazar. El rechazo no cambia de
  estado —la mano sigue quieta, la ventana sigue siendo estable—, solo suspende
  la clasificación unos frames. Así el sistema reintenta pronto sin quemar CPU.
  `config.py` valida que el cooldown de rechazo no supere al de emisión.

**Regla de letras dobles.** Una letra **distinta** a la anterior se emite en cuanto
la ventana vuelve a ser estable. Para repetir la **misma** letra se exige que la
mano haya salido de STABLE desde la emisión anterior.

Deletrear "carro" o "llave" obliga a emitir dos veces seguidas la misma letra, así
que pedir movimiento entre *todas* las letras no es viable: convertiría el
deletreo normal en un ejercicio de sacudir la mano. Pedirlo solo entre letras
iguales cubre los dos casos y coincide con cómo se ejecuta la dactilología real,
donde los dobles se marcan con un rebote pequeño.

En el código el cerrojo es `pending_repeat`, que guarda la última letra emitida:

- se pone al emitir;
- se libera en cuanto un frame supera `velocity_threshold` —el rebote— o se pierde
  la mano;
- mientras esté puesto, una predicción con esa misma etiqueta produce
  `WindowRejected(REPEATED_LETTER)` en vez de una emisión.

El `emit_cooldown_frames` sigue existiendo y hace lo suyo: evitar treinta
emisiones por segundo mientras la ventana sigue estable. El cerrojo resuelve un
problema distinto, el de la repetición sostenida.

Aquí solo vive la lógica de estados. El buffer de deletreo —acumular letras en
palabras, espacio, borrado, y qué hacer con los dígrafos LL y RR— es `spelling.py`
y llega en la Fase 3.

Es una función pura sobre un flujo de frames ya detectados: no abre la cámara, no
lee disco y no importa el clasificador, que llega inyectado. Eso es lo que permite
ejercitar cada transición con secuencias sintéticas en CI.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, TypeAlias

from lsm.config import Config
from lsm.features import ExtractionRejected, extract_sequence_features
from lsm.types import FrameSlot, Prediction, RawFrame, Sequence

#: Versión del contrato de segmentación: la §6 de `docs/feature-spec.md` (cómo se
#: mide la velocidad) y la máquina de estados de este módulo.
#:
#: **Es independiente de `FEATURE_SPEC_VERSION` a propósito.** Las dos cosas
#: cambian por motivos distintos: el vector de features cambia cuando cambia lo que
#: consume el clasificador —y entonces hay que reentrenar—, mientras que la
#: segmentación cambia cuando se ajusta cómo se decide que una mano está quieta, lo
#: que no invalida ningún modelo. Acoplarlas obligaría a reentrenar cada vez que se
#: afina un umbral, que es absurdo.
#:
#: `segmentation.ts` deberá reproducir la §6 igual que `features.ts` reproduce las
#: §1 a §5. Ver `docs/adr/0004-contrato-de-segmentacion.md`.
SEGMENTATION_SPEC_VERSION: Final = 1


class State(StrEnum):
    """Estados de `ARQUITECTURA.md` §4.2.

    `EMIT` es el estado de cooldown posterior a una emisión: la letra se emite al
    entrar, y lo que dura es la espera.
    """

    IDLE = "IDLE"
    TRACKING = "TRACKING"
    STABLE = "STABLE"
    EMIT = "EMIT"


class RejectionReason(StrEnum):
    """Por qué una ventana estable no produjo letra."""

    #: σ por encima de `config.quality.max_dispersion`: la mano todavía se estaba
    #: acomodando. Se rechaza **antes** de llamar al clasificador.
    UNSTABLE_WINDOW = "UNSTABLE_WINDOW"
    #: El clasificador devolvió UNKNOWN o una confianza insuficiente.
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    #: La ventana no se pudo procesar (escala degenerada en algún frame).
    EXTRACTION_FAILED = "EXTRACTION_FAILED"
    #: La letra es la misma que la última emitida y la mano no ha salido de STABLE
    #: desde entonces. Ver la regla de letras dobles en el encabezado del módulo.
    REPEATED_LETTER = "REPEATED_LETTER"


@dataclass(frozen=True, slots=True)
class HandAcquired:
    """Apareció una mano y empezó el seguimiento."""

    frame_index: int


@dataclass(frozen=True, slots=True)
class HandLost:
    """Se perdió la mano durante `missing_frames_to_idle` frames."""

    frame_index: int


@dataclass(frozen=True, slots=True)
class StateChanged:
    """Transición de la máquina. Es lo que hace auditable el flujo."""

    frame_index: int
    previous: State
    current: State


@dataclass(frozen=True, slots=True)
class WindowStable:
    """La ventana se estabilizó y es candidata a clasificarse."""

    frame_index: int
    window: Sequence
    dispersion: float


@dataclass(frozen=True, slots=True)
class WindowRejected:
    """La ventana no produjo letra, y se dice por qué."""

    frame_index: int
    reason: RejectionReason


@dataclass(frozen=True, slots=True)
class LetterEmitted:
    """Una letra con confianza suficiente. Lo único que llega al texto."""

    frame_index: int
    prediction: Prediction
    window: Sequence


#: Eventos tipados, nunca cadenas: quien consume esto hace `match` sobre tipos y
#: el verificador atrapa el caso que se olvidó.
SegmentationEvent: TypeAlias = (
    HandAcquired
    | HandLost
    | StateChanged
    | WindowStable
    | WindowRejected
    | LetterEmitted
)

#: El clasificador entra inyectado para que este módulo no dependa de
#: `lsm.classifiers` y siga siendo puro y testeable sin modelo.
Classify: TypeAlias = Callable[[Sequence], Prediction]


def run_segmentation(
    stream: Iterable[FrameSlot],
    config: Config,
    classify: Classify,
) -> Iterator[SegmentationEvent]:
    """Recorre un flujo de frames y emite los eventos de la máquina de estados.

    `stream` puede ser finito (una grabación) o infinito (la cámara en vivo): se
    consume perezosamente, un frame a la vez, sin acumular nada más que el buffer
    circular.
    """
    settings = config.segmentation
    buffer: deque[RawFrame] = deque(maxlen=settings.buffer_size)

    state = State.IDLE
    missing = 0
    stable_run = 0
    cooldown = 0
    suppressed = 0
    #: Última letra emitida mientras la mano no ha vuelto a moverse. Cadena vacía
    #: significa que no hay cerrojo puesto y cualquier letra puede emitirse.
    pending_repeat = ""

    for index, slot in enumerate(stream):
        frame = _usable_frame(slot, settings.min_detection_score)

        if frame is None:
            # Un hueco interrumpe la secuencia: el buffer se vacía en vez de coser
            # los dos tramos (`feature-spec.md` §0.3).
            buffer.clear()
            stable_run = 0
            if state is State.IDLE:
                continue
            missing += 1
            if missing >= settings.missing_frames_to_idle:
                yield HandLost(frame_index=index)
                yield StateChanged(
                    frame_index=index, previous=state, current=State.IDLE
                )
                state = State.IDLE
                missing = 0
                cooldown = 0
                suppressed = 0
                pending_repeat = ""
            continue

        missing = 0
        buffer.append(frame)

        if state is State.IDLE:
            yield HandAcquired(frame_index=index)
            yield StateChanged(
                frame_index=index, previous=State.IDLE, current=State.TRACKING
            )
            state = State.TRACKING
            stable_run = 0
            continue

        window = Sequence(frames=tuple(buffer))
        features = extract_sequence_features(window, config)

        if isinstance(features, ExtractionRejected):
            yield WindowRejected(
                frame_index=index, reason=RejectionReason.EXTRACTION_FAILED
            )
            buffer.clear()
            stable_run = 0
            if state is State.STABLE:
                yield StateChanged(
                    frame_index=index, previous=State.STABLE, current=State.TRACKING
                )
                state = State.TRACKING
            continue

        # La velocidad se evalúa en todos los estados, también durante el cooldown
        # de EMIT: si el rebote entre dos letras iguales cayera entero dentro del
        # cooldown y no se mirara, el cerrojo de repetición no se liberaría y la
        # segunda letra quedaría bloqueada sin que quien firma pueda hacer nada.
        moving = bool(features.velocities) and (
            features.velocities[-1] >= settings.velocity_threshold
        )
        if moving:
            pending_repeat = ""

        if state is State.EMIT:
            cooldown -= 1
            if cooldown <= 0:
                yield StateChanged(
                    frame_index=index, previous=State.EMIT, current=State.TRACKING
                )
                state = State.TRACKING
                stable_run = 0
            continue

        if not features.velocities:
            # Un solo frame en el buffer: todavía no hay velocidad que medir.
            continue

        if moving:
            stable_run = 0
            if state is State.STABLE:
                yield StateChanged(
                    frame_index=index, previous=State.STABLE, current=State.TRACKING
                )
                state = State.TRACKING
                suppressed = 0
            continue

        stable_run += 1

        if state is State.TRACKING:
            if stable_run < settings.stable_frames:
                continue
            yield StateChanged(
                frame_index=index, previous=State.TRACKING, current=State.STABLE
            )
            state = State.STABLE
            suppressed = 0
            yield WindowStable(
                frame_index=index,
                window=window,
                dispersion=features.static.dispersion,
            )

        if suppressed > 0:
            suppressed -= 1
            continue

        if features.static.dispersion > config.quality.max_dispersion:
            yield WindowRejected(
                frame_index=index, reason=RejectionReason.UNSTABLE_WINDOW
            )
            suppressed = settings.reject_cooldown_frames
            continue

        prediction = classify(window)
        if prediction.is_unknown or prediction.confidence < settings.min_confidence:
            yield WindowRejected(
                frame_index=index, reason=RejectionReason.LOW_CONFIDENCE
            )
            suppressed = settings.reject_cooldown_frames
            continue

        if prediction.label == pending_repeat:
            # Misma letra que la anterior y la mano no se ha movido desde entonces:
            # se exige el rebote. Una letra distinta sí sale de inmediato.
            yield WindowRejected(
                frame_index=index, reason=RejectionReason.REPEATED_LETTER
            )
            suppressed = settings.reject_cooldown_frames
            continue

        yield LetterEmitted(frame_index=index, prediction=prediction, window=window)
        yield StateChanged(frame_index=index, previous=State.STABLE, current=State.EMIT)
        state = State.EMIT
        cooldown = settings.emit_cooldown_frames
        stable_run = 0
        pending_repeat = prediction.label


def _usable_frame(slot: FrameSlot, min_detection_score: float) -> RawFrame | None:
    """Un frame por debajo del score mínimo cuenta como ausencia de mano.

    Es el único `None` del módulo y no sale de aquí: por fuera, la ausencia de
    mano se expresa con `InvalidFrame` y con los eventos tipados.
    """
    if not isinstance(slot, RawFrame):
        return None
    if slot.detection_score < min_detection_score:
        return None
    return slot
