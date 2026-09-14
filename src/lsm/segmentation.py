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

  Se resuelve con **dos cooldowns distintos**: `emit_cooldown_ms` tras emitir
  y `reject_cooldown_ms`, más corto, tras rechazar. El rechazo no cambia de
  estado —la mano sigue quieta, la ventana sigue siendo estable—, solo suspende
  la clasificación unos frames. Así el sistema reintenta pronto sin quemar CPU.
  `config.py` valida que el cooldown de rechazo no supere al de emisión.

**Qué ventana se clasifica, y cuándo se emite** (v2 del contrato,
`docs/feature-spec.md` §6.4 y §6.6, `docs/adr/0013-la-ventana-mezclada.md`):

- La ventana es el **tramo verificado estable** —`min(max(stable_run,
  stable_frames), T_max)` frames— y no el buffer entero. La v1 comprobaba quietud
  sobre una cola del buffer y clasificaba el buffer completo, así que con un
  tránsito más corto que el buffer la ventana mezclaba dos manos y salían letras
  que nadie firmó. Ver `_stable_window`.
- La emisión es **progresiva**: desde `stable_frames` se clasifica en cada frame
  mientras la ventana crece. Por encima de `high_confidence` se emite ya; entre
  ese umbral y `min_confidence` se **acumula** —`EvidenceAccumulated`, sin
  cooldown, porque acumular no es rechazar— hasta despegarse o hasta agotar la
  ventana. De ahí sale la latencia adaptativa: rápido donde puede permitírselo,
  prudente solo donde hace falta.

**Los umbrales temporales llegan en milisegundos** y se convierten a cuadros con
la tasa de la sesión (`FrameThresholds`, §6.5). En cuadros, el mismo
`config.yaml` significaba cosas distintas en máquinas distintas: medido, esta
tubería sostiene 17.8 fps y no los 30 que suponían los comentarios, así que cada
umbral duraba 1.7 veces lo que decía.

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

El `emit_cooldown_ms` sigue existiendo y hace lo suyo: evitar treinta
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

import math
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
#:
#: **v2** (`docs/adr/0013-la-ventana-mezclada.md`): la ventana que se clasifica
#: pasa a ser el tramo verificado estable en vez del buffer entero (§6.4), se
#: añade la emisión progresiva con dos umbrales de confianza (§6.6) y los
#: umbrales temporales pasan a milisegundos derivados de la tasa medida (§6.5).
#: `FEATURE_SPEC_VERSION` **no** cambia: el promedio del §2 es el mismo, lo que
#: cambia es qué frames entran, así que ningún modelo entrenado se invalida.
SEGMENTATION_SPEC_VERSION: Final = 2


def frames_from_ms(ms: float, fps: float) -> int:
    """Convierte una duración a cuadros: `floor(ms · fps / 1000 + 0.5)`, mínimo 1.

    **La regla de redondeo es parte del contrato** y por eso se escribe explícita
    en vez de llamar a `round`: `round` de Python redondea al par más cercano
    (`round(4.5) == 4`) y `Math.round` de JavaScript redondea hacia arriba
    (`Math.round(4.5) === 5`). Con un empate exacto, `segmentation.ts` derivaría
    un cuadro distinto que Python leyendo el mismo `config.yaml`, y la
    discrepancia aparecería solo en algunas combinaciones de umbral y tasa: la
    peor clase de discrepancia para depurar. Se adopta la de JavaScript.

    El piso de un cuadro no es cosmético: un cooldown de cero cuadros no es un
    cooldown, y una ventana de cero frames no se puede clasificar.
    """
    return max(1, math.floor(ms * fps / 1000.0 + 0.5))


@dataclass(frozen=True, slots=True)
class FrameThresholds:
    """Los umbrales de `config.segmentation`, resueltos a cuadros para una tasa.

    Existe porque la configuración habla en **milisegundos** y la máquina de
    estados cuenta **cuadros**. La conversión ocurre una sola vez, antes del
    primer frame, y a partir de ahí la máquina consume enteros: dada una tasa, el
    comportamiento sigue siendo determinista bit a bit, que es lo que permite
    seguir validando contra golden vectors y lo que `segmentation.ts` tendrá que
    reproducir.

    La tasa se congela al arrancar y no se re-deriva a mitad de sesión. Que los
    umbrales cambiaran mientras alguien deletrea haría que la misma seña se
    comportara distinto según lo que la máquina estuviera haciendo un segundo
    antes, y volvería imposible reproducir una sesión a partir de su grabación.
    """

    #: La tasa con la que se resolvió, en cuadros por segundo. Se guarda para que
    #: quien lea un evento pueda saber contra qué se decidió.
    fps: float
    buffer_size: int
    stable_frames: int
    emit_cooldown_frames: int
    reject_cooldown_frames: int
    missing_frames_to_idle: int

    @classmethod
    def from_config(cls, config: Config, fps: float) -> FrameThresholds:
        if fps <= 0.0:
            msg = f"la tasa ({fps}) tiene que ser positiva"
            raise ValueError(msg)
        settings = config.segmentation
        return cls(
            fps=fps,
            buffer_size=frames_from_ms(settings.buffer_ms, fps),
            stable_frames=frames_from_ms(settings.stable_ms, fps),
            emit_cooldown_frames=frames_from_ms(settings.emit_cooldown_ms, fps),
            reject_cooldown_frames=frames_from_ms(settings.reject_cooldown_ms, fps),
            missing_frames_to_idle=frames_from_ms(settings.missing_to_idle_ms, fps),
        )


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
class EvidenceAccumulated:
    """La ventana se clasificó por encima del piso pero sin holgura para emitir.

    No es un rechazo y no cuesta cooldown: la mano sigue quieta, la ventana
    sigue creciendo y en el frame siguiente se vuelve a clasificar con un frame
    más de evidencia. Existe como evento —en vez de no decir nada— porque es lo
    que el preview enseña mientras una letra no sale: sin él, una letra que está
    acumulando y una mano que el detector no encuentra se ven exactamente igual
    en pantalla, que es el mismo argumento por el que el HUD muestra σ.
    """

    frame_index: int
    prediction: Prediction
    window: Sequence


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
    | EvidenceAccumulated
    | LetterEmitted
)

#: El clasificador entra inyectado para que este módulo no dependa de
#: `lsm.classifiers` y siga siendo puro y testeable sin modelo.
Classify: TypeAlias = Callable[[Sequence], Prediction]


def run_segmentation(
    stream: Iterable[FrameSlot],
    config: Config,
    classify: Classify,
    *,
    fps: float | None = None,
) -> Iterator[SegmentationEvent]:
    """Recorre un flujo de frames y emite los eventos de la máquina de estados.

    `stream` puede ser finito (una grabación) o infinito (la cámara en vivo): se
    consume perezosamente, un frame a la vez, sin acumular nada más que el buffer
    circular.

    `fps` es la tasa con la que se convierten a cuadros los umbrales en
    milisegundos de `config.segmentation`. **La sesión en vivo pasa la tasa
    medida**; sin ella se usa la nominal, `capture.camera_fps`, que es lo
    correcto para reproducir una grabación o para un test determinista, donde no
    hay ninguna tasa real que medir.
    """
    thresholds = FrameThresholds.from_config(config, fps or config.capture.camera_fps)
    # Dos fuentes con nombres distintos a propósito: `thresholds` son los que
    # dependen de la tasa y ya vienen convertidos a cuadros; `settings` los que
    # no dependen de ella —scores y confianzas— y se leen tal cual.
    settings = config.segmentation
    buffer: deque[RawFrame] = deque(maxlen=thresholds.buffer_size)

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
            if missing >= thresholds.missing_frames_to_idle:
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

        if state is State.TRACKING and stable_run < thresholds.stable_frames:
            continue

        # La ventana que se clasifica es el TRAMO ESTABLE, no el buffer entero.
        # Ver `_stable_window` y `docs/feature-spec.md` §6.4.
        window = _stable_window(buffer, stable_run, thresholds)
        window_features = extract_sequence_features(window, config)

        if isinstance(window_features, ExtractionRejected):
            # No debería poder ocurrir —el tramo estable es un sufijo del buffer,
            # que acaba de extraerse sin rechazo— pero el tipo lo admite y
            # tragárselo en silencio sería peor que gastar esta rama.
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

        dispersion = window_features.static.dispersion

        if state is State.TRACKING:
            yield StateChanged(
                frame_index=index, previous=State.TRACKING, current=State.STABLE
            )
            state = State.STABLE
            suppressed = 0
            yield WindowStable(
                frame_index=index,
                window=window,
                dispersion=dispersion,
            )

        if suppressed > 0:
            suppressed -= 1
            continue

        if dispersion > config.quality.max_dispersion:
            yield WindowRejected(
                frame_index=index, reason=RejectionReason.UNSTABLE_WINDOW
            )
            suppressed = thresholds.reject_cooldown_frames
            continue

        prediction = classify(window)
        if prediction.is_unknown or prediction.confidence < settings.min_confidence:
            yield WindowRejected(
                frame_index=index, reason=RejectionReason.LOW_CONFIDENCE
            )
            suppressed = thresholds.reject_cooldown_frames
            continue

        if prediction.label == pending_repeat:
            # Misma letra que la anterior y la mano no se ha movido desde entonces:
            # se exige el rebote. Una letra distinta sí sale de inmediato.
            #
            # Va ANTES de acumular a propósito: acumular evidencia de una
            # letra que el cerrojo no va a dejar salir gastaría la ventana
            # entera para terminar en este mismo rechazo.
            yield WindowRejected(
                frame_index=index, reason=RejectionReason.REPEATED_LETTER
            )
            suppressed = thresholds.reject_cooldown_frames
            continue

        if prediction.confidence < settings.high_confidence and len(
            window
        ) < _max_window(thresholds):
            # Emisión progresiva: hay evidencia suficiente para no rechazar, pero
            # no para dejar de mirar. Se acumula un frame más y se reclasifica en
            # el siguiente, SIN cooldown — acumular no es rechazar. Ver el
            # encabezado del módulo.
            yield EvidenceAccumulated(
                frame_index=index, prediction=prediction, window=window
            )
            continue

        yield LetterEmitted(frame_index=index, prediction=prediction, window=window)
        yield StateChanged(frame_index=index, previous=State.STABLE, current=State.EMIT)
        state = State.EMIT
        cooldown = thresholds.emit_cooldown_frames
        stable_run = 0
        pending_repeat = prediction.label


def _stable_window(
    buffer: deque[RawFrame], stable_run: int, thresholds: FrameThresholds
) -> Sequence:
    """El tramo verificado estable: los últimos `stable_run` frames del buffer.

        longitud = min(stable_run, T_max)   con piso en stable_frames

    y `T_max = buffer_size`, que es todo lo que el buffer circular recuerda.

    **Aquí la invariante del ADR 0004 deja de comprobarse y pasa a cumplirse por
    construcción.** No hay ningún bucle que revise que los frames de la ventana
    son estables: si la ventana *es* el tramo estable, lo son por definición, y
    el caso en que fallaría no existe. La implementación anterior comprobaba
    quietud sobre los últimos `stable_run` frames y clasificaba los
    `buffer_size` del buffer: hasta 18 frames sin comprobar, que con un tránsito
    más corto que el buffer eran la mano viajando. Ver
    `docs/adr/0013-la-ventana-mezclada.md`.

    Se toman `stable_run` frames y no `stable_run + 1`, aunque `stable_run`
    pares estables involucren un frame más: el más viejo de esos es el frame al
    que la mano **llegó** —su propia entrada pudo ser rápida— y dejarlo fuera es
    la lectura conservadora de la definición.

    Consecuencia aritmética de esa elección: como `stable_run` cuenta pares, con
    un buffer lleno de N frames hay N−1 pares, así que **el tope alcanzable es
    `buffer_size − 1`** y el frame más viejo del buffer nunca entra en la
    ventana clasificada. `T_max` sigue escrito como `buffer_size` porque es el
    límite del contrato; que no se alcance es propiedad de la definición, no un
    error de una unidad.

    El piso en `stable_frames` es redundante mientras `stable_run` no pueda ser
    menor al llegar aquí, y se escribe igual porque es parte de la definición del
    contrato, no una defensa contra un estado imposible.
    """
    longitud = min(
        max(stable_run, thresholds.stable_frames), thresholds.buffer_size, len(buffer)
    )
    return Sequence(frames=tuple(buffer)[-longitud:])


def _max_window(thresholds: FrameThresholds) -> int:
    """El tope alcanzable de la ventana de clasificación, en frames.

    `T_max` es `buffer_size`, pero `stable_run` cuenta **pares**: con un buffer
    lleno de N frames hay N−1 pares, así que la ventana nunca llega a N. Ese
    N−1 es el punto en el que ya no hay más evidencia que acumular, y por tanto
    el punto en el que una confianza media deja de esperar y emite.

    Se calcula aquí y no en línea para que «ventana agotada» tenga un solo
    sitio donde estar definida, en Python y en la §6.4 que `segmentation.ts`
    tendrá que reproducir.
    """
    return thresholds.buffer_size - 1


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
