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

**El camino dinámico** (v3, `docs/feature-spec.md` §6.7, ADR 0015). Una letra
dinámica nunca dispara STABLE —el movimiento *es* la seña—, así que tiene su propio
camino:

    TRACKING ──(racha de movimiento ≥ motion_min)────> DYNAMIC_CANDIDATE
    DYNAMIC_CANDIDATE ──(reposo ≥ motion_confirm_low)─> DYNAMIC_EMIT
    DYNAMIC_CANDIDATE ──(trazo > motion_max)──────────> TRACKING (descarta)
    DYNAMIC_EMIT ──(dinámico acepta)──────────────────> EMIT
    DYNAMIC_EMIT ──(dinámico rechaza)─────────────────> TRACKING

Movimiento es la misma `v_t` del §6.1 contra `motion_threshold`; no hay una
segunda métrica. Los dos caminos son **excluyentes con histéresis**: mientras dura
el candidato, STABLE está suspendido —el freno de un cambio de dirección de la Z o
del gancho de la J deja la velocidad cerca de cero un par de frames, y sin esta
suspensión se leería como letra estática—, y solo un reposo de
`motion_confirm_low_ms`, más largo que esos frenos, cierra el trazo. Al revés, una
parada que el camino estático ya llama quietud (`stable_ms`) corta la racha que
todavía no llegó a candidato: el movimiento anterior era un tránsito.

La ruta de la ventana la decide esta máquina y viaja al clasificador como
`WindowOrigin`: el registry no vuelve a medir nada.

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
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, TypeAlias

from lsm.config import Config
from lsm.features import (
    ExtractionRejected,
    extract_sequence_features,
    pair_velocity,
)
from lsm.gaps import GapPolicy, can_bridge, interpolate_frames
from lsm.one_euro import OneEuroFilter, OneEuroParams
from lsm.plausibility import PlausibilityParams, filter_stream, is_implausible
from lsm.types import (
    FrameSlot,
    InvalidReason,
    Prediction,
    RawFrame,
    Sequence,
    WindowOrigin,
)

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
#:
#: **v3** (`docs/adr/0015-el-camino-dinamico-de-la-segmentacion.md`): el camino
#: dinámico del §6.7 —DYNAMIC_CANDIDATE y DYNAMIC_EMIT—, que suspende STABLE
#: mientras dura un trazo, y el origen de la ventana como argumento del
#: clasificador. Tampoco cambia `FEATURE_SPEC_VERSION`: el trazo se extrae con
#: las mismas §1 a §3 que ya existían.
#:
#: **v4** (ADR 0017): `velocity_threshold_per_s` y `motion_threshold_per_s` están
#: en unidades de mano **por segundo** y se convierten a por cuadro con la tasa
#: congelada de la sesión. `v_t` sigue siendo la del §6.1.
#:
#: **v5** (ADR 0017): la velocidad con la que decide la máquina es la del §6.1
#: entre el cuadro actual y el de hace `velocity_window_ms`, dividida entre los
#: cuadros que los separan. El temblor de MediaPipe es por cuadro: entre pares
#: consecutivos, por segundo, crecía con la tasa. `velocity_threshold_per_s`
#: pasa a 0.55, fijado con la prueba de reposo.
#:
#: **v6** (ADR 0019): la velocidad del §6.1 se divide entre el **tamaño de
#: palma**, no entre la escala del paso 4, que se colapsa con la palma de canto.
#: El cierre de DYNAMIC_CANDIDATE se mide con su propia ventana
#: (`closing_window_ms`, 150 ms), y tras un `DYNAMIC_TOO_LONG` puede nacer otra
#: racha a los `motion_exhausted_ms` aunque la mano no haya reposado.
#:
#: **v7** (Bloque 2, ADR 0021): dentro de DYNAMIC_CANDIDATE, un hueco de como
#: mucho `dynamic_max_gap_ms` entre dos frames de la misma mano se rellena
#: interpolando los landmarks crudos (`lsm.gaps`), en vez de cortar el trazo.
SEGMENTATION_SPEC_VERSION: Final = 7


def window_velocity(
    buffer: deque[RawFrame], window_frames: int, config: Config
) -> float | None:
    """La velocidad del §6.1 contra el cuadro de hace `window_frames`, por cuadro.

    El desplazamiento `v` entre el último cuadro del buffer y el de
    `window_frames` antes, dividido entre los cuadros que los separan: en
    unidades de mano por cuadro, las mismas de los umbrales ya convertidos. Si
    el buffer todavía no llega tan atrás se usa el más antiguo que tenga. `None`
    con menos de dos cuadros.
    """
    if len(buffer) < 2:
        return None
    atras = min(window_frames, len(buffer) - 1)
    # Sin pasar por el paso 5 (FEATURE_SPEC 3): la mano de canto también se mueve.
    velocidad = pair_velocity(buffer[-1 - atras], buffer[-1], config)
    if velocidad is None:
        return None
    return velocidad / atras


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
    #: Camino dinámico (§6.7): frames móviles para volverse candidato, reposo
    #: continuado que cierra el trazo, y largo máximo del trazo.
    motion_min_frames: int
    motion_confirm_low_frames: int
    motion_max_frames: int
    #: Los dos umbrales de velocidad, de unidades de mano por segundo a por
    #: cuadro con esta tasa (v4).
    velocity_threshold: float
    motion_threshold: float
    #: Cuántos cuadros atrás está el cuadro contra el que se mide la velocidad
    #: (v5, `velocity_window_ms`).
    velocity_window_frames: int
    #: La ventana del cierre del trazo (v6, `closing_window_ms`).
    closing_window_frames: int
    #: La espera máxima tras un `DYNAMIC_TOO_LONG` (v6, `motion_exhausted_ms`).
    motion_exhausted_frames: int

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
            motion_min_frames=frames_from_ms(settings.motion_min_ms, fps),
            motion_confirm_low_frames=frames_from_ms(
                settings.motion_confirm_low_ms, fps
            ),
            motion_max_frames=frames_from_ms(settings.motion_max_ms, fps),
            velocity_threshold=settings.velocity_threshold_per_s / fps,
            motion_threshold=settings.motion_threshold_per_s / fps,
            velocity_window_frames=frames_from_ms(settings.velocity_window_ms, fps),
            closing_window_frames=frames_from_ms(settings.closing_window_ms, fps),
            motion_exhausted_frames=frames_from_ms(settings.motion_exhausted_ms, fps),
        )


class State(StrEnum):
    """Estados de `ARQUITECTURA.md` §4.2, más el camino dinámico del §6.7.

    `EMIT` es el estado de cooldown posterior a una emisión: la letra se emite al
    entrar, y lo que dura es la espera. Lo comparten los dos caminos.

    `DYNAMIC_EMIT` es **transitorio**: se entra y se sale en el mismo frame, hacia
    `EMIT` si el clasificador dinámico aceptó el trazo o hacia `TRACKING` si lo
    rechazó. Existe como estado —y no solo como evento— porque es donde se decide
    la ruta de la ventana, y el flujo de `StateChanged` tiene que dejarlo escrito.
    """

    IDLE = "IDLE"
    TRACKING = "TRACKING"
    STABLE = "STABLE"
    EMIT = "EMIT"
    DYNAMIC_CANDIDATE = "DYNAMIC_CANDIDATE"
    DYNAMIC_EMIT = "DYNAMIC_EMIT"


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
    #: El candidato dinámico superó `motion_max_ms` sin que la mano se detuviera.
    #: Se descarta **sin clasificar**: nadie tarda eso en trazar una letra.
    DYNAMIC_TOO_LONG = "DYNAMIC_TOO_LONG"
    #: Un hueco (mano perdida o escala degenerada) cortó el trazo a medias. No se
    #: cose (`feature-spec.md` §0.3): el trazo se descarta entero.
    DYNAMIC_INTERRUPTED = "DYNAMIC_INTERRUPTED"
    #: El trazo se cerró, pero más de `dynamic_max_interpolated_fraction` de sus
    #: frames eran interpolados (Bloque 2): se descarta sin clasificar.
    DYNAMIC_TOO_MUCH_INTERPOLATED = "DYNAMIC_TOO_MUCH_INTERPOLATED"
    #: El tramo estable no tiene ningún frame con la palma de frente
    #: (FEATURE_SPEC 3, paso 5): no se puede clasificar como estática.
    PALM_EDGE_ON = "PALM_EDGE_ON"


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
class WindowDynamic:
    """El trazo de un candidato dinámico terminó y va al clasificador (§6.7).

    Es el hermano de `WindowStable` en el otro camino. `window` es el trazo
    **crudo** completo —desde el frame anterior al primer par en movimiento hasta
    el último frame en movimiento—, sin el reposo que lo cerró y sin remuestrear:
    el remuestreo del §3.2 es cosa del clasificador.
    """

    frame_index: int
    window: Sequence
    #: Frames de `window` rellenados por interpolación (Bloque 2).
    interpolated_frames: int = 0
    #: Índice, en el flujo, del primer frame de `window` (Bloque 4). Con él la
    #: captura guarda exactamente el tramo que el clasificador recibe.
    start_frame_index: int = 0
    #: De los rellenados, los que sustituyen a un cuadro invalidado por
    #: plausibilidad (§0.4, ADR 0027).
    implausible_frames: int = 0


@dataclass(frozen=True, slots=True)
class GapResolved:
    """Se cerró un hueco que la máquina sostuvo (Paso 4, ADR 0029).

    Mientras dura un hueco corto en TRACKING, STABLE o DYNAMIC_CANDIDATE la
    máquina **sostiene** el último cuadro válido: no cuenta ausencia ni decide
    nada. Si la mano vuelve a tiempo, los cuadros sostenidos se sustituyen por
    interpolación (`filled`); si no, el hueco se procesa como siempre. Llega en
    el cuadro que lo resuelve, antes que cualquier otro evento de ese cuadro.
    """

    frame_index: int
    #: El estado de la máquina cuando empezó el hueco.
    state: State
    #: Cuadros sostenidos.
    frames: int
    #: De ellos, los que invalidó la plausibilidad (§0.4).
    implausible: int
    filled: bool


@dataclass(frozen=True, slots=True)
class LetterEmitted:
    """Una letra con confianza suficiente. Lo único que llega al texto."""

    frame_index: int
    prediction: Prediction
    window: Sequence
    #: De qué camino salió. Lo usa la demo para decir qué clasificador habló.
    origin: WindowOrigin = WindowOrigin.STABLE
    #: Frames de `window` rellenados por interpolación (Bloque 2; en el camino
    #: estático desde el Paso 4).
    interpolated_frames: int = 0
    #: De los rellenados, los que sustituyen a un cuadro implausible (§0.4).
    implausible_frames: int = 0


#: Eventos tipados, nunca cadenas: quien consume esto hace `match` sobre tipos y
#: el verificador atrapa el caso que se olvidó.
SegmentationEvent: TypeAlias = (
    HandAcquired
    | HandLost
    | StateChanged
    | WindowStable
    | WindowDynamic
    | WindowRejected
    | EvidenceAccumulated
    | LetterEmitted
    | GapResolved
)

#: El clasificador entra inyectado para que este módulo no dependa de
#: `lsm.classifiers` y siga siendo puro y testeable sin modelo.
#:
#: Recibe **de qué camino salió la ventana** (`WindowOrigin`). Esa decisión la
#: toma esta máquina y solo esta máquina: el registry la obedece en vez de
#: volver a medir el movimiento, que sería un segundo criterio capaz de
#: discrepar con el primero sin que nada lo detectara.
Classify: TypeAlias = Callable[[Sequence, WindowOrigin], Prediction]


@dataclass(slots=True)
class _Stroke:
    """La racha de movimiento en curso: el trazo que un candidato entregaría.

    Guarda frames **crudos** —no features— porque lo que llega al clasificador
    dinámico es una `Sequence` `(T, 21, 3)` como cualquier otra (`CLAUDE.md`,
    regla 1); el remuestreo del §3.2 lo hace él.
    """

    frames: list[RawFrame] = field(default_factory=list)
    #: Frames en movimiento dentro de la racha. No consecutivos: ver
    #: `motion_min_ms` en `config.py`.
    moving: int = 0
    #: Longitud del trazo en su último frame en movimiento. El reposo que cierra
    #: el candidato no es parte de la letra y se corta aquí.
    end: int = 0
    #: Por frame, si se rellenó por interpolación (Bloque 2).
    filled: list[bool] = field(default_factory=list)
    #: Por frame, si el relleno sustituye a un cuadro implausible (§0.4).
    implausible: list[bool] = field(default_factory=list)
    #: Índice, en el flujo, del frame de partida del trazo (Bloque 4).
    start_index: int = 0

    @property
    def active(self) -> bool:
        return bool(self.frames)

    def __len__(self) -> int:
        return len(self.frames)

    def start(self, origin: RawFrame, index: int) -> None:
        self.frames = [origin]
        self.filled = [False]
        self.implausible = [False]
        self.start_index = index
        self.moving = 0
        self.end = 1

    def add(
        self,
        frame: RawFrame,
        *,
        moving: bool,
        filled: bool = False,
        implausible: bool = False,
    ) -> None:
        self.frames.append(frame)
        self.filled.append(filled)
        self.implausible.append(implausible)
        if moving:
            self.moving += 1
            self.end = len(self.frames)

    def trace(self) -> Sequence:
        return Sequence(frames=tuple(self.frames[: self.end]))

    def interpolated(self) -> int:
        """Frames interpolados dentro de `trace()`."""
        return sum(self.filled[: self.end])

    def implausible_count(self) -> int:
        """De los interpolados, los que sustituyen a un cuadro implausible."""
        return sum(self.implausible[: self.end])

    def reset(self) -> None:
        self.frames = []
        self.filled = []
        self.implausible = []
        self.moving = 0
        self.end = 0


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
    circular y, mientras dure, el trazo del candidato dinámico — acotado por
    `motion_max_ms`.

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
    #: Por frame del buffer: (rellenado, sustituye a un implausible). Paso 4: la
    #: ventana estable también puede llevar rellenados.
    marcas: deque[tuple[bool, bool]] = deque(maxlen=thresholds.buffer_size)

    state = State.IDLE
    missing = 0
    stable_run = 0
    cooldown = 0
    suppressed = 0
    #: Última letra emitida mientras la mano no ha vuelto a moverse. Cadena vacía
    #: significa que no hay cerrojo puesto y cualquier letra puede emitirse.
    pending_repeat = ""

    # -- Camino dinámico (§6.7) ------------------------------------------------
    #: La racha de movimiento en curso, desde su primer par en movimiento.
    stroke = _Stroke()
    #: Frames consecutivos con `v_t < motion_threshold`: lo que cierra un trazo.
    low_run = 0
    #: Frames consecutivos con `v_t < velocity_threshold`, en cualquier estado.
    #: Es la quietud que el camino estático llamaría parada, y parte la racha.
    still_run = 0
    #: Tras descartar un candidato por largo, no nace otra racha hasta que la
    #: mano repose: sin esto, alguien gesticulando entraría y saldría de
    #: DYNAMIC_CANDIDATE una y otra vez.
    exhausted = False
    #: Cuadro en que se puso `exhausted`: a los `motion_exhausted_frames` se
    #: quita solo (v6), sin esperar el reposo que acababa de faltar.
    exhausted_at = 0
    #: Tras emitir una dinámica, su pose final no es una letra nueva: la J acaba
    #: en la mano de la I, la LL en la de la L. Mientras la mano no se mueva, el
    #: camino estático no promueve a STABLE. Se libera como `pending_repeat`.
    dynamic_lock = False

    # -- Huecos dentro de un trazo (Bloque 2) -----------------------------------
    politica = GapPolicy.from_config(config, thresholds.fps)

    # -- Plausibilidad (§0.4, FEATURE_SPEC 4) ----------------------------------
    # Antes que todo lo demás: un cuadro imposible llega a la máquina como un
    # hueco, con `IMPLAUSIBLE`, y se rellena o interrumpe como cualquier otro.
    stream = filter_stream(
        stream, PlausibilityParams.from_config(config), thresholds.fps
    )

    # -- One Euro (§4) -----------------------------------------------------------
    # Sobre cada frame que llega a la máquina, rellenados incluidos, con el tiempo
    # de su marca o del índice a la tasa congelada. Un hueco que no se rellena
    # corta la secuencia y con ella el estado del filtro.
    suavizado = OneEuroFilter(OneEuroParams.from_config(config))
    paso_ms = 1000.0 / thresholds.fps

    #: Hueco más largo que se sostiene, en cuadros, según el estado en que
    #: empieza (Paso 4, ADR 0029). En IDLE y EMIT no se sostiene.
    def _limite(ms: float) -> int:
        return 0 if ms == 0.0 else frames_from_ms(ms, thresholds.fps)

    limites = {
        State.DYNAMIC_CANDIDATE: politica.max_gap_frames,
        State.TRACKING: _limite(settings.tracking_max_gap_ms),
        State.STABLE: _limite(settings.stable_max_gap_ms),
    }
    #: Los huecos sostenidos que el generador cerró, para cederlos como eventos.
    resueltos: list[GapResolved] = []

    def con_huecos_rellenos() -> Iterator[tuple[int, FrameSlot, bool, bool]]:
        """El flujo, con los huecos cortos ya rellenos (Bloque 2, Paso 4).

        Si se pierde la mano en TRACKING, STABLE o DYNAMIC_CANDIDATE, los frames
        inválidos se **sostienen**: se retienen sin procesar, y la máquina se
        queda en el último cuadro válido. Si la mano vuelve antes de pasar el
        límite del estado en que empezó el hueco y se puede interpolar
        (`gaps.can_bridge`), se ceden los frames interpolados —con el índice de
        los que reemplazan— y después el real; si no, se ceden los inválidos tal
        cual y la máquina procesa el hueco como siempre. Mientras hay un hueco
        abierto la máquina va hasta ese límite por detrás.

        Lee `state` en cada paso: el cuerpo del bucle ya procesó el cuadro
        anterior cuando se le pide el siguiente. Cede `(índice, slot, relleno,
        implausible)`: el último dice si el relleno sustituye a un cuadro que
        invalidó la plausibilidad.
        """
        retenidos: list[tuple[int, FrameSlot]] = []
        limite = 0
        estado_hueco = State.IDLE
        ultimo: RawFrame | None = None

        def cerrar(indice: int, *, relleno: bool) -> None:
            resueltos.append(
                GapResolved(
                    frame_index=indice,
                    state=estado_hueco,
                    frames=len(retenidos),
                    implausible=sum(1 for _, x in retenidos if is_implausible(x)),
                    filled=relleno,
                )
            )

        for i, s in enumerate(stream):
            usable = _usable_frame(s, settings.min_detection_score)
            if usable is None:
                if not retenidos and ultimo is not None and limites.get(state, 0):
                    limite = limites[state]
                    estado_hueco = state
                if retenidos or limite:
                    retenidos.append((i, s))
                    if len(retenidos) > limite:
                        cerrar(i, relleno=False)
                        yield from ((k, x, False, False) for k, x in retenidos)
                        retenidos = []
                        limite = 0
                        ultimo = None
                    continue
                ultimo = None
                yield i, s, False, False
                continue
            if retenidos:
                if ultimo is not None and can_bridge(ultimo, usable):
                    cerrar(i, relleno=True)
                    rellenos = interpolate_frames(
                        ultimo,
                        usable,
                        len(retenidos),
                        tuple(x.timestamp_ms for _, x in retenidos),
                    )
                    for (k, x), relleno in zip(retenidos, rellenos, strict=True):
                        yield k, relleno, True, is_implausible(x)
                else:
                    cerrar(i, relleno=False)
                    yield from ((k, x, False, False) for k, x in retenidos)
                retenidos = []
                limite = 0
            ultimo = usable
            yield i, s, False, False
        if retenidos:
            cerrar(retenidos[-1][0], relleno=False)
        yield from ((k, x, False, False) for k, x in retenidos)

    for index, slot, relleno, implausible in con_huecos_rellenos():
        while resueltos:
            yield resueltos.pop(0)
        frame = _usable_frame(slot, settings.min_detection_score)
        if frame is None:
            suavizado.reset()
        else:
            frame = suavizado.step(
                frame,
                frame.timestamp_ms
                if frame.timestamp_ms is not None
                else index * paso_ms,
            )

        if frame is None:
            # Un hueco interrumpe la secuencia: el buffer se vacía en vez de coser
            # los dos tramos (`feature-spec.md` §0.3). Por lo mismo, el trazo en
            # curso se descarta entero: coserlo inventaría un movimiento que
            # nadie observó.
            buffer.clear()
            marcas.clear()
            stable_run = 0
            low_run = 0
            still_run = 0
            exhausted = False
            stroke.reset()
            if state is State.DYNAMIC_CANDIDATE:
                yield WindowRejected(
                    frame_index=index, reason=RejectionReason.DYNAMIC_INTERRUPTED
                )
                yield StateChanged(
                    frame_index=index,
                    previous=State.DYNAMIC_CANDIDATE,
                    current=State.TRACKING,
                )
                state = State.TRACKING
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
                dynamic_lock = False
            continue

        missing = 0
        buffer.append(frame)
        marcas.append((relleno, implausible))

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

        # Una ventana con la palma de canto (FEATURE_SPEC 3) no es un fallo
        # estructural: no se puede clasificar como estática, pero la mano está y
        # se mueve. Solo la escala degenerada reinicia la máquina.
        if (
            isinstance(features, ExtractionRejected)
            and features.reason is not InvalidReason.PALM_EDGE_ON
        ):
            yield WindowRejected(
                frame_index=index, reason=RejectionReason.EXTRACTION_FAILED
            )
            buffer.clear()
            marcas.clear()
            stable_run = 0
            low_run = 0
            still_run = 0
            stroke.reset()
            if state in (State.STABLE, State.DYNAMIC_CANDIDATE):
                yield StateChanged(
                    frame_index=index, previous=state, current=State.TRACKING
                )
                state = State.TRACKING
            continue

        # La velocidad se evalúa en todos los estados, también durante el cooldown
        # de EMIT: si el rebote entre dos letras iguales cayera entero dentro del
        # cooldown y no se mirara, el cerrojo de repetición no se liberaría y la
        # segunda letra quedaría bloqueada sin que quien firma pueda hacer nada.
        velocity = window_velocity(buffer, thresholds.velocity_window_frames, config)
        moving = velocity is not None and velocity >= thresholds.velocity_threshold
        motion = velocity is not None and velocity >= thresholds.motion_threshold
        if moving:
            pending_repeat = ""
            dynamic_lock = False
        if (
            state is State.DYNAMIC_CANDIDATE
            and velocity is not None
            and thresholds.closing_window_frames != thresholds.velocity_window_frames
        ):
            # El cierre del trazo (§6.1.2, v6): dentro del candidato, el reposo
            # que lo cierra se mide con su propia ventana. Solo cambia `motion`:
            # `moving`, y con él `still_run` y STABLE, siguen con `velocity`.
            cierre = window_velocity(buffer, thresholds.closing_window_frames, config)
            motion = cierre is not None and cierre >= thresholds.motion_threshold
        if exhausted and index - exhausted_at >= thresholds.motion_exhausted_frames:
            # Tercera salida de la cascada (v6): una espera fija que no depende
            # del reposo que acaba de faltar ni de que la mano se vaya.
            exhausted = False

        # La racha de movimiento también avanza en todos los estados salvo IDLE:
        # un trazo que empieza durante el cooldown de la letra anterior sigue
        # siendo el mismo trazo cuando el cooldown acaba.
        if velocity is not None:
            if motion:
                low_run = 0
                if not stroke.active and not exhausted:
                    # El trazo arranca en el frame ANTERIOR al primer par en
                    # movimiento: ese es el punto de partida del recorrido, y
                    # sin él τ tendría su origen ya desplazado.
                    # `buffer[-2]` es el frame anterior del flujo: fuera de
                    # un candidato no hay huecos rellenados que desplacen índices.
                    stroke.start(buffer[-2], index - 1)
                if stroke.active:
                    stroke.add(
                        frame, moving=True, filled=relleno, implausible=implausible
                    )
            else:
                low_run += 1
                if stroke.active:
                    stroke.add(
                        frame, moving=False, filled=relleno, implausible=implausible
                    )
                if low_run >= thresholds.motion_confirm_low_frames:
                    exhausted = False
                    if state is not State.DYNAMIC_CANDIDATE:
                        stroke.reset()
            still_run = 0 if moving else still_run + 1
            if (
                still_run >= thresholds.stable_frames
                and state is not State.DYNAMIC_CANDIDATE
            ):
                # La mano se detuvo tanto como para que el camino estático lo
                # llame parada: el movimiento anterior fue un tránsito, no el
                # principio de un trazo. Sin este corte, los tránsitos cortos de
                # un deletreo rápido se sumarían entre sí hasta parecer uno largo.
                stroke.reset()

        if state is State.EMIT:
            cooldown -= 1
            if cooldown <= 0:
                yield StateChanged(
                    frame_index=index, previous=State.EMIT, current=State.TRACKING
                )
                state = State.TRACKING
                stable_run = 0
            continue

        if velocity is None:
            # Un solo frame en el buffer: todavía no hay velocidad que medir.
            continue

        if state is State.DYNAMIC_CANDIDATE:
            # STABLE está suspendido: el freno de un cambio de dirección no es
            # una letra estática. `stable_run` se sigue contando para que, si el
            # trazo se rechaza, el camino estático retome sin esperar de nuevo.
            stable_run = 0 if moving else stable_run + 1

            if len(stroke) > thresholds.motion_max_frames:
                yield WindowRejected(
                    frame_index=index, reason=RejectionReason.DYNAMIC_TOO_LONG
                )
                yield StateChanged(
                    frame_index=index,
                    previous=State.DYNAMIC_CANDIDATE,
                    current=State.TRACKING,
                )
                state = State.TRACKING
                stroke.reset()
                exhausted = low_run < thresholds.motion_confirm_low_frames
                exhausted_at = index
                continue

            if low_run < thresholds.motion_confirm_low_frames:
                continue

            trazo = stroke.trace()
            interpolados = stroke.interpolated()
            implausibles = stroke.implausible_count()
            inicio_trazo = stroke.start_index
            stroke.reset()
            if interpolados > politica.max_fraction * len(trazo.frames):
                # Demasiado reconstruido (Bloque 2): como el trazo demasiado
                # largo, se descarta sin llegar al clasificador.
                yield WindowRejected(
                    frame_index=index,
                    reason=RejectionReason.DYNAMIC_TOO_MUCH_INTERPOLATED,
                )
                yield StateChanged(
                    frame_index=index,
                    previous=State.DYNAMIC_CANDIDATE,
                    current=State.TRACKING,
                )
                state = State.TRACKING
                continue
            yield StateChanged(
                frame_index=index,
                previous=State.DYNAMIC_CANDIDATE,
                current=State.DYNAMIC_EMIT,
            )
            yield WindowDynamic(
                frame_index=index,
                window=trazo,
                interpolated_frames=interpolados,
                start_frame_index=inicio_trazo,
                implausible_frames=implausibles,
            )

            prediction = classify(trazo, WindowOrigin.DYNAMIC)
            if prediction.is_unknown or prediction.confidence < settings.min_confidence:
                # Un tránsito largo entre dos letras acaba aquí, y es lo normal:
                # la mano ya reposa sobre la letra siguiente y `stable_run`
                # conserva ese reposo, así que el camino estático la toma en el
                # frame siguiente en vez de esperar otra vez.
                yield WindowRejected(
                    frame_index=index, reason=RejectionReason.LOW_CONFIDENCE
                )
                yield StateChanged(
                    frame_index=index,
                    previous=State.DYNAMIC_EMIT,
                    current=State.TRACKING,
                )
                state = State.TRACKING
                continue

            yield LetterEmitted(
                frame_index=index,
                prediction=prediction,
                window=trazo,
                origin=WindowOrigin.DYNAMIC,
                interpolated_frames=interpolados,
                implausible_frames=implausibles,
            )
            yield StateChanged(
                frame_index=index, previous=State.DYNAMIC_EMIT, current=State.EMIT
            )
            state = State.EMIT
            cooldown = thresholds.emit_cooldown_frames
            stable_run = 0
            pending_repeat = prediction.label
            dynamic_lock = True
            continue

        if moving:
            stable_run = 0
            if state is State.STABLE:
                yield StateChanged(
                    frame_index=index, previous=State.STABLE, current=State.TRACKING
                )
                state = State.TRACKING
                suppressed = 0
            if stroke.moving >= thresholds.motion_min_frames:
                yield StateChanged(
                    frame_index=index,
                    previous=State.TRACKING,
                    current=State.DYNAMIC_CANDIDATE,
                )
                state = State.DYNAMIC_CANDIDATE
            continue

        stable_run += 1

        if dynamic_lock:
            # La pose en la que termina una dinámica no es una letra nueva. Ver
            # `dynamic_lock` arriba.
            continue

        if state is State.TRACKING and stable_run < thresholds.stable_frames:
            continue

        # La ventana que se clasifica es el TRAMO ESTABLE, no el buffer entero.
        # Ver `_stable_window` y `docs/feature-spec.md` §6.4.
        window = _stable_window(buffer, stable_run, thresholds)
        window_features = extract_sequence_features(window, config)

        if (
            isinstance(window_features, ExtractionRejected)
            and window_features.reason is InvalidReason.PALM_EDGE_ON
        ):
            # El tramo estable tiene la palma de canto de principio a fin
            # (FEATURE_SPEC 3): no hay rotación con qué clasificarlo. No se
            # reinicia nada; la mano sigue y la máquina espera otra ventana.
            yield WindowRejected(frame_index=index, reason=RejectionReason.PALM_EDGE_ON)
            continue

        if isinstance(window_features, ExtractionRejected):
            # No debería poder ocurrir —el tramo estable es un sufijo del buffer,
            # que acaba de extraerse sin rechazo— pero el tipo lo admite y
            # tragárselo en silencio sería peor que gastar esta rama.
            yield WindowRejected(
                frame_index=index, reason=RejectionReason.EXTRACTION_FAILED
            )
            buffer.clear()
            marcas.clear()
            stable_run = 0
            if state is State.STABLE:
                yield StateChanged(
                    frame_index=index, previous=State.STABLE, current=State.TRACKING
                )
                state = State.TRACKING
            continue

        dispersion = window_features.static.dispersion
        rellenados = list(marcas)[-len(window) :]
        interpolados_ventana = sum(1 for r, _ in rellenados if r)
        implausibles_ventana = sum(1 for _, i in rellenados if i)

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

        if interpolados_ventana > politica.max_fraction * len(window):
            # Demasiado reconstruida para clasificarla todavía (Paso 4): no es
            # un rechazo ni cuesta cooldown; la ventana crece y se vuelve a mirar.
            continue

        if dispersion > config.quality.max_dispersion:
            yield WindowRejected(
                frame_index=index, reason=RejectionReason.UNSTABLE_WINDOW
            )
            suppressed = thresholds.reject_cooldown_frames
            continue

        prediction = classify(window, WindowOrigin.STABLE)
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

        yield LetterEmitted(
            frame_index=index,
            prediction=prediction,
            window=window,
            interpolated_frames=interpolados_ventana,
            implausible_frames=implausibles_ventana,
        )
        yield StateChanged(frame_index=index, previous=State.STABLE, current=State.EMIT)
        state = State.EMIT
        cooldown = thresholds.emit_cooldown_frames
        stable_run = 0
        pending_repeat = prediction.label

    # Un hueco sostenido que el flujo dejó abierto al acabarse.
    yield from resueltos


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
