"""Lógica de la recolección de dataset, sin cámara y sin disco.

`src/lsm/cli/capture.py` es el envoltorio: abre la cámara, dibuja el preview y
escribe archivos. Todo lo que decide **si una grabación sirve como muestra** vive
aquí, y por eso se puede ejercitar entero en CI con secuencias sintéticas: el
criterio de aceptación de la Fase 1 —veinte muestras de una letra, features
re-derivadas idénticas— no debería depender de que haya una webcam conectada.

La separación no es simetría estética. Durante la captura se toman decisiones que
después no se pueden deshacer: una ventana inestable aceptada por error es una
muestra mala que nadie va a volver a mirar, y aparecerá en la Fase 2 como una
clase que no separa. Conviene poder probar ese criterio con un test.

Este módulo es código puro: sin OpenCV, sin MediaPipe, sin disco, sin cámara.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import pairwise
from typing import Final

from lsm.config import Config
from lsm.features import (
    ExtractionRejected,
    extract_sequence_features,
    pair_velocity,
    reference_scales,
    scale_to_pixels,
    split_valid_runs,
)
from lsm.gaps import GapPolicy, GapRejected, fill_gaps
from lsm.segmentation import (
    RejectionReason,
    WindowDynamic,
    WindowRejected,
    frames_from_ms,
    run_segmentation,
)
from lsm.types import (
    FrameSlot,
    FrameStream,
    Handedness,
    Landmark,
    Prediction,
    RawFrame,
    SampleKind,
    Sequence,
    TrajectoryChannel,
    WindowOrigin,
)

#: Versión del criterio de aceptación de una muestra.
#:
#: Se versiona aparte de `FEATURE_SPEC_VERSION` y de `SEGMENTATION_SPEC_VERSION`
#: por el mismo motivo que aquellas se versionan entre sí: cambian por razones
#: distintas. Ajustar cuánta quietud se le exige a quien graba no invalida ningún
#: modelo ni ninguna muestra ya grabada —los landmarks crudos siguen ahí—, pero sí
#: cambia qué se aceptó, y conviene poder leerlo en el archivo años después.
CAPTURE_SPEC_VERSION: Final = 2


class Rejection(StrEnum):
    """Por qué una ventana no se guarda como muestra.

    Es un tipo cerrado y no un `bool` porque quien graba necesita saber qué
    corregir: "muévete menos" y "sal del contraluz para que no se pierda la mano"
    son instrucciones distintas, y una sesión de cuarenta minutos se arruina si el
    programa solo sabe decir que no.
    """

    #: Menos frames de los que el modo exige.
    TOO_FEW_FRAMES = "TOO_FEW_FRAMES"
    #: El detector perdió la mano en algún punto de la ventana.
    HAS_GAPS = "HAS_GAPS"
    #: σ por encima de `config.quality.max_dispersion`. Solo aplica a estáticas.
    UNSTABLE = "UNSTABLE"
    #: Algún frame tiene escala degenerada (paso 4 de `feature-spec.md`).
    DEGENERATE_SCALE = "DEGENERATE_SCALE"
    #: Más frames de los que el modo admite. Solo aplica a dinámicas.
    TOO_MANY_FRAMES = "TOO_MANY_FRAMES"
    #: La trayectoria no recorrió lo suficiente. Solo aplica a dinámicas: es el
    #: espejo de `UNSTABLE`, y rechaza a la que **no** se movió.
    TRAJECTORY_TOO_SHORT = "TRAJECTORY_TOO_SHORT"
    #: Se alcanzó `capture.dynamic_max_ms` sin que la máquina de estados
    #: entregara el trazo cerrado (Bloque 4): la mano no se detuvo al terminar.
    STROKE_NOT_CLOSED = "STROKE_NOT_CLOSED"
    #: La máquina descartó el trazo por pasar de `motion_max_ms` (Bloque 4).
    STROKE_TOO_LONG = "STROKE_TOO_LONG"


#: Qué hacer ante cada rechazo, en palabras de quien está frente a la cámara.
#:
#: Vive en el núcleo puro y no en la capa de dibujo porque lo consumen dos sitios
#: —el HUD del preview y los mensajes de la terminal— y porque el objetivo es que
#: la instrucción sea accionable: "no se aceptó" no le sirve de nada a nadie a los
#: cuarenta minutos de sesión; "la mano se salió del encuadre" sí.
#:
#: Sin acentos a propósito: los dibuja la fuente de OpenCV, que es ASCII y
#: convierte cualquier `ó` en un signo de interrogación.
_INSTRUCCIONES: Final[dict[Rejection, str]] = {
    Rejection.TOO_FEW_FRAMES: "faltan frames: manten la sena un momento mas",
    Rejection.HAS_GAPS: "se perdio la mano: acercala o mejora la luz",
    Rejection.UNSTABLE: "demasiado movimiento: sosten la mano mas quieta",
    Rejection.DEGENERATE_SCALE: "mano degenerada: alejala un poco de la camara",
    Rejection.TOO_MANY_FRAMES: "grabacion demasiado larga: repite el trazo mas corto",
    Rejection.TRAJECTORY_TOO_SHORT: "falta recorrido: marca el trazo de la letra",
    Rejection.STROKE_NOT_CLOSED: "no se cerro el trazo: deten la mano al terminar",
    Rejection.STROKE_TOO_LONG: "trazo demasiado largo: hazlo de una vez y detente",
}


def explain(rejection: Rejection) -> str:
    """Qué corregir para que la siguiente ventana sí se acepte."""
    return _INSTRUCCIONES[rejection]


@dataclass(frozen=True, slots=True)
class BufferedFrame:
    """Un frame recién salido del detector, con lo que la imagen aportó.

    `luminance` viaja pegada al frame y no se recalcula después porque para
    entonces la imagen ya no existe: el buffer guarda landmarks, que ocupan
    kilobytes, no cuadros de video, que ocupan megabytes. Es la medida objetiva
    que acompaña a `light_level` en `docs/adr/0005-taxonomias-de-metadatos-de-
    captura.md`.
    """

    slot: FrameSlot
    #: Luminancia media del cuadro, en `[0, 1]`.
    luminance: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.luminance <= 1.0:
            raise ValueError(f"luminancia fuera de [0, 1]: {self.luminance}")


@dataclass
class FrameBuffer:
    """Buffer circular de los últimos frames vistos.

    En modo estático se mira su cola: quien graba sostiene la seña y pulsa la
    tecla cuando le parece, y lo que se guarda son los `static_frames` anteriores
    a la pulsación. Grabar *hacia adelante* desde la tecla obligaría a sostener la
    seña otro segundo después de haber decidido que ya estaba bien, que es justo
    cuando la mano se relaja.
    """

    capacity: int
    _frames: deque[BufferedFrame] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError(f"capacidad de buffer inválida: {self.capacity}")
        self._frames = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._frames)

    def push(self, slot: FrameSlot, luminance: float) -> None:
        self._frames.append(BufferedFrame(slot=slot, luminance=luminance))

    def tail(self, count: int) -> tuple[BufferedFrame, ...]:
        """Los últimos `count` frames, o los que haya si son menos."""
        if count < 0:
            raise ValueError(f"count debe ser ≥ 0, no {count}")
        if count == 0:
            return ()
        frames = tuple(self._frames)
        return frames[max(0, len(frames) - count) :]

    def clear(self) -> None:
        self._frames.clear()


@dataclass(frozen=True, slots=True)
class WindowQuality:
    """Qué tan buena es una ventana, y si se acepta.

    Los números se calculan **siempre que se pueda**, incluso cuando la ventana se
    rechaza: el preview los muestra en vivo para que quien graba vea acercarse el
    umbral antes de pulsar la tecla, en vez de descubrir después de veinte
    repeticiones que todas salieron temblorosas.
    """

    frame_count: int
    #: σ del `feature-spec.md` §2. `None` si no se pudo extraer nada.
    dispersion: float | None
    #: Media de las luminancias de los cuadros de la ventana.
    mean_luminance: float
    #: Escala del paso 4 en píxeles, promediada. `None` si no se pudo extraer.
    mean_scale_px: float | None
    #: Longitud de arco de τ, en unidades de mano. `None` si no se pudo extraer.
    #: Es a las dinámicas lo que `dispersion` a las estáticas.
    arc_length: float | None
    #: Lateralidad reportada, si fue la misma en toda la ventana.
    handedness: Handedness | None
    #: Motivo del rechazo, o `None` si la ventana se acepta.
    rejection: Rejection | None

    @property
    def accepted(self) -> bool:
        return self.rejection is None


def trajectory_arc_length(trajectory: TrajectoryChannel) -> float:
    """Cuánto recorrió la muñeca a lo largo de la secuencia, en unidades de mano.

    Suma de las distancias entre puntos consecutivos de τ (`feature-spec.md`
    §3.1). Como τ ya viene dividida por la escala media de la mano, el mismo trazo
    mide igual ejecutado cerca o lejos de la cámara.

    **Es longitud de arco y no desplazamiento neto, y la diferencia es la razón de
    ser de la medida.** Una Ñ y una Q son rotaciones de ida y vuelta; una Z y una X
    vuelven cerca de donde empezaron. Para todas ellas el desplazamiento neto es
    casi cero mientras que el recorrido es largo, así que medir el neto rechazaría
    justo las letras que hay que aceptar.

    El precio es que el arco **acumula el temblor del detector**: una mano quieta
    durante noventa frames suma noventa pequeñas sacudidas y no da cero. Por eso el
    umbral tiene que dejar margen sobre ese ruido de fondo y por eso
    `capture.min_trajectory_arc` se calibra con trazos reales en la Fase 2, no a
    ojo.

    Vive aquí y no en `features.py` porque no entra en ningún vector: es un
    criterio de calidad, versionado por `CAPTURE_SPEC_VERSION`. Cambiarlo no
    invalida ningún modelo ni ninguna muestra ya grabada.
    """
    total = 0.0
    for (x0, y0), (x1, y1) in pairwise(trajectory.points):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def minimum_frames(kind: SampleKind, config: Config) -> int:
    """Cuántos frames exige cada modo para aceptar una muestra."""
    if kind is SampleKind.STATIC:
        return config.capture.static_frames
    return config.capture.dynamic_min_frames


def maximum_frames(kind: SampleKind, config: Config) -> int | None:
    """Tope de frames de una muestra, o `None` si el modo no tiene.

    Una estática no lo necesita: su ventana tiene el tamaño que por construcción
    mide `static_frames`. Una dinámica sí: desde el Bloque 4 es el tope de tiempo
    de la grabación, `dynamic_max_ms`, a la tasa nominal.
    """
    if kind is SampleKind.STATIC:
        return None
    return frames_from_ms(config.capture.dynamic_max_ms, config.capture.camera_fps)


def evaluate_window(
    frames: tuple[BufferedFrame, ...], kind: SampleKind, config: Config
) -> WindowQuality:
    """Decide si una ventana de frames se guarda como muestra, y si no, por qué.

    El orden de las comprobaciones va de lo estructural a lo gradual, y no es
    indiferente: si la ventana tiene huecos, la σ calculada sobre el trozo
    superviviente describiría una muestra que no es la que se grabó, y mostrarla
    en pantalla sería peor que no mostrar nada.

    **Cada modo tiene su criterio gradual, y son opuestos.**

    - Una estática se rechaza si se movió **de más**: σ por encima de
      `quality.max_dispersion`.
    - Una dinámica se rechaza si se movió **de menos**: longitud de arco de τ por
      debajo de `capture.min_trajectory_arc`.

    Aplicarle a una dinámica el umbral de σ rechazaría exactamente las muestras
    correctas, porque en una J o una Z el movimiento *es* la seña
    (`feature-spec.md` §6.3). Pero dejarla sin ningún criterio era el otro
    extremo: una "J" en la que la mano apenas se desplazó es indistinguible de una
    "I" en el dataset, y no hay forma de detectarla después mirando el archivo.

    Los dos números se calculan siempre, y ninguno de los dos decide fuera de su
    modo: la σ de una dinámica y el arco de una estática se guardan igual porque
    sirven para depurar.
    """
    luminance = _mean_luminance(frames)
    count = len(frames)

    def rechazo(motivo: Rejection) -> WindowQuality:
        """Un rechazo estructural: no hay números que enseñar todavía."""
        return WindowQuality(
            frame_count=count,
            dispersion=None,
            mean_luminance=luminance,
            mean_scale_px=None,
            arc_length=None,
            handedness=None,
            rejection=motivo,
        )

    if count < minimum_frames(kind, config):
        return rechazo(Rejection.TOO_FEW_FRAMES)

    tope = maximum_frames(kind, config)
    if tope is not None and count > tope:
        return rechazo(Rejection.TOO_MANY_FRAMES)

    stream: FrameStream = tuple(buffered.slot for buffered in frames)
    runs = split_valid_runs(stream)
    if len(runs) == 1 and len(runs[0]) == count:
        sequence = runs[0]
    elif kind is SampleKind.DYNAMIC:
        # Bloque 2 (ADR 0021): una dinámica con huecos cortos se guarda —con los
        # huecos tal como ocurrieron— si `to_sample` podrá reconstruirla. Se mide
        # sobre la reconstruida, que es lo que verá el entrenamiento.
        reconstruida = fill_gaps(
            stream, GapPolicy.from_config(config, config.capture.camera_fps)
        )
        if isinstance(reconstruida, GapRejected):
            return rechazo(Rejection.HAS_GAPS)
        sequence = reconstruida.sequence
    else:
        return rechazo(Rejection.HAS_GAPS)

    # La mano es la declarada de la sesión, igual en todos los frames por
    # construcción (ADR 0017): ya no hay «lateralidad mixta» que rechazar.
    handedness = sequence.frames[0].handedness

    extraction = extract_sequence_features(sequence, config)
    if isinstance(extraction, ExtractionRejected):
        return WindowQuality(
            frame_count=count,
            dispersion=None,
            mean_luminance=luminance,
            mean_scale_px=None,
            arc_length=None,
            handedness=handedness,
            rejection=Rejection.DEGENERATE_SCALE,
        )

    dispersion = extraction.static.dispersion
    # El metadato de distancia sigue con muñeca → nudillo 9, con la que se
    # calibró `Distance`; desde FEATURE_SPEC 3 no es la escala de las features.
    scale_px = _mean_scale_px(sequence, reference_scales(sequence))
    arc = trajectory_arc_length(extraction.trajectory)

    rejection: Rejection | None = None
    if kind is SampleKind.STATIC:
        if dispersion > config.quality.max_dispersion:
            rejection = Rejection.UNSTABLE
    elif arc < config.capture.min_trajectory_arc:
        rejection = Rejection.TRAJECTORY_TOO_SHORT

    return WindowQuality(
        frame_count=count,
        dispersion=dispersion,
        mean_luminance=luminance,
        mean_scale_px=scale_px,
        arc_length=arc,
        handedness=handedness,
        rejection=rejection,
    )


def _mean_luminance(frames: Iterable[BufferedFrame]) -> float:
    """Media de las luminancias. Una ventana vacía es negra, no un error.

    Devolver `0.0` en vez de lanzar mantiene al preview dibujando durante el
    primer segundo de la sesión, cuando el buffer todavía se está llenando.
    """
    total = 0.0
    count = 0
    for buffered in frames:
        total += buffered.luminance
        count += 1
    return total / count if count else 0.0


def _mean_scale_px(sequence: Sequence, scales: tuple[float, ...]) -> float:
    """`mean_scale_px` de `docs/dataset-schema.md`: el paso 4, en píxeles.

    Cada frame se convierte con **su propio** alto y no con uno asumido: la
    resolución que se le pide a la cámara es una petición, no una garantía, y una
    grabación puede cambiar de tamaño si el driver decide otra cosa a media
    sesión.
    """
    total = 0.0
    for frame, scale in zip(sequence.frames, scales, strict=True):
        total += scale_to_pixels(scale, frame.height)
    return total / len(scales)


# --------------------------------------------------------------------------- #
# Preview — geometría del dibujo, sin dibujar
# --------------------------------------------------------------------------- #


def preview_position(
    landmark: Landmark, width: int, height: int, *, mirrored: bool
) -> tuple[int, int]:
    """Dónde cae un landmark en el preview, en píxeles.

    **Aquí, y solo aquí, ocurre el espejado**, y es la razón de que esta función
    exista en vez de estar embebida en el código de dibujo. `feature-spec.md` §0.3
    prohíbe alimentar al detector con la imagen espejada: si se hiciera, la
    lateralidad reportada se invertiría y el paso 2 corrompería el vector sin que
    nada fallara. De modo que el detector ve el mundo tal cual, el preview lo ve
    reflejado —que es lo que espera quien se mira en pantalla— y la diferencia se
    salva justo antes de pintar, con `x ← 1 - x`.

    `y` no se toca: los landmarks ya vienen en convención de imagen, con el eje
    apuntando hacia abajo, igual que el arreglo de píxeles.

    Los valores pueden caer fuera del cuadro y no se recortan: MediaPipe extrapola
    fuera del encuadre a propósito, y un dedo que se sale por el borde debe
    dibujarse saliéndose, no pegado al borde fingiendo que sigue dentro.
    """
    x = 1.0 - landmark.x if mirrored else landmark.x
    return (round(x * width), round(landmark.y * height))


# --------------------------------------------------------------------------- #
# Bloque 4 — la muestra dinámica la delimita la máquina de estados
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DelimitedStroke:
    """Dónde está, en el flujo, el trazo que entregó la máquina de estados.

    `start` es el primer frame del trazo y `end` el frame en que la máquina lo
    entregó (DYNAMIC_EMIT), los dos incluidos. Entre `start + stroke_frames` y
    `end` está el reposo que lo cerró: se guarda con la muestra para que
    reproducirla cierre el trazo sola, sin fabricar ningún reposo (Bloque 4).
    """

    start: int
    end: int
    #: Frames del trazo propiamente dicho: lo que recibe el clasificador.
    stroke_frames: int


def _ningun_clasificador(_window: Sequence, _origin: WindowOrigin) -> Prediction:
    """La captura no reconoce letras: solo quiere saber dónde está el trazo."""
    return Prediction.unknown()


def delimit_dynamic_stroke(
    stream: Iterable[FrameSlot], config: Config, fps: float
) -> DelimitedStroke | Rejection:
    """El primer trazo que la máquina de estados entregaría en este flujo.

    Es **la misma** máquina que usa la demo (`run_segmentation`), con la misma
    tasa, así que el trazo guardado es exactamente el segmento que el
    clasificador va a recibir en vivo (Bloque 4). Consume `stream` perezosamente
    y se detiene en cuanto hay trazo: en vivo, `stream` es la cámara, y termina
    cuando se acaba el tiempo de la grabación.

    - un trazo entregado → `DelimitedStroke`;
    - la máquina lo descartó por largo → `STROKE_TOO_LONG`;
    - `stream` se acabó sin trazo → `STROKE_NOT_CLOSED`.

    Un candidato interrumpido por un hueco largo no rechaza: la grabación sigue
    y quien firma puede volver a hacer la letra antes del tope.
    """
    for evento in run_segmentation(stream, config, _ningun_clasificador, fps=fps):
        if isinstance(evento, WindowDynamic):
            return DelimitedStroke(
                start=evento.start_frame_index,
                end=evento.frame_index,
                stroke_frames=len(evento.window.frames),
            )
        if (
            isinstance(evento, WindowRejected)
            and evento.reason is RejectionReason.DYNAMIC_TOO_LONG
        ):
            return Rejection.STROKE_TOO_LONG
    return Rejection.STROKE_NOT_CLOSED


def final_velocity(frames: FrameStream, config: Config) -> float | None:
    """La velocidad de cierre (§6.1.2) del último frame, en unidades de mano por
    segundo a la tasa nominal: si la mano seguía moviéndose al cortar.

    `None` si el flujo no tiene frames válidos suficientes para medirla.
    """
    fps = float(config.capture.camera_fps)
    k = frames_from_ms(config.segmentation.closing_window_ms, fps)
    validos = [f for f in frames if isinstance(f, RawFrame)]
    if len(validos) <= k:
        return None
    velocidad = pair_velocity(validos[-1 - k], validos[-1], config)
    return None if velocidad is None else velocidad / k * fps


def is_truncated(frames: FrameStream, config: Config) -> bool:
    """Una dinámica grabada antes del Bloque 4 que acaba con la mano en marcha."""
    velocidad = final_velocity(frames, config)
    return (
        velocidad is not None
        and velocidad >= config.segmentation.motion_threshold_per_s
    )
