"""Diagnóstico de pérdidas de tracking (Fase 5.1, Bloque 0).

## Por qué existe

En vivo las dinámicas fallan porque la mano se pierde a mitad del trazo, y un
hueco de un solo frame descarta el candidato entero (`feature-spec.md` §0.3,
`RejectionReason.DYNAMIC_INTERRUPTED`). Antes de tocar esa regla hay que saber
**por qué** se pierde la mano. Las hipótesis compiten y piden arreglos distintos:

- **Desenfoque por movimiento con poca luz.** La cámara alarga la exposición, la
  mano rápida sale borrosa y MediaPipe re-detecta desde cero. Si es esto, las
  pérdidas se concentran en frames rápidos y oscuros, y más luz las quita.
- **La cámara baja la tasa y repite cuadros.** Con poca luz muchas webcams pasan
  de 30 a 15 fps y entregan cada cuadro dos veces. Los duplicados no pierden la
  mano, pero duplican el salto entre cuadros reales.
- **El reloj que recibe MediaPipe no es el real.** `io/hands.py` le pasa
  marcas de tiempo nominales (`1000 / camera_fps` por cuadro); a 17.8 fps reales
  el rastreador cree que entre cuadros pasa menos tiempo del que pasa.

Este módulo registra cada cuadro de una sesión en vivo y produce el reporte que
separa esas hipótesis: tasa de detección dentro de los trazos contra mano
quieta, duración de los huecos, correlación de las pérdidas con velocidad y
luminancia, tasa real descontando duplicados y los dos relojes.

## Lo que MediaPipe no expone

La API de Tasks de MediaPipe (1.0.1) devuelve por mano `handedness`,
`hand_landmarks` y `hand_world_landmarks`, y **un solo** número de confianza: el
de la clasificación de lateralidad. Las confianzas de detección de palma y de
presencia se consumen dentro del grafo (`min_hand_detection_confidence`,
`min_hand_presence_confidence`) y no salen. Por eso se registra si hubo
detección, el motivo del frame inválido y el score de lateralidad; no hay score
de presencia que registrar.

## Pureza

Código puro, como `telemetry.py`: sin cámara, sin disco, sin `time`. Los
tiempos llegan ya medidos, los cuadros ya detectados y la miniatura ya
calculada. Quien mide y escribe es `cli/demo.py`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import pairwise
from typing import Any, Final

from lsm.config import Config
from lsm.features import (
    MIN_SCALE,
    SequenceFeatures,
    canonicalize_handedness,
    correct_aspect_and_orientation,
    extract_sequence_features,
    mean_displacement,
    reference_scale,
    smooth_sequence,
    translate_to_origin,
)
from lsm.types import FrameSlot, InvalidFrame, RawFrame
from lsm.types import Sequence as FrameSequence

#: Estado de la máquina que marca «dentro del trazo». Se compara por nombre para
#: no importar `segmentation` —el diagnóstico observa la máquina, no la usa—.
CANDIDATE_STATE: Final = "DYNAMIC_CANDIDATE"

#: Prefijo de las posturas de la prueba de reposo en `FrameRecord.prompt`.
REST_PREFIX: Final = "REPOSO_"

#: Las posturas de la prueba de reposo y lo que se le pide a quien firma.
#: Sin acentos: el HUD los dibuja con OpenCV, que no los tiene.
REST_POSES: Final[dict[str, str]] = {
    "REPOSO_ESTATICA": "una estatica (la A) con la mano quieta",
    "REPOSO_K": "la K en su posicion inicial, quieta",
    "REPOSO_X": "la X (indice en gancho), quieta",
    "REPOSO_Q": "la Q en su posicion inicial, quieta",
}

#: Nombres de los dos juegos de puntos de la prueba de reposo.
ALL_POINTS: Final = "21"
STABLE_POINTS: Final = "estables"

#: Versión del formato de `diagnostico.json`.
REPORT_SCHEMA_VERSION: Final = 1


class Context(StrEnum):
    """Dónde estaba la mano cuando llegó un cuadro."""

    #: Dentro de un trazo: en DYNAMIC_CANDIDATE, en los `pre_candidate_ms`
    #: previos a entrar, o en un hueco que empezó dentro de uno de esos dos.
    DYNAMIC = "DINAMICO"
    #: Mano quieta: la velocidad del último par válido bajo `velocity_threshold`.
    QUIET = "QUIETA"
    #: Todo lo demás: tránsitos, la mano entrando o saliendo del encuadre.
    OTHER = "OTRO"


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """Un cuadro de la sesión, reducido a lo que el diagnóstico necesita."""

    index: int
    #: Milisegundos de reloj real desde el primer cuadro.
    wall_ms: float
    #: La marca de tiempo que recibió MediaPipe para este cuadro, en ms.
    detector_ms: float
    detected: bool
    #: Motivo del frame inválido (`InvalidReason`), o `None` si hubo mano.
    invalid_reason: str | None
    #: El único score que expone MediaPipe Tasks: el de lateralidad.
    handedness_score: float | None
    handedness: str | None
    luminance: float
    #: La miniatura es idéntica a la del cuadro anterior: la cámara lo repitió.
    duplicate: bool
    #: Diferencia media absoluta con la miniatura anterior, en `[0, 1]`. Deja
    #: ver si los «casi duplicados» se agolpan cerca de cero.
    thumbnail_diff: float | None
    #: `v_t` (§6.1) entre este cuadro y el anterior, si los dos fueron válidos.
    velocity: float | None
    #: Estado de la máquina de estados **al llegar** el cuadro.
    state: str
    #: Letra pedida en la sesión guiada y número de repetición (desde 1).
    prompt: str | None = None
    repetition: int | None = None
    #: Cuadros repetidos que la cámara descartó justo antes de éste
    #: (`capture.drop_duplicate_frames`, Bloque 1).
    skipped_duplicates: int = 0
    #: Velocidad entre este cuadro y el más reciente con al menos
    #: `segmentation.velocity_window_ms` de antigüedad, sin huecos entre
    #: ellos, en unidades de mano **por segundo** de reloj real.
    velocity_window: float | None = None
    #: Solo en la prueba de reposo: (juego de puntos, ventana en ms, velocidad
    #: por segundo) para cada variante de `diagnostics.rest_windows_ms` ×
    #: {21, estables}.
    rest_variants: tuple[tuple[str, float, float], ...] = ()


@dataclass(frozen=True, slots=True)
class SessionEvent:
    """Un evento de la segmentación que el reporte necesita por repetición."""

    frame_index: int
    #: `WindowDynamic`, `DYNAMIC_INTERRUPTED`, `DYNAMIC_TOO_LONG` o `LetterEmitted`.
    kind: str
    #: La letra emitida, para `LetterEmitted`.
    label: str | None = None
    prompt: str | None = None
    repetition: int | None = None


@dataclass
class TrackingRecorder:
    """Acumula los cuadros de una sesión. La única mutación del módulo.

    Se alimenta un cuadro a la vez con `observe`, en el orden en que llegan.
    """

    config: Config
    records: list[FrameRecord] = field(default_factory=list)
    events: list[SessionEvent] = field(default_factory=list)
    _previous_frame: RawFrame | None = field(default=None, repr=False)
    _previous_thumbnail: bytes = field(default=b"", repr=False)
    _origin_ms: float | None = field(default=None, repr=False)
    #: Cuadros con mano desde el último hueco, con su reloj, para la ventana.
    _recent: list[tuple[float, RawFrame]] = field(default_factory=list, repr=False)

    def observe(
        self,
        slot: FrameSlot,
        *,
        wall_ms: float,
        detector_ms: float,
        luminance: float,
        thumbnail: bytes,
        state: str,
        prompt: str | None = None,
        repetition: int | None = None,
        skipped_duplicates: int = 0,
    ) -> FrameRecord:
        """Registra un cuadro y devuelve su registro.

        `wall_ms` es el reloj real en cualquier origen: se resta el del primer
        cuadro. La velocidad se calcula con la misma §6.1 que usa la máquina de
        estados, sobre el par (anterior, actual), y solo si los dos tuvieron mano.
        """
        if self._origin_ms is None:
            self._origin_ms = wall_ms
        frame = slot if isinstance(slot, RawFrame) else None

        velocity: float | None = None
        if frame is not None and self._previous_frame is not None:
            features = extract_sequence_features(
                FrameSequence(frames=(self._previous_frame, frame)), self.config
            )
            if isinstance(features, SequenceFeatures) and features.velocities:
                velocity = features.velocities[0]

        velocity_window = self._window_velocity(frame, wall_ms)
        rest_variants = (
            self._rest_variants(frame, wall_ms)
            if prompt is not None and prompt.startswith(REST_PREFIX)
            else ()
        )

        duplicate = bool(thumbnail) and thumbnail == self._previous_thumbnail
        diff = _thumbnail_diff(self._previous_thumbnail, thumbnail)

        record = FrameRecord(
            index=len(self.records),
            wall_ms=wall_ms - self._origin_ms,
            detector_ms=detector_ms,
            detected=frame is not None,
            invalid_reason=(
                str(slot.reason) if isinstance(slot, InvalidFrame) else None
            ),
            handedness_score=frame.handedness_score if frame is not None else None,
            # Lo que dijo MediaPipe, no la mano declarada: sus cambios de
            # opinión son lo que este diagnóstico mide (ADR 0017).
            handedness=(
                str(frame.detected_handedness or frame.handedness)
                if frame is not None
                else None
            ),
            luminance=luminance,
            duplicate=duplicate,
            thumbnail_diff=diff,
            velocity=velocity,
            state=state,
            prompt=prompt,
            repetition=repetition,
            skipped_duplicates=skipped_duplicates,
            velocity_window=velocity_window,
            rest_variants=rest_variants,
        )
        self.records.append(record)
        self._previous_frame = frame
        self._previous_thumbnail = thumbnail
        return record

    def _window_velocity(self, frame: RawFrame | None, wall_ms: float) -> float | None:
        """Desplazamiento contra el cuadro de hace una ventana, por segundo."""
        if frame is None:
            self._recent.clear()
            return None
        # La historia guarda lo necesario para la ventana más larga que se mida:
        # del más antiguo solo hace falta el más reciente que ya la cumpla.
        mas_larga = max(
            self.config.segmentation.velocity_window_ms,
            *self.config.diagnostics.rest_windows_ms,
        )
        while len(self._recent) > 1 and wall_ms - self._recent[1][0] >= mas_larga:
            self._recent.pop(0)
        base = self._base(wall_ms, self.config.segmentation.velocity_window_ms)
        self._recent.append((wall_ms, frame))
        if base is None:
            return None
        velocidad = velocity_between(base[1], frame, self.config)
        if velocidad is None:
            return None
        return velocidad * 1000.0 / (wall_ms - base[0])

    def _base(self, wall_ms: float, window_ms: float) -> tuple[float, RawFrame] | None:
        """El cuadro más reciente de la historia con al menos `window_ms`."""
        for entrada in reversed(self._recent):
            if wall_ms - entrada[0] >= window_ms:
                return entrada
        return None

    def _rest_variants(
        self, frame: RawFrame | None, wall_ms: float
    ) -> tuple[tuple[str, float, float], ...]:
        """Cada variante de la prueba de reposo para este cuadro.

        Se llama después de `_window_velocity`, que ya añadió el cuadro actual a
        la historia: la base se busca sin él.
        """
        if frame is None:
            return ()
        variantes: list[tuple[str, float, float]] = []
        estables = self.config.diagnostics.stable_landmarks
        for ventana in self.config.diagnostics.rest_windows_ms:
            base = self._base(wall_ms, ventana)
            if base is None:
                continue
            for nombre, puntos in ((ALL_POINTS, None), (STABLE_POINTS, estables)):
                v = velocity_between(base[1], frame, self.config, puntos)
                if v is not None:
                    variantes.append(
                        (nombre, ventana, v * 1000.0 / (wall_ms - base[0]))
                    )
        return tuple(variantes)

    def note(
        self,
        kind: str,
        frame_index: int,
        *,
        label: str | None = None,
        prompt: str | None = None,
        repetition: int | None = None,
    ) -> None:
        self.events.append(
            SessionEvent(
                frame_index=frame_index,
                kind=kind,
                label=label,
                prompt=prompt,
                repetition=repetition,
            )
        )


def velocity_between(
    before: RawFrame,
    after: RawFrame,
    config: Config,
    landmarks: Sequence[int] | None = None,
) -> float | None:
    """La velocidad del §6.1 entre dos frames cualesquiera, en unidades de mano.

    Con `landmarks=None` es exactamente la de `extract_sequence_features` sobre el
    par: el mismo suavizado del §4, los pasos 1 y 2 y la escala del paso 4 de
    cada frame. Con una lista de índices, el desplazamiento medio se toma solo
    sobre esos puntos; la escala sigue siendo la de la mano entera, para que las
    dos medidas estén en la misma unidad. `None` si la escala es degenerada.
    """
    suavizados = smooth_sequence(
        FrameSequence(frames=(before, after)), config.smoothing.alpha
    ).frames
    geometria: list[tuple[tuple[tuple[float, float, float], ...], float]] = []
    for f in suavizados:
        paso_2 = canonicalize_handedness(
            correct_aspect_and_orientation(f.points(), f.aspect_ratio), f.handedness
        )
        escala = reference_scale(translate_to_origin(paso_2))
        if escala < MIN_SCALE:
            return None
        geometria.append((paso_2, escala))
    (a, s_a), (b, s_b) = geometria
    if landmarks is not None:
        a = tuple(a[i] for i in landmarks)
        b = tuple(b[i] for i in landmarks)
    return mean_displacement(a, b) / ((s_a + s_b) / 2.0)


def _thumbnail_diff(before: bytes, after: bytes) -> float | None:
    """Diferencia media absoluta entre dos miniaturas, normalizada a `[0, 1]`."""
    if not before or not after or len(before) != len(after):
        return None
    total = 0
    for a, b in zip(before, after, strict=True):
        total += abs(a - b)
    return total / (255.0 * len(after))


# --------------------------------------------------------------------------- #
# Contexto de cada cuadro
# --------------------------------------------------------------------------- #


def classify_contexts(
    records: Sequence[FrameRecord], config: Config
) -> tuple[Context, ...]:
    """Asigna a cada cuadro su contexto.

    - DINAMICO: estado DYNAMIC_CANDIDATE al llegar, o a menos de
      `pre_candidate_ms` antes de un cuadro que llegó en DYNAMIC_CANDIDATE
      viniendo de otro estado (la entrada al candidato).
    - Un hueco (frames sin mano) hereda el contexto del último cuadro con mano
      antes de él, **salvo** que empezara en DYNAMIC_CANDIDATE, que es
      exactamente el caso que descarta el trazo.
    - QUIETA: con mano y velocidad bajo `velocity_threshold_per_s`, medida
      como la máquina desde la v5: contra el cuadro de hace `velocity_window_ms`,
      por segundo de reloj real (`FrameRecord.velocity_window`).
    - OTRO: el resto.
    """
    pre_ms = config.diagnostics.pre_candidate_ms
    threshold = config.segmentation.velocity_threshold_per_s

    entradas = [
        r.wall_ms
        for i, r in enumerate(records)
        if r.state == CANDIDATE_STATE
        and (i == 0 or records[i - 1].state != CANDIDATE_STATE)
    ]

    def cerca_de_una_entrada(ms: float) -> bool:
        return any(0.0 <= entrada - ms <= pre_ms for entrada in entradas)

    contextos: list[Context] = []
    heredado = Context.OTHER
    for i, r in enumerate(records):
        dinamico = r.state == CANDIDATE_STATE or cerca_de_una_entrada(r.wall_ms)
        if r.detected:
            if dinamico:
                contexto = Context.DYNAMIC
            elif r.velocity_window is not None and r.velocity_window < threshold:
                contexto = Context.QUIET
            else:
                contexto = Context.OTHER
            heredado = contexto
        else:
            inicio_de_hueco = i == 0 or records[i - 1].detected
            if inicio_de_hueco:
                heredado = Context.DYNAMIC if dinamico else heredado
            contexto = heredado
        contextos.append(contexto)
    return tuple(contextos)


# --------------------------------------------------------------------------- #
# Huecos
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Gap:
    """Una racha de cuadros sin mano, cerrada por un cuadro con mano."""

    start: int
    frames: int
    #: Del primer cuadro sin mano al primero con mano de vuelta.
    duration_ms: float
    context: Context
    #: `v_t` del último par válido antes del hueco.
    velocity_before: float | None
    luminance: float


def find_gaps(
    records: Sequence[FrameRecord], contexts: Sequence[Context]
) -> tuple[Gap, ...]:
    """Los huecos cerrados de la sesión.

    Uno que sigue abierto al terminar no tiene duración y queda fuera; uno que
    empieza en el primer cuadro no tiene «antes» y también.
    """
    huecos: list[Gap] = []
    i = 0
    while i < len(records):
        if records[i].detected or i == 0:
            i += 1
            continue
        inicio = i
        while i < len(records) and not records[i].detected:
            i += 1
        if i >= len(records):
            break
        previo = records[inicio - 1]
        luminancias = [r.luminance for r in records[inicio:i]]
        huecos.append(
            Gap(
                start=inicio,
                frames=i - inicio,
                duration_ms=records[i].wall_ms - records[inicio].wall_ms,
                context=contexts[inicio],
                velocity_before=previo.velocity,
                luminance=sum(luminancias) / len(luminancias),
            )
        )
    return tuple(huecos)


# --------------------------------------------------------------------------- #
# Estadística mínima
# --------------------------------------------------------------------------- #


def percentile(values: Sequence[float], q: float) -> float | None:
    """Rango más cercano: un valor que existe en la muestra. `None` si vacía."""
    if not values:
        return None
    ordenados = sorted(values)
    indice = min(len(ordenados) - 1, max(0, math.ceil(q * len(ordenados)) - 1))
    return ordenados[indice]


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Correlación de Pearson; `None` si alguna serie es constante o corta."""
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sxx = syy = 0.0
    for x, y in zip(xs, ys, strict=True):
        sxy += (x - mx) * (y - my)
        sxx += (x - mx) ** 2
        syy += (y - my) ** 2
    if sxx <= 0.0 or syy <= 0.0:
        return None
    return sxy / math.sqrt(sxx * syy)


@dataclass(frozen=True, slots=True)
class Bin:
    """Un cuartil de una variable y la tasa de pérdida dentro de él."""

    low: float
    high: float
    frames: int
    loss_rate: float


def quartile_bins(pairs: Sequence[tuple[float, bool]]) -> tuple[Bin, ...]:
    """Tasa de pérdida por cuartil de la variable. Sin umbrales elegidos a mano."""
    if len(pairs) < 4:
        return ()
    ordenados = sorted(pairs, key=lambda p: p[0])
    n = len(ordenados)
    bins: list[Bin] = []
    for k in range(4):
        tramo = ordenados[k * n // 4 : (k + 1) * n // 4]
        if not tramo:
            continue
        bins.append(
            Bin(
                low=tramo[0][0],
                high=tramo[-1][0],
                frames=len(tramo),
                loss_rate=sum(1 for _, perdida in tramo if perdida) / len(tramo),
            )
        )
    return tuple(bins)


# --------------------------------------------------------------------------- #
# El reporte
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ContextRate:
    frames: int
    detected: int

    @property
    def rate(self) -> float | None:
        return self.detected / self.frames if self.frames else None


@dataclass(frozen=True, slots=True)
class RepetitionSummary:
    """Qué pasó en una repetición de la sesión guiada."""

    prompt: str
    repetition: int
    frames: int
    gaps_in_stroke: int
    strokes: int
    interrupted: int
    too_long: int
    emitted: tuple[str, ...]
    #: Por cada trazo entregado (`WindowDynamic`): ms de reloj real entre el
    #: último cuadro con velocidad de ventana ≥ `motion_threshold_per_s` y la
    #: entrega. Es la espera de `motion_confirm_low_ms` tal como se vivió.
    stroke_latency_ms: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class LetterSummary:
    """Una letra de la sesión guiada, sumando sus intentos."""

    letter: str
    attempts: int
    #: Intentos con exactamente un trazo entregado.
    whole: int
    #: Intentos con más de uno: la letra se partió.
    split: int
    #: Intentos sin ninguno.
    lost: int
    stroke_latency_ms: tuple[float, ...]


def summarize_letters(
    repetitions: Sequence[RepetitionSummary],
) -> tuple[LetterSummary, ...]:
    """Trazos enteros, partidos y perdidos por letra, en orden de aparición."""
    filas: list[LetterSummary] = []
    for letra in dict.fromkeys(r.prompt for r in repetitions):
        propias = [r for r in repetitions if r.prompt == letra]
        filas.append(
            LetterSummary(
                letter=letra,
                attempts=len(propias),
                whole=sum(1 for r in propias if r.strokes == 1),
                split=sum(1 for r in propias if r.strokes > 1),
                lost=sum(1 for r in propias if r.strokes == 0),
                stroke_latency_ms=tuple(
                    ms for r in propias for ms in r.stroke_latency_ms
                ),
            )
        )
    return tuple(filas)


@dataclass(frozen=True, slots=True)
class RestSummary:
    """Una postura de la prueba de reposo: cuánto se mueve la mano quieta."""

    pose: str
    frames: int
    detected: int
    #: `v_t` de cada par consecutivo con mano, entre su intervalo real, en
    #: unidades de mano por segundo: lo que la máquina compara hoy con
    #: `velocity_threshold_per_s` (v4).
    pair_per_s: tuple[float, ...]
    #: La misma velocidad contra el cuadro de hace `velocity_window_ms`: la
    #: que usa la máquina desde la v5, con el reloj real.
    window_per_s: tuple[float, ...]
    #: Tasa real durante la postura, en cuadros por segundo.
    fps: float | None
    #: (juego de puntos, ventana en ms, velocidades por segundo) de cada
    #: variante, en el orden de `diagnostics.rest_windows_ms`.
    variants: tuple[tuple[str, float, tuple[float, ...]], ...] = ()


@dataclass(frozen=True, slots=True)
class TrackingReport:
    frames: int
    duration_ms: float
    rates: dict[Context, ContextRate]
    gaps: tuple[Gap, ...]
    #: Pérdida en el cuadro siguiente, por cuartil de `v_t` y de luminancia.
    velocity_bins: tuple[Bin, ...]
    luminance_bins: tuple[Bin, ...]
    velocity_correlation: float | None
    luminance_correlation: float | None
    #: `v_t` del último par válido antes de cada pérdida, y de todos los pares.
    velocity_before_loss: tuple[float, ...]
    velocity_all: tuple[float, ...]
    duplicates: int
    duplicate_bins: tuple[Bin, ...]
    thumbnail_diffs: tuple[float, ...]
    detector_interval_ms: float | None
    wall_interval_ms: float | None
    repetitions: tuple[RepetitionSummary, ...]
    #: Repetidos que la cámara descartó antes de MediaPipe (Bloque 1). Con
    #: descarte activo, `duplicates` debería quedar cerca de cero y este número
    #: es el que dice cuántos había.
    skipped_duplicates: int = 0
    rest: tuple[RestSummary, ...] = ()

    @property
    def fps(self) -> float | None:
        if self.duration_ms <= 0.0 or self.frames < 2:
            return None
        return (self.frames - 1) * 1000.0 / self.duration_ms

    @property
    def fps_unique(self) -> float | None:
        if self.duration_ms <= 0.0 or self.frames < 2:
            return None
        return (self.frames - 1 - self.duplicates) * 1000.0 / self.duration_ms


def analyze(
    records: Sequence[FrameRecord],
    events: Sequence[SessionEvent],
    config: Config,
) -> TrackingReport:
    """Todo el reporte, desde los registros crudos de la sesión."""
    contexts = classify_contexts(records, config)
    tasas: dict[Context, ContextRate] = {}
    for contexto in Context:
        del_contexto = [
            r for r, c in zip(records, contexts, strict=True) if c is contexto
        ]
        tasas[contexto] = ContextRate(
            frames=len(del_contexto),
            detected=sum(1 for r in del_contexto if r.detected),
        )

    # Pérdida = el cuadro siguiente a uno con mano llega sin mano.
    con_siguiente = [
        (r, not records[i + 1].detected)
        for i, r in enumerate(records[:-1])
        if r.detected
    ]
    por_velocidad = [
        (r.velocity, p) for r, p in con_siguiente if r.velocity is not None
    ]
    por_luz = [(r.luminance, p) for r, p in con_siguiente]

    antes_de_perder = tuple(
        r.velocity for r, p in con_siguiente if p and r.velocity is not None
    )
    todas = tuple(r.velocity for r in records if r.velocity is not None)

    duplicados = [(r.luminance, r.duplicate) for r in records[1:]]

    intervalos_reales = [b.wall_ms - a.wall_ms for a, b in pairwise(records)]
    intervalos_detector = [b.detector_ms - a.detector_ms for a, b in pairwise(records)]

    return TrackingReport(
        frames=len(records),
        duration_ms=records[-1].wall_ms - records[0].wall_ms if records else 0.0,
        rates=tasas,
        gaps=find_gaps(records, contexts),
        velocity_bins=quartile_bins(por_velocidad),
        luminance_bins=quartile_bins(por_luz),
        velocity_correlation=pearson(
            [v for v, _ in por_velocidad], [float(p) for _, p in por_velocidad]
        ),
        luminance_correlation=pearson(
            [lum for lum, _ in por_luz], [float(p) for _, p in por_luz]
        ),
        velocity_before_loss=antes_de_perder,
        velocity_all=todas,
        duplicates=sum(1 for r in records if r.duplicate),
        duplicate_bins=quartile_bins(duplicados),
        thumbnail_diffs=tuple(
            r.thumbnail_diff for r in records if r.thumbnail_diff is not None
        ),
        detector_interval_ms=(
            sum(intervalos_detector) / len(intervalos_detector)
            if intervalos_detector
            else None
        ),
        wall_interval_ms=(
            sum(intervalos_reales) / len(intervalos_reales)
            if intervalos_reales
            else None
        ),
        repetitions=summarize_repetitions(
            records,
            contexts,
            events,
            motion_threshold_per_s=config.segmentation.motion_threshold_per_s,
        ),
        skipped_duplicates=sum(r.skipped_duplicates for r in records),
        rest=summarize_rest(records, config.diagnostics.rest_settle_ms),
    )


def summarize_rest(
    records: Sequence[FrameRecord], settle_ms: float = 0.0
) -> tuple[RestSummary, ...]:
    """Las posturas de reposo, en el orden en que se hicieron.

    De cada postura cuenta solo el **último intento** —el tramo contiguo final;
    uno reiniciado con BACKSPACE es el que salió mal— y sin sus primeros
    `settle_ms`, en que la mano todavía se acomoda.
    """
    filas: list[RestSummary] = []
    for pose in dict.fromkeys(
        r.prompt for r in records if r.prompt and r.prompt.startswith(REST_PREFIX)
    ):
        todos = [i for i, r in enumerate(records) if r.prompt == pose]
        inicio = todos[-1]
        while inicio - 1 >= 0 and records[inicio - 1].prompt == pose:
            inicio -= 1
        origen = records[inicio].wall_ms
        propios = [
            i for i in todos if i >= inicio and records[i].wall_ms - origen >= settle_ms
        ]
        if not propios:
            continue
        pares = [
            v * 1000.0 / (records[i].wall_ms - records[i - 1].wall_ms)
            for i in propios
            if i > 0
            and i - 1 >= propios[0]
            and (v := records[i].velocity) is not None
            and records[i].wall_ms > records[i - 1].wall_ms
        ]
        ventana = tuple(
            v for i in propios if (v := records[i].velocity_window) is not None
        )
        duracion = records[propios[-1]].wall_ms - records[propios[0]].wall_ms
        variantes: dict[tuple[str, float], list[float]] = {}
        for i in propios:
            for nombre, ms, v in records[i].rest_variants:
                variantes.setdefault((nombre, ms), []).append(v)
        filas.append(
            RestSummary(
                pose=pose,
                frames=len(propios),
                detected=sum(1 for i in propios if records[i].detected),
                pair_per_s=tuple(pares),
                window_per_s=ventana,
                fps=(len(propios) - 1) * 1000.0 / duracion if duracion > 0 else None,
                variants=tuple(
                    (nombre, ms, tuple(vs))
                    for (nombre, ms), vs in sorted(
                        variantes.items(),
                        key=lambda kv: (kv[0][0] != ALL_POINTS, kv[0][1]),
                    )
                ),
            )
        )
    return tuple(filas)


def summarize_repetitions(
    records: Sequence[FrameRecord],
    contexts: Sequence[Context],
    events: Sequence[SessionEvent],
    *,
    motion_threshold_per_s: float | None = None,
) -> tuple[RepetitionSummary, ...]:
    """Una fila por **intento**: cada tramo contiguo de la misma repetición.

    Una repetición descartada con BACKSPACE y vuelta a hacer aparece dos veces,
    una por intento, cada una con sus propios cuadros y eventos. Agruparlas por
    (letra, repetición) sumaría el intento malo al bueno.
    """
    tramos: list[tuple[str, int, int, int]] = []  # letra, rep, inicio, fin
    for i, r in enumerate(records):
        if r.prompt is None or r.repetition is None or r.prompt.startswith(REST_PREFIX):
            continue
        if tramos and tramos[-1][:2] == (r.prompt, r.repetition) and tramos[-1][3] == i:
            letra, rep, inicio, _ = tramos[-1]
            tramos[-1] = (letra, rep, inicio, i + 1)
        else:
            tramos.append((r.prompt, r.repetition, i, i + 1))

    filas: list[RepetitionSummary] = []
    for prompt, repeticion, inicio, fin in tramos:
        huecos = sum(
            1
            for i in range(inicio, fin)
            if not records[i].detected
            and contexts[i] is Context.DYNAMIC
            and (i == 0 or records[i - 1].detected)
        )
        propios = [e for e in events if inicio <= e.frame_index < fin]
        filas.append(
            RepetitionSummary(
                prompt=prompt,
                repetition=repeticion,
                frames=fin - inicio,
                gaps_in_stroke=huecos,
                strokes=sum(1 for e in propios if e.kind == "WindowDynamic"),
                interrupted=sum(1 for e in propios if e.kind == "DYNAMIC_INTERRUPTED"),
                too_long=sum(1 for e in propios if e.kind == "DYNAMIC_TOO_LONG"),
                emitted=tuple(
                    e.label or "" for e in propios if e.kind == "LetterEmitted"
                ),
                stroke_latency_ms=(
                    ()
                    if motion_threshold_per_s is None
                    else _stroke_latencies(
                        records, propios, inicio, motion_threshold_per_s
                    )
                ),
            )
        )
    return tuple(filas)


def _stroke_latencies(
    records: Sequence[FrameRecord],
    events: Sequence[SessionEvent],
    start: int,
    motion_threshold_per_s: float,
) -> tuple[float, ...]:
    """Fin del trazo → entrega, en ms, por cada `WindowDynamic`."""
    latencias: list[float] = []
    for e in events:
        if e.kind != "WindowDynamic" or e.frame_index >= len(records):
            continue
        fin = next(
            (
                i
                for i in range(e.frame_index - 1, start - 1, -1)
                if (v := records[i].velocity_window) is not None
                and v >= motion_threshold_per_s
            ),
            None,
        )
        if fin is not None:
            latencias.append(records[e.frame_index].wall_ms - records[fin].wall_ms)
    return tuple(latencias)


# --------------------------------------------------------------------------- #
# Sesión guiada
# --------------------------------------------------------------------------- #


@dataclass
class GuidedSession:
    """Qué letra pedir y cuántas veces. `ESPACIO` avanza; `BACKSPACE` repite.

    Antes de las letras van las posturas de reposo, si las hay: cada una se
    arranca con ESPACIO, dura `rest_ms` y avanza sola (`tick`). Mientras no se
    arranca, sus cuadros no se atribuyen a la postura (`recording`).
    """

    letters: tuple[str, ...]
    repetitions: int
    _position: int = 0
    #: Repeticiones descartadas con BACKSPACE, para el reporte.
    discarded: list[tuple[str, int]] = field(default_factory=list)
    rest_poses: tuple[str, ...] = ()
    rest_ms: float = 5000.0
    _rest_started_ms: float | None = None

    @property
    def _total(self) -> int:
        return len(self.rest_poses) + len(self.letters) * self.repetitions

    @property
    def finished(self) -> bool:
        return self._position >= self._total

    @property
    def in_rest(self) -> bool:
        return self._position < len(self.rest_poses)

    @property
    def recording(self) -> bool:
        """Si los cuadros de ahora cuentan para lo que se está pidiendo."""
        return not self.in_rest or self._rest_started_ms is not None

    def current(self) -> tuple[str, int] | None:
        """(letra o postura, repetición desde 1), o `None` al terminar."""
        if self.finished:
            return None
        if self.in_rest:
            return self.rest_poses[self._position], 1
        posicion = self._position - len(self.rest_poses)
        letra = self.letters[posicion // self.repetitions]
        return letra, posicion % self.repetitions + 1

    def advance(self) -> None:
        if not self.finished:
            self._position += 1
            self._rest_started_ms = None

    def press_next(self, now_ms: float) -> None:
        """ESPACIO: arranca la postura de reposo, o da por hecha la repetición."""
        if self.in_rest:
            if self._rest_started_ms is None:
                self._rest_started_ms = now_ms
            return
        self.advance()

    def tick(self, now_ms: float) -> None:
        """Avanza sola la postura de reposo cuando cumple `rest_ms`."""
        if (
            self.in_rest
            and self._rest_started_ms is not None
            and now_ms - self._rest_started_ms >= self.rest_ms
        ):
            self.advance()

    def discard_last(self) -> None:
        """Vuelve a pedir la repetición anterior y la anota como descartada.

        En una postura de reposo en curso, la reinicia."""
        if self.in_rest and self._rest_started_ms is not None:
            self._rest_started_ms = None
            return
        if self._position == 0:
            return
        self._position -= 1
        self._rest_started_ms = None
        actual = self.current()
        if actual is not None:
            self.discarded.append(actual)

    def instruction(self) -> str:
        actual = self.current()
        if actual is None:
            return "diagnostico terminado: pulsa q"
        letra, n = actual
        if self.in_rest:
            que = REST_POSES.get(letra, letra)
            if self._rest_started_ms is None:
                segundos = self.rest_ms / 1000
                return f"REPOSO  pon {que}   ESPACIO: empezar ({segundos:g} s)"
            return f"REPOSO  {que}: no te muevas...   BACKSPACE: reiniciar"
        return (
            f"DIAGNOSTICO  haz la {letra}  ({n}/{self.repetitions})   "
            "ESPACIO: hecha   BACKSPACE: repetir   q: terminar"
        )


# --------------------------------------------------------------------------- #
# Salida
# --------------------------------------------------------------------------- #


def _fmt(value: float | None, pattern: str = "{:.3f}") -> str:
    return "—" if value is None else pattern.format(value)


def _q(values: Sequence[float], pattern: str = "{:.3f}") -> str:
    return " · ".join(
        f"p{int(q * 100)} {_fmt(percentile(values, q), pattern)}"
        for q in (0.1, 0.5, 0.9, 0.99)
    )


def _tabla(cabeceras: Sequence[str], filas: Iterable[Sequence[str]]) -> str:
    lineas = ["| " + " | ".join(cabeceras) + " |"]
    lineas.append("|" + "|".join("---" for _ in cabeceras) + "|")
    lineas += ["| " + " | ".join(fila) + " |" for fila in filas]
    return "\n".join(lineas)


def _tabla_bins(nombre: str, bins: Sequence[Bin], tasa: str) -> str:
    if not bins:
        return f"_Sin datos suficientes para cuartiles de {nombre}._"
    return _tabla(
        [f"cuartil de {nombre}", "cuadros", tasa],
        (
            [f"{b.low:.3f} – {b.high:.3f}", str(b.frames), f"{b.loss_rate:.3f}"]
            for b in bins
        ),
    )


def render_report(report: TrackingReport, metadata: dict[str, Any]) -> str:
    """El reporte en Markdown, para leer después de la sesión."""
    partes = ["# Diagnóstico de pérdidas de tracking", ""]
    partes += [f"- **{k}**: {v}" for k, v in metadata.items()]

    partes += [
        "",
        "## 1. Tasa de detección por contexto",
        "",
        _tabla(
            ["contexto", "cuadros", "con mano", "tasa"],
            (
                [c.value, str(t.frames), str(t.detected), _fmt(t.rate)]
                for c, t in report.rates.items()
            ),
        ),
        "",
        (
            "DINAMICO: en DYNAMIC_CANDIDATE, en los "
            f"{metadata.get('pre_candidate_ms', '?')} ms previos a entrar, o en "
            "un hueco que empezó ahí. QUIETA: mano con v_t bajo "
            "velocity_threshold_per_s. Un hueco hereda el contexto del cuadro anterior."
        ),
        "",
        "## 2. Huecos",
        "",
    ]
    for contexto in Context:
        huecos = [g for g in report.gaps if g.context is contexto]
        partes.append(
            f"- **{contexto.value}**: {len(huecos)} huecos · frames "
            f"{_q([float(g.frames) for g in huecos], '{:.0f}')} · ms "
            f"{_q([g.duration_ms for g in huecos], '{:.0f}')}"
        )
    dinamicos = [g for g in report.gaps if g.context is Context.DYNAMIC]
    if dinamicos:
        cortos = {1: 0, 2: 0, 3: 0}
        for g in dinamicos:
            if g.frames in cortos:
                cortos[g.frames] += 1
        partes.append(
            "- huecos DINAMICO de 1, 2 y 3 frames: "
            f"{cortos[1]}, {cortos[2]}, {cortos[3]} de {len(dinamicos)}"
        )

    partes += [
        "",
        "## 3. ¿Qué acompaña a una pérdida?",
        "",
        "Pérdida = un cuadro con mano seguido de uno sin mano.",
        "",
        (
            f"- correlación (Pearson) pérdida ~ velocidad: "
            f"{_fmt(report.velocity_correlation)}; pérdida ~ luminancia: "
            f"{_fmt(report.luminance_correlation)}"
        ),
        f"- v_t antes de cada pérdida: {_q(report.velocity_before_loss)}",
        f"- v_t de todos los pares: {_q(report.velocity_all)}",
        "",
        _tabla_bins("v_t", report.velocity_bins, "tasa de pérdida"),
        "",
        _tabla_bins("luminancia", report.luminance_bins, "tasa de pérdida"),
        "",
        "## 4. Tasa real y relojes",
        "",
        (
            f"- {report.frames} cuadros en {report.duration_ms / 1000.0:.1f} s: "
            f"**{_fmt(report.fps, '{:.1f}')} fps entregados, "
            f"{_fmt(report.fps_unique, '{:.1f}')} fps sin duplicados** "
            f"({report.duplicates} duplicados)"
        ),
        (
            "- repetidos descartados por la cámara antes de MediaPipe: "
            f"{report.skipped_duplicates}"
        ),
        (
            f"- intervalo real medio {_fmt(report.wall_interval_ms, '{:.1f}')} ms; "
            "intervalo que recibe MediaPipe "
            f"{_fmt(report.detector_interval_ms, '{:.1f}')} ms"
        ),
        (
            "- diferencia de miniatura entre cuadros: "
            f"{_q(report.thumbnail_diffs, '{:.4f}')}"
        ),
        "",
        _tabla_bins("luminancia", report.duplicate_bins, "fracción duplicada"),
    ]

    if report.repetitions:
        partes += [
            "",
            "## 5. Sesión guiada",
            "",
            _tabla(
                [
                    "letra",
                    "rep",
                    "cuadros",
                    "huecos en trazo",
                    "trazos",
                    "interrumpidos",
                    "demasiado largos",
                    "emitidas",
                ],
                (
                    [
                        r.prompt,
                        str(r.repetition),
                        str(r.frames),
                        str(r.gaps_in_stroke),
                        str(r.strokes),
                        str(r.interrupted),
                        str(r.too_long),
                        " ".join(r.emitted) or "—",
                    ]
                    for r in report.repetitions
                ),
            ),
            "",
            "### 5.1 Por letra",
            "",
            "**Enteros**: intentos con exactamente un trazo entregado; **partidos**: "
            "más de uno; **perdidos**: ninguno. **Latencia**: ms de reloj real entre "
            "el último cuadro en movimiento (velocidad de ventana ≥ "
            "`motion_threshold_per_s`) y la entrega del trazo, p50 / p90 / máx.",
            "",
            _tabla(
                [
                    "letra",
                    "intentos",
                    "enteros",
                    "partidos",
                    "perdidos",
                    "latencia (ms)",
                ],
                (
                    [
                        f.letter,
                        str(f.attempts),
                        str(f.whole),
                        str(f.split),
                        str(f.lost),
                        _latencias(f.stroke_latency_ms),
                    ]
                    for f in summarize_letters(report.repetitions)
                ),
            ),
        ]
    if report.rest:
        partes += _render_rest(report.rest, metadata)
    return "\n".join(partes) + "\n"


def _latencias(values: Sequence[float]) -> str:
    if not values:
        return "—"
    return (
        f"{_fmt(percentile(values, 0.5), '{:.0f}')} / "
        f"{_fmt(percentile(values, 0.9), '{:.0f}')} / {max(values):.0f}"
    )


def _q3(values: Sequence[float]) -> str:
    if not values:
        return "—"
    return (
        f"{_fmt(percentile(values, 0.5), '{:.2f}')} / "
        f"{_fmt(percentile(values, 0.95), '{:.2f}')} / {max(values):.2f}"
    )


def _sobre(values: Sequence[float], umbral: float) -> str:
    if not values:
        return "—"
    return f"{sum(1 for v in values if v >= umbral) / len(values):.2f}"


def _render_rest(rest: Sequence[RestSummary], metadata: dict[str, Any]) -> list[str]:
    quieto = float(metadata.get("segmentation.velocity_threshold_per_s", 0.0))
    movimiento = float(metadata.get("segmentation.motion_threshold_per_s", 0.0))
    ventana = metadata.get("segmentation.velocity_window_ms", "?")
    asiento = metadata.get("diagnostics.rest_settle_ms", "?")
    return [
        "",
        "## 6. Prueba de reposo",
        "",
        "Mano quieta. Velocidad en unidades de mano por segundo, p50 / p95 / máx. "
        "**Pares**: cada cuadro contra el anterior, entre su intervalo real —lo "
        f"que la máquina comparaba hasta la v4—. **Ventana**: contra el cuadro de "
        f"hace {ventana} ms, la de la v5. Cuenta el último intento de cada postura "
        f"sin sus primeros {asiento} ms. «≥ reposo» y «≥ movimiento»: fracción "
        "de pares por encima "
        f"de `velocity_threshold_per_s` = {quieto:g} y `motion_threshold_per_s` "
        f"= {movimiento:g}.",
        "",
        _tabla(
            [
                "postura",
                "cuadros",
                "con mano",
                "fps",
                "pares",
                "ventana",
                "pares ≥ reposo",
                "pares ≥ movimiento",
                "ventana ≥ reposo",
            ],
            (
                [
                    r.pose,
                    str(r.frames),
                    str(r.detected),
                    _fmt(r.fps, "{:.1f}"),
                    _q3(r.pair_per_s),
                    _q3(r.window_per_s),
                    _sobre(r.pair_per_s, quieto),
                    _sobre(r.pair_per_s, movimiento),
                    _sobre(r.window_per_s, quieto),
                ]
                for r in rest
            ),
        ),
        "",
        "Si el temblor es por cuadro, «pares» queda muy por encima de «ventana»: "
        "al medir por segundo, el ruido crece con la tasa.",
        *_render_rest_variants(rest, quieto, movimiento),
    ]


def _render_rest_variants(
    rest: Sequence[RestSummary], quieto: float, movimiento: float
) -> list[str]:
    filas = [
        [
            r.pose,
            nombre,
            f"{ms:g}",
            _q3(vs),
            _sobre(vs, movimiento),
            _sobre(vs, quieto),
        ]
        for r in rest
        for nombre, ms, vs in r.variants
    ]
    if not filas:
        return []
    return [
        "",
        "### 6.1 Variantes: puntos × ventana",
        "",
        "Misma grabación, medida de varias formas. **21**: los 21 landmarks, como "
        "la máquina hoy. **estables**: solo muñeca y nudillos "
        "(`diagnostics.stable_landmarks`). Velocidad en unidades de mano por "
        "segundo, p50 / p95 / máx. «≥ movimiento»: fracción de cuadros que no "
        f"contarían como reposo para cerrar un trazo (`motion_threshold_per_s` = "
        f"{movimiento:g}); «≥ reposo»: sobre `velocity_threshold_per_s` = {quieto:g}.",
        "",
        _tabla(
            [
                "postura",
                "puntos",
                "ventana (ms)",
                "velocidad",
                "≥ movimiento",
                "≥ reposo",
            ],
            filas,
        ),
    ]


def report_to_json(
    report: TrackingReport,
    records: Sequence[FrameRecord],
    events: Sequence[SessionEvent],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Todo, crudo incluido: el reporte se puede rehacer sin volver a grabar."""
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "metadata": metadata,
        "summary": {
            "frames": report.frames,
            "duration_ms": report.duration_ms,
            "fps": report.fps,
            "fps_unique": report.fps_unique,
            "duplicates": report.duplicates,
            "skipped_duplicates": report.skipped_duplicates,
            "detection_rate": {c.value: t.rate for c, t in report.rates.items()},
            "velocity_correlation": report.velocity_correlation,
            "luminance_correlation": report.luminance_correlation,
            "detector_interval_ms": report.detector_interval_ms,
            "wall_interval_ms": report.wall_interval_ms,
            "letters": [
                {
                    "letter": f.letter,
                    "attempts": f.attempts,
                    "whole": f.whole,
                    "split": f.split,
                    "lost": f.lost,
                    "stroke_latency_ms": list(f.stroke_latency_ms),
                }
                for f in summarize_letters(report.repetitions)
            ],
            "rest": [
                {
                    "pose": r.pose,
                    "frames": r.frames,
                    "detected": r.detected,
                    "fps": r.fps,
                    "pair_per_s": list(r.pair_per_s),
                    "window_per_s": list(r.window_per_s),
                    "variants": [
                        {"points": nombre, "window_ms": ms, "values": list(vs)}
                        for nombre, ms, vs in r.variants
                    ],
                }
                for r in report.rest
            ],
        },
        "gaps": [
            {
                "start": g.start,
                "frames": g.frames,
                "duration_ms": g.duration_ms,
                "context": g.context.value,
                "velocity_before": g.velocity_before,
                "luminance": g.luminance,
            }
            for g in report.gaps
        ],
        "records": [
            {
                "index": r.index,
                "wall_ms": r.wall_ms,
                "detector_ms": r.detector_ms,
                "detected": r.detected,
                "invalid_reason": r.invalid_reason,
                "handedness_score": r.handedness_score,
                "handedness": r.handedness,
                "luminance": r.luminance,
                "duplicate": r.duplicate,
                "thumbnail_diff": r.thumbnail_diff,
                "velocity": r.velocity,
                "state": r.state,
                "prompt": r.prompt,
                "repetition": r.repetition,
                "skipped_duplicates": r.skipped_duplicates,
                "velocity_window": r.velocity_window,
                "rest_variants": [list(v) for v in r.rest_variants],
            }
            for r in records
        ],
        "events": [
            {
                "frame_index": e.frame_index,
                "kind": e.kind,
                "label": e.label,
                "prompt": e.prompt,
                "repetition": e.repetition,
            }
            for e in events
        ],
    }


# --------------------------------------------------------------------------- #
# Sondeo de la cámara (Fase 5.1, Bloque 1)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CameraProbe:
    """Una configuración de cámara medida: cuántos cuadros y cuántos nuevos."""

    #: Lo que se pidió: backend, formato y resolución.
    requested: str
    #: Lo que el driver dijo haber aceptado, o el error si no abrió.
    negotiated: str
    frames: int
    seconds: float
    duplicates: int
    #: Milisegundos que tardó cada cuadro en el bucle, p50 y p95.
    interval_p50_ms: float | None
    interval_p95_ms: float | None
    error: str | None = None
    #: Luminancia media de los cuadros, en `[0, 1]`: con exposición manual es
    #: lo que dice si la imagen sigue siendo utilizable.
    mean_luminance: float | None = None

    @property
    def fps(self) -> float | None:
        return (
            (self.frames - 1) / self.seconds
            if self.seconds > 0 and self.frames > 1
            else None
        )

    @property
    def fps_unique(self) -> float | None:
        if self.seconds <= 0 or self.frames < 2:
            return None
        return (self.frames - 1 - self.duplicates) / self.seconds


def summarize_probe(
    requested: str,
    negotiated: str,
    wall_ms: Sequence[float],
    thumbnails: Sequence[bytes],
    luminances: Sequence[float] = (),
) -> CameraProbe:
    """Tasa entregada y tasa de cuadros nuevos de una configuración.

    Un duplicado es una miniatura idéntica a la anterior, el mismo criterio que
    `io/camera.py` usa para descartarlos.
    """
    intervalos = [b - a for a, b in pairwise(wall_ms)]
    return CameraProbe(
        requested=requested,
        negotiated=negotiated,
        frames=len(wall_ms),
        seconds=(wall_ms[-1] - wall_ms[0]) / 1000.0 if len(wall_ms) > 1 else 0.0,
        duplicates=sum(1 for a, b in pairwise(thumbnails) if a and a == b),
        interval_p50_ms=percentile(intervalos, 0.5),
        interval_p95_ms=percentile(intervalos, 0.95),
        mean_luminance=sum(luminances) / len(luminances) if luminances else None,
    )


def render_probes(probes: Sequence[CameraProbe], metadata: dict[str, Any]) -> str:
    """La tabla del sondeo, de mejor a peor por cuadros nuevos por segundo."""
    ordenados = sorted(
        probes, key=lambda p: -(p.fps_unique or 0.0) if p.error is None else 1.0
    )
    partes = ["# Sondeo de la cámara", ""]
    partes += [f"- **{k}**: {v}" for k, v in metadata.items()]
    partes += [
        "",
        _tabla(
            [
                "pedido",
                "aceptado por el driver",
                "fps entregados",
                "fps nuevos",
                "% repetidos",
                "intervalo p50 / p95 (ms)",
                "luminancia",
            ],
            (
                [p.requested, p.error or p.negotiated, "—", "—", "—", "—", "—"]
                if p.error is not None
                else [
                    p.requested,
                    p.negotiated,
                    _fmt(p.fps, "{:.1f}"),
                    f"**{_fmt(p.fps_unique, '{:.1f}')}**",
                    _fmt(100.0 * p.duplicates / max(p.frames - 1, 1), "{:.1f}"),
                    f"{_fmt(p.interval_p50_ms, '{:.1f}')} / "
                    f"{_fmt(p.interval_p95_ms, '{:.1f}')}",
                    _fmt(p.mean_luminance, "{:.2f}"),
                ]
                for p in ordenados
            ),
        ),
        "",
        (
            "«fps nuevos» es lo que importa: cuadros que no repiten al anterior. "
            "La configuración de arriba es la candidata para `capture.backend`, "
            "`capture.fourcc`, `capture.frame_width` y `capture.frame_height`."
        ),
    ]
    return "\n".join(partes) + "\n"
