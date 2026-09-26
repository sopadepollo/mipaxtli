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
from lsm.features import SequenceFeatures, extract_sequence_features
from lsm.types import FrameSlot, InvalidFrame, RawFrame
from lsm.types import Sequence as FrameSequence

#: Estado de la máquina que marca «dentro del trazo». Se compara por nombre para
#: no importar `segmentation` —el diagnóstico observa la máquina, no la usa—.
CANDIDATE_STATE: Final = "DYNAMIC_CANDIDATE"

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
        )
        self.records.append(record)
        self._previous_frame = frame
        self._previous_thumbnail = thumbnail
        return record

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
    - QUIETA: con mano y `v_t < velocity_threshold`.
    - OTRO: el resto.
    """
    pre_ms = config.diagnostics.pre_candidate_ms
    threshold = config.segmentation.velocity_threshold

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
            elif r.velocity is not None and r.velocity < threshold:
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
        repetitions=summarize_repetitions(records, contexts, events),
        skipped_duplicates=sum(r.skipped_duplicates for r in records),
    )


def summarize_repetitions(
    records: Sequence[FrameRecord],
    contexts: Sequence[Context],
    events: Sequence[SessionEvent],
) -> tuple[RepetitionSummary, ...]:
    """Una fila por **intento**: cada tramo contiguo de la misma repetición.

    Una repetición descartada con BACKSPACE y vuelta a hacer aparece dos veces,
    una por intento, cada una con sus propios cuadros y eventos. Agruparlas por
    (letra, repetición) sumaría el intento malo al bueno.
    """
    tramos: list[tuple[str, int, int, int]] = []  # letra, rep, inicio, fin
    for i, r in enumerate(records):
        if r.prompt is None or r.repetition is None:
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
            )
        )
    return tuple(filas)


# --------------------------------------------------------------------------- #
# Sesión guiada
# --------------------------------------------------------------------------- #


@dataclass
class GuidedSession:
    """Qué letra pedir y cuántas veces. `ESPACIO` avanza; `BACKSPACE` repite."""

    letters: tuple[str, ...]
    repetitions: int
    _position: int = 0
    #: Repeticiones descartadas con BACKSPACE, para el reporte.
    discarded: list[tuple[str, int]] = field(default_factory=list)

    @property
    def finished(self) -> bool:
        return self._position >= len(self.letters) * self.repetitions

    def current(self) -> tuple[str, int] | None:
        """(letra, repetición desde 1), o `None` al terminar."""
        if self.finished:
            return None
        letra = self.letters[self._position // self.repetitions]
        return letra, self._position % self.repetitions + 1

    def advance(self) -> None:
        if not self.finished:
            self._position += 1

    def discard_last(self) -> None:
        """Vuelve a pedir la repetición anterior y la anota como descartada."""
        if self._position == 0:
            return
        self._position -= 1
        actual = self.current()
        if actual is not None:
            self.discarded.append(actual)

    def instruction(self) -> str:
        actual = self.current()
        if actual is None:
            return "diagnostico terminado: pulsa q"
        letra, n = actual
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
            "velocity_threshold. Un hueco hereda el contexto del cuadro anterior."
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
        ]
    return "\n".join(partes) + "\n"


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
            ],
            (
                [p.requested, p.error or p.negotiated, "—", "—", "—", "—"]
                if p.error is not None
                else [
                    p.requested,
                    p.negotiated,
                    _fmt(p.fps, "{:.1f}"),
                    f"**{_fmt(p.fps_unique, '{:.1f}')}**",
                    _fmt(100.0 * p.duplicates / max(p.frames - 1, 1), "{:.1f}"),
                    f"{_fmt(p.interval_p50_ms, '{:.1f}')} / "
                    f"{_fmt(p.interval_p95_ms, '{:.1f}')}",
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
