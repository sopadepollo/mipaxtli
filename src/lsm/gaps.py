"""Tolerancia a huecos en el camino dinámico (Fase 5.1, Bloque 2, ADR 0021).

La regla general del contrato es que un frame inválido **interrumpe** la
secuencia: coser dos tramos inventaría un movimiento que nadie observó
(`feature-spec.md` §0.3). En vivo esa regla descartaba trazos dinámicos enteros
por perder la mano uno o dos cuadros: la mitad de los huecos dentro de un trazo
duran ≤ 100 ms (ADR 0021).

La excepción, **solo para el camino dinámico**: un hueco corto entre dos frames
válidos se rellena interpolando linealmente los **landmarks crudos** —antes del
paso 1—, para que todos los canales derivados (forma, τ, y los del Bloque 3)
salgan consistentes entre sí. Condiciones:

- el hueco dura como mucho `max_gap_frames` cuadros; uno más largo no se rellena;
- la mano y la resolución coinciden en los dos extremos;
- los huecos al principio y al final de la secuencia no se rellenan: se recortan;
- si la fracción de frames interpolados supera `max_fraction`, la secuencia se
  rechaza.

En el camino estático no cambia nada: los inválidos siguen interrumpiendo.

Código puro, como `features.py`: sin cámara ni disco. `segmentation.py` lo usa
cuadro a cuadro dentro de DYNAMIC_CANDIDATE, y `io/dataset.py` para las muestras
dinámicas guardadas con huecos.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from lsm.config import Config
from lsm.types import FrameStream, InvalidFrame, Landmark, RawFrame, Sequence


class GapRejection(StrEnum):
    """Por qué una secuencia dinámica con huecos no se puede reconstruir."""

    #: Un hueco interno dura más de `max_gap_frames`.
    GAP_TOO_LONG = "GAP_TOO_LONG"
    #: La mano o la resolución cambian a través del hueco: interpolar entre dos
    #: frames que no son la misma mano en el mismo encuadre no tiene sentido.
    HAND_CHANGED = "HAND_CHANGED"
    #: Los frames interpolados superan `max_fraction` de la secuencia.
    TOO_MUCH_INTERPOLATED = "TOO_MUCH_INTERPOLATED"
    #: No quedó ningún frame válido.
    NO_VALID_FRAMES = "NO_VALID_FRAMES"


@dataclass(frozen=True, slots=True)
class GapPolicy:
    """Los límites de la reconstrucción, ya convertidos a cuadros."""

    max_gap_frames: int
    max_fraction: float

    @classmethod
    def from_config(cls, config: Config, fps: float) -> GapPolicy:
        # Import local: `segmentation` importa este módulo.
        from lsm.segmentation import frames_from_ms

        settings = config.segmentation
        return cls(
            max_gap_frames=frames_from_ms(settings.dynamic_max_gap_ms, fps),
            max_fraction=settings.dynamic_max_interpolated_fraction,
        )


@dataclass(frozen=True, slots=True)
class GapFilled:
    """La secuencia reconstruida y cuánto se reconstruyó."""

    sequence: Sequence
    #: Frames interpolados dentro de `sequence`.
    interpolated: int
    #: Frames inválidos recortados al principio y al final.
    trimmed_start: int
    trimmed_end: int


@dataclass(frozen=True, slots=True)
class GapRejected:
    reason: GapRejection
    #: Índice, en el flujo de entrada, del primer frame del hueco que decidió.
    frame_index: int


def can_bridge(before: RawFrame, after: RawFrame) -> bool:
    """Si un hueco entre estos dos frames se puede rellenar.

    Compara la mano **declarada** (`handedness`), que es la que canoniza el paso
    2 desde `FEATURE_SPEC_VERSION` 2; la etiqueta de MediaPipe cambia de opinión a
    mitad de un trazo y no decide nada (ADR 0017). En una sesión en vivo la
    declarada es la misma en todos los frames; la condición protege los flujos
    que mezclan manos y los cambios de resolución.
    """
    return before.handedness is after.handedness and (before.width, before.height) == (
        after.width,
        after.height,
    )


def interpolate_frames(
    before: RawFrame, after: RawFrame, count: int
) -> tuple[RawFrame, ...]:
    """Los `count` frames entre `before` y `after`, por interpolación lineal.

    El frame `k` (desde 1) usa `t = k / (count + 1)` y cada coordenada cruda se
    calcula como `a + (b - a) · t`, en ese orden de operaciones (paridad con
    TypeScript). Hereda la resolución y la mano de `before`; los scores son el
    menor de los dos extremos —un frame inventado no es más fiable que los reales
    que lo rodean— y `detected_handedness` queda en `None`: nadie lo detectó.
    """
    frames: list[RawFrame] = []
    score_mano = min(before.handedness_score, after.handedness_score)
    score_deteccion = min(before.detection_score, after.detection_score)
    for k in range(1, count + 1):
        t = k / (count + 1)
        frames.append(
            RawFrame(
                landmarks=tuple(
                    Landmark(
                        x=a.x + (b.x - a.x) * t,
                        y=a.y + (b.y - a.y) * t,
                        z=a.z + (b.z - a.z) * t,
                    )
                    for a, b in zip(before.landmarks, after.landmarks, strict=True)
                ),
                width=before.width,
                height=before.height,
                handedness=before.handedness,
                handedness_score=score_mano,
                detection_score=score_deteccion,
                detected_handedness=None,
            )
        )
    return tuple(frames)


def fill_gaps(stream: FrameStream, policy: GapPolicy) -> GapFilled | GapRejected:
    """Reconstruye una secuencia dinámica con huecos cortos.

    Orden: se recortan los inválidos del principio y del final; cada hueco
    interno se rellena si cumple las condiciones del módulo, y si alguno no, la
    secuencia se rechaza por el primero que falla; al final se comprueba la
    fracción interpolada sobre la secuencia resultante.
    """
    inicio = 0
    while inicio < len(stream) and not isinstance(stream[inicio], RawFrame):
        inicio += 1
    if inicio == len(stream):
        return GapRejected(reason=GapRejection.NO_VALID_FRAMES, frame_index=0)
    fin = len(stream)
    while not isinstance(stream[fin - 1], RawFrame):
        fin -= 1

    frames: list[RawFrame] = []
    interpolados = 0
    i = inicio
    while i < fin:
        slot = stream[i]
        if isinstance(slot, RawFrame):
            frames.append(slot)
            i += 1
            continue
        # Un hueco interno: hay un válido antes (frames[-1]) y otro después.
        j = i
        while isinstance(stream[j], InvalidFrame):
            j += 1
        largo = j - i
        antes = frames[-1]
        despues = stream[j]
        assert isinstance(despues, RawFrame)
        if largo > policy.max_gap_frames:
            return GapRejected(reason=GapRejection.GAP_TOO_LONG, frame_index=i)
        if not can_bridge(antes, despues):
            return GapRejected(reason=GapRejection.HAND_CHANGED, frame_index=i)
        frames.extend(interpolate_frames(antes, despues, largo))
        interpolados += largo
        i = j

    if interpolados / len(frames) > policy.max_fraction:
        return GapRejected(
            reason=GapRejection.TOO_MUCH_INTERPOLATED, frame_index=inicio
        )
    return GapFilled(
        sequence=Sequence(frames=tuple(frames)),
        interpolated=interpolados,
        trimmed_start=inicio,
        trimmed_end=len(stream) - fin,
    )
