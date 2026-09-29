"""Del flujo crudo guardado a la secuencia que se entrena (§0.3 y §0.4).

Una muestra en disco es el flujo tal como salió del detector, huecos incluidos
(`io/dataset.py`). Lo que se entrena tiene que pasar por **lo mismo** que pasa un
trazo en vivo antes de llegar al clasificador, en el mismo orden:

1. el filtro de plausibilidad (§0.4): los cuadros imposibles pasan a huecos;
2. el relleno de huecos (§0.3): solo en las dinámicas; en una estática,
   cualquier hueco sigue siendo un error.

Este módulo es ese recorrido, en un solo sitio, para que el entrenamiento, la
evaluación y la captura no lo reimplementen cada uno a su manera. En vivo lo
mismo ocurre cuadro a cuadro dentro de `segmentation.run_segmentation`.

Código puro.
"""

from __future__ import annotations

from dataclasses import dataclass

from lsm.config import Config
from lsm.gaps import GapFilled, GapPolicy, GapRejected, fill_gaps
from lsm.plausibility import PlausibilityParams, filter_stream, is_implausible
from lsm.types import FrameStream, RawFrame, SampleKind, Sequence


@dataclass(frozen=True, slots=True)
class Preprocessing:
    """Todo lo que hace falta para reconstruir una muestra, a una tasa dada.

    `fps` es la tasa con que se fechan los cuadros sin marca de tiempo y con que
    se convierten los límites en ms: la nominal para el dataset grabado.
    """

    plausibility: PlausibilityParams
    gaps: GapPolicy
    fps: float

    @classmethod
    def from_config(cls, config: Config, fps: float) -> Preprocessing:
        return cls(
            plausibility=PlausibilityParams.from_config(config),
            gaps=GapPolicy.from_config(config, fps),
            fps=fps,
        )


@dataclass(frozen=True, slots=True)
class Reconstructed:
    """La secuencia lista para extraer features, y qué costó obtenerla."""

    sequence: Sequence
    #: Frames de `sequence` rellenados por interpolación.
    interpolated: int
    #: De esos, los que sustituyen a un cuadro invalidado por plausibilidad.
    implausible: int


@dataclass(frozen=True, slots=True)
class NotReconstructed:
    """Por qué no: `GAP` si un hueco no se puede rellenar (con el rechazo de
    `lsm.gaps`), `STATIC_GAP` si es una estática con cualquier hueco."""

    reason: str
    detail: str


def reconstruct(
    stream: FrameStream, kind: SampleKind, pre: Preprocessing
) -> Reconstructed | NotReconstructed:
    """Plausibilidad y relleno sobre un flujo guardado, con estado nuevo.

    Una estática no pasa por la plausibilidad: sin relleno en el camino
    estático, un solo cuadro invalidado la haría ilegible. En el dataset de hoy
    le pasaría a 1 de 2900 muestras (ADR 0027).
    """
    filtrado = (
        tuple(filter_stream(stream, pre.plausibility, pre.fps))
        if kind is SampleKind.DYNAMIC
        else stream
    )
    if all(isinstance(slot, RawFrame) for slot in filtrado):
        return Reconstructed(
            sequence=Sequence(
                frames=tuple(f for f in filtrado if isinstance(f, RawFrame))
            ),
            interpolated=0,
            implausible=0,
        )
    if kind is not SampleKind.DYNAMIC:
        huecos = sum(1 for s in filtrado if not isinstance(s, RawFrame))
        implausibles = sum(1 for s in filtrado if is_implausible(s))
        return NotReconstructed(
            reason="STATIC_GAP",
            detail=(
                f"{huecos} de {len(filtrado)} frames no son válidos "
                f"({implausibles} por plausibilidad)"
            ),
        )
    relleno = fill_gaps(filtrado, pre.gaps)
    if isinstance(relleno, GapRejected):
        return NotReconstructed(
            reason="GAP",
            detail=f"{relleno.reason} en el frame {relleno.frame_index}",
        )
    return Reconstructed(
        sequence=relleno.sequence,
        interpolated=relleno.interpolated,
        implausible=_implausibles_rellenados(filtrado, relleno),
    )


def _implausibles_rellenados(filtrado: FrameStream, relleno: GapFilled) -> int:
    """Los cuadros implausibles que quedaron dentro de la secuencia rellenada:
    los de los bordes se recortan, no se rellenan."""
    interior = filtrado[relleno.trimmed_start : len(filtrado) - relleno.trimmed_end]
    return sum(1 for slot in interior if is_implausible(slot))
