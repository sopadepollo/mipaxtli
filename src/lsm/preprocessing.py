"""Del flujo crudo guardado a la secuencia que se entrena (§0.3, §0.4 y §4).

Una muestra en disco es el flujo tal como salió del detector, huecos incluidos
(`io/dataset.py`). Lo que se entrena tiene que pasar por **lo mismo** que pasa un
trazo en vivo antes de llegar al clasificador, en el mismo orden:

1. el filtro de plausibilidad (§0.4): los cuadros imposibles pasan a huecos;
2. el relleno de huecos (§0.3): en una dinámica con el límite del trazo, en
   una estática con el de STABLE (Paso 4); sin límite, cualquier hueco de una
   estática sigue siendo un error;
3. el filtro One Euro (§4), desde el primer frame de la secuencia resultante.

Este módulo es ese recorrido, en un solo sitio, para que el entrenamiento, la
evaluación y la captura no lo reimplementen cada uno a su manera. En vivo lo
mismo ocurre cuadro a cuadro dentro de `segmentation.run_segmentation`; la
diferencia es que allí el One Euro llega al trazo ya caliente, con la historia
anterior al trazo, y aquí arranca en su primer frame.

Código puro.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lsm.config import Config
from lsm.gaps import GapFilled, GapPolicy, GapRejected, fill_gaps
from lsm.one_euro import OneEuroParams, filter_sequence
from lsm.plausibility import PlausibilityParams, filter_stream, is_implausible
from lsm.timing import frame_times_ms
from lsm.types import FrameStream, RawFrame, SampleKind, Sequence


@dataclass(frozen=True, slots=True)
class Preprocessing:
    """Todo lo que hace falta para reconstruir una muestra, a una tasa dada.

    `fps` es la tasa con que se fechan los cuadros sin marca de tiempo y con que
    se convierten los límites en ms: la nominal para el dataset grabado.
    """

    plausibility: PlausibilityParams
    gaps: GapPolicy
    one_euro: OneEuroParams
    fps: float
    #: El relleno de una estática, con `segmentation.stable_max_gap_ms` (Paso 4).
    #: `None` si ese límite es 0: cualquier hueco de una estática es un error.
    static_gaps: GapPolicy | None = None

    @classmethod
    def from_config(cls, config: Config, fps: float) -> Preprocessing:
        # Import local: `segmentation` importa este módulo a través de `gaps`.
        from lsm.segmentation import frames_from_ms

        estable = config.segmentation.stable_max_gap_ms
        return cls(
            plausibility=PlausibilityParams.from_config(config),
            gaps=GapPolicy.from_config(config, fps),
            one_euro=OneEuroParams.from_config(config),
            fps=fps,
            static_gaps=(
                None
                if estable == 0.0
                else GapPolicy(
                    max_gap_frames=frames_from_ms(estable, fps),
                    max_fraction=config.segmentation.dynamic_max_interpolated_fraction,
                )
            ),
        )


def preprocessing_record(config: Config) -> dict[str, Any]:
    """Lo que la configuración decide del preprocesado, como JSON.

    Viaja en las muestras (con qué se calculó su σ) y en los modelos exportados
    (con qué se entrenaron): los dos se comparan contra el de ahora.
    """
    return {
        "plausibility": PlausibilityParams.from_config(config).to_json(),
        "one_euro": OneEuroParams.from_config(config).to_json(),
        "dynamic_max_gap_ms": config.segmentation.dynamic_max_gap_ms,
        "tracking_max_gap_ms": config.segmentation.tracking_max_gap_ms,
        "stable_max_gap_ms": config.segmentation.stable_max_gap_ms,
        "dynamic_max_interpolated_fraction": (
            config.segmentation.dynamic_max_interpolated_fraction
        ),
    }


def config_from_record(record: dict[str, Any]) -> dict[str, Any]:
    """Las secciones de configuración de un `preprocessing_record`, para
    reconstruir la configuración con que se entrenó un modelo."""
    return {
        "plausibility": dict(record["plausibility"]),
        "smoothing": dict(record["one_euro"]),
        "segmentation": {
            "dynamic_max_gap_ms": record["dynamic_max_gap_ms"],
            "tracking_max_gap_ms": record["tracking_max_gap_ms"],
            "stable_max_gap_ms": record["stable_max_gap_ms"],
            "dynamic_max_interpolated_fraction": record[
                "dynamic_max_interpolated_fraction"
            ],
        },
    }


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
    """Plausibilidad, relleno y One Euro sobre un flujo guardado, con estado nuevo.

    Sin relleno estático (`static_gaps` en `None`) una estática no pasa por la
    plausibilidad: un solo cuadro invalidado la haría ilegible (ADR 0027).
    """
    politica = pre.gaps if kind is SampleKind.DYNAMIC else pre.static_gaps
    filtrado = (
        tuple(filter_stream(stream, pre.plausibility, pre.fps))
        if politica is not None
        else stream
    )
    if all(isinstance(slot, RawFrame) for slot in filtrado):
        return Reconstructed(
            sequence=_suavizada(
                Sequence(frames=tuple(f for f in filtrado if isinstance(f, RawFrame))),
                pre,
            ),
            interpolated=0,
            implausible=0,
        )
    if politica is None:
        huecos = sum(1 for s in filtrado if not isinstance(s, RawFrame))
        implausibles = sum(1 for s in filtrado if is_implausible(s))
        return NotReconstructed(
            reason="STATIC_GAP",
            detail=(
                f"{huecos} de {len(filtrado)} frames no son válidos "
                f"({implausibles} por plausibilidad)"
            ),
        )
    relleno = fill_gaps(filtrado, politica)
    if isinstance(relleno, GapRejected):
        return NotReconstructed(
            reason="GAP",
            detail=f"{relleno.reason} en el frame {relleno.frame_index}",
        )
    return Reconstructed(
        sequence=_suavizada(relleno.sequence, pre),
        interpolated=relleno.interpolated,
        implausible=_implausibles_rellenados(filtrado, relleno),
    )


def _suavizada(sequence: Sequence, pre: Preprocessing) -> Sequence:
    """El One Euro sobre la secuencia, con el tiempo de `lsm.timing`."""
    if not pre.one_euro.enabled:
        return sequence
    return filter_sequence(
        sequence, frame_times_ms(sequence.frames, pre.fps), pre.one_euro
    )


def _implausibles_rellenados(filtrado: FrameStream, relleno: GapFilled) -> int:
    """Los cuadros implausibles que quedaron dentro de la secuencia rellenada:
    los de los bordes se recortan, no se rellenan."""
    interior = filtrado[relleno.trimmed_start : len(filtrado) - relleno.trimmed_end]
    return sum(1 for slot in interior if is_implausible(slot))
