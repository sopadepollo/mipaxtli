"""Genera `tests/fixtures/golden_features.json`.

Estos vectores son el contrato ejecutable entre Python y TypeScript. La
implementación de Python es la **referencia normativa** (`feature-spec.md` §5.1):
`web/tests/features.test.ts` cargará este mismo archivo y verificará que su
reimplementación coincide dentro de `1e-6`.

Se generan en Fase 0, mucho antes de que exista el código web, y a propósito: el
momento de fijar el contrato es cuando todavía hay una sola implementación.

El archivo tiene tres bloques:

- `cases` — el formato del §5.1: un frame de entrada y sus 42 componentes. Cubre
  la tabla de cobertura mínima obligatoria.
- `sequence_cases` — lo que un frame suelto no puede cubrir: el canal de
  trayectoria (§3.1), el remuestreo a 24 (§3.2), la ponderación de `g_t` (§3.3),
  la dispersión σ (§2) y el comportamiento ante secuencias interrumpidas.
- `metadata` — versiones y parámetros con los que se generó todo lo anterior.

Este módulo vive en `cli/` porque escribe en disco. La geometría que usa es la de
`lsm.synthetic`, la misma que consumen los tests: si cada uno construyera sus
propios landmarks, los golden vectors dejarían de describir lo que los tests
verifican.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lsm.config import Config
from lsm.features import (
    FEATURE_SPEC_VERSION,
    MIN_SCALE,
    RESAMPLE_LENGTH,
    DynamicUnavailable,
    ExtractionRejected,
    SequenceFeatures,
    extract_sequence_features,
    resample,
    split_valid_runs,
)
from lsm.io.hands import dump_frame_stream
from lsm.synthetic import (
    arc_offsets,
    canonical_hand,
    collapsed_scale,
    fist_hand,
    mirrored_x,
    moving_sequence,
    rotated,
    scaled,
    to_frame,
    translated,
)
from lsm.types import (
    FrameSlot,
    FrameStream,
    Handedness,
    InvalidFrame,
    InvalidReason,
    Points3,
    RawFrame,
    Sequence,
)

#: Tolerancia del contrato entre implementaciones (`feature-spec.md`).
TOLERANCE = 1e-6

#: Cobertura mínima obligatoria de la tabla del §5.1.
MINIMUM_CASES = 20

WIDE = (1280, 720)
SQUARE = (720, 720)
PORTRAIT = (720, 1280)
CLASSIC = (1024, 768)

CENTER = (640.0, 400.0)

#: Caso de referencia. Todos los pares de invariancia apuntan aquí.
BASE_ID = "right_hand_upright_open"


@dataclass(frozen=True)
class FrameCase:
    """Un frame de entrada y qué paso del contrato valida."""

    id: str
    description: str
    validates: str
    points_px: Points3
    size: tuple[int, int] = WIDE
    handedness: Handedness = Handedness.RIGHT
    #: Si está, las features de este caso deben coincidir con las de ese otro
    #: caso dentro de la tolerancia. Es la parte más valiosa del archivo: verifica
    #: que las invariancias se cumplen de verdad, no solo que el código corre.
    same_features_as: str | None = None

    def to_frame(self) -> RawFrame:
        width, height = self.size
        return to_frame(
            self.points_px, width=width, height=height, handedness=self.handedness
        )


@dataclass(frozen=True)
class SequenceCase:
    """Una secuencia de entrada, huecos incluidos, y todo lo que produce."""

    id: str
    description: str
    validates: str
    stream: FrameStream


def frame_cases() -> tuple[FrameCase, ...]:
    """Los casos por frame. Cubren la tabla de cobertura mínima del §5.1."""
    hand = translated(canonical_hand(), *CENTER)
    fist = translated(fist_hand(), *CENTER)

    return (
        FrameCase(
            id=BASE_ID,
            description="Mano derecha, vertical, centrada, encuadre 16:9.",
            validates="Camino base de los pasos 1 a 7.",
            points_px=hand,
        ),
        FrameCase(
            id="left_hand_same_sign",
            description="La misma seña ejecutada con la mano izquierda.",
            validates="Paso 2: canonicalización de lateralidad.",
            points_px=mirrored_x(hand),
            handedness=Handedness.LEFT,
            same_features_as=BASE_ID,
        ),
        FrameCase(
            id="rotated_45",
            description="Muñeca inclinada 45° en el plano de la imagen.",
            validates="Paso 5: rotación al eje +Y.",
            points_px=rotated(hand, 45.0),
            same_features_as=BASE_ID,
        ),
        FrameCase(
            id="rotated_90",
            description="Muñeca inclinada 90°: la mano apunta de lado.",
            validates="Paso 5.",
            points_px=rotated(hand, 90.0),
            same_features_as=BASE_ID,
        ),
        FrameCase(
            id="rotated_minus_45",
            description="Inclinación de 45° en el sentido contrario.",
            validates="Paso 5, signo del ángulo.",
            points_px=rotated(hand, -45.0),
            same_features_as=BASE_ID,
        ),
        FrameCase(
            id="rotated_180",
            description="Mano de cabeza.",
            validates="Paso 5, ángulo en el otro cuadrante.",
            points_px=rotated(hand, 180.0),
            same_features_as=BASE_ID,
        ),
        FrameCase(
            id="scaled_2x",
            description="La misma mano al doble de tamaño: firmando más cerca.",
            validates="Paso 4: normalización de escala.",
            points_px=scaled(hand, 2.0),
            same_features_as=BASE_ID,
        ),
        FrameCase(
            id="scaled_half",
            description="La misma mano a la mitad: firmando desde más lejos.",
            validates="Paso 4.",
            points_px=scaled(hand, 0.5),
            same_features_as=BASE_ID,
        ),
        FrameCase(
            id="corner_top_left",
            description="Mano en la esquina superior izquierda del encuadre.",
            validates="Paso 3: traslación al origen.",
            points_px=translated(canonical_hand(), 120.0, 260.0),
            same_features_as=BASE_ID,
        ),
        FrameCase(
            id="corner_top_right",
            description="Mano en la esquina superior derecha.",
            validates="Paso 3.",
            points_px=translated(canonical_hand(), 1160.0, 260.0),
            same_features_as=BASE_ID,
        ),
        FrameCase(
            id="corner_bottom_left",
            description="Mano en la esquina inferior izquierda.",
            validates="Paso 3.",
            points_px=translated(canonical_hand(), 120.0, 660.0),
            same_features_as=BASE_ID,
        ),
        FrameCase(
            id="corner_bottom_right",
            description="Mano en la esquina inferior derecha.",
            validates="Paso 3.",
            points_px=translated(canonical_hand(), 1160.0, 660.0),
            same_features_as=BASE_ID,
        ),
        FrameCase(
            id="frame_1_1",
            description="La misma mano, en píxeles, filmada en un encuadre 1:1.",
            validates="Paso 1: corrección de relación de aspecto.",
            points_px=translated(canonical_hand(), 360.0, 400.0),
            size=SQUARE,
            same_features_as="frame_16_9_same_pixels",
        ),
        FrameCase(
            id="frame_16_9_same_pixels",
            description="Los mismos píxeles que el caso 1:1, en un encuadre 16:9.",
            validates="Paso 1.",
            points_px=translated(canonical_hand(), 360.0, 400.0),
            size=WIDE,
            same_features_as=BASE_ID,
        ),
        FrameCase(
            id="frame_9_16_portrait",
            description="Encuadre vertical, como el de un celular.",
            validates="Paso 1 con a < 1.",
            points_px=translated(canonical_hand(), 360.0, 400.0),
            size=PORTRAIT,
            same_features_as=BASE_ID,
        ),
        FrameCase(
            id="frame_4_3",
            description="Encuadre 4:3, como el de muchas webcams viejas.",
            validates="Paso 1 con otra relación de aspecto.",
            points_px=translated(canonical_hand(), 360.0, 400.0),
            size=CLASSIC,
            same_features_as=BASE_ID,
        ),
        FrameCase(
            id="landmarks_below_zero",
            description="Mano parcialmente fuera del encuadre: x e y negativas.",
            validates="MediaPipe extrapola fuera del frame; no debe romper.",
            points_px=translated(canonical_hand(), -80.0, 40.0),
            same_features_as=BASE_ID,
        ),
        FrameCase(
            id="landmarks_above_one",
            description="Mano fuera del encuadre por el otro lado: x e y > 1.",
            validates="Extrapolación de MediaPipe.",
            points_px=translated(canonical_hand(), 1500.0, 900.0),
            same_features_as=BASE_ID,
        ),
        FrameCase(
            id="combined_transforms",
            description=(
                "Mano izquierda, girada 30°, al doble de tamaño, en una esquina y "
                "en encuadre 1:1: todas las invariancias a la vez."
            ),
            validates="Composición de los pasos 1 a 5.",
            points_px=translated(
                mirrored_x(rotated(scaled(canonical_hand(), 2.0), 30.0)), 300.0, 500.0
            ),
            size=(1000, 1000),
            handedness=Handedness.LEFT,
            same_features_as=BASE_ID,
        ),
        FrameCase(
            id="fist_configuration",
            description="Otra configuración de mano: dedos flexionados.",
            validates=(
                "Control negativo: sus features deben diferir del caso base, o el "
                "resto de la tabla se satisfaría devolviendo ceros."
            ),
            points_px=fist,
        ),
        FrameCase(
            id="left_hand_fist",
            description="El puño ejecutado con la mano izquierda.",
            validates="Paso 2 sobre una configuración distinta.",
            points_px=mirrored_x(fist),
            handedness=Handedness.LEFT,
            same_features_as="fist_configuration",
        ),
        FrameCase(
            id="degenerate_scale",
            description="Nudillo del dedo medio sobre la muñeca: escala nula.",
            validates="Paso 4: el frame debe marcarse inválido.",
            points_px=collapsed_scale(hand),
        ),
        FrameCase(
            id="degenerate_scale_left_hand",
            description="La misma degeneración con la mano izquierda.",
            validates="Paso 4: el rechazo no depende de la lateralidad.",
            points_px=collapsed_scale(mirrored_x(hand)),
            handedness=Handedness.LEFT,
        ),
        FrameCase(
            id="scale_just_below_minimum",
            description=f"Escala justo por debajo de MIN_SCALE ({MIN_SCALE}).",
            validates="Paso 4: el umbral es estricto, no aproximado.",
            points_px=_with_middle_mcp_at(hand, MIN_SCALE * WIDE[0] * 0.5),
        ),
    )


def _with_middle_mcp_at(points: Points3, offset_px: float) -> Points3:
    """Coloca p_9 a `offset_px` de la muñeca, para probar el umbral de escala."""
    mutated = list(points)
    wrist = points[0]
    mutated[9] = (wrist[0] + offset_px, wrist[1], points[9][2])
    return tuple(mutated)


def sequence_cases() -> tuple[SequenceCase, ...]:
    """Los casos de secuencia: trayectoria, remuestreo, σ e interrupciones."""
    hand = canonical_hand()
    arc = arc_offsets(10)
    still = moving_sequence(hand, ((0.0, 0.0),) * 8).frames
    arc_frames = moving_sequence(hand, arc).frames
    hook: FrameStream = arc_frames

    return (
        SequenceCase(
            id="still_sequence",
            description="Una seña estática: ocho frames de la misma mano quieta.",
            validates="§2: el promedio es el propio frame y σ ≈ 0. §3.1: τ = 0.",
            stream=still,
        ),
        SequenceCase(
            id="trajectory_arc",
            description=(
                "La misma configuración de mano recorriendo un gancho, como el "
                "trazo de la J."
            ),
            validates=(
                "§3.1: el canal de trayectoria toma la muñeca ANTES de la "
                "traslación del paso 3. Medido después saldría cero y esta "
                "secuencia sería idéntica a `still_sequence`."
            ),
            stream=hook,
        ),
        SequenceCase(
            id="trajectory_arc_translated",
            description="El mismo trazo, ejecutado en otra parte del encuadre.",
            validates="§3.1: τ es invariante a la posición en el encuadre.",
            stream=moving_sequence(translated(hand, 420.0, 180.0), arc).frames,
        ),
        SequenceCase(
            id="trajectory_arc_closer",
            description="El mismo trazo, ejecutado más cerca de la cámara.",
            validates="§3.1: τ es invariante a la distancia; va en unidades de mano.",
            stream=moving_sequence(
                scaled(hand, 2.0), tuple((dx * 2.0, dy * 2.0) for dx, dy in arc)
            ).frames,
        ),
        SequenceCase(
            id="trajectory_arc_left_hand",
            description="El trazo espejado, ejecutado con la mano izquierda.",
            validates="§3.1 y paso 2: la lateralidad se canoniza también en τ.",
            stream=moving_sequence(
                mirrored_x(hand),
                tuple((-dx, dy) for dx, dy in arc),
                handedness=Handedness.LEFT,
            ).frames,
        ),
        SequenceCase(
            id="too_few_source_frames",
            description="Dos frames: por debajo de `dtw.min_source_frames`.",
            validates=(
                "§3.2: la secuencia se rechaza en vez de estirarse a 24. "
                "Interpolar dos frames inventa una trayectoria que nadie ejecutó."
            ),
            stream=moving_sequence(hand, ((0.0, 0.0), (12.0, 0.0))).frames,
        ),
        SequenceCase(
            id="exactly_24_frames",
            description="Una secuencia que ya mide 24 frames.",
            validates="§3.2: el remuestreo es la identidad, sin desplazar filas.",
            stream=moving_sequence(hand, arc_offsets(RESAMPLE_LENGTH)).frames,
        ),
        SequenceCase(
            id="interrupted_at_start",
            description="El detector pierde la mano en el primer frame.",
            validates=(
                "§0.3: la secuencia empieza más tarde y el origen de τ se mueve "
                "al primer frame válido. El trazo restante se mide desde otro punto."
            ),
            stream=(InvalidFrame(reason=InvalidReason.NO_HAND), *hook[1:]),
        ),
        SequenceCase(
            id="interrupted_in_middle",
            description="El detector pierde la mano a la mitad del trazo.",
            validates=(
                "§0.3: quedan DOS secuencias, no una con un salto. Coserlas "
                "inventaría un movimiento entre dos posiciones no observadas."
            ),
            stream=(*hook[:5], InvalidFrame(reason=InvalidReason.NO_HAND), *hook[6:]),
        ),
        SequenceCase(
            id="interrupted_at_end",
            description="El detector pierde la mano en el último frame.",
            validates="§0.3: la secuencia simplemente se corta antes.",
            stream=(*hook[:-1], InvalidFrame(reason=InvalidReason.NO_HAND)),
        ),
    )


# --------------------------------------------------------------------------- #
# Serialización
# --------------------------------------------------------------------------- #


def _frame_input(frame: RawFrame) -> dict[str, Any]:
    return {
        "width": frame.width,
        "height": frame.height,
        "handedness": str(frame.handedness),
        "landmarks": [[point.x, point.y, point.z] for point in frame.landmarks],
    }


def _slot_input(slot: FrameSlot) -> dict[str, Any]:
    """Los frames inválidos viajan con un centinela explícito, nunca como `null`."""
    if isinstance(slot, InvalidFrame):
        return {"valid": False, "reason": str(slot.reason)}
    return {"valid": True, **_frame_input(slot)}


def build_frame_case(case: FrameCase, config: Config) -> dict[str, Any]:
    frame = case.to_frame()
    outcome = extract_sequence_features(Sequence(frames=(frame,)), config)

    payload: dict[str, Any] = {
        "id": case.id,
        "description": case.description,
        "validates": case.validates,
        "input": _frame_input(frame),
    }
    if isinstance(outcome, ExtractionRejected):
        payload["expected_features"] = None
        payload["expected_invalid_reason"] = str(outcome.reason)
    else:
        payload["expected_features"] = list(outcome.frames[0].values)
        payload["expected_invalid_reason"] = None
    if case.same_features_as is not None:
        payload["same_features_as"] = case.same_features_as
    return payload


def _run_expectation(features: SequenceFeatures, length: int) -> dict[str, Any]:
    dynamic = features.dynamic
    payload: dict[str, Any] = {
        "length": length,
        "frame_features": [list(vector.values) for vector in features.frames],
        "dispersion": features.static.dispersion,
        "static_features": list(features.static.shape.values),
        "mean_scale": features.trajectory.mean_scale,
        "trajectory": [list(point) for point in features.trajectory.points],
    }
    if isinstance(dynamic, DynamicUnavailable):
        payload["dynamic_unavailable_reason"] = str(dynamic.reason)
        payload["resampled_trajectory"] = None
        payload["dynamic_rows"] = None
    else:
        payload["dynamic_unavailable_reason"] = None
        payload["resampled_trajectory"] = [
            list(point) for point in resample(features.trajectory.points)
        ]
        payload["dynamic_rows"] = [list(row) for row in dynamic.rows]
    return payload


def build_sequence_case(case: SequenceCase, config: Config) -> dict[str, Any]:
    runs = split_valid_runs(case.stream)
    expectations: list[dict[str, Any]] = []
    for run in runs:
        outcome = extract_sequence_features(run, config)
        if isinstance(outcome, ExtractionRejected):
            expectations.append(
                {
                    "length": len(run),
                    "rejected_reason": str(outcome.reason),
                    "rejected_frame_index": outcome.frame_index,
                }
            )
            continue
        expectations.append(_run_expectation(outcome, len(run)))

    return {
        "id": case.id,
        "description": case.description,
        "validates": case.validates,
        "input": {"frames": [_slot_input(slot) for slot in case.stream]},
        "expected": {"run_count": len(runs), "runs": expectations},
    }


def build_document(config: Config) -> dict[str, Any]:
    """Arma el documento completo de golden vectors."""
    cases = [build_frame_case(case, config) for case in frame_cases()]
    if len(cases) < MINIMUM_CASES:
        msg = (
            f"la cobertura mínima del §5.1 son {MINIMUM_CASES} casos y solo hay "
            f"{len(cases)}"
        )
        raise ValueError(msg)

    return {
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "tolerance": TOLERANCE,
        "generated_by": "lsm.cli.golden — Python es la referencia normativa",
        "config": {
            "trajectory_weight": config.features.trajectory_weight,
            "min_source_frames": config.dtw.min_source_frames,
            "smoothing_alpha": config.smoothing.alpha,
            "resample_length": RESAMPLE_LENGTH,
            "min_scale": MIN_SCALE,
        },
        "cases": cases,
        "sequence_cases": [
            build_sequence_case(case, config) for case in sequence_cases()
        ],
    }


def _example_stream() -> FrameStream:
    """Grabación sintética de ejemplo para `FakeHandDetector`, con un hueco."""
    frames = moving_sequence(canonical_hand(), arc_offsets(10)).frames
    return (
        *frames[:4],
        InvalidFrame(reason=InvalidReason.NO_HAND, detail="la mano salió del encuadre"),
        *frames[4:],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Genera los golden vectors de referencia. Python es la implementación "
            "normativa: la de TypeScript se valida contra este archivo."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/golden_features.json"),
        help="ruta del archivo de golden vectors",
    )
    parser.add_argument(
        "--sequences",
        type=Path,
        default=Path("tests/fixtures/sequences/ejemplo_trazo_j.json"),
        help="ruta de la grabación sintética de ejemplo",
    )
    args = parser.parse_args(argv)

    document = build_document(Config())
    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    sequences: Path = args.sequences
    sequences.parent.mkdir(parents=True, exist_ok=True)
    dump_frame_stream(_example_stream(), sequences)

    print(
        f"{output}: {len(document['cases'])} casos por frame, "
        f"{len(document['sequence_cases'])} casos de secuencia\n"
        f"{sequences}: grabación de ejemplo"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
