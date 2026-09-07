"""Implementación normativa de `docs/feature-spec.md`.

Este módulo **es** el contrato. La reimplementación en TypeScript de la Fase 7
debe reproducir sus resultados dentro de `1e-6`, verificado contra
`tests/fixtures/golden_features.json`.

De ahí tres decisiones que en otro proyecto serían raras:

1. **Aritmética en Python puro, sin numpy.** El §5 del contrato prohíbe reordenar
   operaciones, y las reducciones de numpy suman por pares: `numpy.mean` no da los
   mismos bits que sumar en orden temporal ascendente. Con `T ≤ 30` y 42
   componentes, el costo de hacerlo a mano es irrelevante y a cambio el código se
   traduce a TypeScript línea por línea.
2. **Un paso, una función.** Aunque fusionar traslación y rotación en una matriz
   sea algebraicamente equivalente, numéricamente no lo es (§5.5).
3. **Ninguna API pública recibe un frame suelto.** El tipo de entrada es
   `Sequence`. La composición por frame existe, pero es privada: una seña estática
   es una secuencia corta, no un frame.

Código puro: sin OpenCV, sin MediaPipe, sin disco, sin cámara.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Final, TypeAlias

from lsm.config import Config
from lsm.types import (
    NUM_FEATURES,
    FeatureVector,
    FrameStream,
    Handedness,
    InvalidReason,
    Landmark,
    LandmarkIndex,
    Point2,
    Points2,
    Points3,
    RawFrame,
    Sequence,
    TrajectoryChannel,
)

# --------------------------------------------------------------------------- #
# Constantes del contrato
#
# No son configurables a propósito: definen el formato del vector, no una
# preferencia ajustable. Cambiar cualquiera de ellas obliga a incrementar
# FEATURE_SPEC_VERSION, regenerar los golden vectors, reentrenar todos los
# modelos y escribir un ADR (`CLAUDE.md` §4).
# --------------------------------------------------------------------------- #

#: Versión del contrato implementado por este módulo.
FEATURE_SPEC_VERSION: Final = 1

#: T_ref del §3.2: toda secuencia dinámica se remuestrea a esta longitud.
RESAMPLE_LENGTH: Final = 24

#: Umbral de escala degenerada del paso 4. Por debajo, el frame es inválido.
MIN_SCALE: Final = 1e-6


class DynamicRejection(StrEnum):
    """Por qué una secuencia no produjo canal dinámico."""

    #: Menos frames de origen que `config.dtw.min_source_frames` (§3.2).
    TOO_FEW_SOURCE_FRAMES = "TOO_FEW_SOURCE_FRAMES"


@dataclass(frozen=True, slots=True)
class StaticFeatures:
    """Agregación del §2: vector de forma y su indicador de calidad."""

    #: F: promedio temporal de los vectores por frame.
    shape: FeatureVector
    #: σ: dispersión media por componente. **No es una feature**: es el criterio
    #: de calidad que decide si la ventana se clasifica o se descarta.
    dispersion: float


@dataclass(frozen=True, slots=True)
class DynamicFeatures:
    """Matriz `(24, 44)` del §3.3, lista para el DTW."""

    rows: tuple[tuple[float, ...], ...]
    #: Cuántos frames válidos tenía la secuencia antes de remuestrear. Útil para
    #: depurar: 24 filas no significan 24 frames observados.
    source_length: int


@dataclass(frozen=True, slots=True)
class DynamicUnavailable:
    """La secuencia no da para canal dinámico, pero sigue siendo válida.

    Marcador explícito en vez de `None`: una letra estática se resuelve con pocos
    frames, y confundir "no aplica" con "falló" haría descartar señas buenas.
    """

    reason: DynamicRejection


@dataclass(frozen=True, slots=True)
class SequenceFeatures:
    """Todo lo que la tubería extrae de una secuencia válida."""

    #: f_t para cada frame, en orden temporal.
    frames: tuple[FeatureVector, ...]
    static: StaticFeatures
    trajectory: TrajectoryChannel
    dynamic: DynamicFeatures | DynamicUnavailable
    #: Velocidad entre frames consecutivos, en unidades de mano por frame.
    #: Longitud `T - 1`. Insumo de la máquina de estados, no del clasificador.
    velocities: tuple[float, ...]
    spec_version: int


@dataclass(frozen=True, slots=True)
class ExtractionRejected:
    """La secuencia no se puede procesar, y se dice en qué frame se rompió."""

    reason: InvalidReason
    frame_index: int


ExtractionOutcome: TypeAlias = SequenceFeatures | ExtractionRejected


# --------------------------------------------------------------------------- #
# §1 — Pipeline por frame, un paso por función
# --------------------------------------------------------------------------- #


def correct_aspect_and_orientation(points: Points3, aspect_ratio: float) -> Points3:
    """Paso 1: corrige la relación de aspecto e invierte el eje `y`.

    MediaPipe normaliza `x` por el ancho e `y` por el alto, así que en un frame
    16:9 la mano sale deformada horizontalmente. Multiplicar `x` por `a` la
    devuelve a proporciones isotrópicas. La inversión de `y` lleva el eje a
    apuntar hacia arriba, que es la convención del resto del contrato.
    """
    return tuple((x * aspect_ratio, -y, z * aspect_ratio) for x, y, z in points)


def canonicalize_handedness(points: Points3, handedness: Handedness) -> Points3:
    """Paso 2: lleva toda muestra a una mano derecha canónica.

    Evita duplicar el dataset y permite que una persona zurda use un modelo
    entrenado por diestros. Depende de que el detector haya recibido el frame
    **sin espejar** (§0.3): con la imagen espejada, la lateralidad reportada se
    invierte y este paso corrompe el vector en silencio.
    """
    if handedness is Handedness.RIGHT:
        return points
    return tuple((-x, y, z) for x, y, z in points)


def translate_to_origin(points: Points3) -> Points3:
    """Paso 3: resta la muñeca a todos los puntos.

    Aquí se destruye, a propósito, dónde estaba la mano en el encuadre. Lo que
    importa para las señas dinámicas se recupera aparte en `trajectory_channel`.
    """
    wrist_x, wrist_y, wrist_z = points[LandmarkIndex.WRIST]
    return tuple((x - wrist_x, y - wrist_y, z - wrist_z) for x, y, z in points)


def reference_scale(points: Points3) -> float:
    """Paso 4 (medida): distancia muñeca → nudillo del dedo medio, en 2D.

    Se espera `points` ya trasladado. Se ignora `z` deliberadamente, y se elige
    esta distancia porque es estable frente a la flexión de los dedos: cualquier
    medida que involucre una punta cambia con la propia seña.
    """
    x, y, _ = points[LandmarkIndex.MIDDLE_MCP]
    return math.sqrt(x * x + y * y)


def apply_scale(points: Points3, scale: float) -> Points3:
    """Paso 4 (división). Tras esto, ‖p_9‖₂ = 1 en el plano XY."""
    return tuple((x / scale, y / scale, z / scale) for x, y, z in points)


def rotate_to_axis(points: Points3) -> Points3:
    """Paso 5: alinea el eje de la palma con +Y. Tolera muñecas inclinadas.

    `z` no se modifica. Los senos y cosenos se calculan una sola vez, fuera del
    bucle: recalcularlos por punto daría los mismos bits, pero esto deja claro
    que el ángulo es uno solo para todo el frame.
    """
    x9, y9, _ = points[LandmarkIndex.MIDDLE_MCP]
    theta = math.atan2(y9, x9)
    phi = math.pi / 2 - theta
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    return tuple(
        (x * cos_phi - y * sin_phi, x * sin_phi + y * cos_phi, z) for x, y, z in points
    )


def drop_z(points: Points3) -> Points2:
    """Paso 6: descarta `z`.

    La `z` de MediaPipe es una profundidad relativa estimada, ruidosa e
    inconsistente entre frames y entre cámaras. Ver
    `docs/adr/0002-formato-de-features.md`. El canal se calcula igual a lo largo
    de toda la tubería para poder reactivarlo como spec v2 sin rediseñar nada.
    """
    return tuple((x, y) for x, y, _ in points)


def flatten(points: Points2) -> FeatureVector:
    """Paso 7: aplana a ℝ⁴², por índice ascendente y `x` antes que `y`.

    Se conservan las componentes constantes (`p_0 = (0,0)`, `p_9 = (0,1)`):
    aportan distancia cero en cualquier métrica y mantener los 21 índices
    alineados con la numeración de MediaPipe elimina una fuente crónica de
    errores off-by-one al depurar y al reimplementar en TypeScript.
    """
    values: list[float] = []
    for x, y in points:
        values.append(x)
        values.append(y)
    return FeatureVector(values=tuple(values), spec_version=FEATURE_SPEC_VERSION)


# --------------------------------------------------------------------------- #
# Composición por frame — privada: el tipo de entrada del sistema es la secuencia
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _FrameGeometry:
    """Resultados intermedios de un frame que la secuencia necesita después."""

    features: FeatureVector
    #: Muñeca tras el paso 2, **antes** de trasladar. Es lo que alimenta el canal
    #: de trayectoria del §3.1; tomarla después del paso 3 daría siempre (0, 0).
    wrist: Point2
    #: s_t del paso 4.
    scale: float
    #: Puntos tras el paso 2, sin trasladar ni escalar. Base de la velocidad.
    canonical: Points3


def _frame_geometry(frame: RawFrame) -> _FrameGeometry | InvalidReason:
    """Aplica los pasos 1 a 7 en el orden exacto del contrato."""
    step_1 = correct_aspect_and_orientation(frame.points(), frame.aspect_ratio)
    step_2 = canonicalize_handedness(step_1, frame.handedness)
    step_3 = translate_to_origin(step_2)
    scale = reference_scale(step_3)
    if scale < MIN_SCALE:
        return InvalidReason.SCALE_TOO_SMALL
    step_4 = apply_scale(step_3, scale)
    step_5 = rotate_to_axis(step_4)
    step_6 = drop_z(step_5)
    wrist_x, wrist_y, _ = step_2[LandmarkIndex.WRIST]
    return _FrameGeometry(
        features=flatten(step_6),
        wrist=(wrist_x, wrist_y),
        scale=scale,
        canonical=step_2,
    )


def _sequence_geometry(
    sequence: Sequence,
) -> tuple[_FrameGeometry, ...] | ExtractionRejected:
    geometries: list[_FrameGeometry] = []
    for index, frame in enumerate(sequence.frames):
        geometry = _frame_geometry(frame)
        if isinstance(geometry, InvalidReason):
            return ExtractionRejected(reason=geometry, frame_index=index)
        geometries.append(geometry)
    return tuple(geometries)


# --------------------------------------------------------------------------- #
# §4 — Suavizado temporal, antes del paso 1
# --------------------------------------------------------------------------- #


def smooth_sequence(sequence: Sequence, alpha: float) -> Sequence:
    """Media móvil exponencial sobre los landmarks crudos (§4).

    `p̃_t = α · p_t + (1 - α) · p̃_{t-1}`, con `p̃_0 = p_0`: la serie arranca en el
    primer frame observado. El contrato no fijaba la inicialización; se elige
    ésta porque cualquier otra (arrancar en cero, por ejemplo) inventaría un
    movimiento desde el origen del encuadre que nadie ejecutó. Queda anotado en
    `docs/feature-spec.md` §4.

    Con `α = 1.0` la secuencia se devuelve tal cual, sin recorrerla: el suavizado
    está desactivado por defecto porque emborrona justo los movimientos rápidos
    que distinguen a las letras dinámicas.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha debe estar en (0, 1], no {alpha}")
    if alpha == 1.0:
        return sequence

    complement = 1.0 - alpha
    smoothed: list[RawFrame] = []
    previous: Points3 | None = None
    for frame in sequence.frames:
        points = frame.points()
        if previous is None:
            current = points
        else:
            current = tuple(
                (
                    alpha * x + complement * px,
                    alpha * y + complement * py,
                    alpha * z + complement * pz,
                )
                for (x, y, z), (px, py, pz) in zip(points, previous, strict=True)
            )
        smoothed.append(_with_points(frame, current))
        previous = current
    return Sequence(frames=tuple(smoothed))


def _with_points(frame: RawFrame, points: Points3) -> RawFrame:
    """Copia un frame cambiándole los landmarks y conservando los metadatos."""
    return RawFrame(
        landmarks=tuple(Landmark(x=x, y=y, z=z) for x, y, z in points),
        width=frame.width,
        height=frame.height,
        handedness=frame.handedness,
        handedness_score=frame.handedness_score,
        detection_score=frame.detection_score,
    )


# --------------------------------------------------------------------------- #
# §2 — Agregación estática
# --------------------------------------------------------------------------- #


def aggregate_static(vectors: tuple[FeatureVector, ...]) -> StaticFeatures:
    """`F = mean_t(f_t)` y `σ = mean_j(std_t(f_t[j]))` (§2).

    Las sumas recorren el tiempo en orden ascendente y dividen al final, tal como
    exige el §5.4: la suma en punto flotante no es asociativa.

    `std` es la desviación **poblacional** (ddof = 0). El contrato no lo decía y
    σ no viaja en los golden vectors de un frame, así que una discrepancia con
    TypeScript aquí sería invisible para los tests y aparecería como ventanas
    rechazadas en el navegador que en escritorio pasaban. Queda fijado en
    `docs/feature-spec.md` §2.
    """
    if not vectors:
        raise ValueError("no se puede agregar una ventana vacía")
    count = len(vectors)

    means: list[float] = []
    for component in range(NUM_FEATURES):
        total = 0.0
        for vector in vectors:
            total += vector.values[component]
        means.append(total / count)

    total_std = 0.0
    for component in range(NUM_FEATURES):
        squared = 0.0
        for vector in vectors:
            deviation = vector.values[component] - means[component]
            squared += deviation * deviation
        total_std += math.sqrt(squared / count)

    return StaticFeatures(
        shape=FeatureVector(values=tuple(means), spec_version=FEATURE_SPEC_VERSION),
        dispersion=total_std / NUM_FEATURES,
    )


# --------------------------------------------------------------------------- #
# §3 — Canal de trayectoria y remuestreo
# --------------------------------------------------------------------------- #


def _trajectory_channel(geometries: tuple[_FrameGeometry, ...]) -> TrajectoryChannel:
    """`τ_t = (w_t - w_0) / s̄` (§3.1).

    `w_t` es la muñeca tras el paso 2 —aspecto corregido, `y` invertida,
    lateralidad canonizada— y **antes** de la traslación del paso 3. Ese detalle
    es la razón de ser de todo el §3: tomada después de trasladar, `w_t` sería
    idénticamente cero y una J resultaría indistinguible de una I.

    El origen en `w_0` la hace invariante a la posición en el encuadre; la
    división por `s̄` la hace invariante a la distancia a la cámara.
    """
    total_scale = 0.0
    for geometry in geometries:
        total_scale += geometry.scale
    mean_scale = total_scale / len(geometries)

    origin_x, origin_y = geometries[0].wrist
    points = tuple(
        (
            (geometry.wrist[0] - origin_x) / mean_scale,
            (geometry.wrist[1] - origin_y) / mean_scale,
        )
        for geometry in geometries
    )
    return TrajectoryChannel(points=points, mean_scale=mean_scale)


def resample(
    rows: tuple[tuple[float, ...], ...], length: int = RESAMPLE_LENGTH
) -> tuple[tuple[float, ...], ...]:
    """Remuestreo temporal lineal a `length` filas (§3.2).

    Mapeo de índices, componente a componente:

        pos_j = (j · (T_src − 1)) / (length − 1)
        i = floor(pos_j);  frac = pos_j − i
        out_j = rows[i] + frac · (rows[i+1] − rows[i])

    El producto va **antes** que la división a propósito: `(j·(T−1))/(L−1)`
    devuelve el índice exacto cuando `T == L`, mientras que `(j/(L−1))·(T−1)`
    arrastra el redondeo del cociente intermedio y desplaza filas que deberían
    quedarse quietas. Los extremos se preservan exactamente en ambos casos.

    Con una sola fila de origen se replica: no hay trayectoria que interpolar.
    """
    if length < 1:
        raise ValueError(f"length debe ser ≥ 1, no {length}")
    if not rows:
        raise ValueError("no se puede remuestrear una secuencia vacía")

    source_length = len(rows)
    if source_length == 1 or length == 1:
        return tuple(rows[0] for _ in range(length))

    last = source_length - 1
    steps = length - 1
    output: list[tuple[float, ...]] = []
    for step in range(length):
        position = (step * last) / steps
        lower = math.floor(position)
        if lower >= last:
            output.append(rows[last])
            continue
        fraction = position - lower
        start = rows[lower]
        end = rows[lower + 1]
        output.append(
            tuple(a + fraction * (b - a) for a, b in zip(start, end, strict=True))
        )
    return tuple(output)


def _dynamic_features(
    frame_vectors: tuple[FeatureVector, ...],
    trajectory: TrajectoryChannel,
    config: Config,
) -> DynamicFeatures | DynamicUnavailable:
    """`g_t = concat(f_t, w_τ · τ_t) ∈ ℝ⁴⁴`, remuestreado a 24 filas (§3.2-§3.3).

    Se remuestrean por separado el canal de forma y el de trayectoria, y la
    ponderación se aplica **después** de interpolar. Multiplicar antes daría
    resultados distintos bit a bit.
    """
    source_length = len(frame_vectors)
    if source_length < config.dtw.min_source_frames:
        return DynamicUnavailable(reason=DynamicRejection.TOO_FEW_SOURCE_FRAMES)

    shape_rows = resample(tuple(vector.values for vector in frame_vectors))
    trajectory_rows = resample(trajectory.points)
    weight = config.features.trajectory_weight

    rows = tuple(
        (*shape, weight * point[0], weight * point[1])
        for shape, point in zip(shape_rows, trajectory_rows, strict=True)
    )
    return DynamicFeatures(rows=rows, source_length=source_length)


# --------------------------------------------------------------------------- #
# Velocidad — insumo de la segmentación, no del clasificador
# --------------------------------------------------------------------------- #


def mean_displacement(before: Points3, after: Points3) -> float:
    """Desplazamiento medio de los landmarks entre dos frames, en 2D.

    Ignora `z` por la misma razón que el paso 4: es ruido. No está normalizado
    por escala; de eso se encarga quien lo llama.
    """
    total = 0.0
    for (x0, y0, _), (x1, y1, _) in zip(before, after, strict=True):
        total += math.hypot(x1 - x0, y1 - y0)
    return total / len(before)


def _velocities(
    geometries: tuple[_FrameGeometry, ...], mean_scale: float
) -> tuple[float, ...]:
    """Velocidad por par de frames consecutivos, en unidades de mano por frame.

    Se mide sobre los puntos del paso 2 —sin trasladar—, de modo que cuenta tanto
    el cambio de configuración de la mano como su desplazamiento por el encuadre:
    la máquina de estados necesita saber que la mano está quieta, no solo que su
    forma no cambió. Dividir por `s̄` la hace invariante a la distancia: la misma
    seña ejecutada más cerca de la cámara no parece más rápida.

    Esta función no forma parte del vector de features, pero sí del contrato con
    la implementación web: `segmentation.ts` tiene que medir la velocidad igual.
    """
    return tuple(
        mean_displacement(previous.canonical, current.canonical) / mean_scale
        for previous, current in pairwise(geometries)
    )


# --------------------------------------------------------------------------- #
# Composición
# --------------------------------------------------------------------------- #


def extract_sequence_features(sequence: Sequence, config: Config) -> ExtractionOutcome:
    """Tubería completa: de `(T, 21, 3)` crudos a todo lo que consume el sistema.

    Orden: suavizado (§4) → pasos 1-7 por frame (§1) → agregación estática (§2) →
    canal de trayectoria (§3.1) → remuestreo y ponderación (§3.2, §3.3).

    Devuelve `ExtractionRejected` —no lanza, no devuelve `None`— cuando algún
    frame tiene escala degenerada, e indica cuál.
    """
    smoothed = smooth_sequence(sequence, config.smoothing.alpha)
    geometries = _sequence_geometry(smoothed)
    if isinstance(geometries, ExtractionRejected):
        return geometries

    frame_vectors = tuple(geometry.features for geometry in geometries)
    trajectory = _trajectory_channel(geometries)
    return SequenceFeatures(
        frames=frame_vectors,
        static=aggregate_static(frame_vectors),
        trajectory=trajectory,
        dynamic=_dynamic_features(frame_vectors, trajectory, config),
        velocities=_velocities(geometries, trajectory.mean_scale),
        spec_version=FEATURE_SPEC_VERSION,
    )


def split_valid_runs(stream: FrameStream) -> tuple[Sequence, ...]:
    """Parte un flujo crudo en sus secuencias válidas máximas.

    `feature-spec.md` §0.3: los frames inválidos **no se interpolan**, interrumpen
    la secuencia. Dónde cae el hueco importa y no solo cuántos hay:

    - al inicio, la secuencia empieza más tarde y el origen del canal de
      trayectoria se mueve al primer frame válido;
    - al final, la secuencia simplemente se corta antes;
    - en medio, hay **dos** secuencias, no una con un salto. Coserlas inventaría
      un movimiento entre dos posiciones que nunca se observó.
    """
    runs: list[Sequence] = []
    current: list[RawFrame] = []
    for slot in stream:
        if isinstance(slot, RawFrame):
            current.append(slot)
            continue
        if current:
            runs.append(Sequence(frames=tuple(current)))
            current = []
    if current:
        runs.append(Sequence(frames=tuple(current)))
    return tuple(runs)
