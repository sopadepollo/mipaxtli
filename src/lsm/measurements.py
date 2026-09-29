"""Las mediciones que fijan los umbrales de la tolerancia (Paso 1, ADR 0026).

Se miden sobre grabaciones reales —las sesiones de diagnóstico y las muestras
del dataset— y no se decide nada aquí: el filtro de plausibilidad, el One Euro
y el relleno de huecos toman sus umbrales de estos números.

Una `Recording` es un flujo de cuadros con, por cuadro, la letra que se estaba
haciendo y la **fase** del intento:

- `ANTES` del trazo: la mano llegando a la pose inicial;
- `DURANTE`: el trazo, desde `pre_candidate_ms` antes de que la máquina entrara
  a DYNAMIC_CANDIDATE hasta que salió;
- `FINAL`: la pose final, tras el trazo;
- `SIN_TRAZO`: un intento en el que la máquina nunca vio un trazo;
- `REPOSO`: la prueba de reposo; `ESTATICA`: una muestra estática.

Código puro: los datos llegan ya leídos.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeVar

from lsm.landmark_stats import (
    BONES,
    FINGERS,
    MCP_JOINTS,
    bone_lengths,
    canonical_points,
    finger_speed_per_s,
    mcp_dorsal_elevation,
    median,
    palm_jump_per_s,
)
from lsm.timing import frame_times_ms
from lsm.types import FrameStream, Points3, RawFrame


class Phase(StrEnum):
    ANTES = "antes"
    DURANTE = "durante"
    FINAL = "final"
    SIN_TRAZO = "sin_trazo"
    REPOSO = "reposo"
    ESTATICA = "estatica"


@dataclass(frozen=True, slots=True)
class Recording:
    """Un flujo con la letra y la fase de cada cuadro.

    `group` agrupa las grabaciones que comparten mediana de huesos: la misma
    persona en la misma sesión. `fps` es la tasa con la que se fecha un cuadro
    sin marca de tiempo.
    """

    source: str
    group: str
    stream: FrameStream
    letters: tuple[str | None, ...]
    phases: tuple[Phase | None, ...]
    fps: float
    #: Estado de la máquina al llegar cada cuadro, si se sabe (sesiones en
    #: vivo). Vacío si no.
    states: tuple[str, ...] = ()


def phases_from_states(
    states: Sequence[str],
    wall_ms: Sequence[float],
    attempt_bounds: Sequence[tuple[int, int]],
    pre_candidate_ms: float,
    candidate_state: str = "DYNAMIC_CANDIDATE",
) -> tuple[Phase | None, ...]:
    """La fase de cada cuadro de una sesión guiada.

    Por intento `[inicio, fin)`: el trazo va de `pre_candidate_ms` antes del
    primer cuadro en DYNAMIC_CANDIDATE hasta el último; antes es ANTES y después
    FINAL. Si el intento no tuvo candidato, todo es SIN_TRAZO. Fuera de los
    intentos, `None`.
    """
    fases: list[Phase | None] = [None] * len(states)
    for inicio, fin in attempt_bounds:
        candidatos = [i for i in range(inicio, fin) if states[i] == candidate_state]
        if not candidatos:
            for i in range(inicio, fin):
                fases[i] = Phase.SIN_TRAZO
            continue
        desde = wall_ms[candidatos[0]] - pre_candidate_ms
        hasta = candidatos[-1]
        for i in range(inicio, fin):
            if i > hasta:
                fases[i] = Phase.FINAL
            elif wall_ms[i] >= desde:
                fases[i] = Phase.DURANTE
            else:
                fases[i] = Phase.ANTES
    return tuple(fases)


def attempt_bounds(
    prompts: Sequence[str | None], repetitions: Sequence[int | None]
) -> tuple[tuple[int, int], ...]:
    """Los tramos contiguos con la misma (letra, repetición), como el reporte
    de diagnóstico: un intento repetido con BACKSPACE es otro tramo."""
    tramos: list[tuple[int, int]] = []
    clave_anterior: tuple[str | None, int | None] | None = None
    for i, clave in enumerate(zip(prompts, repetitions, strict=True)):
        if clave[0] is None:
            clave_anterior = None
            continue
        if clave == clave_anterior and tramos and tramos[-1][1] == i:
            tramos[-1] = (tramos[-1][0], i + 1)
        else:
            tramos.append((i, i + 1))
        clave_anterior = clave
    return tuple(tramos)


# --------------------------------------------------------------------------- #
# Acumulación
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class GapRecord:
    letter: str
    phase: Phase
    #: Estado de la máquina en el primer cuadro del hueco, si se sabe.
    state: str | None
    frames: int
    #: Del primer cuadro sin mano al primero con mano de vuelta.
    duration_ms: float
    #: Scores del detector en los tres cuadros válidos anteriores (el más
    #: cercano primero); faltan si no los hay.
    scores_before: tuple[float, ...]


@dataclass
class Tally:
    """Todo lo medido, por letra. La única mutación del módulo."""

    #: (letra, fase) → [cuadros, con mano]
    detection: dict[tuple[str, Phase], list[int]] = field(default_factory=dict)
    gaps: list[GapRecord] = field(default_factory=list)
    #: letra → scores de todos los cuadros válidos.
    scores: dict[str, list[float]] = field(default_factory=dict)
    #: (letra, "2d"|"3d", "sesion"|"movil") → desviaciones |L/ref − 1| de cada
    #: hueso, y el máximo por cuadro.
    bone_dev: dict[tuple[str, str, str], list[float]] = field(default_factory=dict)
    bone_dev_max: dict[tuple[str, str, str], list[float]] = field(default_factory=dict)
    #: (letra, "2d"|"3d") → max_hueso |L − ref| contra la mediana móvil, en
    #: palmas, por cuadro: la diferencia absoluta y no la relativa.
    bone_abs_max: dict[tuple[str, str], list[float]] = field(default_factory=dict)
    #: (letra, dedo) → elevación dorsal de la falange proximal, en grados.
    mcp_dorsal: dict[tuple[str, str], list[float]] = field(default_factory=dict)
    #: letra → salto de la palma entre cuadros válidos consecutivos, palmas/s.
    jumps: dict[str, list[float]] = field(default_factory=dict)
    #: letra → el mismo salto, pero a través de un hueso (tras un hueco).
    jumps_after_gap: dict[str, list[float]] = field(default_factory=dict)
    #: (postura, dedo) → velocidad del dedo relativa a la muñeca, palmas/s.
    tremor: dict[tuple[str, str], list[float]] = field(default_factory=dict)
    #: (postura, "palma") → velocidad del centro de la palma, palmas/s.
    tremor_palm: dict[str, list[float]] = field(default_factory=dict)


_K = TypeVar("_K")


def _add(tabla: dict[_K, list[float]], clave: _K, valor: float) -> None:
    tabla.setdefault(clave, []).append(valor)


def measure(
    recordings: Iterable[Recording],
    tally: Tally,
    *,
    rolling_frames: int,
    rest_window_ms: float,
    rest_settle_ms: float,
) -> Tally:
    """Acumula en `tally` las medidas de cada grabación.

    `rolling_frames` es la ventana de la mediana móvil de los huesos (los
    últimos cuadros válidos); `rest_window_ms` la ventana de la velocidad de
    reposo, y `rest_settle_ms` lo que se descarta al principio de cada postura.
    Las medianas de sesión se calculan por `group` sobre todas sus grabaciones.
    """
    grabaciones = list(recordings)
    geometria = [_geometry(g) for g in grabaciones]

    # Medianas de sesión, por grupo y por hueso.
    por_grupo: dict[tuple[str, str], list[list[float]]] = {}
    for g, geo in zip(grabaciones, geometria, strict=True):
        for dim in ("2d", "3d"):
            columnas = por_grupo.setdefault((g.group, dim), [[] for _ in BONES])
            for largos in geo.lengths[dim]:
                if largos is None:
                    continue
                for k, valor in enumerate(largos):
                    columnas[k].append(valor)
    medianas = {
        clave: tuple(median(col) if col else math.nan for col in columnas)
        for clave, columnas in por_grupo.items()
    }

    for g, geo in zip(grabaciones, geometria, strict=True):
        _measure_one(
            g,
            geo,
            tally,
            medianas,
            rolling_frames=rolling_frames,
            rest_window_ms=rest_window_ms,
            rest_settle_ms=rest_settle_ms,
        )
    return tally


@dataclass(frozen=True, slots=True)
class _Geometry:
    points: tuple[Points3 | None, ...]
    lengths: dict[str, tuple[tuple[float, ...] | None, ...]]
    times: tuple[float, ...]


def _geometry(g: Recording) -> _Geometry:
    puntos = tuple(
        canonical_points(s) if isinstance(s, RawFrame) else None for s in g.stream
    )
    return _Geometry(
        points=puntos,
        lengths={
            "2d": tuple(
                None if p is None else bone_lengths(p, depth=False) for p in puntos
            ),
            "3d": tuple(
                None if p is None else bone_lengths(p, depth=True) for p in puntos
            ),
        },
        times=frame_times_ms(g.stream, g.fps),
    )


def _measure_one(
    g: Recording,
    geo: _Geometry,
    tally: Tally,
    medianas: dict[tuple[str, str], tuple[float, ...]],
    *,
    rolling_frames: int,
    rest_window_ms: float,
    rest_settle_ms: float,
) -> None:
    n = len(g.stream)
    tiempos = geo.times

    # Detección y scores.
    for i in range(n):
        letra, fase = g.letters[i], g.phases[i]
        if letra is None or fase is None:
            continue
        fila = tally.detection.setdefault((letra, fase), [0, 0])
        fila[0] += 1
        slot = g.stream[i]
        if isinstance(slot, RawFrame):
            fila[1] += 1
            _add(tally.scores, letra, slot.detection_score)

    # Huecos cerrados: del primer cuadro sin mano al primero con mano de vuelta.
    i = 0
    while i < n:
        if isinstance(g.stream[i], RawFrame) or i == 0:
            i += 1
            continue
        inicio = i
        while i < n and not isinstance(g.stream[i], RawFrame):
            i += 1
        if i >= n:
            break
        letra, fase = g.letters[inicio], g.phases[inicio]
        if letra is None or fase is None:
            continue
        previos: list[float] = []
        k = inicio - 1
        while k >= 0 and len(previos) < 3:
            s = g.stream[k]
            if not isinstance(s, RawFrame):
                break
            previos.append(s.detection_score)
            k -= 1
        tally.gaps.append(
            GapRecord(
                letter=letra,
                phase=fase,
                frames=i - inicio,
                state=g.states[inicio] if g.states else None,
                duration_ms=tiempos[i] - tiempos[inicio],
                scores_before=tuple(previos),
            )
        )

    # Huesos contra la mediana de la sesión y contra la móvil.
    for dim in ("2d", "3d"):
        referencia = medianas.get((g.group, dim))
        recientes: list[tuple[float, ...]] = []
        for i, largos in enumerate(geo.lengths[dim]):
            letra = g.letters[i]
            if largos is None or letra is None or g.phases[i] is None:
                if largos is not None:
                    recientes.append(largos)
                continue
            if referencia is not None:
                desv = [
                    abs(v / r - 1.0) for v, r in zip(largos, referencia, strict=True)
                ]
                tally.bone_dev.setdefault((letra, dim, "sesion"), []).extend(desv)
                _add(tally.bone_dev_max, (letra, dim, "sesion"), max(desv))
            if len(recientes) >= rolling_frames:
                ventana = recientes[-rolling_frames:]
                movil = [median([f[k] for f in ventana]) for k in range(len(BONES))]
                desv = [abs(v / r - 1.0) for v, r in zip(largos, movil, strict=True)]
                tally.bone_dev.setdefault((letra, dim, "movil"), []).extend(desv)
                _add(tally.bone_dev_max, (letra, dim, "movil"), max(desv))
                _add(
                    tally.bone_abs_max,
                    (letra, dim),
                    max(abs(v - r) for v, r in zip(largos, movil, strict=True)),
                )
            recientes.append(largos)
            del recientes[:-rolling_frames]

    # Elevación dorsal de las falanges proximales.
    for i, p in enumerate(geo.points):
        letra = g.letters[i]
        if p is None or letra is None or g.phases[i] is None:
            continue
        elevaciones = mcp_dorsal_elevation(p)
        if elevaciones is None:
            continue
        for (dedo, *_), grados in zip(MCP_JOINTS, elevaciones, strict=True):
            _add(tally.mcp_dorsal, (letra, dedo), grados)

    # Saltos de la palma.
    anterior: int | None = None
    hubo_hueco = False
    for i, p in enumerate(geo.points):
        if p is None:
            hubo_hueco = anterior is not None
            continue
        letra = g.letters[i]
        if anterior is not None and letra is not None and g.phases[i] is not None:
            previo = geo.points[anterior]
            assert previo is not None
            salto = palm_jump_per_s(previo, p, tiempos[i] - tiempos[anterior])
            if salto is not None:
                _add(tally.jumps_after_gap if hubo_hueco else tally.jumps, letra, salto)
        anterior = i
        hubo_hueco = False

    # Temblor en reposo: cada cuadro contra el más reciente con al menos
    # `rest_window_ms` de antigüedad, sin huecos entre medias.
    inicio_postura: dict[str, float] = {}
    historia: list[int] = []
    for i, p in enumerate(geo.points):
        letra = g.letters[i]
        if g.phases[i] is not Phase.REPOSO or letra is None or p is None:
            historia = []
            continue
        inicio_postura.setdefault(letra, tiempos[i])
        historia.append(i)
        if tiempos[i] - inicio_postura[letra] < rest_settle_ms:
            continue
        base = next(
            (
                j
                for j in reversed(historia[:-1])
                if tiempos[i] - tiempos[j] >= rest_window_ms
            ),
            None,
        )
        if base is None:
            continue
        previo = geo.points[base]
        assert previo is not None
        dt = tiempos[i] - tiempos[base]
        for dedo in FINGERS:
            v = finger_speed_per_s(previo, p, dt, dedo)
            if v is not None:
                _add(tally.tremor, (letra, dedo), v)
        salto = palm_jump_per_s(previo, p, dt)
        if salto is not None:
            _add(tally.tremor_palm, letra, salto)


def percentile(values: Sequence[float], q: float) -> float | None:
    """Rango más cercano, como `tracking_diagnostics.percentile`."""
    if not values:
        return None
    ordenados = sorted(values)
    return ordenados[min(len(ordenados) - 1, max(0, math.ceil(q * len(ordenados)) - 1))]
