"""Medidas de la geometría de una mano detectada (Paso 1 de la tolerancia).

Antes de decidir que un cuadro es físicamente imposible hay que saber cuánto
varía una mano real de un cuadro a otro. Este módulo mide, sobre landmarks
crudos, lo que el filtro de plausibilidad va a vigilar:

- **el largo de cada hueso** —los 20 segmentos del árbol de la mano— en unidades
  de palma, en 2D (como el resto del contrato) y en 3D (con la `z` de
  MediaPipe): un hueso real no cambia de largo, su proyección sí;
- **cuánto se levanta cada falange proximal hacia el dorso** en el nudillo
  (MCP), respecto del plano de la palma: una falange doblada hacia atrás más
  allá de lo posible;
- **el salto de la palma** entre dos cuadros, en unidades de palma por segundo.

Todas las medidas empiezan por los pasos 1 y 2 del contrato (`feature-spec.md`
§1): relación de aspecto, `y` hacia arriba y la mano llevada a derecha. La `z`
se escala con la relación de aspecto igual que `x`, como hace el paso 1.

Código puro: sin cámara, sin disco, sin reloj.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

from lsm.features import (
    MIN_SCALE,
    PALM_SEGMENTS,
    canonicalize_handedness,
    correct_aspect_and_orientation,
    palm_size,
)
from lsm.types import LandmarkIndex, Points3, RawFrame

L = LandmarkIndex

#: Los 20 huesos del árbol de la mano: cada landmark salvo la muñeca, unido a su
#: padre. Los cuatro metacarpianos cuelgan de la muñeca; el pulgar, de su CMC.
BONES: Final[tuple[tuple[int, int], ...]] = (
    (L.WRIST, L.THUMB_CMC),
    (L.THUMB_CMC, L.THUMB_MCP),
    (L.THUMB_MCP, L.THUMB_IP),
    (L.THUMB_IP, L.THUMB_TIP),
    (L.WRIST, L.INDEX_MCP),
    (L.INDEX_MCP, L.INDEX_PIP),
    (L.INDEX_PIP, L.INDEX_DIP),
    (L.INDEX_DIP, L.INDEX_TIP),
    (L.WRIST, L.MIDDLE_MCP),
    (L.MIDDLE_MCP, L.MIDDLE_PIP),
    (L.MIDDLE_PIP, L.MIDDLE_DIP),
    (L.MIDDLE_DIP, L.MIDDLE_TIP),
    (L.WRIST, L.RING_MCP),
    (L.RING_MCP, L.RING_PIP),
    (L.RING_PIP, L.RING_DIP),
    (L.RING_DIP, L.RING_TIP),
    (L.WRIST, L.PINKY_MCP),
    (L.PINKY_MCP, L.PINKY_PIP),
    (L.PINKY_PIP, L.PINKY_DIP),
    (L.PINKY_DIP, L.PINKY_TIP),
)

#: Los cinco dedos, con sus landmarks de la base a la punta.
FINGERS: Final[dict[str, tuple[int, ...]]] = {
    "pulgar": (L.THUMB_CMC, L.THUMB_MCP, L.THUMB_IP, L.THUMB_TIP),
    "indice": (L.INDEX_MCP, L.INDEX_PIP, L.INDEX_DIP, L.INDEX_TIP),
    "medio": (L.MIDDLE_MCP, L.MIDDLE_PIP, L.MIDDLE_DIP, L.MIDDLE_TIP),
    "anular": (L.RING_MCP, L.RING_PIP, L.RING_DIP, L.RING_TIP),
    "menique": (L.PINKY_MCP, L.PINKY_PIP, L.PINKY_DIP, L.PINKY_TIP),
}

#: Los nudillos de los cuatro dedos largos: (dedo, nudillo, articulación
#: siguiente). La falange proximal va del nudillo al PIP.
MCP_JOINTS: Final[tuple[tuple[str, int, int], ...]] = (
    ("indice", L.INDEX_MCP, L.INDEX_PIP),
    ("medio", L.MIDDLE_MCP, L.MIDDLE_PIP),
    ("anular", L.RING_MCP, L.RING_PIP),
    ("menique", L.PINKY_MCP, L.PINKY_PIP),
)

#: Muñeca y nudillos: los puntos cuyo centro es el «centro de la palma».
PALM_POINTS: Final[tuple[int, ...]] = (
    L.WRIST,
    L.INDEX_MCP,
    L.MIDDLE_MCP,
    L.RING_MCP,
    L.PINKY_MCP,
)


def canonical_points(frame: RawFrame) -> Points3:
    """Pasos 1 y 2 del contrato: aspecto, `y` hacia arriba, mano derecha."""
    return canonicalize_handedness(
        correct_aspect_and_orientation(frame.points(), frame.aspect_ratio),
        frame.handedness,
    )


def _dist(
    a: tuple[float, float, float], b: tuple[float, float, float], depth: bool
) -> float:
    dz = (a[2] - b[2]) if depth else 0.0
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + dz * dz)


def palm_size_3d(points: Points3) -> float:
    """El tamaño de palma del contrato, pero con `z`: la mayor distancia 3D
    entre los `PALM_SEGMENTS`. No se encoge al girar la palma."""
    return max(_dist(points[a], points[b], True) for a, b in PALM_SEGMENTS)


def bone_lengths(points: Points3, *, depth: bool) -> tuple[float, ...] | None:
    """El largo de los 20 `BONES`, dividido entre el tamaño de palma.

    Con `depth` los largos y la palma son 3D; sin él, 2D (la palma es la del
    paso 4). `None` si la palma es degenerada.
    """
    palma = palm_size_3d(points) if depth else palm_size(points)
    if palma < MIN_SCALE:
        return None
    return tuple(_dist(points[a], points[b], depth) / palma for a, b in BONES)


def _resta(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cruz(
    u: tuple[float, float, float], v: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def _punto(u: tuple[float, float, float], v: tuple[float, float, float]) -> float:
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def mcp_dorsal_elevation(points: Points3) -> tuple[float, ...] | None:
    """Cuánto se levanta hacia el dorso cada falange proximal, en grados.

    `n = unit((p_5 − p_0) × (p_17 − p_0))` es la normal de la palma de la mano
    canónica (tras el paso 2), que apunta al dorso. Para cada `MCP_JOINTS`, con
    `u = p_nudillo − p_0` (el metacarpiano) y `v = p_pip − p_nudillo`:

        elevación = asin(v̂ · n) − asin(û · n)

    la de la falange sobre el plano de la palma, menos la del propio
    metacarpiano. Negativa doblando hacia la palma (un puño da −25° a −80°),
    positiva hacia el dorso. Usa la `z` de MediaPipe: con la palma de canto la
    normal es casi toda `z` y el número se vuelve ruido (ADR 0026). `None` si la
    palma o algún segmento son degenerados.
    """
    n = _cruz(
        _resta(points[L.INDEX_MCP], points[L.WRIST]),
        _resta(points[L.PINKY_MCP], points[L.WRIST]),
    )
    norma = math.sqrt(_punto(n, n))
    if norma < MIN_SCALE:
        return None
    normal = (n[0] / norma, n[1] / norma, n[2] / norma)
    elevaciones: list[float] = []
    for _, nudillo, pip in MCP_JOINTS:
        u = _resta(points[nudillo], points[L.WRIST])
        v = _resta(points[pip], points[nudillo])
        nu, nv = math.sqrt(_punto(u, u)), math.sqrt(_punto(v, v))
        if nu < MIN_SCALE or nv < MIN_SCALE:
            return None
        elevaciones.append(
            math.degrees(
                math.asin(_acotado(_punto(v, normal) / nv))
                - math.asin(_acotado(_punto(u, normal) / nu))
            )
        )
    return tuple(elevaciones)


def _acotado(c: float) -> float:
    """Clamp a `[-1, 1]` antes de `asin` (`feature-spec.md` §5.2)."""
    return max(-1.0, min(1.0, c))


def palm_center(points: Points3) -> tuple[float, float]:
    """El centro 2D de la muñeca y los cuatro nudillos."""
    n = len(PALM_POINTS)
    return (
        sum(points[i][0] for i in PALM_POINTS) / n,
        sum(points[i][1] for i in PALM_POINTS) / n,
    )


def palm_jump_per_s(before: Points3, after: Points3, dt_ms: float) -> float | None:
    """Cuánto se desplazó el centro de la palma, en palmas por segundo.

    El divisor es la media de las dos palmas 2D. `None` si alguna es degenerada
    o si `dt_ms` no es positivo.
    """
    if dt_ms <= 0.0:
        return None
    m_a, m_b = palm_size(before), palm_size(after)
    if min(m_a, m_b) < MIN_SCALE:
        return None
    (xa, ya), (xb, yb) = palm_center(before), palm_center(after)
    return math.hypot(xb - xa, yb - ya) / ((m_a + m_b) / 2.0) * 1000.0 / dt_ms


def finger_speed_per_s(
    before: Points3, after: Points3, dt_ms: float, finger: str
) -> float | None:
    """Desplazamiento medio 2D de los puntos de un dedo, relativo a la muñeca,
    en palmas por segundo: el movimiento del dedo sin el de la mano entera."""
    if dt_ms <= 0.0:
        return None
    m_a, m_b = palm_size(before), palm_size(after)
    if min(m_a, m_b) < MIN_SCALE:
        return None
    wa, wb = before[L.WRIST], after[L.WRIST]
    total = 0.0
    puntos = FINGERS[finger]
    for i in puntos:
        ax, ay = before[i][0] - wa[0], before[i][1] - wa[1]
        bx, by = after[i][0] - wb[0], after[i][1] - wb[1]
        total += math.hypot(bx - ax, by - ay)
    return total / len(puntos) / ((m_a + m_b) / 2.0) * 1000.0 / dt_ms


def median(values: Sequence[float]) -> float:
    """Mediana; con un número par de valores, la media de los dos centrales."""
    if not values:
        raise ValueError("mediana de una lista vacía")
    ordenados = sorted(values)
    medio = len(ordenados) // 2
    if len(ordenados) % 2:
        return ordenados[medio]
    return (ordenados[medio - 1] + ordenados[medio]) / 2.0
