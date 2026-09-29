"""Filtro de plausibilidad anatómica (`feature-spec.md` §0.4, FEATURE_SPEC 4).

MediaPipe a veces entrega una mano que no puede existir: una falange que mide el
doble que un cuadro antes, la mano entera que salta media pantalla en 30 ms. Ese
cuadro no se corrige —no se enderezan dedos ni se inventan posiciones—: se marca
**inválido** (`InvalidReason.IMPLAUSIBLE`) y lo resuelve el relleno de huecos
igual que un cuadro sin mano.

Un cuadro con mano es inválido si, en este orden:

1. **Hueso** (`BONE`): algún hueso de los 20 (`landmark_stats.BONES`) se aleja
   de su referencia más de `bone_max_deviation` palmas. Largo 3D en unidades de
   palma 3D (`landmark_stats.bone_lengths(depth=True)`); la referencia de cada
   hueso es la mediana de ese hueso en los últimos `bone_reference_frames`
   cuadros **aceptados**, y solo se comprueba con la referencia llena.
2. **Articulación** (`JOINT`), si `max_mcp_dorsal_deg` no es `None`: alguna
   falange proximal se levanta hacia el dorso más de ese ángulo
   (`landmark_stats.mcp_dorsal_elevation`). **Desactivado** por defecto: en
   grabaciones limpias MediaPipe da elevaciones que una mano real no puede hacer
   (ADR 0026), así que no hay umbral que no tire poses buenas.
3. **Salto** (`JUMP`): el centro de la palma se desplazó, desde el último cuadro
   aceptado, más de `max_palm_speed_per_s` palmas por segundo de tiempo real.

Un cuadro inválido no toca la referencia ni el «último aceptado». Si los cuadros
con mano se rechazan sin interrupción durante `reset_after_ms` o más —la
referencia ya no describe a la mano que hay delante—, la referencia se vacía y
ese cuadro se acepta: empieza otra. Un cuadro con palma degenerada no se juzga
aquí (lo invalida el paso 4, `SCALE_TOO_SMALL`) y no entra en la referencia.

Código puro: el tiempo llega en el cuadro (`lsm.timing`).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum

from lsm.config import Config
from lsm.landmark_stats import (
    BONES,
    bone_lengths,
    canonical_points,
    mcp_dorsal_elevation,
    median,
    palm_jump_per_s,
)
from lsm.types import FrameSlot, InvalidFrame, InvalidReason, Points3, RawFrame


class ImplausibleKind(StrEnum):
    """Qué comprobación falló. Va en el `detail` del `InvalidFrame`."""

    BONE = "BONE"
    JOINT = "JOINT"
    JUMP = "JUMP"


@dataclass(frozen=True, slots=True)
class PlausibilityParams:
    """Los umbrales de `config.plausibility`."""

    enabled: bool
    bone_max_deviation: float
    bone_reference_frames: int
    max_mcp_dorsal_deg: float | None
    max_palm_speed_per_s: float
    reset_after_ms: float

    @classmethod
    def from_config(cls, config: Config) -> PlausibilityParams:
        p = config.plausibility
        return cls(
            enabled=p.enabled,
            bone_max_deviation=p.bone_max_deviation,
            bone_reference_frames=p.bone_reference_frames,
            max_mcp_dorsal_deg=p.max_mcp_dorsal_deg,
            max_palm_speed_per_s=p.max_palm_speed_per_s,
            reset_after_ms=p.reset_after_ms,
        )


@dataclass
class PlausibilityFilter:
    """El filtro cuadro a cuadro. `check` decide y actualiza el estado."""

    params: PlausibilityParams
    _huesos: deque[tuple[float, ...]] = field(init=False, repr=False)
    _ultimo: tuple[Points3, float] | None = field(default=None, repr=False)
    #: Tiempo del primer cuadro de la racha de rechazos en curso.
    _racha_desde: float | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._huesos = deque(maxlen=self.params.bone_reference_frames)

    def reset(self) -> None:
        self._huesos.clear()
        self._ultimo = None
        self._racha_desde = None

    def check(self, frame: RawFrame, t_ms: float) -> ImplausibleKind | None:
        """`None` si el cuadro es plausible; si no, qué comprobación falló."""
        if not self.params.enabled:
            return None
        puntos = canonical_points(frame)
        largos = bone_lengths(puntos, depth=True)
        if largos is None:
            return None
        motivo = self._juzgar(puntos, largos, t_ms)
        if motivo is not None:
            if self._racha_desde is None:
                self._racha_desde = t_ms
            if t_ms - self._racha_desde < self.params.reset_after_ms:
                return motivo
            # La referencia ya no describe a la mano de delante: se empieza otra
            # con este cuadro.
            self.reset()
        self._racha_desde = None
        self._huesos.append(largos)
        self._ultimo = (puntos, t_ms)
        return None

    def _juzgar(
        self, puntos: Points3, largos: tuple[float, ...], t_ms: float
    ) -> ImplausibleKind | None:
        p = self.params
        if len(self._huesos) == p.bone_reference_frames:
            for k in range(len(BONES)):
                referencia = median([h[k] for h in self._huesos])
                if abs(largos[k] - referencia) > p.bone_max_deviation:
                    return ImplausibleKind.BONE
        if p.max_mcp_dorsal_deg is not None:
            elevaciones = mcp_dorsal_elevation(puntos)
            if elevaciones is not None and max(elevaciones) > p.max_mcp_dorsal_deg:
                return ImplausibleKind.JOINT
        if self._ultimo is not None:
            anteriores, t_anterior = self._ultimo
            salto = palm_jump_per_s(anteriores, puntos, t_ms - t_anterior)
            if salto is not None and salto > p.max_palm_speed_per_s:
                return ImplausibleKind.JUMP
        return None


def filter_stream(
    stream: Iterable[FrameSlot], params: PlausibilityParams, fps: float
) -> Iterator[FrameSlot]:
    """El flujo con los cuadros implausibles convertidos en `InvalidFrame`.

    El tiempo de un cuadro sin marca es `índice · 1000 / fps`
    (`lsm.timing.frame_times_ms`), igual que en el resto de la tubería. El
    cuadro inválido conserva la marca del original, y su `detail` dice qué
    comprobación falló.
    """
    filtro = PlausibilityFilter(params)
    paso = 1000.0 / fps
    for indice, slot in enumerate(stream):
        if not isinstance(slot, RawFrame):
            yield slot
            continue
        t = slot.timestamp_ms if slot.timestamp_ms is not None else indice * paso
        motivo = filtro.check(slot, t)
        if motivo is None:
            yield slot
        else:
            yield InvalidFrame(
                reason=InvalidReason.IMPLAUSIBLE,
                detail=motivo.value,
                timestamp_ms=slot.timestamp_ms,
            )


def is_implausible(slot: FrameSlot) -> bool:
    return isinstance(slot, InvalidFrame) and slot.reason is InvalidReason.IMPLAUSIBLE
