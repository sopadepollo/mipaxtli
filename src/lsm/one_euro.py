"""Filtro One Euro sobre los landmarks crudos (`feature-spec.md` §4, v5).

Casiez, Roussel y Vogel (2012), «1 € Filter: A Simple Speed-based Low-pass
Filter for Noisy Input in Interactive Systems». Un pasa-bajos de primer orden
cuya frecuencia de corte sube con la velocidad: con la mano quieta corta fuerte
y quita el temblor de MediaPipe; con la mano en movimiento corta poco y casi no
retrasa el trazo. Sustituye a la media exponencial de α fijo, que para quitar el
temblor tenía que emborronar justo los trazos que distinguen a las dinámicas.

Cada coordenada de cada landmark (x, y, z de los 21) es una señal aparte, con su
propio estado. Para la muestra `k` de una señal, con `Δ_k` el tiempo real entre
cuadros en segundos (`lsm.timing`: la marca del cuadro, o el índice a la tasa
nominal):

    α(f, Δ)  = r / (r + 1),   r = 2π · f · Δ

    k = 0:   x̂_0 = x_0 ;  d̂_0 = 0
    k ≥ 1:   d_k  = (x_k − x̂_{k−1}) / Δ_k
             d̂_k  = α(d_cutoff, Δ_k) · d_k + (1 − α(d_cutoff, Δ_k)) · d̂_{k−1}
             f_k  = min_cutoff + β · |d̂_k| · e / m_k
             x̂_k  = α(f_k, Δ_k) · x_k + (1 − α(f_k, Δ_k)) · x̂_{k−1}

**Una desviación del original, a propósito:** la velocidad que abre el filtro,
`|d̂_k| · e / m_k`, va en **palmas por segundo** y no en unidades crudas. `e` es
el factor del paso 1 para ese eje (`a = ancho/alto` para x y z, 1 para y) y
`m_k` el tamaño de palma del cuadro `k` en unidades del paso 1 (`features.
palm_size`). Así `β` significa lo mismo con la mano cerca o lejos de la cámara y
en cualquier resolución, como los umbrales de velocidad de la segmentación. Si la
palma del cuadro es degenerada se usa la del último cuadro que no lo era.

El estado se reinicia con cada secuencia: un hueco que no se rellena corta la
secuencia (§0.3) y con ella el filtro. Código puro; el tiempo llega en el cuadro.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace

from lsm.config import Config
from lsm.features import (
    MIN_SCALE,
    correct_aspect_and_orientation,
    palm_size,
)
from lsm.types import Landmark, RawFrame, Sequence


@dataclass(frozen=True, slots=True)
class OneEuroParams:
    """Los tres parámetros del filtro. `enabled = False` lo vuelve la identidad."""

    enabled: bool
    #: Frecuencia de corte con la mano quieta, en Hz. Más baja, más suave y más
    #: retraso en reposo.
    min_cutoff: float
    #: Cuánto sube la frecuencia de corte por cada palma por segundo de
    #: velocidad. Más alto, menos retraso en los trazos y más temblor.
    beta: float
    #: Frecuencia de corte del filtro de la derivada, en Hz.
    d_cutoff: float

    @classmethod
    def from_config(cls, config: Config) -> OneEuroParams:
        s = config.smoothing
        return cls(
            enabled=s.enabled,
            min_cutoff=s.min_cutoff,
            beta=s.beta,
            d_cutoff=s.d_cutoff,
        )

    def to_json(self) -> dict[str, float | bool]:
        return {
            "enabled": self.enabled,
            "min_cutoff": self.min_cutoff,
            "beta": self.beta,
            "d_cutoff": self.d_cutoff,
        }


def smoothing_factor(cutoff_hz: float, dt_s: float) -> float:
    """`α(f, Δ) = r / (r + 1)` con `r = 2π · f · Δ`."""
    r = 2.0 * math.pi * cutoff_hz * dt_s
    return r / (r + 1.0)


@dataclass
class OneEuroFilter:
    """El filtro sobre un flujo de frames, cuadro a cuadro.

    `step` recibe cada frame con su tiempo en ms y devuelve el frame filtrado,
    con los mismos metadatos. `reset` lo deja como recién creado: lo llama
    quien sabe que la secuencia se cortó.
    """

    params: OneEuroParams
    _x: list[float] = field(default_factory=list, repr=False)
    _d: list[float] = field(default_factory=list, repr=False)
    _t_ms: float | None = field(default=None, repr=False)
    _palma: float | None = field(default=None, repr=False)

    def reset(self) -> None:
        self._x = []
        self._d = []
        self._t_ms = None
        self._palma = None

    def step(self, frame: RawFrame, t_ms: float) -> RawFrame:
        if not self.params.enabled:
            return frame
        crudos = [c for lm in frame.landmarks for c in (lm.x, lm.y, lm.z)]
        palma = palm_size(
            correct_aspect_and_orientation(frame.points(), frame.aspect_ratio)
        )
        if palma >= MIN_SCALE:
            self._palma = palma
        if self._t_ms is None or not self._x:
            self._x = crudos
            self._d = [0.0] * len(crudos)
            self._t_ms = t_ms
            return frame
        dt_s = (t_ms - self._t_ms) / 1000.0
        if dt_s <= 0.0:
            msg = f"el tiempo del filtro no avanza: {self._t_ms} → {t_ms} ms"
            raise ValueError(msg)
        a = frame.aspect_ratio
        alfa_d = smoothing_factor(self.params.d_cutoff, dt_s)
        filtrados: list[float] = []
        for k, x in enumerate(crudos):
            previo = self._x[k]
            d = (x - previo) / dt_s
            d_hat = alfa_d * d + (1.0 - alfa_d) * self._d[k]
            eje = 1.0 if k % 3 == 1 else a
            velocidad = (
                abs(d_hat) * eje / self._palma if self._palma is not None else 0.0
            )
            corte = self.params.min_cutoff + self.params.beta * velocidad
            alfa = smoothing_factor(corte, dt_s)
            x_hat = alfa * x + (1.0 - alfa) * previo
            self._d[k] = d_hat
            filtrados.append(x_hat)
        self._x = filtrados
        self._t_ms = t_ms
        return replace(
            frame,
            landmarks=tuple(
                Landmark(x=filtrados[i], y=filtrados[i + 1], z=filtrados[i + 2])
                for i in range(0, len(filtrados), 3)
            ),
        )


def filter_sequence(
    sequence: Sequence, times_ms: Iterable[float], params: OneEuroParams
) -> Sequence:
    """Filtra una secuencia desde su primer frame, con estado nuevo."""
    if not params.enabled:
        return sequence
    filtro = OneEuroFilter(params)
    return Sequence(
        frames=tuple(
            filtro.step(frame, t)
            for frame, t in zip(sequence.frames, times_ms, strict=True)
        )
    )


def filter_frames(
    frames: Iterable[tuple[RawFrame, float]], params: OneEuroParams
) -> Iterator[RawFrame]:
    """Filtra `(frame, t_ms)` en orden, sin reiniciar."""
    filtro = OneEuroFilter(params)
    for frame, t in frames:
        yield filtro.step(frame, t)
