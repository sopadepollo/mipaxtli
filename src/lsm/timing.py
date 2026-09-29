"""El tiempo de cada cuadro (ADR 0018).

Los cuadros llevan la marca que les puso `io/hands.py` —el mismo reloj que
recibe MediaPipe— o no llevan ninguna: el dataset anterior al esquema 5 y los
flujos sintéticos. Este módulo es el único sitio que decide qué hora es un
cuadro, para que el filtro One Euro, la duración de un hueco y cualquier otro
consumidor lean el mismo reloj.

Invariantes de un flujo con marcas, comprobados al leerlo de disco:

1. **Todo o nada**: o todos los cuadros llevan marca o ninguno. Mezclar tiempo
   real con tiempo derivado del índice daría saltos que no ocurrieron.
2. **Estrictamente creciente**: un cuadro que retrocede o repite marca es un
   error de captura y se rechaza, no se ordena.

Código puro: no lee ningún reloj. El tiempo llega en el cuadro.
"""

from __future__ import annotations

from collections.abc import Sequence

from lsm.types import FrameSlot


class TimestampError(ValueError):
    """Un flujo con marcas que no cumplen los invariantes del módulo."""


def check_timestamps(slots: Sequence[FrameSlot]) -> None:
    """Levanta `TimestampError` si el flujo mezcla o desordena marcas."""
    con_marca = [s.timestamp_ms is not None for s in slots]
    if any(con_marca) and not all(con_marca):
        primero = con_marca.index(not con_marca[0])
        msg = (
            f"el flujo mezcla cuadros con marca de tiempo y sin ella "
            f"(el cuadro {primero} rompe la racha del 0)"
        )
        raise TimestampError(msg)
    anterior: float | None = None
    for indice, slot in enumerate(slots):
        marca = slot.timestamp_ms
        if marca is None:
            return
        if anterior is not None and marca <= anterior:
            msg = (
                f"la marca de tiempo del cuadro {indice} ({marca} ms) no es "
                f"posterior a la del anterior ({anterior} ms)"
            )
            raise TimestampError(msg)
        anterior = marca


def frame_times_ms(slots: Sequence[FrameSlot], fps: float) -> tuple[float, ...]:
    """El tiempo de cada cuadro, en ms: su marca, o `índice · 1000 / fps`.

    Sin marcas se usa la tasa que se pase —la nominal para reproducir un
    dataset o un test, que es lo que la máquina de estados ya usa para
    convertir sus umbrales (feature-spec §6.5)—. Un flujo sin marcas da así
    exactamente los intervalos que daba antes de que existieran.
    """
    if fps <= 0.0:
        msg = f"la tasa ({fps}) tiene que ser positiva"
        raise ValueError(msg)
    check_timestamps(slots)
    if slots and slots[0].timestamp_ms is not None:
        return tuple(float(s.timestamp_ms or 0.0) for s in slots)
    paso = 1000.0 / fps
    return tuple(indice * paso for indice in range(len(slots)))
