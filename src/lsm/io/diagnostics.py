"""Lectura de las sesiones de diagnóstico de `data/diagnostico/`.

`lsm-demo diagnosticar` escribe por sesión `diagnostico.json` —un registro por
cuadro: reloj real, estado de la máquina, letra pedida— y `flujo.json` —los
landmarks crudos de esos mismos cuadros, en el mismo orden—. Este módulo junta
las dos cosas para reprocesar una sesión sin cámara: cada frame del flujo recibe
como `timestamp_ms` el reloj real de su registro (ADR 0018), que es lo que las
sesiones anteriores al esquema 2 del fixture no guardaban en el flujo.

Toca disco; no importa OpenCV ni MediaPipe.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from lsm.io.hands import load_frame_stream
from lsm.timing import check_timestamps
from lsm.types import FrameStream


@dataclass(frozen=True, slots=True)
class DiagnosticRecord:
    """Lo que el reprocesado necesita de un registro de `diagnostico.json`."""

    index: int
    #: Reloj real, en ms desde el primer cuadro de la sesión.
    wall_ms: float
    detected: bool
    handedness_score: float | None
    #: Estado de la máquina **al llegar** el cuadro, durante la sesión en vivo.
    state: str
    prompt: str | None
    repetition: int | None
    #: Velocidad contra la ventana de `velocity_window_ms`, por segundo.
    velocity_window: float | None


@dataclass(frozen=True, slots=True)
class DiagnosticEvent:
    frame_index: int
    kind: str
    label: str | None
    prompt: str | None
    repetition: int | None


@dataclass(frozen=True, slots=True)
class DiagnosticSession:
    name: str
    metadata: dict[str, Any]
    records: tuple[DiagnosticRecord, ...]
    events: tuple[DiagnosticEvent, ...]
    #: El flujo de la sesión, con `timestamp_ms` = `wall_ms` de cada registro.
    stream: FrameStream


def load_diagnostic_session(folder: Path) -> DiagnosticSession:
    """Lee una carpeta de diagnóstico. Exige que flujo y registros coincidan."""
    payload: Any = json.loads((folder / "diagnostico.json").read_text("utf-8"))
    registros = tuple(
        DiagnosticRecord(
            index=r["index"],
            wall_ms=float(r["wall_ms"]),
            detected=bool(r["detected"]),
            handedness_score=r.get("handedness_score"),
            state=r["state"],
            prompt=r.get("prompt"),
            repetition=r.get("repetition"),
            velocity_window=r.get("velocity_window"),
        )
        for r in payload["records"]
    )
    flujo = load_frame_stream(folder / "flujo.json")
    if len(flujo) != len(registros):
        msg = (
            f"{folder}: {len(flujo)} cuadros en flujo.json y {len(registros)} "
            "registros en diagnostico.json"
        )
        raise ValueError(msg)
    if all(slot.timestamp_ms is None for slot in flujo):
        flujo = tuple(
            replace(slot, timestamp_ms=r.wall_ms)
            for slot, r in zip(flujo, registros, strict=True)
        )
    check_timestamps(flujo)
    eventos = tuple(
        DiagnosticEvent(
            frame_index=e["frame_index"],
            kind=e["kind"],
            label=e.get("label"),
            prompt=e.get("prompt"),
            repetition=e.get("repetition"),
        )
        for e in payload.get("events", [])
    )
    return DiagnosticSession(
        name=folder.name,
        metadata=dict(payload.get("metadata", {})),
        records=registros,
        events=eventos,
        stream=flujo,
    )


def iter_diagnostic_folders(root: Path) -> Iterator[Path]:
    """Las carpetas de sesión de `root`, en orden: las que tienen los dos archivos."""
    if not root.is_dir():
        return
    yield from sorted(
        carpeta
        for carpeta in root.iterdir()
        if (carpeta / "diagnostico.json").is_file()
        and (carpeta / "flujo.json").is_file()
    )
