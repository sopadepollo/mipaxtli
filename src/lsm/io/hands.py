"""Frontera con el detector de manos.

`CLAUDE.md` §3: **MediaPipe solo puede importarse en este módulo.** El resto del
proyecto consume la interfaz definida aquí y no sabe qué hay detrás. Eso es lo que
permite correr entrenamiento, evaluación y la suite completa de tests en una
máquina sin webcam y sin MediaPipe instalado.

En Fase 0 no hay implementación real: hay un `Protocol` y un doble que reproduce
secuencias grabadas. `MediaPipeHandDetector` llega en la Fase 1 y será el único
sitio del repositorio con `import mediapipe`.

**Sobre el espejado.** Quien implemente el detector real tiene que alimentarlo con
el frame **sin espejar** (`feature-spec.md` §0.3). El espejado del preview se hace
por comodidad de quien firma y vive en la capa de visualización. Si se le pasa la
imagen espejada, la lateralidad reportada se invierte y el paso 2 de la
especificación corrompe el vector sin que nada falle.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol, TypeAlias, runtime_checkable

from lsm.types import (
    NUM_LANDMARKS,
    FrameSlot,
    FrameStream,
    Handedness,
    InvalidFrame,
    InvalidReason,
    Landmark,
    RawFrame,
)

#: Versión del formato de los fixtures de secuencias en disco.
FIXTURE_SCHEMA_VERSION: Final = 1

#: Un frame de video crudo, tal como lo entrega `io/camera.py`. Se tipa laxo a
#: propósito: el tipo real es un arreglo de OpenCV, y nombrarlo aquí metería
#: `numpy` y `cv2` en la interfaz que consume el núcleo puro.
VideoImage: TypeAlias = Any


@runtime_checkable
class HandDetector(Protocol):
    """Convierte imágenes en frames de landmarks.

    Devuelve siempre un `FrameSlot`: o una mano (`RawFrame`) o el marcador
    explícito de por qué no la hay. Nunca `None` y nunca una excepción por el caso
    normal de que en el encuadre no haya nadie.

    Si el detector encuentra varias manos se queda con la de mayor
    `detection_score`: el alfabeto dactilológico de LSM es monomanual.
    """

    def detect(self, image: VideoImage) -> FrameSlot:
        """Procesa una imagen y devuelve la mano detectada, o el motivo."""
        ...

    def close(self) -> None:
        """Libera los recursos del detector."""
        ...


@dataclass
class FakeHandDetector:
    """Detector de mentira: reproduce una secuencia grabada, ignora la imagen.

    Es lo que permite ejercitar la tubería completa —segmentación incluida— en CI
    y en Docker, sin cámara y sin MediaPipe. Cuando la grabación se agota devuelve
    frames inválidos, que es exactamente lo que devolvería una cámara apuntando a
    una habitación vacía.
    """

    slots: FrameStream
    _position: int = field(default=0, init=False, repr=False)

    @classmethod
    def from_fixture(cls, path: Path | str) -> FakeHandDetector:
        return cls(slots=load_frame_stream(path))

    def detect(self, image: VideoImage = None) -> FrameSlot:  # noqa: ARG002 — grabado
        if self._position >= len(self.slots):
            return InvalidFrame(
                reason=InvalidReason.NO_HAND, detail="la grabación se agotó"
            )
        slot = self.slots[self._position]
        self._position += 1
        return slot

    def stream(self) -> Iterator[FrameSlot]:
        """Recorre la grabación entera desde el principio, sin consumir `detect`."""
        yield from self.slots

    def close(self) -> None:
        self._position = 0


# --------------------------------------------------------------------------- #
# Fixtures en disco
# --------------------------------------------------------------------------- #


def dump_frame_stream(stream: FrameStream, path: Path | str) -> None:
    """Escribe un flujo de frames como JSON.

    Se guardan los **landmarks crudos**, no las features: si cambia la
    normalización, se re-deriva todo sin volver a grabar (`ARQUITECTURA.md` §4.7).

    Los huecos se escriben como huecos. Si se perdieran al guardar, una secuencia
    interrumpida se convertiría en una continua y el dataset mentiría sobre lo que
    ocurrió frente a la cámara.
    """
    payload = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "frames": [_slot_to_json(slot) for slot in stream],
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_frame_stream(path: Path | str) -> FrameStream:
    """Lee un flujo de frames grabado. Rechaza formatos de otra versión."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = payload.get("schema_version")
    if version != FIXTURE_SCHEMA_VERSION:
        msg = (
            f"{path}: schema_version {version} incompatible; "
            f"este código lee la versión {FIXTURE_SCHEMA_VERSION}"
        )
        raise ValueError(msg)
    return tuple(_slot_from_json(entry) for entry in payload["frames"])


def _slot_to_json(slot: FrameSlot) -> dict[str, Any]:
    if isinstance(slot, InvalidFrame):
        return {"valid": False, "reason": str(slot.reason), "detail": slot.detail}
    return {
        "valid": True,
        "width": slot.width,
        "height": slot.height,
        "handedness": str(slot.handedness),
        "handedness_score": slot.handedness_score,
        "detection_score": slot.detection_score,
        "landmarks": [[point.x, point.y, point.z] for point in slot.landmarks],
    }


def _slot_from_json(entry: dict[str, Any]) -> FrameSlot:
    if not entry["valid"]:
        return InvalidFrame(
            reason=InvalidReason(entry["reason"]), detail=entry.get("detail", "")
        )
    landmarks = entry["landmarks"]
    if len(landmarks) != NUM_LANDMARKS:
        msg = f"un frame lleva {NUM_LANDMARKS} landmarks, no {len(landmarks)}"
        raise ValueError(msg)
    return RawFrame(
        landmarks=tuple(Landmark(x=x, y=y, z=z) for x, y, z in landmarks),
        width=entry["width"],
        height=entry["height"],
        handedness=Handedness(entry["handedness"]),
        handedness_score=entry["handedness_score"],
        detection_score=entry["detection_score"],
    )
