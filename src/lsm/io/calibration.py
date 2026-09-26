"""Registro de calibración de la cámara: que el cuadro llega al detector sin espejar.

**v2** (ADR 0017). Hasta la v1 esto comprobaba qué **nombre** le ponía MediaPipe
a la mano levantada, porque ese nombre decidía el espejo del paso 2 de
`feature-spec.md`. Desde `FEATURE_SPEC_VERSION` 2 el espejo lo decide la mano
**declarada** por quien firma, y el nombre de MediaPipe es solo diagnóstico —con
la palma de lado cambia de opinión a mitad de una J—. Lo que sigue importando, y
sigue sin producir síntomas si falla, es otra cosa: **que la imagen que recibe el
detector no esté espejada** (§0.3). Hay drivers de webcam que espejan por su
cuenta. Con la entrada espejada, una mano derecha declarada se vería como una
izquierda, el vector entero saldría reflejado y el modelo entrenaría igual de
bien sobre datos al revés.

La comprobación es geométrica y no depende de MediaPipe: quien calibra levanta
su mano derecha junto a su hombro derecho. Delante de una cámara, el lado
derecho de la persona queda a la **izquierda** de la imagen sin espejar. Si la
muñeca aparece en la mitad izquierda, la entrada está bien; en la derecha, algo
la espejó. Ver `lsm.capture.input_looks_unmirrored`.

Como antes, dos defensas mecánicas porque la vigilancia humana no funciona:
comprobarlo una vez por cámara (`lsm-capture calibrar`) y exigir el registro
después (`grabar`, este módulo).

La calibración es del **equipo**, no de quien firma: va por cámara.

Este módulo toca disco pero no importa OpenCV ni MediaPipe: es JSON y rutas.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

#: Versión del formato del registro de calibración.
CALIBRATION_SCHEMA_VERSION: Final = 2

#: Versiones que se leen. Una entrada v1 se lee como **no verificada**: comprobó
#: otra cosa, y `grabar` pide recalibrar en vez de fallar por el formato.
READABLE_CALIBRATION_SCHEMAS: Final = frozenset({1, 2})

#: Nombre del registro, en la raíz del dataset.
CALIBRATION_FILENAME: Final = "calibracion.json"


class CalibrationError(RuntimeError):
    """No hay calibración utilizable para esta cámara."""


def camera_key(*, index: int, width: int, height: int, backend: str) -> str:
    """Identificador de la cámara calibrada.

    No existe forma portable de preguntarle a OpenCV el número de serie de una
    webcam, así que se compone con lo que sí se puede leer y sí cambia el
    resultado:

    - el **índice**, que es lo que distingue la cámara integrada del portátil de
      la externa que alguien enchufó;
    - la **resolución realmente entregada**, que no siempre es la pedida, y que
      cambia el recorte y la relación de aspecto que ve el detector;
    - el **backend** de OpenCV (`V4L2`, `DSHOW`, `AVFOUNDATION`…), porque el mismo
      dispositivo a través de dos backends distintos no se comporta igual.

    Es una heurística y no una identidad. Puede confundir dos webcams idénticas
    conectadas en el mismo orden, y eso es aceptable: el coste de un falso positivo
    es una calibración de más, y la comprobación cuesta un segundo.
    """
    return f"{backend}:{index}@{width}x{height}"


@dataclass(frozen=True, slots=True)
class Calibration:
    """Lo que se comprobó, cuándo y quién lo miró."""

    camera: str
    #: Se comprobó que el cuadro llega al detector sin espejar. `False` en las
    #: entradas v1, que comprobaban la etiqueta de MediaPipe y no esto.
    entrada_sin_espejar: bool
    fecha: datetime
    width: int
    height: int
    #: Quién levantó la mano y confirmó.
    confirmado_por: str = ""

    @property
    def vigente(self) -> bool:
        return self.entrada_sin_espejar


def calibration_path(root: Path) -> Path:
    return root / CALIBRATION_FILENAME


def load_calibrations(root: Path) -> dict[str, Calibration]:
    """Lee el registro. Sin archivo, no hay ninguna cámara calibrada."""
    path = calibration_path(root)
    if not path.is_file():
        return {}
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("schema_version")
    if version not in READABLE_CALIBRATION_SCHEMAS:
        msg = (
            f"{path}: schema_version {version} incompatible; este código lee las "
            f"versiones {sorted(READABLE_CALIBRATION_SCHEMAS)}"
        )
        raise CalibrationError(msg)
    return {
        camera: Calibration(
            camera=camera,
            entrada_sin_espejar=version >= 2 and bool(entry["entrada_sin_espejar"]),
            fecha=datetime.fromisoformat(entry["fecha"]),
            width=int(entry["width"]),
            height=int(entry["height"]),
            confirmado_por=entry.get("confirmado_por", ""),
        )
        for camera, entry in payload.get("camaras", {}).items()
    }


def save_calibration(root: Path, calibration: Calibration) -> Path:
    """Registra o reemplaza la calibración de una cámara."""
    registro = load_calibrations(root)
    registro[calibration.camera] = calibration

    root.mkdir(parents=True, exist_ok=True)
    path = calibration_path(root)
    payload = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "camaras": {
            camera: {
                "entrada_sin_espejar": entry.entrada_sin_espejar,
                "fecha": entry.fecha.isoformat(),
                "width": entry.width,
                "height": entry.height,
                "confirmado_por": entry.confirmado_por,
            }
            for camera, entry in sorted(registro.items())
        },
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def has_any_calibration(root: Path) -> bool:
    """Si hay al menos una cámara calibrada, sea cual sea.

    Existe para poder rechazar **antes de tocar el hardware** el caso más común de
    todos: nadie ha calibrado nunca. La comprobación de verdad es por cámara y
    necesita la resolución que el driver entrega, así que llega inevitablemente
    con el dispositivo ya abierto; ésta se adelanta y evita encender una webcam
    para nada.
    """
    return bool(load_calibrations(root))


def current_calibration(root: Path, camera: str) -> Calibration | None:
    """La calibración vigente de una cámara, o `None` si no la hay.

    Vigente es que se comprobó la entrada sin espejar con esta cámara. No caduca
    por tiempo: se invalida al cambiar de cámara o de resolución, y las dos
    cosas cambian la clave.
    """
    calibration = load_calibrations(root).get(camera)
    if calibration is None or not calibration.vigente:
        return None
    return calibration


def require_calibration(root: Path, camera: str) -> Calibration:
    """La calibración vigente, o un error que dice cómo obtenerla.

    Se llama antes de grabar y **rechaza**, no avisa: el dataset que sale de una
    sesión con la entrada espejada no tiene ningún síntoma que la delate.
    """
    calibration = current_calibration(root, camera)
    if calibration is not None:
        return calibration

    existente = load_calibrations(root).get(camera)
    if existente is None:
        detalle = f"la cámara {camera} no está calibrada"
    else:
        detalle = (
            f"la calibración de {camera} es anterior a la mano declarada (ADR "
            "0017): comprobaba la etiqueta de MediaPipe, no que la entrada llegue "
            "sin espejar"
        )
    msg = (
        f"{detalle}. Sin esa comprobación, una cámara que espeja por su cuenta "
        "reflejaría el dataset entero sin ningún síntoma.\n"
        "  lsm-capture calibrar"
    )
    raise CalibrationError(msg)
