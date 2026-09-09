"""Registro de calibración de la cámara: qué mano ve, y con qué convención.

Existe por un solo motivo, y conviene tenerlo delante para entender por qué un
archivo tan pequeño bloquea una sesión entera.

`hands.mediapipe_reports_mirrored_handedness` decide si se invierte la lateralidad
que reporta el detector. **Elegir mal ese interruptor no rompe nada observable**:
el paso 2 de `feature-spec.md` canoniza *todas* las muestras hacia la otra mano,
las dos poblaciones de vectores difieren por un espejo global y nada más. El
modelo entrena igual de bien, infiere igual de bien, y la matriz de confusión sale
idéntica.

Duele en un único escenario, y es el de la Fase 7: **dos implementaciones con
convenciones distintas.** MediaPipe JS traerá la suya, y los golden vectors no
cubren esto —reciben la lateralidad ya resuelta como entrada—, así que el test de
paridad pasaría en verde mientras la app web confunde cada seña con su espejo.

Contra un error que no produce síntomas solo hay dos defensas, y las dos son
mecánicas porque la vigilancia humana no funciona:

1. **Comprobarlo con un ojo humano una vez por cámara**, que es lo que hace
   `lsm-capture calibrar`: levantar la mano derecha y mirar si el preview dice
   `RIGHT`. Es la única forma de saberlo de verdad en una cámara concreta.
2. **Que el resultado quede escrito y se exija después**, que es este módulo.

La calibración es del **equipo**, no de quien firma: la misma persona con dos
cámaras necesita dos calibraciones, y dos personas con la misma cámara comparten
una. Por eso el registro va por cámara y no por `signer_id`, al revés que el
consentimiento.

Este módulo toca disco pero no importa OpenCV ni MediaPipe: es JSON y rutas.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from lsm.types import HANDEDNESS_CONVENTION, HandednessConvention

#: Versión del formato del registro de calibración.
CALIBRATION_SCHEMA_VERSION: Final = 1

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
    """Lo que se comprobó, cuándo, y con qué ajuste.

    `swap_handedness` es el valor **efectivo** del interruptor en el momento de
    calibrar, no lo que dice `config.yaml` ahora. Esa distinción es todo el
    mecanismo: si alguien toca la configuración, la calibración deja de coincidir
    y hay que rehacerla.
    """

    camera: str
    #: Valor de `hands.mediapipe_reports_mirrored_handedness` que se confirmó.
    swap_handedness: bool
    #: Qué mano nombra `handedness` con ese ajuste. Hoy siempre `SIGNER`; viaja
    #: explícito para que un registro viejo siga siendo legible si algún día se
    #: añade otra convención.
    convention: HandednessConvention
    fecha: datetime
    width: int
    height: int
    #: Quién confirmó a ojo que el preview decía lo que debía.
    confirmado_por: str = ""

    def matches(self, *, swap_handedness: bool) -> bool:
        """Si esta calibración sigue describiendo la configuración actual."""
        return (
            self.swap_handedness == swap_handedness
            and self.convention is HANDEDNESS_CONVENTION
        )


def calibration_path(root: Path) -> Path:
    return root / CALIBRATION_FILENAME


def load_calibrations(root: Path) -> dict[str, Calibration]:
    """Lee el registro. Sin archivo, no hay ninguna cámara calibrada."""
    path = calibration_path(root)
    if not path.is_file():
        return {}

    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("schema_version")
    if version != CALIBRATION_SCHEMA_VERSION:
        msg = (
            f"{path}: schema_version {version} incompatible; este código lee la "
            f"versión {CALIBRATION_SCHEMA_VERSION}"
        )
        raise CalibrationError(msg)

    return {
        camera: Calibration(
            camera=camera,
            swap_handedness=bool(entry["swap_handedness"]),
            convention=HandednessConvention(entry["convention"]),
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
                "swap_handedness": entry.swap_handedness,
                "convention": str(entry.convention),
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


def current_calibration(
    root: Path, camera: str, *, swap_handedness: bool
) -> Calibration | None:
    """La calibración vigente de una cámara, o `None` si no la hay.

    «Vigente» no es solo «existe»: tiene que haberse confirmado **con el ajuste que
    está puesto ahora**. Cambiar
    `hands.mediapipe_reports_mirrored_handedness` en `config.yaml` invalida las
    calibraciones anteriores, que es exactamente lo que debe pasar — el registro
    dice que alguien miró la pantalla y vio `RIGHT` con *aquel* ajuste, y con el
    contrario habría visto `LEFT`.

    No hay caducidad por tiempo. Una calibración no se estropea sola: se estropea
    cuando cambia la configuración o cuando se cambia de cámara, y las dos cosas se
    detectan aquí.
    """
    calibration = load_calibrations(root).get(camera)
    if calibration is None:
        return None
    return calibration if calibration.matches(swap_handedness=swap_handedness) else None


def require_calibration(
    root: Path, camera: str, *, swap_handedness: bool
) -> Calibration:
    """La calibración vigente, o un error que dice cómo obtenerla.

    Se llama antes de grabar y **rechaza**, no avisa. Un aviso en una terminal a
    las nueve de la mañana, antes de cuarenta minutos de grabación con otra
    persona delante, no lo lee nadie; y el dataset que sale de esa sesión no tiene
    ningún síntoma que delate el problema.
    """
    calibration = current_calibration(root, camera, swap_handedness=swap_handedness)
    if calibration is not None:
        return calibration

    existente = load_calibrations(root).get(camera)
    if existente is None:
        detalle = f"la cámara {camera} no está calibrada"
    else:
        detalle = (
            f"la calibración de {camera} se hizo con "
            f"mediapipe_reports_mirrored_handedness = "
            f"{str(existente.swap_handedness).lower()}, y ahora vale "
            f"{str(swap_handedness).lower()}"
        )

    msg = (
        f"{detalle}. Sin calibración no se puede saber si la lateralidad que "
        "reporta el detector es la mano real, y equivocarse canoniza el dataset "
        "entero hacia la mano contraria sin ningún síntoma.\n"
        "  lsm-capture calibrar"
    )
    raise CalibrationError(msg)
