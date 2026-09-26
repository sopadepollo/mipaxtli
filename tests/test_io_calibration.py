"""El registro que impide grabar con la entrada espejada.

Todo lo de aquí protege un error que **no produce ningún síntoma**: si el cuadro
llega espejado al detector, una mano derecha declarada se ve como una izquierda,
el paso 2 canoniza el dataset entero hacia el reflejo, el modelo entrena igual de
bien y la precisión es idéntica. Desde el ADR 0017 eso —y no el nombre que
MediaPipe le pone a la mano— es lo que la calibración comprueba.

Contra un error invisible no sirve la vigilancia: sirve un cerrojo. Estos tests
son los del cerrojo.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lsm.hand_check import input_looks_unmirrored
from lsm.io.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    Calibration,
    CalibrationError,
    calibration_path,
    camera_key,
    current_calibration,
    load_calibrations,
    require_calibration,
    save_calibration,
)
from lsm.synthetic import canonical_hand, to_frame, translated
from lsm.types import Handedness

FECHA = datetime(2026, 9, 8, 9, 15, tzinfo=UTC)
CAMARA = camera_key(index=0, width=1280, height=720, backend="V4L2")


def calibracion(**cambios: object) -> Calibration:
    base: dict[str, object] = {
        "camera": CAMARA,
        "entrada_sin_espejar": True,
        "fecha": FECHA,
        "width": 1280,
        "height": 720,
        "confirmado_por": "quien grabó",
    }
    base.update(cambios)
    return Calibration(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Identidad de la cámara
# --------------------------------------------------------------------------- #


def test_la_clave_distingue_indice_resolucion_y_backend() -> None:
    """Las tres cosas cambian lo que ve el detector, así que ninguna calibración
    vale para la otra combinación."""
    base = camera_key(index=0, width=1280, height=720, backend="V4L2")

    assert base != camera_key(index=1, width=1280, height=720, backend="V4L2")
    assert base != camera_key(index=0, width=640, height=480, backend="V4L2")
    assert base != camera_key(index=0, width=1280, height=720, backend="DSHOW")


def test_la_clave_es_estable_entre_llamadas() -> None:
    """Si no lo fuera, cada arranque exigiría recalibrar y nadie lo haría."""
    assert camera_key(index=0, width=1280, height=720, backend="V4L2") == CAMARA


# --------------------------------------------------------------------------- #
# La comprobación geométrica
# --------------------------------------------------------------------------- #


def test_la_mano_derecha_de_la_persona_queda_a_la_izquierda_sin_espejar() -> None:
    """Frente a la cámara, el lado derecho de quien firma queda a la izquierda de
    la imagen tal como sale del sensor. Si aparece a la derecha, algo la espejó.
    No depende de la etiqueta de MediaPipe."""
    izquierda_de_la_imagen = to_frame(
        translated(canonical_hand(), 300.0, 400.0), width=1280, height=720
    )
    derecha_de_la_imagen = to_frame(
        translated(canonical_hand(), 980.0, 400.0), width=1280, height=720
    )

    assert input_looks_unmirrored(izquierda_de_la_imagen, Handedness.RIGHT)
    assert not input_looks_unmirrored(derecha_de_la_imagen, Handedness.RIGHT)
    assert input_looks_unmirrored(derecha_de_la_imagen, Handedness.LEFT)


# --------------------------------------------------------------------------- #
# Registro
# --------------------------------------------------------------------------- #


def test_sin_archivo_no_hay_ninguna_camara_calibrada(tmp_path: Path) -> None:
    assert load_calibrations(tmp_path) == {}
    assert current_calibration(tmp_path, CAMARA) is None


def test_una_calibracion_sobrevive_el_viaje_a_disco(tmp_path: Path) -> None:
    save_calibration(tmp_path, calibracion())

    assert load_calibrations(tmp_path)[CAMARA] == calibracion()


def test_calibrar_una_camara_no_borra_las_demas(tmp_path: Path) -> None:
    otra = camera_key(index=1, width=640, height=480, backend="V4L2")
    save_calibration(tmp_path, calibracion())
    save_calibration(tmp_path, calibracion(camera=otra, width=640, height=480))

    assert set(load_calibrations(tmp_path)) == {CAMARA, otra}


def test_recalibrar_reemplaza_el_registro_anterior(tmp_path: Path) -> None:
    save_calibration(tmp_path, calibracion(confirmado_por="primera"))
    save_calibration(tmp_path, calibracion(confirmado_por="segunda"))

    assert load_calibrations(tmp_path)[CAMARA].confirmado_por == "segunda"


def test_un_registro_de_una_version_futura_se_rechaza(tmp_path: Path) -> None:
    save_calibration(tmp_path, calibracion())
    ruta = calibration_path(tmp_path)
    payload = json.loads(ruta.read_text(encoding="utf-8"))
    payload["schema_version"] = CALIBRATION_SCHEMA_VERSION + 1
    ruta.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CalibrationError, match="schema_version"):
        load_calibrations(tmp_path)


def test_una_calibracion_v1_se_lee_pero_no_esta_vigente(tmp_path: Path) -> None:
    """El registro de antes del ADR 0017 comprobaba la etiqueta de MediaPipe, no
    la entrada sin espejar. Se lee —el archivo de la Fase 1 existe— pero no
    habilita a grabar, y el error dice por qué."""
    ruta = calibration_path(tmp_path)
    ruta.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "camaras": {
                    CAMARA: {
                        "swap_handedness": False,
                        "convention": "SIGNER",
                        "fecha": FECHA.isoformat(),
                        "width": 1280,
                        "height": 720,
                        "confirmado_por": "alguien",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert load_calibrations(tmp_path)[CAMARA].entrada_sin_espejar is False
    assert current_calibration(tmp_path, CAMARA) is None
    with pytest.raises(CalibrationError, match="anterior a la mano declarada"):
        require_calibration(tmp_path, CAMARA)


# --------------------------------------------------------------------------- #
# Vigencia y cerrojo
# --------------------------------------------------------------------------- #


def test_una_camara_distinta_no_hereda_la_calibracion(tmp_path: Path) -> None:
    save_calibration(tmp_path, calibracion())
    otra = camera_key(index=2, width=1280, height=720, backend="V4L2")

    assert current_calibration(tmp_path, otra) is None


def test_la_vigencia_no_caduca_por_tiempo(tmp_path: Path) -> None:
    save_calibration(tmp_path, calibracion(fecha=datetime(2020, 1, 1, tzinfo=UTC)))

    assert current_calibration(tmp_path, CAMARA) is not None


def test_sin_calibracion_require_lanza_y_dice_como_obtenerla(tmp_path: Path) -> None:
    with pytest.raises(CalibrationError, match="lsm-capture calibrar") as error:
        require_calibration(tmp_path, CAMARA)

    assert "no está calibrada" in str(error.value)


def test_con_calibracion_vigente_require_la_devuelve(tmp_path: Path) -> None:
    save_calibration(tmp_path, calibracion())

    devuelta = require_calibration(tmp_path, CAMARA)

    assert devuelta.camera == CAMARA
    assert devuelta.confirmado_por == "quien grabó"
