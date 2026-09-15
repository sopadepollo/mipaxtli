"""El registro que impide grabar a ciegas.

Todo lo de aquí protege un error que **no produce ningún síntoma**: si
`hands.mediapipe_reports_mirrored_handedness` está al revés, el dataset entero
queda canonizado hacia la mano contraria, el modelo entrena igual de bien y la
precisión es idéntica. Solo duele en la Fase 7, cuando MediaPipe JS use la
convención contraria y la app web confunda cada seña con su espejo — y los golden
vectors no lo detectan, porque reciben la lateralidad ya resuelta.

Contra un error invisible no sirve la vigilancia: sirve un cerrojo. Estos tests
son los del cerrojo.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

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
from lsm.types import HANDEDNESS_CONVENTION, HandednessConvention

FECHA = datetime(2026, 9, 8, 9, 15, tzinfo=UTC)
CAMARA = camera_key(index=0, width=1280, height=720, backend="V4L2")


def calibracion(**cambios: object) -> Calibration:
    base: dict[str, object] = {
        "camera": CAMARA,
        "swap_handedness": True,
        "convention": HANDEDNESS_CONVENTION,
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
# Registro
# --------------------------------------------------------------------------- #


def test_sin_archivo_no_hay_ninguna_camara_calibrada(tmp_path: Path) -> None:
    assert load_calibrations(tmp_path) == {}
    assert current_calibration(tmp_path, CAMARA, swap_handedness=True) is None


def test_una_calibracion_sobrevive_el_viaje_a_disco(tmp_path: Path) -> None:
    save_calibration(tmp_path, calibracion())

    recuperada = load_calibrations(tmp_path)[CAMARA]

    assert recuperada == calibracion()


def test_calibrar_una_camara_no_borra_las_demas(tmp_path: Path) -> None:
    """La misma persona con dos cámaras necesita dos calibraciones, y cambiar de
    una a otra no debería obligar a rehacer la anterior."""
    otra = camera_key(index=1, width=640, height=480, backend="V4L2")
    save_calibration(tmp_path, calibracion())
    save_calibration(tmp_path, calibracion(camera=otra, width=640, height=480))

    assert set(load_calibrations(tmp_path)) == {CAMARA, otra}


def test_recalibrar_reemplaza_el_registro_anterior(tmp_path: Path) -> None:
    save_calibration(tmp_path, calibracion(swap_handedness=True))
    save_calibration(tmp_path, calibracion(swap_handedness=False))

    assert load_calibrations(tmp_path)[CAMARA].swap_handedness is False


def test_un_registro_de_otra_version_se_rechaza(tmp_path: Path) -> None:
    save_calibration(tmp_path, calibracion())
    ruta = calibration_path(tmp_path)
    payload = json.loads(ruta.read_text(encoding="utf-8"))
    payload["schema_version"] = CALIBRATION_SCHEMA_VERSION + 1
    ruta.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CalibrationError, match="schema_version"):
        load_calibrations(tmp_path)


# --------------------------------------------------------------------------- #
# Vigencia
# --------------------------------------------------------------------------- #


def test_una_calibracion_del_ajuste_actual_es_vigente(tmp_path: Path) -> None:
    save_calibration(tmp_path, calibracion(swap_handedness=True))

    assert current_calibration(tmp_path, CAMARA, swap_handedness=True) is not None


def test_cambiar_el_interruptor_invalida_la_calibracion(tmp_path: Path) -> None:
    """El mecanismo entero, en un test.

    El registro dice que alguien miró la pantalla y vio `RIGHT` **con aquel
    ajuste**. Con el contrario habría visto `LEFT`, así que la comprobación ya no
    dice nada y hay que rehacerla. Sin esto, tocar `config.yaml` a mitad del
    proyecto dejaría medio dataset con una convención y medio con la otra, todo
    con la misma pinta.
    """
    save_calibration(tmp_path, calibracion(swap_handedness=True))

    assert current_calibration(tmp_path, CAMARA, swap_handedness=False) is None


def test_una_camara_distinta_no_hereda_la_calibracion(tmp_path: Path) -> None:
    save_calibration(tmp_path, calibracion())
    otra = camera_key(index=2, width=1280, height=720, backend="V4L2")

    assert current_calibration(tmp_path, otra, swap_handedness=True) is None


def test_una_convencion_distinta_invalida_la_calibracion(tmp_path: Path) -> None:
    """Si algún día el proyecto cambiara de convención, las calibraciones viejas
    describirían otra cosa. Es el mismo patrón que `feature_spec_version`.

    Se escribe `IMAGE` a pelo, y no "la contraria a la del proyecto", porque hoy
    solo hay dos valores y calcularlo obligaría a un condicional que mypy sabe
    resolver: el test dejaría de comprobar nada en cuanto alguien añadiera un
    tercero.
    """
    save_calibration(tmp_path, calibracion(convention=HandednessConvention.IMAGE))

    assert current_calibration(tmp_path, CAMARA, swap_handedness=True) is None


def test_la_vigencia_no_caduca_por_tiempo(tmp_path: Path) -> None:
    """Una calibración no se estropea sola. Se estropea al cambiar la
    configuración o la cámara, y las dos cosas ya se detectan. Caducarla por
    fecha añadiría un umbral arbitrario y una molestia periódica sin tapar ningún
    fallo real."""
    save_calibration(tmp_path, calibracion(fecha=datetime(2020, 1, 1, tzinfo=UTC)))

    assert current_calibration(tmp_path, CAMARA, swap_handedness=True) is not None


# --------------------------------------------------------------------------- #
# El cerrojo
# --------------------------------------------------------------------------- #


def test_sin_calibracion_require_lanza_y_dice_como_obtenerla(tmp_path: Path) -> None:
    with pytest.raises(CalibrationError, match="lsm-capture calibrar") as error:
        require_calibration(tmp_path, CAMARA, swap_handedness=True)

    assert "no está calibrada" in str(error.value)


def test_con_el_interruptor_cambiado_el_error_dice_exactamente_eso(
    tmp_path: Path,
) -> None:
    """Los dos fallos piden acciones distintas —calibrar por primera vez o revisar
    qué se tocó en `config.yaml`— así que el mensaje los distingue."""
    save_calibration(tmp_path, calibracion(swap_handedness=True))

    with pytest.raises(CalibrationError) as error:
        require_calibration(tmp_path, CAMARA, swap_handedness=False)

    assert "mediapipe_reports_mirrored_handedness" in str(error.value)


def test_con_calibracion_vigente_require_la_devuelve(tmp_path: Path) -> None:
    save_calibration(tmp_path, calibracion())

    devuelta = require_calibration(tmp_path, CAMARA, swap_handedness=True)

    assert devuelta.camera == CAMARA
    assert devuelta.confirmado_por == "quien grabó"
