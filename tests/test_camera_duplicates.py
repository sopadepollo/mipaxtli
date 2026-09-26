"""Bloque 1: la cámara descarta los cuadros repetidos antes de MediaPipe.

En la webcam de referencia el 44.5% de los cuadros repetía exactamente al
anterior (ADR 0017). `Camera.read` los reconoce por su miniatura y los salta. Se
prueba con una captura falsa; necesita OpenCV y numpy, que solo están con las
dependencias de captura, así que en CI se salta.
"""

from __future__ import annotations

from typing import Any

import pytest

from lsm.config import Config
from lsm.tracking_diagnostics import render_probes, summarize_probe


class _CapturaFalsa:
    """Devuelve una lista de imágenes en orden, como `cv2.VideoCapture.read`."""

    def __init__(self, imagenes: list[Any]) -> None:
        self.imagenes = imagenes

    def read(self) -> tuple[bool, Any]:
        return (True, self.imagenes.pop(0)) if self.imagenes else (False, None)


def _camara(imagenes: list[Any], **opciones: Any) -> Any:
    from lsm.io.camera import Camera

    base: dict[str, Any] = {"thumbnail_px": 8, "drop_duplicates": True}
    base.update(opciones)
    camara = Camera(index=0, width=64, height=48, fps=30, **base)
    camara._capture = _CapturaFalsa(imagenes)
    return camara


def _imagen(valor: int) -> Any:
    np = pytest.importorskip("numpy")
    return np.full((48, 64, 3), valor, dtype=np.uint8)


def test_los_cuadros_repetidos_no_llegan_al_detector() -> None:
    pytest.importorskip("cv2")
    a, b = _imagen(10), _imagen(200)
    camara = _camara([a, a.copy(), a.copy(), b])

    primero = camara.read()
    segundo = camara.read()

    assert primero.skipped_duplicates == 0
    assert segundo.skipped_duplicates == 2
    assert int(segundo.bgr[0, 0, 0]) == 200


def test_una_imagen_congelada_no_cuelga_el_bucle() -> None:
    """Tras `max_consecutive_duplicates` se entrega el repetido: una cámara
    congelada se ve congelada, en vez de bloquear la demo."""
    pytest.importorskip("cv2")
    a = _imagen(10)
    camara = _camara([a] + [a.copy() for _ in range(5)], max_consecutive_duplicates=3)

    camara.read()
    congelado = camara.read()

    assert congelado.skipped_duplicates == 3


def test_sin_descarte_los_repetidos_pasan() -> None:
    pytest.importorskip("cv2")
    a = _imagen(10)
    camara = _camara([a, a.copy()], drop_duplicates=False)

    camara.read()

    assert camara.read().skipped_duplicates == 0


def test_la_configuracion_de_fabrica_descarta_repetidos() -> None:
    config = Config()

    assert config.capture.drop_duplicate_frames is True
    assert config.capture.fourcc is None  # hasta medir con `medir-camara`


def test_un_fourcc_que_no_tiene_cuatro_letras_se_rechaza() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Config.model_validate({"capture": {"fourcc": "MJPEG"}})


# --------------------------------------------------------------------------- #
# El sondeo
# --------------------------------------------------------------------------- #


def test_el_sondeo_separa_cuadros_entregados_de_cuadros_nuevos() -> None:
    """Cuatro miniaturas en alternancia real/repetida, a 30 fps: 30 entregados y
    ~15 nuevos, el patrón que midió el diagnóstico."""
    a, b = b"\x01" * 4, b"\x02" * 4
    miniaturas = [a, a, b, b, a, a, b, b, a]
    tiempos = [i * 1000.0 / 30.0 for i in range(len(miniaturas))]

    sondeo = summarize_probe("auto driver 1280x720", "MSMF", tiempos, miniaturas)

    assert sondeo.fps == pytest.approx(30.0)
    assert sondeo.duplicates == 4
    assert sondeo.fps_unique == pytest.approx(15.0)


def test_el_reporte_del_sondeo_ordena_de_mejor_a_peor() -> None:
    lento = summarize_probe("lenta", "x", [0.0, 100.0, 200.0], [b"a", b"b", b"c"])
    rapido = summarize_probe("rápida", "y", [0.0, 33.0, 66.0], [b"a", b"b", b"c"])

    texto = render_probes([lento, rapido], {"cámara": 0})

    assert texto.index("rápida") < texto.index("lenta")
