"""El dato que necesita el HUD de la demo en vivo, probado sin cámara.

`draw_demo_hud` necesita OpenCV para pintar, así que lo que se prueba aquí no es
el dibujo sino `DemoHudState`: que lleva confianza y estado de la máquina de
segmentación siempre presentes, aunque no haya letra todavía. Sin eso, una
ventana rechazada por confianza baja y una mano que el detector no encuentra se
ven exactamente igual en pantalla, y depurar la demo se vuelve adivinar.
"""

from __future__ import annotations

from lsm.io.preview import DemoHudState
from lsm.segmentation import State
from lsm.types import Prediction


def test_el_hud_de_la_demo_dice_confianza_y_estado() -> None:
    """Confianza y estado siempre visibles: es lo que hace depurable la demo, y
    sin ellos un rechazo y un fallo de deteccion se ven igual."""
    state = DemoHudState(
        texto="casa me",
        palabra="me",
        estado=State.STABLE,
        ultima=Prediction(label="E", confidence=0.91),
        dispersion=0.019,
        mensaje="",
    )

    assert state.estado is State.STABLE
    assert state.ultima is not None
    assert state.ultima.confidence == 0.91


def test_el_hud_de_la_demo_admite_no_haber_predicho_todavia() -> None:
    """Antes de la primera prediccion no hay letra ni sigma que mostrar, pero el
    estado de la maquina sigue siendo un dato de primera clase: IDLE con mano no
    detectada y STABLE rechazado no deberian confundirse, y esa distincion vive
    en `estado`, no en `ultima`."""
    state = DemoHudState(
        texto="",
        palabra="",
        estado=State.IDLE,
        ultima=None,
        dispersion=None,
        mensaje="",
    )

    assert state.ultima is None
    assert state.dispersion is None
    assert state.estado is State.IDLE
