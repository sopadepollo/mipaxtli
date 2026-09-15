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


def test_el_hud_de_la_demo_lleva_la_tasa_de_cuadros() -> None:
    """El fps va en pantalla y no solo en el volcado de `--medir-fps`.

    Todos los umbrales de la segmentacion estan en frames, asi que la tasa es la
    unidad en la que estan expresados: sin verla, "tarda en confirmar" no se
    puede separar de "la tuberia va a 9 fps". Ver
    `docs/adr/0013-la-ventana-mezclada.md`.
    """
    state = DemoHudState(
        texto="",
        palabra="",
        estado=State.TRACKING,
        ultima=None,
        dispersion=None,
        mensaje="",
        fps_entrega=27.4,
        fps_procesamiento=41.2,
    )

    assert state.fps_entrega == 27.4
    assert state.fps_procesamiento == 41.2


def test_el_hud_de_la_demo_no_exige_haber_medido_la_tasa() -> None:
    """`--desde-dataset` no abre camara y no mide nada: ahi la tasa es `None` y
    el HUD tiene que poder dibujarse igual."""
    state = DemoHudState(
        texto="",
        palabra="",
        estado=State.IDLE,
        ultima=None,
        dispersion=None,
        mensaje="",
    )

    assert state.fps_entrega is None
    assert state.fps_procesamiento is None


def test_el_hud_de_la_demo_lleva_el_minimo_contra_el_que_avisar() -> None:
    """El umbral viaja en el dato y no lo lee el dibujo de `config`: `preview.py`
    pinta, y quien decide que ensenar es el CLI."""
    state = DemoHudState(
        texto="",
        palabra="",
        estado=State.TRACKING,
        ultima=None,
        dispersion=None,
        mensaje="",
        fps_entrega=9.2,
        fps_procesamiento=11.0,
        fps_minimo=12.0,
    )

    assert state.fps_minimo == 12.0
