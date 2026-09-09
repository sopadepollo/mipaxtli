"""Demo en vivo: señas → texto (`ARQUITECTURA.md` §5, Fase 3).

Cablea cámara → MediaPipe → features → segmentación → clasificador → spelling.
Todas las piezas ya existían; esto es el cable.

## Quién mueve el bucle

`run_segmentation` **consume** el flujo, así que la demo no puede iterar frame a
frame por fuera. En vez de cambiar la segmentación —cuyo contrato está versionado
(`docs/adr/0004-contrato-de-segmentacion.md`)—, el flujo que se le pasa es un
generador que en cada `next()` captura el cuadro, dibuja el HUD con el estado que
dejó el frame anterior, lee el teclado y cede el `FrameSlot`. Iterar los eventos
mueve el bucle entero.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lsm.classifiers.static_knn import StaticKnnClassifier
from lsm.config import Config, load_config
from lsm.io.camera import Camera, CameraError
from lsm.io.dataset import iter_sample_paths, read_sample
from lsm.io.hands import HandDetector, build_detector
from lsm.io.preview import DemoHudState
from lsm.segmentation import (
    LetterEmitted,
    SegmentationEvent,
    State,
    StateChanged,
    WindowRejected,
    WindowStable,
    run_segmentation,
)
from lsm.spelling import (
    Backspace,
    CommitText,
    HandAbsent,
    HandPresent,
    LetterSignal,
    LetterWritten,
    NothingToDelete,
    Signal,
    SpaceWritten,
    SpellingState,
    SymbolDeleted,
    TextCommitted,
    render_text,
    render_word,
    step,
)
from lsm.types import FrameSlot, InvalidFrame, InvalidReason, Prediction
from lsm.vocabulary import Label, spec

DEFAULT_MODEL = Path("data/models/static_knn.json")

#: Códigos de `cv2.waitKey`. ESC y `q` hacen lo mismo, igual que en el CLI de
#: captura: quien está delante de la cámara no recuerda cuál era.
_SALIR = frozenset({ord("q"), 27})
#: Borrar el último símbolo: BACKSPACE (8) y DEL (127). Las dos porque el código
#: que entrega `waitKey` depende del backend de ventanas y del teclado, y
#: descubrir cuál es la buena probando delante de la cámara es un mal rato.
_BORRAR = frozenset({8, 127})
#: ENTER. 13 en Windows, 10 en Linux.
_CERRAR_FRASE = frozenset({13, 10})


@dataclass
class Sesion:
    """El estado mutable de una sesión de demo. Solo aquí hay mutación.

    `spelling.py` es puro y devuelve estados nuevos; esto es lo que los sostiene
    entre frames y lo que el HUD lee para dibujar.
    """

    config: Config
    state: SpellingState = field(default_factory=SpellingState)
    estado_maquina: State = State.IDLE
    ultima: Prediction | None = None
    dispersion: float | None = None
    mensaje: str = ""

    def aplicar(self, signal: Signal) -> None:
        resultado = step(self.state, signal, self.config)
        self.state = resultado.state
        match resultado.event:
            case LetterWritten(label=label):
                self.mensaje = f"letra {spec(label).display}"
            case SpaceWritten():
                self.mensaje = "palabra cerrada"
            case SymbolDeleted(label=label):
                self.mensaje = f"borrada {spec(label).display}"
            case NothingToDelete():
                self.mensaje = "nada que borrar"
            case TextCommitted(text=texto):
                print(texto)
                self.mensaje = f"frase cerrada: {texto}"
            case None:
                pass

    def hud(self) -> DemoHudState:
        return DemoHudState(
            texto=render_text(self.state),
            palabra=render_word(self.state),
            estado=self.estado_maquina,
            ultima=self.ultima,
            dispersion=self.dispersion,
            mensaje=self.mensaje,
        )


def aplicar_evento(sesion: Sesion, evento: SegmentationEvent) -> None:
    """Traduce un evento de la máquina de estados a lo que la demo hace con él.

    `WindowRejected` lleva el motivo pero **no** la predicción: la ventana pudo
    rechazarse antes de clasificarla. Por eso el motivo va al mensaje y la última
    predicción se deja como estaba.
    """
    match evento:
        case LetterEmitted(prediction=prediction):
            sesion.ultima = prediction
            sesion.aplicar(LetterSignal(label=Label(prediction.label)))
        case WindowStable(dispersion=dispersion):
            sesion.dispersion = dispersion
        case WindowRejected(reason=reason):
            sesion.mensaje = f"rechazo: {reason}"
        case StateChanged(current=current):
            sesion.estado_maquina = current
        case _:
            pass


def cargar_clasificador(path: Path) -> StaticKnnClassifier:
    """Carga el modelo exportado.

    `from_export` rechaza un `feature_spec_version` o un
    `handedness_convention` que no coincidan, que es la defensa contra predecir
    en silencio con una normalización distinta a la del entrenamiento.
    """
    if not path.exists():
        raise SystemExit(
            f"no hay modelo en {path}. Entrenalo primero:\n"
            "  uv run lsm-train --sin-sintetico"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return StaticKnnClassifier.from_export(payload)


def flujo_desde_dataset(raiz: Path, config: Config) -> Iterator[Any]:
    """Las muestras de `raiz`, en orden, separadas por ausencia de mano.

    El hueco entre muestras no es adorno: sin él las señas se fundirían en una
    sola ventana y la segmentación no vería dónde acaba una y empieza la
    siguiente. Su longitud es la del espacio, para que además cierre la palabra
    igual que lo haría en vivo.
    """
    hueco = config.spelling.space_after_absent_frames
    for ruta in sorted(iter_sample_paths(raiz)):
        yield from read_sample(ruta).frames
        for _ in range(hueco):
            yield InvalidFrame(reason=InvalidReason.NO_HAND, detail="entre muestras")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lsm-demo",
        description=(
            "Demo de deletreo manual: reconoce las señas del abecedario por "
            "cámara y las convierte en texto. El procesamiento es local."
        ),
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--modelo",
        type=Path,
        default=DEFAULT_MODEL,
        help="modelo exportado por lsm-train",
    )
    parser.add_argument(
        "--desde-dataset",
        type=Path,
        default=None,
        dest="desde_dataset",
        help=(
            "reproduce las muestras de esa ruta en vez de abrir la cámara. "
            "No necesita webcam ni MediaPipe."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    classifier = cargar_clasificador(args.modelo)
    sesion = Sesion(config=config)

    if args.desde_dataset is not None:
        flujo = flujo_desde_dataset(args.desde_dataset, config)
        for evento in run_segmentation(flujo, config, classifier.predict):
            aplicar_evento(sesion, evento)
        print(render_text(sesion.state))
        return 0

    return _sesion_en_vivo(config, classifier, sesion)


def _sesion_en_vivo(
    config: Config, classifier: StaticKnnClassifier, sesion: Sesion
) -> int:
    """Abre la cámara y deletrea hasta que se pulse `q`.

    Quien mueve el bucle es el generador de frames, no un `while` de este cuerpo:
    ver el encabezado del módulo. En cada `next()` se captura un cuadro, se dibuja
    el HUD con el estado que dejó el frame anterior, se lee el teclado y se cede el
    `FrameSlot`. `run_segmentation` lo consume y sus eventos actualizan el estado
    que el siguiente cuadro dibujará.

    La presencia de mano se aplica **aquí**, un `HandPresent`/`HandAbsent` por
    cuadro, y no en `aplicar_evento`: la segmentación no emite un evento por
    frame, así que por esa vía la sesión no vería las ausencias y el espacio entre
    palabras no llegaría nunca.
    """
    import cv2

    from lsm.io.preview import draw_demo_hud, draw_landmarks

    ventana = "demo LSM — deletreo manual"

    def flujo(camera: Camera, detector: HandDetector) -> Iterator[FrameSlot]:
        # El `return` de la tecla de salida agota el generador, y agotarlo termina
        # `run_segmentation`: no hace falta ninguna bandera compartida.
        while True:
            frame = camera.read()
            slot = detector.detect(frame.rgb)

            imagen = frame.bgr
            if config.capture.preview_mirror:
                imagen = cv2.flip(imagen, 1)
            if isinstance(slot, InvalidFrame):
                sesion.aplicar(HandAbsent())
            else:
                draw_landmarks(imagen, slot, mirrored=config.capture.preview_mirror)
                sesion.aplicar(HandPresent())
            draw_demo_hud(imagen, sesion.hud())
            cv2.imshow(ventana, imagen)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla in _SALIR:
                return
            if tecla in _BORRAR:
                sesion.aplicar(Backspace())
            elif tecla in _CERRAR_FRASE:
                sesion.aplicar(CommitText())

            yield slot

    try:
        with (
            Camera.from_config(config.capture) as camera,
            build_detector(config) as detector,
        ):
            try:
                for evento in run_segmentation(
                    flujo(camera, detector), config, classifier.predict
                ):
                    aplicar_evento(sesion, evento)
            finally:
                cv2.destroyAllWindows()
    except CameraError as error:
        print(error)
        return 1

    # Lo que quedó sin cerrar con ENTER se imprime igual: quien termina la sesión
    # pulsando `q` no debería perder lo que acaba de deletrear.
    texto = render_text(sesion.state)
    if texto:
        print(texto)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
