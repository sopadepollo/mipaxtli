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
from lsm.io.dataset import iter_sample_paths, read_sample
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
from lsm.types import InvalidFrame, InvalidReason, Prediction
from lsm.vocabulary import Label, spec

DEFAULT_MODEL = Path("data/models/static_knn.json")


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
    config: Config,  # noqa: ARG001 -- firma fija para la tarea 7, cuerpo aún no
    classifier: StaticKnnClassifier,  # noqa: ARG001
    sesion: Sesion,  # noqa: ARG001
) -> int:
    raise SystemExit("la sesión en vivo llega en la tarea 7")


if __name__ == "__main__":
    raise SystemExit(main())
