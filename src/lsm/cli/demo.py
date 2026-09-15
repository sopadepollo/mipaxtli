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

## Dónde se mide el tiempo

Aquí, porque aquí está el bucle y el reloj. La aritmética —percentiles, reparto
por etapas— vive en `lsm.telemetry`, que es código puro y recibe el reloj
inyectado; este módulo solo llama a `time.perf_counter` en los cinco sitios que
delimitan las etapas.

Que quien mueve el bucle sea el generador tiene una consecuencia útil: el tiempo
que `run_segmentation` tarda en consumir un cuadro es exactamente lo que pasa
entre el `yield` y el `next()` siguiente, así que se mide sin instrumentar la
segmentación por dentro. El precio es que la etapa de un cuadro se cierra al
principio de la iteración siguiente, y que el último cuadro de la sesión —el que
se cede antes de salir— no llega a contarse.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lsm.classifiers.static_knn import StaticKnnClassifier
from lsm.cli import MENSAJE_SIN_EXTRAS
from lsm.config import Config, load_config
from lsm.io.camera import Camera, CameraError
from lsm.io.dataset import iter_sample_paths, read_sample
from lsm.io.hands import HandDetector, build_detector
from lsm.io.preview import DemoHudState
from lsm.segmentation import (
    EvidenceAccumulated,
    FrameThresholds,
    LetterEmitted,
    SegmentationEvent,
    State,
    StateChanged,
    WindowRejected,
    WindowStable,
    frames_from_ms,
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
from lsm.telemetry import (
    Cronometro,
    FrameTiming,
    Medicion,
    Stage,
    render_resumen,
    segundos_restantes,
    tasa_insuficiente,
)
from lsm.types import FrameSlot, InvalidFrame, InvalidReason, Prediction
from lsm.vocabulary import Label, spec

DEFAULT_MODEL = Path("data/models/static_knn.json")

#: `--desde-dataset` con una ruta sin muestras terminaba imprimiendo una linea en
#: blanco y saliendo con 0, que es indistinguible de "la tubería no reconoció
#: nada". La distinción importa porque el error es fácil de cometer: la raíz que
#: `iter_sample_paths` espera es la del dataset —firmante/sesion/letra/NNN.json—
#: y apuntar a una sesión concreta, que es lo que uno haría, no encuentra nada.
SIN_MUESTRAS = (
    "no hay muestras en {raiz}.\n"
    "La ruta tiene que ser la RAIZ del dataset, no una sesion: dentro se buscan\n"
    "  <firmante>/<sesion>/<letra>/NNN.json\n"
    "Prueba con  --desde-dataset data/raw"
)

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

    #: Media móvil de la tasa de cuadros, para el HUD. Se queda vacía en
    #: `--desde-dataset`: ahí no hay bucle en vivo y medirlo no significaría nada.
    ventana_fps: Medicion = field(init=False)

    #: La tasa con la que se convierten a cuadros los umbrales en milisegundos.
    #: Se mide al arrancar la sesión en vivo y **no se vuelve a tocar**: que los
    #: umbrales cambiaran a mitad de deletreo haría que la misma seña se
    #: comportara distinto según lo que la máquina llevara haciendo un segundo
    #: antes. `None` es «usa la nominal», que es lo correcto sin cámara.
    fps: float | None = None

    def __post_init__(self) -> None:
        self.ventana_fps = Medicion(ventana=self.config.telemetry.fps_window_frames)

    def registrar(self, timing: FrameTiming) -> None:
        """Anota un cuadro medido. Lo que el HUD leerá en el cuadro siguiente."""
        self.ventana_fps.agregar(timing)

    def aplicar(self, signal: Signal) -> None:
        resultado = step(self.state, signal, self.config, fps=self.fps)
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
            fps_entrega=self.ventana_fps.fps_entrega,
            fps_procesamiento=self.ventana_fps.fps_procesamiento,
            fps_minimo=self.config.telemetry.min_fps,
        )


@dataclass(frozen=True, slots=True)
class Medida:
    """Una medición de duración fija: `--medir-fps`.

    Va junta porque las dos cosas solo tienen sentido juntas, y así el bucle no
    tiene que defenderse de un acumulador sin duración o al revés.
    """

    medicion: Medicion
    #: Segundos de bucle que se van a medir.
    duracion: float


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
        case EvidenceAccumulated(prediction=prediction, window=window):
            # La confianza subiendo en vivo es lo que hace diagnosticable la demo
            # cuando una letra no sale: sin esto, «está acumulando evidencia sobre
            # una C» y «el detector no encuentra la mano» se ven igual.
            sesion.ultima = prediction
            sesion.mensaje = f"acumulando {prediction.label}: {len(window)} frames"
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
    # A la tasa NOMINAL, que es la que corresponde: reproducir un dataset no mide
    # ninguna tasa, y el hueco tiene que ser el mismo que la máquina de estados
    # va a exigir al recorrer ese mismo flujo.
    hueco = frames_from_ms(
        config.spelling.space_after_absent_ms, config.capture.camera_fps
    )
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
    parser.add_argument(
        "--medir-fps",
        action="store_true",
        dest="medir_fps",
        help=(
            "corre la sesión el tiempo de telemetry.benchmark_seconds y vuelca "
            "el resumen estadístico de la tasa de cuadros. Necesita cámara."
        ),
    )
    parser.add_argument(
        "--medir-segundos",
        type=float,
        default=None,
        dest="medir_segundos",
        help="cuánto dura --medir-fps, si no telemetry.benchmark_seconds",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)

    medida = _medida_pedida(args, config)
    if isinstance(medida, str):
        print(medida)
        return 2

    # Antes de cargar el modelo: si no hay muestras que reproducir, cargarlo no
    # sirve para nada y el mensaje llega igual de rápido.
    if args.desde_dataset is not None and not sorted(
        iter_sample_paths(args.desde_dataset)
    ):
        print(SIN_MUESTRAS.format(raiz=args.desde_dataset))
        return 1

    classifier = cargar_clasificador(args.modelo)
    sesion = Sesion(config=config)

    if args.desde_dataset is not None:
        flujo = flujo_desde_dataset(args.desde_dataset, config)
        for evento in run_segmentation(flujo, config, classifier.predict):
            aplicar_evento(sesion, evento)
        print(render_text(sesion.state))
        return 0

    return _sesion_en_vivo(config, classifier, sesion, medida)


def _medida_pedida(args: argparse.Namespace, config: Config) -> Medida | str | None:
    """Qué medición pidieron los argumentos, o el error que hay que imprimir.

    `--medir-fps` mide el bucle **en vivo**: cuánto tarda la cámara en entregar
    un cuadro y cuánto tarda la tubería en procesarlo. Sobre un dataset ya
    grabado no hay ni cámara ni bucle, así que el número no diría nada sobre la
    máquina que corre la demo y se rechaza en vez de medir otra cosa parecida.
    """
    if args.medir_segundos is not None and not args.medir_fps:
        return "--medir-segundos solo tiene sentido junto con --medir-fps"
    if not args.medir_fps:
        return None
    if args.desde_dataset is not None:
        return (
            "--medir-fps mide el bucle en vivo y no se puede combinar con "
            "--desde-dataset: sin cámara no hay tasa de entrega que medir"
        )
    duracion = args.medir_segundos or config.telemetry.benchmark_seconds
    if duracion <= 0.0:
        return f"--medir-segundos tiene que ser positivo, no {duracion}"
    return Medida(medicion=Medicion(), duracion=duracion)


def _resumen_de_tasa(config: Config, umbrales: FrameThresholds) -> str:
    """Qué tasa se midió y en cuántos cuadros se tradujo cada umbral.

    Se imprime al arrancar porque es la información con la que hay que leer todo
    lo que pase después: con la misma configuración, dos máquinas distintas
    corren con ventanas de distinto número de cuadros, y eso tiene que estar a la
    vista y no deducirse.
    """
    lineas = [
        f"tasa medida: {umbrales.fps:.1f} fps",
        f"  ventana maxima {umbrales.buffer_size} cuadros"
        f" ({config.segmentation.buffer_ms:.0f} ms)",
        f"  estabilidad {umbrales.stable_frames} cuadros"
        f" ({config.segmentation.stable_ms:.0f} ms)",
        f"  cooldown de emision {umbrales.emit_cooldown_frames} cuadros"
        f" ({config.segmentation.emit_cooldown_ms:.0f} ms)",
    ]
    if tasa_insuficiente(umbrales.fps, minimo=config.telemetry.min_fps):
        lineas.append(
            f"AVISO: por debajo de telemetry.min_fps ({config.telemetry.min_fps:.0f})."
            " El reconocimiento va a ir a tirones y no es culpa de la sena."
        )
    return "\n".join(lineas)


def _cerrar_cuadro(
    crono: Cronometro, sesion: Sesion, medida: Medida | None, arranque: float
) -> bool:
    """Cierra el cuadro ya medido y dice si el bucle debe seguir.

    Devuelve `False` solo cuando `--medir-fps` agotó su tiempo. En una sesión
    normal la medición alimenta el HUD y nada más: nunca corta el bucle.
    """
    timing = crono.cerrar()
    sesion.registrar(timing)
    if medida is None:
        return True

    medida.medicion.agregar(timing)
    restante = segundos_restantes(
        duracion=medida.duracion, transcurrido=time.perf_counter() - arranque
    )
    # La cuenta atrás ocupa la línea de mensajes mientras dura la medición. Es
    # el modo en el que se está mirando el reloj, no los rechazos.
    sesion.mensaje = f"midiendo fps: faltan {restante:.0f} s"
    return restante > 0.0


def _sesion_en_vivo(
    config: Config,
    classifier: StaticKnnClassifier,
    sesion: Sesion,
    medida: Medida | None = None,
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
    ventana = "demo LSM — deletreo manual"

    # Los tres fallos de arranque —sin OpenCV, sin modelo de MediaPipe, sin
    # cámara— caen dentro del mismo `try` **incluidos los imports**: `import cv2`
    # es el primero que se rompe en una máquina que solo hizo `make setup`, y
    # dejarlo fuera lo convertiría en el único que sí escupe un traceback.
    try:
        import cv2

        from lsm.io.preview import draw_demo_hud, draw_landmarks

        def flujo(camera: Camera, detector: HandDetector) -> Iterator[FrameSlot]:
            # El `return` de la tecla de salida agota el generador, y agotarlo
            # termina `run_segmentation`: no hace falta ninguna bandera compartida.
            crono = Cronometro(reloj=time.perf_counter)
            arranque = time.perf_counter()
            while True:
                if crono.abierto:
                    # La etapa del cuadro ANTERIOR: lo que `run_segmentation`
                    # tardó en consumirlo es exactamente lo que pasó entre su
                    # `yield` y este `next()`, y no se sabe hasta aquí.
                    crono.marcar(Stage.SEGMENTACION)
                    if not _cerrar_cuadro(crono, sesion, medida, arranque):
                        return

                crono.iniciar()
                frame = camera.read()
                crono.marcar(Stage.CAMARA)
                slot = detector.detect(frame.rgb)
                crono.marcar(Stage.DETECCION)

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

                # La marca va después del teclado y no antes: las cuatro etapas
                # tienen que cubrir el bucle entero o el ciclo dejaría de ser el
                # tiempo de pared y el fps saldría más alto que el real.
                crono.marcar(Stage.PREVIEW)
                yield slot

        def calentar(camera: Camera, detector: HandDetector) -> float:
            """Mide la tasa ANTES de arrancar la máquina de estados.

            Los umbrales de `config.segmentation` están en milisegundos y hay que
            convertirlos a cuadros antes del primer frame, así que la tasa tiene
            que conocerse antes de que la máquina exista. De ahí este par de
            segundos de calentamiento, que además son los que la cámara suele
            tardar en estabilizar exposición y enfoque: medir el primer cuadro de
            todos daría un número peor que el de la sesión real.

            Mide el bucle **sin la segmentación**, que todavía no corre. En la
            máquina de referencia esa etapa es el 3% del ciclo (1.7 ms de 53.5),
            así que la tasa sale ligeramente optimista y por tanto los umbrales,
            ligeramente largos. Es el lado seguro del error.
            """
            medicion = Medicion()
            crono = Cronometro(reloj=time.perf_counter)
            cuadros = config.telemetry.fps_window_frames
            for numero in range(cuadros):
                crono.iniciar()
                frame = camera.read()
                crono.marcar(Stage.CAMARA)
                slot = detector.detect(frame.rgb)
                crono.marcar(Stage.DETECCION)

                imagen = frame.bgr
                if config.capture.preview_mirror:
                    imagen = cv2.flip(imagen, 1)
                if not isinstance(slot, InvalidFrame):
                    draw_landmarks(imagen, slot, mirrored=config.capture.preview_mirror)
                sesion.mensaje = f"midiendo la tasa: {numero + 1}/{cuadros}"
                draw_demo_hud(imagen, sesion.hud())
                cv2.imshow(ventana, imagen)
                cv2.waitKey(1)
                crono.marcar(Stage.PREVIEW)
                # Sin segmentación que medir: la máquina de estados todavía no
                # existe, y su etapa se cierra en cero para que el ciclo siga
                # siendo la suma de las cuatro.
                crono.marcar(Stage.SEGMENTACION)
                timing = crono.cerrar()
                medicion.agregar(timing)
                sesion.registrar(timing)

            return medicion.fps_entrega or float(config.capture.camera_fps)

        with (
            Camera.from_config(config.capture) as camera,
            build_detector(config) as detector,
        ):
            try:
                sesion.fps = calentar(camera, detector)
                sesion.mensaje = ""
                umbrales = FrameThresholds.from_config(config, sesion.fps)
                print(_resumen_de_tasa(config, umbrales))

                for evento in run_segmentation(
                    flujo(camera, detector),
                    config,
                    classifier.predict,
                    fps=sesion.fps,
                ):
                    aplicar_evento(sesion, evento)
            finally:
                cv2.destroyAllWindows()
    except ImportError as error:
        # Las dependencias de cámara se instalan aparte a propósito: el resto del
        # proyecto corre sin ellas. El traceback de un módulo ausente no dice eso.
        print(MENSAJE_SIN_EXTRAS.format(modulo=error.name))
        return 1
    except CameraError as error:
        print(f"error de cámara: {error}")
        return 1
    except FileNotFoundError as error:
        # Falta el bundle de MediaPipe. Es el fallo más probable la primera vez:
        # pesa 8 MB y no se versiona, así que un repositorio recién clonado no lo
        # tiene. El mensaje de `io/hands.py` ya dice qué ejecutar; lo único que
        # hacía falta era no enterrarlo bajo un traceback.
        print(f"error: {error}")
        return 1

    # El volcado va antes del texto y siempre que se haya pedido, aunque la
    # sesión se cortara con `q` a los diez segundos: los cuadros medidos son los
    # que son y el resumen dice cuántos fueron.
    if medida is not None:
        print(render_resumen(medida.medicion.resumen()))

    # Lo que quedó sin cerrar con ENTER se imprime igual: quien termina la sesión
    # pulsando `q` no debería perder lo que acaba de deletrear.
    texto = render_text(sesion.state)
    if texto:
        print(texto)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
