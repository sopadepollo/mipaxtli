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
import importlib.metadata
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from lsm.classifiers.base import IncompatibleModelError, check_preprocessing
from lsm.classifiers.registry import ClassifierRegistry
from lsm.cli import AYUDA_MANO, MANOS, MENSAJE_SIN_EXTRAS
from lsm.config import Config, load_config
from lsm.hand_check import HandMismatchWatcher
from lsm.io.camera import Camera, CameraError
from lsm.io.corpus import git_commit
from lsm.io.dataset import iter_sample_paths, now, read_sample
from lsm.io.hands import HandDetector, build_detector, dump_frame_stream
from lsm.io.preview import DemoHudState
from lsm.one_euro import OneEuroFilter, OneEuroParams
from lsm.preprocessing import preprocessing_record
from lsm.segmentation import (
    EvidenceAccumulated,
    FrameThresholds,
    GapResolved,
    LetterEmitted,
    RejectionReason,
    SegmentationEvent,
    State,
    StateChanged,
    WindowDynamic,
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
from lsm.tracking_diagnostics import (
    REST_POSES,
    CameraProbe,
    GuidedSession,
    TrackingRecorder,
    analyze,
    render_probes,
    render_report,
    report_to_json,
    summarize_probe,
)
from lsm.types import (
    FrameSlot,
    Handedness,
    InvalidFrame,
    InvalidReason,
    Prediction,
    WindowOrigin,
)
from lsm.vocabulary import ALPHABET, DYNAMIC_LABELS, Label, spec

DEFAULT_MODEL = Path("data/models/static_knn.json")
#: El modelo dinámico es OPCIONAL: sin él, la demo sigue deletreando estáticas y
#: los trazos que la segmentación entregue se rechazan en vez de caer al
#: clasificador estático. Ver `classifiers/registry.py`.
DEFAULT_DYNAMIC_MODEL = Path("data/models/dynamic_dtw.json")
#: Dónde se escriben las sesiones de diagnóstico de tracking. Dentro de `data/`,
#: que no se versiona: llevan landmarks de quien firma.
DEFAULT_DIAGNOSTIC_DIR = Path("data/diagnostico")

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
#: ESPACIO: en la sesión guiada de diagnóstico, «repetición hecha».
_SIGUIENTE = 32


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
    #: Lo que la sesión guiada de diagnóstico pide ahora. Vacío fuera de ella.
    instruccion: str = ""
    #: MediaPipe contradice a la mano declarada de forma sostenida.
    aviso_mano: bool = False

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
            instruccion=self.instruccion,
            aviso_mano=self.aviso_mano,
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


@dataclass
class Diagnostico:
    """Una sesión instrumentada para el diagnóstico de pérdidas de tracking.

    `guiada` es `None` en `--diagnostico`, donde se deletrea libremente y solo
    se registra; en `diagnosticar` pide cada letra dinámica N veces.
    """

    etiqueta: str
    salida: Path
    recorder: TrackingRecorder
    guiada: GuidedSession | None = None
    #: El flujo de landmarks tal como salió del detector. Se guarda para poder
    #: reproducir los huecos reales de esta sesión sin cámara (Bloque 2).
    flujo: list[FrameSlot] = field(default_factory=list)


def letras_dinamicas() -> tuple[str, ...]:
    """Las ocho dinámicas en el orden del glosario, LL y RR incluidas.

    Aquí van todas: lo que se diagnostica es si la mano se pierde durante el
    trazo, no si el clasificador la reconoce, y LL y RR también son trazos.
    """
    return tuple(label.value for label in ALPHABET if label in DYNAMIC_LABELS)


def anotar_evento(diagnostico: Diagnostico, evento: SegmentationEvent) -> None:
    """Lo que el reporte necesita de la segmentación, por repetición."""
    tipo: str | None = None
    etiqueta: str | None = None
    interpolados = 0
    implausibles = 0
    match evento:
        case WindowDynamic(interpolated_frames=n, implausible_frames=m):
            tipo = "WindowDynamic"
            interpolados = n
            implausibles = m
        case GapResolved(state=estado, frames=n, implausible=m, filled=relleno):
            # Paso 4: el hueco que la máquina sostuvo, en qué estado y si se
            # rellenó (`label`: «STABLE/relleno», «TRACKING/cortado»...).
            tipo = "GapResolved"
            etiqueta = f"{estado.value}/{'relleno' if relleno else 'cortado'}"
            interpolados = n
            implausibles = m
        case WindowRejected(reason=reason) if reason in (
            RejectionReason.DYNAMIC_INTERRUPTED,
            RejectionReason.DYNAMIC_TOO_LONG,
        ):
            tipo = str(reason)
        case LetterEmitted(prediction=prediction):
            tipo = "LetterEmitted"
            etiqueta = prediction.label
        case _:
            return
    registros = diagnostico.recorder.records
    registro = (
        registros[evento.frame_index] if evento.frame_index < len(registros) else None
    )
    diagnostico.recorder.note(
        tipo,
        evento.frame_index,
        label=etiqueta,
        prompt=registro.prompt if registro else None,
        repetition=registro.repetition if registro else None,
        interpolated=interpolados,
        implausible=implausibles,
    )


def escribir_diagnostico(
    diagnostico: Diagnostico, config: Config, extra: dict[str, object]
) -> Path:
    """Vuelca el reporte, los registros crudos y el flujo de landmarks."""
    carpeta = diagnostico.salida / (
        f"{now().strftime('%Y-%m-%d-%H%M%S')}-{diagnostico.etiqueta}"
    )
    carpeta.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "iluminacion": diagnostico.etiqueta,
        "mano declarada": extra.pop("mano declarada", "—"),
        "fecha": now().isoformat(),
        "commit": git_commit(Path.cwd()),
        "mediapipe": _version("mediapipe"),
        # Todos los parámetros del detector, no solo los tres umbrales: el
        # reporte tiene que poder compararse con uno hecho tras cambiar
        # cualquiera de ellos (Bloque 1, barrido de min_tracking_confidence).
        **{
            f"hands.{campo}": valor
            for campo, valor in config.hands.model_dump(mode="json").items()
        },
        "segmentation.min_detection_score": config.segmentation.min_detection_score,
        "segmentation.motion_threshold_per_s": (
            config.segmentation.motion_threshold_per_s
        ),
        "segmentation.velocity_threshold_per_s": (
            config.segmentation.velocity_threshold_per_s
        ),
        "segmentation.velocity_window_ms": config.segmentation.velocity_window_ms,
        "segmentation.closing_window_ms": config.segmentation.closing_window_ms,
        "segmentation.motion_exhausted_ms": config.segmentation.motion_exhausted_ms,
        "segmentation.motion_min_ms": config.segmentation.motion_min_ms,
        "segmentation.motion_confirm_low_ms": config.segmentation.motion_confirm_low_ms,
        "capture.camera_fps (pedidos)": config.capture.camera_fps,
        "pre_candidate_ms": config.diagnostics.pre_candidate_ms,
        "capture.exposure": config.capture.exposure,
        # Plausibilidad, relleno y One Euro de la sesión (ADR 0027 a 0029): el
        # reporte de un barrido tiene que decir con qué red se midió.
        **{
            f"preprocesado.{clave}": valor
            for clave, valor in preprocessing_record(config).items()
        },
        **extra,
    }
    if diagnostico.guiada is not None:
        metadata["letras"] = " ".join(diagnostico.guiada.letters)
        metadata["repeticiones por letra"] = diagnostico.guiada.repetitions
        if diagnostico.guiada.rest_poses:
            metadata["posturas de reposo"] = " ".join(diagnostico.guiada.rest_poses)
            metadata["diagnostics.rest_ms"] = config.diagnostics.rest_ms
            metadata["diagnostics.rest_settle_ms"] = config.diagnostics.rest_settle_ms
            metadata["diagnostics.rest_windows_ms"] = " ".join(
                f"{w:g}" for w in config.diagnostics.rest_windows_ms
            )
            metadata["diagnostics.stable_landmarks"] = " ".join(
                str(i) for i in config.diagnostics.stable_landmarks
            )
        metadata["repeticiones descartadas"] = (
            ", ".join(f"{letra}#{n}" for letra, n in diagnostico.guiada.discarded)
            or "ninguna"
        )

    recorder = diagnostico.recorder
    reporte = analyze(recorder.records, recorder.events, config)
    (carpeta / "diagnostico.md").write_text(
        render_report(reporte, metadata), encoding="utf-8"
    )
    (carpeta / "diagnostico.json").write_text(
        json.dumps(
            report_to_json(reporte, recorder.records, recorder.events, metadata),
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    dump_frame_stream(tuple(diagnostico.flujo), carpeta / "flujo.json")
    return carpeta


def _version(paquete: str) -> str:
    try:
        return importlib.metadata.version(paquete)
    except importlib.metadata.PackageNotFoundError:
        return "no instalado"


def aplicar_evento(sesion: Sesion, evento: SegmentationEvent) -> None:
    """Traduce un evento de la máquina de estados a lo que la demo hace con él.

    `WindowRejected` lleva el motivo pero **no** la predicción: la ventana pudo
    rechazarse antes de clasificarla. Por eso el motivo va al mensaje y la última
    predicción se deja como estaba.
    """
    match evento:
        case LetterEmitted(prediction=prediction, origin=origin):
            sesion.ultima = prediction
            sesion.aplicar(LetterSignal(label=Label(prediction.label)))
            if origin is WindowOrigin.DYNAMIC:
                # Qué clasificador habló: una letra que sale por el camino
                # equivocado es el primer síntoma de un umbral de movimiento mal
                # calibrado, y sin esto no se ve.
                sesion.mensaje += " (trazo)"
        case WindowDynamic(window=window):
            sesion.mensaje = f"trazo de {len(window)} frames"
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


def cargar_registro(
    estatico: Path, dinamico: Path | None, config: Config
) -> ClassifierRegistry:
    """Carga los modelos exportados en su ranura del registry.

    `from_export` rechaza un `feature_spec_version` o un
    `detector_input` que no coincidan, que es la defensa contra predecir
    en silencio con una normalización distinta a la del entrenamiento. Aquí se
    rechaza además un modelo entrenado con otro preprocesado —plausibilidad,
    relleno, One Euro— que el de `config` (`base.check_preprocessing`): con
    `--one-euro on` y un modelo entrenado sin él, las plantillas describen otra
    señal.

    El estático es obligatorio; el dinámico no. Sin él la demo avisa y sigue:
    deletrear estáticas no depende de la Fase 5, y los trazos que la
    segmentación entregue salen UNKNOWN en vez de caer al estático.
    """
    if not estatico.exists():
        raise SystemExit(
            f"no hay modelo en {estatico}. Entrenalo primero:\n"
            "  uv run lsm-train --sin-sintetico"
        )
    payload_estatico = json.loads(estatico.read_text(encoding="utf-8"))
    payload_dinamico = None
    if dinamico is not None and dinamico.exists():
        payload_dinamico = json.loads(dinamico.read_text(encoding="utf-8"))
    elif dinamico is not None:
        print(
            f"aviso: no hay modelo dinámico en {dinamico}; las letras con "
            "movimiento no se reconocerán. `uv run lsm-train --sin-sintetico` "
            "entrena los dos."
        )
    runtime = preprocessing_record(config)
    for ruta, payload in ((estatico, payload_estatico), (dinamico, payload_dinamico)):
        if payload is None:
            continue
        try:
            check_preprocessing(payload, runtime)
        except IncompatibleModelError as error:
            raise SystemExit(f"{ruta}: {error}") from None
    return ClassifierRegistry.from_exports(
        static=payload_estatico, dynamic=payload_dinamico
    )


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
    # Sin reposo fabricado (Bloque 4): una dinámica grabada desde el Bloque 4
    # trae el reposo que la cerró, así que se cierra sola. Las grabadas antes
    # acaban con el trazo en marcha y aquí salen interrumpidas: eso es lo que
    # son, y fabricarles un reposo escondía el problema (ADR 0023).
    for ruta in sorted(iter_sample_paths(raiz)):
        frames = read_sample(ruta).frames
        yield from frames
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
        help="modelo estático exportado por lsm-train",
    )
    parser.add_argument(
        "--modelo-dinamico",
        type=Path,
        default=DEFAULT_DYNAMIC_MODEL,
        dest="modelo_dinamico",
        help="modelo dinámico exportado por lsm-train (opcional)",
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
    parser.add_argument(
        "--diagnostico",
        default=None,
        metavar="ETIQUETA",
        help=(
            "registra cada cuadro de la sesión en vivo y al salir escribe el "
            "diagnóstico de pérdidas de tracking. ETIQUETA describe la "
            "iluminación (p. ej. habitual, lampara). Necesita cámara."
        ),
    )
    parser.add_argument(
        "--diagnostico-salida",
        type=Path,
        default=DEFAULT_DIAGNOSTIC_DIR,
        dest="diagnostico_salida",
        help=f"dónde escribir los diagnósticos (por defecto: {DEFAULT_DIAGNOSTIC_DIR})",
    )
    parser.add_argument("--mano", choices=sorted(MANOS), default=None, help=AYUDA_MANO)
    parser.add_argument(
        "--one-euro",
        choices=("on", "off"),
        default=None,
        dest="one_euro",
        help=(
            "activa o desactiva el filtro One Euro (§4) para esta sesión, por "
            "encima de smoothing.enabled de config.yaml"
        ),
    )
    parser.add_argument(
        "--one-euro-params",
        default=None,
        dest="one_euro_params",
        metavar="MIN_CUTOFF,BETA,D_CUTOFF",
        help="los tres parámetros del One Euro para esta sesión, p. ej. 0.5,1,2",
    )
    parser.add_argument(
        "--comparar-one-euro",
        action="store_true",
        dest="comparar_one_euro",
        help=(
            "dibuja a la vez el esqueleto crudo (gris) y el filtrado por el One "
            "Euro (color), para comparar a ojo cuánto temblor quita y cuánto "
            "retrasa. Usa los parámetros efectivos aunque el filtro esté apagado"
        ),
    )
    subcomandos = parser.add_subparsers(dest="comando")
    guiado = subcomandos.add_parser(
        "diagnosticar",
        help=(
            "sesión guiada de diagnóstico: pide cada letra dinámica N veces y "
            "escribe el reporte de pérdidas de tracking"
        ),
    )
    guiado.add_argument(
        "--iluminacion",
        required=True,
        help="cómo está iluminada la escena, p. ej. habitual o lampara",
    )
    guiado.add_argument(
        "--repeticiones",
        type=int,
        default=None,
        help="repeticiones por letra (por defecto: diagnostics.repetitions_per_letter)",
    )
    guiado.add_argument(
        "--letras",
        default=None,
        help=(
            "solo estas dinámicas, separadas por comas (p. ej. X,ENIE,Q). Por "
            "defecto las ocho del glosario"
        ),
    )
    guiado.add_argument(
        "--presencia",
        type=float,
        default=None,
        help=(
            "hands.min_hand_presence_confidence para esta sesión (barrido del "
            "Paso 5, ADR 0030); queda en los metadatos del reporte"
        ),
    )
    guiado.add_argument(
        "--tracking",
        type=float,
        default=None,
        help="hands.min_tracking_confidence para esta sesión (barrido del Paso 5)",
    )
    guiado.add_argument(
        "--sin-reposo",
        action="store_true",
        dest="sin_reposo",
        help="solo las letras, sin la prueba de reposo del principio",
    )
    guiado.add_argument(
        "--solo-reposo",
        action="store_true",
        dest="solo_reposo",
        help=(
            "solo la prueba de reposo (estática e inicio de la J, "
            "diagnostics.rest_ms cada una), sin las letras"
        ),
    )
    sondeo = subcomandos.add_parser(
        "medir-camara",
        help=(
            "prueba cada combinación de backend, formato (MJPG o el del driver) y "
            "resolución, y reporta los fps reales sin cuadros repetidos"
        ),
    )
    sondeo.add_argument(
        "--segundos",
        type=float,
        default=None,
        help="segundos por configuración (por defecto: telemetry.camera_probe_seconds)",
    )
    sondeo.add_argument(
        "--con-deteccion",
        action="store_true",
        dest="con_deteccion",
        help=(
            "pasa cada cuadro por MediaPipe, como en la demo. Sin esto se mide solo "
            "la cámara; con esto, si la tubería es la que limita"
        ),
    )
    sondeo.add_argument(
        "--exposicion",
        default=None,
        help=(
            "barre exposiciones manuales en vez de formatos: valores separados "
            "por comas (p. ej. -5,-6,-7,-8, log2 de segundos en Windows), sobre "
            "la configuración de cámara de config.yaml. Incluye la automática "
            "como referencia"
        ),
    )
    # También aquí, para que `diagnosticar --mano derecha` funcione además de
    # `--mano derecha diagnosticar`. SUPPRESS: si no se da, no pisa el de arriba.
    guiado.add_argument(
        "--mano", choices=sorted(MANOS), default=argparse.SUPPRESS, help=AYUDA_MANO
    )
    # Lo mismo con las opciones del One Euro: `diagnosticar --one-euro on` tiene
    # que funcionar igual que `--one-euro on diagnosticar`.
    guiado.add_argument(
        "--one-euro",
        choices=("on", "off"),
        default=argparse.SUPPRESS,
        dest="one_euro",
        help="activa o desactiva el One Euro para esta sesión",
    )
    guiado.add_argument(
        "--one-euro-params",
        default=argparse.SUPPRESS,
        dest="one_euro_params",
        metavar="MIN_CUTOFF,BETA,D_CUTOFF",
        help="los tres parámetros del One Euro para esta sesión",
    )
    guiado.add_argument(
        "--comparar-one-euro",
        action="store_true",
        default=argparse.SUPPRESS,
        dest="comparar_one_euro",
        help="dibuja el esqueleto crudo (gris) y el filtrado (color)",
    )
    return parser


def con_one_euro(config: Config, estado: str | None, parametros: str | None) -> Config:
    """`config` con el One Euro que pidió la línea de comandos (ADR 0028)."""
    cambios: dict[str, object] = {}
    if estado is not None:
        cambios["enabled"] = estado == "on"
    if parametros is not None:
        try:
            minimo, beta, derivada = (float(v) for v in parametros.split(","))
        except ValueError:
            msg = (
                f"--one-euro-params espera MIN_CUTOFF,BETA,D_CUTOFF, no {parametros!r}"
            )
            raise SystemExit(msg) from None
        cambios |= {"min_cutoff": minimo, "beta": beta, "d_cutoff": derivada}
    if not cambios:
        return config
    crudo = config.model_dump(mode="json")
    crudo["smoothing"] |= cambios
    return Config.model_validate(crudo)


def con_umbrales_de_mediapipe(
    config: Config, presencia: float | None, tracking: float | None
) -> Config:
    """`config` con los umbrales del detector de la línea de comandos (Paso 5)."""
    cambios: dict[str, float] = {}
    if presencia is not None:
        cambios["min_hand_presence_confidence"] = presencia
    if tracking is not None:
        cambios["min_tracking_confidence"] = tracking
    if not cambios:
        return config
    crudo = config.model_dump(mode="json")
    crudo["hands"] |= cambios
    return Config.model_validate(crudo)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = con_one_euro(load_config(args.config), args.one_euro, args.one_euro_params)
    config = con_umbrales_de_mediapipe(
        config, getattr(args, "presencia", None), getattr(args, "tracking", None)
    )

    medida = _medida_pedida(args, config)
    if isinstance(medida, str):
        print(medida)
        return 2
    if getattr(args, "comando", None) == "medir-camara":
        return _medir_camara(config, args)
    diagnostico = _diagnostico_pedido(args, config)
    if isinstance(diagnostico, str):
        print(diagnostico)
        return 2

    # Antes de cargar el modelo: si no hay muestras que reproducir, cargarlo no
    # sirve para nada y el mensaje llega igual de rápido.
    if args.desde_dataset is not None and not sorted(
        iter_sample_paths(args.desde_dataset)
    ):
        print(SIN_MUESTRAS.format(raiz=args.desde_dataset))
        return 1

    registro = cargar_registro(args.modelo, args.modelo_dinamico, config)
    sesion = Sesion(config=config)

    if args.desde_dataset is not None:
        flujo = flujo_desde_dataset(args.desde_dataset, config)
        for evento in run_segmentation(flujo, config, registro):
            aplicar_evento(sesion, evento)
        print(render_text(sesion.state))
        return 0

    if args.mano is None:
        # Sin mano declarada no hay espejo del paso 2 que aplicar (ADR 0017), y
        # adivinarla con la etiqueta de MediaPipe es justo lo que dejó de hacerse.
        print(
            "la sesión en vivo necesita --mano derecha o --mano izquierda: es la "
            "mano con la que vas a firmar (ADR 0017)"
        )
        return 2
    print(
        "One Euro: "
        + ("activo" if config.smoothing.enabled else "apagado")
        + f" (min_cutoff {config.smoothing.min_cutoff:g}, beta "
        f"{config.smoothing.beta:g}, d_cutoff {config.smoothing.d_cutoff:g})"
        + (
            "; el preview compara crudo (gris) y filtrado"
            if args.comparar_one_euro
            else ""
        )
    )
    return _sesion_en_vivo(
        config,
        registro,
        sesion,
        medida,
        diagnostico,
        mano=MANOS[args.mano],
        comparar_one_euro=args.comparar_one_euro,
    )


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


#: Lo que prueba `medir-camara` (Fase 5.1, Bloque 1). Son las hipótesis del ADR
#: 0017 sobre los 16.4 fps reales —formato y transporte, no exposición—, no
#: umbrales: backend × formato × resolución. DSHOW y MSMF solo existen en
#: Windows; en otro sistema esas filas salen con su error.
_SONDEO_BACKENDS = ("auto", "MSMF", "DSHOW")
_SONDEO_FORMATOS: tuple[str | None, ...] = (None, "MJPG")
_SONDEO_RESOLUCIONES = ((1280, 720), (640, 480))


def _medir_camara(config: Config, args: argparse.Namespace) -> int:
    """Mide cada configuración de cámara y escribe el reporte.

    Sin descartar duplicados —es lo que se quiere contar— y tras descartar los
    `telemetry.warmup_discard_frames` primeros cuadros, que llegan lentos.
    """
    segundos = args.segundos or config.telemetry.camera_probe_seconds
    sondeos: list[CameraProbe] = []
    try:
        import cv2  # noqa: F401 — falla aquí, con el mensaje de extras

        detector = build_detector(config) if args.con_deteccion else None
        if detector is not None:
            detector.open()
        try:
            if args.exposicion is not None:
                valores: list[float | None] = [None] + [
                    float(v) for v in args.exposicion.split(",") if v.strip()
                ]
                for exposicion in valores:
                    pedido = (
                        f"{config.capture.backend} "
                        f"{config.capture.frame_width}x{config.capture.frame_height} "
                        f"exposición {'auto' if exposicion is None else exposicion}"
                    )
                    print(f"midiendo {pedido} ...", flush=True)
                    sondeos.append(
                        _sondear(
                            config,
                            pedido,
                            backend=config.capture.backend,
                            formato=config.capture.fourcc,
                            ancho=config.capture.frame_width,
                            alto=config.capture.frame_height,
                            segundos=segundos,
                            detector=detector,
                            exposicion=exposicion,
                        )
                    )
            for backend in _SONDEO_BACKENDS if args.exposicion is None else ():
                for formato in _SONDEO_FORMATOS:
                    for ancho, alto in _SONDEO_RESOLUCIONES:
                        pedido = f"{backend} {formato or 'driver'} {ancho}x{alto}"
                        print(f"midiendo {pedido} ...", flush=True)
                        sondeos.append(
                            _sondear(
                                config,
                                pedido,
                                backend=backend,
                                formato=formato,
                                ancho=ancho,
                                alto=alto,
                                segundos=segundos,
                                detector=detector,
                            )
                        )
        finally:
            if detector is not None:
                detector.close()
    except ImportError as error:
        print(MENSAJE_SIN_EXTRAS.format(modulo=error.name))
        return 1

    metadata: dict[str, object] = {
        "fecha": now().isoformat(),
        "commit": git_commit(Path.cwd()),
        "segundos por configuración": segundos,
        "cuadros de calentamiento descartados": config.telemetry.warmup_discard_frames,
        "con MediaPipe en el bucle": "sí" if args.con_deteccion else "no",
        "cámara": config.capture.camera_index,
        "fps pedidos": config.capture.camera_fps,
    }
    texto = render_probes(sondeos, metadata)
    salida = args.diagnostico_salida
    salida.mkdir(parents=True, exist_ok=True)
    ruta = salida / f"camara-{now().strftime('%Y-%m-%d-%H%M%S')}.md"
    ruta.write_text(texto, encoding="utf-8")
    print(texto)
    print(f"reporte escrito en {ruta}")
    return 0


def _sondear(
    config: Config,
    pedido: str,
    *,
    backend: str,
    formato: str | None,
    ancho: int,
    alto: int,
    segundos: float,
    detector: HandDetector | None,
    exposicion: float | None = None,
) -> CameraProbe:
    camara = Camera(
        index=config.capture.camera_index,
        width=ancho,
        height=alto,
        fps=config.capture.camera_fps,
        thumbnail_px=config.capture.duplicate_thumbnail_px,
        fourcc=formato,
        backend=backend,
        drop_duplicates=False,
        exposure=exposicion,
    )
    try:
        with camara:
            aceptado = camara.negotiated()
            for _ in range(config.telemetry.warmup_discard_frames):
                camara.read()
            tiempos: list[float] = []
            miniaturas: list[bytes] = []
            luces: list[float] = []
            inicio = time.perf_counter()
            while time.perf_counter() - inicio < segundos:
                frame = camara.read()
                if detector is not None:
                    detector.detect(frame.rgb)
                tiempos.append(time.perf_counter() * 1000.0)
                miniaturas.append(frame.thumbnail)
                luces.append(frame.mean_luminance)
    except CameraError as error:
        return CameraProbe(
            requested=pedido,
            negotiated="",
            frames=0,
            seconds=0.0,
            duplicates=0,
            interval_p50_ms=None,
            interval_p95_ms=None,
            error=str(error).split(".")[0],
        )
    return summarize_probe(
        pedido,
        (
            f"{aceptado.backend} {aceptado.fourcc or '?'} "
            f"{aceptado.width}x{aceptado.height} @ {aceptado.fps:.0f} · "
            f"exp {aceptado.exposure:g} (auto {aceptado.auto_exposure:g})"
        ),
        tiempos,
        miniaturas,
        luces,
    )


def _diagnostico_pedido(
    args: argparse.Namespace, config: Config
) -> Diagnostico | str | None:
    """Qué diagnóstico pidieron los argumentos, o el error que hay que imprimir.

    El diagnóstico mide pérdidas de la cámara en vivo: sobre un dataset no hay
    cámara, y combinado con `--medir-fps` la sesión se cortaría a mitad de la
    guía. Se rechazan las dos combinaciones.
    """
    guiado = getattr(args, "comando", None) == "diagnosticar"
    etiqueta = args.iluminacion if guiado else args.diagnostico
    if etiqueta is None:
        return None
    if guiado and args.diagnostico is not None:
        return "diagnosticar ya registra la sesión: sobra --diagnostico"
    if args.desde_dataset is not None:
        return (
            "el diagnóstico mide pérdidas de tracking de la cámara en vivo y no "
            "se puede combinar con --desde-dataset"
        )
    if args.medir_fps:
        return "el diagnóstico no se puede combinar con --medir-fps"
    limpia = "".join(c if c.isalnum() or c in "-_" else "-" for c in etiqueta)
    if not limpia.strip("-_"):
        return f"etiqueta de iluminación inválida: {etiqueta!r}"
    guiada = None
    if guiado:
        repeticiones = args.repeticiones or config.diagnostics.repetitions_per_letter
        if repeticiones < 1:
            return f"--repeticiones tiene que ser positivo, no {repeticiones}"
        letras = letras_dinamicas()
        if args.letras:
            pedidas = tuple(
                x.strip().upper() for x in args.letras.split(",") if x.strip()
            )
            desconocidas = [x for x in pedidas if x not in letras]
            if desconocidas:
                return (
                    f"--letras: {', '.join(desconocidas)} no son dinámicas; las "
                    f"válidas son {', '.join(letras)}"
                )
            letras = pedidas
        guiada = GuidedSession(
            letters=() if args.solo_reposo else letras,
            repetitions=repeticiones,
            rest_poses=() if args.sin_reposo else tuple(REST_POSES),
            rest_ms=config.diagnostics.rest_ms,
        )
    return Diagnostico(
        etiqueta=limpia,
        salida=args.diagnostico_salida,
        recorder=TrackingRecorder(config=config),
        guiada=guiada,
    )


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
    registro: ClassifierRegistry,
    sesion: Sesion,
    medida: Medida | None = None,
    diagnostico: Diagnostico | None = None,
    *,
    mano: Handedness = Handedness.RIGHT,
    comparar_one_euro: bool = False,
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

        # Solo para dibujar (`--comparar-one-euro`): el mismo filtro que aplica
        # la máquina, con los parámetros efectivos aunque esté apagado.
        comparador = (
            OneEuroFilter(replace(OneEuroParams.from_config(config), enabled=True))
            if comparar_one_euro
            else None
        )

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
                recibido_ms = time.perf_counter() * 1000.0
                crono.marcar(Stage.CAMARA)
                slot = detector.detect(frame.rgb)
                crono.marcar(Stage.DETECCION)
                sesion.aviso_mano = vigia.observe(slot)

                if diagnostico is not None:
                    # Se registra ANTES de cederlo: `estado_maquina` es todavía
                    # el que dejó el cuadro anterior, o sea el estado en que
                    # este cuadro encuentra a la máquina.
                    guiada_ahora = diagnostico.guiada
                    if guiada_ahora is not None:
                        guiada_ahora.tick(recibido_ms)
                    actual = (
                        guiada_ahora.current()
                        if guiada_ahora is not None and guiada_ahora.recording
                        else None
                    )
                    diagnostico.recorder.observe(
                        slot,
                        wall_ms=recibido_ms,
                        detector_ms=float(
                            getattr(detector, "timestamp_ms", len(diagnostico.flujo))
                        ),
                        luminance=frame.mean_luminance,
                        thumbnail=frame.thumbnail,
                        skipped_duplicates=frame.skipped_duplicates,
                        state=sesion.estado_maquina.value,
                        prompt=actual[0] if actual else None,
                        repetition=actual[1] if actual else None,
                    )
                    diagnostico.flujo.append(slot)
                    if diagnostico.guiada is not None:
                        sesion.instruccion = diagnostico.guiada.instruction()

                imagen = frame.bgr
                if config.capture.preview_mirror:
                    imagen = cv2.flip(imagen, 1)
                if isinstance(slot, InvalidFrame):
                    if comparador is not None:
                        comparador.reset()
                    sesion.aplicar(HandAbsent())
                elif comparador is not None:
                    espejo = config.capture.preview_mirror
                    draw_landmarks(imagen, slot, mirrored=espejo, ghost=True)
                    filtrado = comparador.step(slot, slot.timestamp_ms or recibido_ms)
                    draw_landmarks(imagen, filtrado, mirrored=espejo)
                    sesion.aplicar(HandPresent())
                else:
                    draw_landmarks(imagen, slot, mirrored=config.capture.preview_mirror)
                    sesion.aplicar(HandPresent())
                draw_demo_hud(imagen, sesion.hud())
                cv2.imshow(ventana, imagen)

                tecla = cv2.waitKey(1) & 0xFF
                if tecla in _SALIR:
                    return
                guiada = diagnostico.guiada if diagnostico is not None else None
                if guiada is not None and tecla == _SIGUIENTE:
                    guiada.press_next(recibido_ms)
                elif guiada is not None and tecla in _BORRAR:
                    # En la sesión guiada BACKSPACE no borra una letra: descarta
                    # la repetición anterior, que se hizo mal, y la vuelve a pedir.
                    guiada.discard_last()
                elif tecla in _BORRAR:
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
            # Los primeros cuadros de una webcam llegan lentos mientras ajusta
            # exposición y enfoque: en el ADR 0017 la tasa congelada al arrancar
            # fue 18.4 fps y la sesión corrió a 29.5, así que todos los umbrales
            # en milisegundos duraron ~1.6 veces menos. Se descartan antes de
            # medir. Con `capture.drop_duplicate_frames` lo que se mide es la
            # tasa de cuadros **nuevos**, que es la que ve la máquina de estados.
            for _ in range(config.telemetry.warmup_discard_frames):
                camera.read()
                cv2.waitKey(1)
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
            Camera.from_config(
                config.capture,
                thumbnail_px=(
                    config.capture.duplicate_thumbnail_px
                    if diagnostico is not None
                    else None
                ),
            ) as camera,
            build_detector(config, declared_hand=mano) as detector,
        ):
            try:
                sesion.fps = calentar(camera, detector)
                vigia = HandMismatchWatcher(
                    config=config, declared=mano, fps=sesion.fps
                )
                sesion.mensaje = ""
                umbrales = FrameThresholds.from_config(config, sesion.fps)
                print(_resumen_de_tasa(config, umbrales))

                for evento in run_segmentation(
                    flujo(camera, detector),
                    config,
                    registro,
                    fps=sesion.fps,
                ):
                    aplicar_evento(sesion, evento)
                    if diagnostico is not None:
                        anotar_evento(diagnostico, evento)
            finally:
                cv2.destroyAllWindows()
                if diagnostico is not None and diagnostico.recorder.records:
                    carpeta = escribir_diagnostico(
                        diagnostico,
                        config,
                        {
                            "camara (backend)": camera.backend_name(),
                            "fps medidos al arrancar": (
                                f"{sesion.fps:.1f}" if sesion.fps else "—"
                            ),
                            "mano declarada": mano.value,
                        },
                    )
                    print(f"diagnóstico escrito en {carpeta}")
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
