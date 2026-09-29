"""CLI de recolección de dataset (`ARQUITECTURA.md` §4.7, Fase 1).

Cuatro subcomandos:

- `calibrar` — comprueba **a ojo** que la lateralidad que reporta el detector es
  la mano real, y lo deja escrito. Es la única defensa contra un error que no
  produce ningún síntoma; ver `lsm.io.calibration`.
- `grabar` — la sesión de captura: preview con landmarks, letra objetivo,
  contador de muestras y el indicador de calidad en vivo. **Se niega a arrancar
  sin calibración vigente**, y en modo formal, sin el glosario validado.
- `consentimiento` — registra qué autorizó cada persona. Es la llave sin la cual
  `--guardar-video` no hace nada.
- `verificar` — relee el dataset, re-deriva las features y comprueba que salen
  idénticas a las del momento de grabar. No necesita cámara.
- `marcar-truncadas` — excluye sin borrar las dinámicas anteriores al Bloque 4
  que acaban en movimiento (ADR 0023).
- `rechazados` — vuelve a segmentar los intentos dinámicos que `grabar` rechazó
  y guardó en `rechazados/`, con la configuración de hoy (ADR 0024).

**Los tres bloqueos de `grabar` son rechazos, no avisos.** Un aviso en una
terminal, antes de cuarenta minutos de grabación con otra persona delante, no lo
lee nadie; y los tres fallos que tapan —lateralidad invertida, glosario sin
validar, consentimiento ausente— comparten la propiedad de no dejar ningún rastro
en el dataset resultante. La única forma de arreglarlos después es volver a grabar
con todo el mundo.

**Qué hace este módulo y qué no.** Aquí se abre la cámara, se dibuja y se
escriben archivos. El criterio de si una ventana sirve como muestra está en
`lsm.capture`, que es código puro y se prueba sin hardware; el formato en disco
está en `lsm.io.dataset`. La separación es la que permite que el criterio de
aceptación de esta fase sea un test y no una sesión de grabación.

**Sobre el espejado**, que es el error caro de esta capa: el detector recibe
siempre el cuadro tal como sale de la cámara (`feature-spec.md` §0.3). El preview
se espeja porque es lo que espera quien se mira en pantalla, y ese espejado ocurre
después de detectar, sobre píxeles que ya no van a ningún sitio.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from lsm.capture import (
    BufferedFrame,
    FrameBuffer,
    Rejection,
    WindowQuality,
    delimit_dynamic_stroke,
    evaluate_window,
    explain,
    final_velocity,
    is_truncated,
    maximum_frames,
    minimum_frames,
    resegment_attempt,
)
from lsm.cli import AYUDA_MANO, MANOS, MENSAJE_SIN_EXTRAS
from lsm.config import Config, load_config
from lsm.features import (
    FEATURE_SPEC_VERSION,
    ExtractionRejected,
    extract_sequence_features,
)
from lsm.gaps import GapPolicy
from lsm.hand_check import HandMismatchWatcher, input_looks_unmirrored
from lsm.io.calibration import (
    Calibration,
    CalibrationError,
    camera_key,
    has_any_calibration,
    require_calibration,
    save_calibration,
)
from lsm.io.camera import Camera, CameraError, VideoRecorder
from lsm.io.dataset import (
    REJECTED_DIRNAME,
    Consent,
    DatasetError,
    RejectedAttempt,
    SampleMetadata,
    StoredSample,
    check_identifier,
    count_samples,
    iter_rejected_attempt_paths,
    iter_sample_paths,
    label_dir,
    may_store_video,
    next_sample_index,
    now,
    read_rejected_attempt,
    read_sample,
    save_consent,
    trial_root,
    write_rejected_attempt,
    write_sample,
    write_truncated_manifest,
)
from lsm.io.glossary import DEFAULT_GLOSSARY, is_validated
from lsm.io.hands import HandDetector, build_detector
from lsm.io.preview import HudState, draw_hud, draw_landmarks
from lsm.types import (
    Distance,
    FrameSlot,
    Handedness,
    LightDirection,
    LightLevel,
    RawFrame,
    SampleKind,
)
from lsm.vocabulary import ALPHABET, Label, spec

#: Códigos de tecla de `cv2.waitKey`. ESC y `q` hacen lo mismo porque en una
#: sesión de cuarenta minutos nadie se acuerda de cuál era.
_SALIR = frozenset({ord("q"), 27})
_GUARDAR = frozenset({ord(" ")})
_SIGUIENTE = frozenset({ord("n")})
_ANTERIOR = frozenset({ord("p")})
_MODO = frozenset({ord("m")})
_REHACER = frozenset({ord("r")})
#: Solo lo usa `calibrar`. Dos teclas porque el teclado de quien graba puede
#: no ser el mismo que el de quien escribió esto.
_CONFIRMAR = frozenset({ord("s"), ord("y")})


@dataclass
class _Sesion:
    """Estado mutable de una sesión de captura.

    Es un objeto y no un puñado de variables sueltas en el bucle porque el bucle
    ya tiene bastante con la cámara, el detector y el teclado: separar el estado
    hace que se pueda leer qué cambia con cada tecla sin seguir treinta líneas de
    condicionales.
    """

    letras: tuple[Label, ...]
    indice: int
    kind: SampleKind
    guardadas: dict[str, int]
    mensaje: str = ""
    #: Frames de la grabación dinámica en curso, o `None` si no hay ninguna.
    #: Es una lista aparte y no el buffer circular porque una dinámica puede durar
    #: más que la ventana estática: el buffer olvida, y aquí no se puede olvidar
    #: el principio del trazo.
    grabando: list[BufferedFrame] | None = None

    @property
    def label(self) -> Label:
        return self.letras[self.indice]

    def mover(self, paso: int) -> None:
        self.indice = (self.indice + paso) % len(self.letras)
        self.kind = _modo_por_defecto(self.label)
        self.grabando = None
        self.mensaje = ""


def _modo_por_defecto(label: Label) -> SampleKind:
    """El glosario decide el modo inicial; la tecla `m` decide el definitivo.

    La clase negativa `NONE` no está en el glosario y se graba de las dos formas
    —mano relajada, mano en tránsito—, así que arranca en estática y se cambia a
    mano cuando toque grabar transiciones.
    """
    if label is Label.NONE:
        return SampleKind.STATIC
    return SampleKind.DYNAMIC if spec(label).es_dinamica else SampleKind.STATIC


# --------------------------------------------------------------------------- #
# grabar
# --------------------------------------------------------------------------- #


def _clave_de_camara(camera: Camera) -> str:
    """Identificador de la cámara ya abierta, para el registro de calibración.

    Se lee **después** de abrir y con las dimensiones que la cámara entrega de
    verdad: la resolución de `config.yaml` es una petición y el driver puede
    devolver otra. Calibrar a 1280x720 y grabar a 640x480 no es la misma
    situación, así que tampoco es la misma calibración.
    """
    frame = camera.read()
    return camera_key(
        index=camera.index,
        width=frame.width,
        height=frame.height,
        backend=camera.backend_name(),
    )


#: Lo que se imprime cuando no hay ninguna cámara calibrada. El caso de la primera
#: vez, y el único que se puede detectar sin abrir el dispositivo.
_MENSAJE_SIN_CALIBRAR = """No hay ninguna cámara calibrada.

Antes de grabar hay que confirmar a ojo que la lateralidad que reporta el detector
es la mano real. Si estuviera invertida, todas las muestras se canonizarían hacia
la mano contraria — y no se notaría: el modelo entrenaría igual de bien y la
precisión sería idéntica. El error solo aparece en la Fase 7, con la app web
reconociendo cada seña al revés, y ninguna prueba automática lo detecta.

Es un minuto, una vez por cámara:
  lsm-capture calibrar"""


#: Lo que se imprime cuando alguien intenta una sesión formal con el glosario sin
#: firmar. Dice qué falta, por qué importa y cuáles son las salidas legítimas.
_MENSAJE_SIN_VALIDAR = """\
La sección 5 de {glosario} está vacía: nadie ha validado el glosario todavía.

Una sesión formal son tres personas x 29 letras x dos sesiones. Hacerlas contra un
glosario que no ha revisado una persona usuaria de LSM o un intérprete es el error
caro de este proyecto, y no se arregla después: si una seña está mal transcrita,
las ~60 repeticiones de esa letra son ruido etiquetado y hay que volver a citar a
todo el mundo.

Salidas:
  - Validar el glosario y anotar fecha, revisor y rol en la sección 5.
  - Grabar en modo prueba, que no entra al dataset:
      lsm-capture grabar --sesion-prueba ..."""


def _cmd_grabar(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    raiz: Path = args.raiz
    signer_id = check_identifier("signer_id", args.firmante)
    session_id = check_identifier("session_id", args.sesion)

    es_prueba: bool = args.sesion_prueba
    destino = trial_root(raiz) if es_prueba else raiz

    if not es_prueba and not is_validated(args.glosario):
        print(_MENSAJE_SIN_VALIDAR.format(glosario=args.glosario))
        return 1

    if not has_any_calibration(raiz):
        # Se adelanta al caso más común —nadie ha calibrado nunca— para no
        # encender una webcam y esperar a que arranque solo para rechazar. La
        # comprobación por cámara sigue estando, más abajo: necesita la
        # resolución que el driver entregue de verdad.
        print(_MENSAJE_SIN_CALIBRAR)
        return 1

    letras = _resolver_letras(args.letras)
    guarda_video = _resolver_video(raiz, signer_id, pedido=args.guardar_video)
    mano = MANOS[args.mano]
    detector = build_detector(config, declared_hand=mano)

    sesion = _Sesion(
        letras=letras,
        indice=0,
        kind=_modo_por_defecto(letras[0]),
        guardadas=count_samples(destino, signer_id, session_id),
    )

    try:
        with Camera.from_config(config.capture) as camera:
            # La calibración se exige con la cámara ya abierta —hace falta su
            # identificador, y ese depende de la resolución que entregue de
            # verdad— pero antes de cargar el modelo y de grabar un solo frame.
            calibracion = require_calibration(raiz, _clave_de_camara(camera))
            modo = "PRUEBA (no entra al dataset)" if es_prueba else "formal"
            video = "SÍ (con consentimiento registrado)" if guarda_video else "no"
            print(
                f"Sesión {session_id} de {signer_id} · {len(letras)} letras · "
                f"meta {config.capture.target_samples_per_label} por letra\n"
                f"Modo: {modo} · destino {destino}\n"
                f"Video: {video}\n"
                f"Mano declarada: {args.mano} ({mano.value})\n"
                f"Calibrada el {calibracion.fecha:%Y-%m-%d} ({calibracion.camera})"
            )
            with detector:
                _bucle(
                    camera=camera,
                    detector=detector,
                    config=config,
                    sesion=sesion,
                    raiz=destino,
                    signer_id=signer_id,
                    session_id=session_id,
                    condiciones=_condiciones(args),
                    guarda_video=guarda_video,
                    vigia=HandMismatchWatcher(
                        config=config,
                        declared=mano,
                        fps=float(config.capture.camera_fps),
                    ),
                )
    except ImportError as error:
        print(MENSAJE_SIN_EXTRAS.format(modulo=error.name))
        return 1
    except CalibrationError as error:
        print(f"sin calibración: {error}")
        return 1
    except CameraError as error:
        print(f"error de cámara: {error}")
        return 1
    except FileNotFoundError as error:
        print(f"error: {error}")
        return 1

    total = sum(sesion.guardadas.values())
    print(f"Sesión terminada: {total} muestras en {destino / signer_id / session_id}")
    return 0


def _bucle(
    *,
    camera: Camera,
    detector: HandDetector,
    config: Config,
    sesion: _Sesion,
    raiz: Path,
    signer_id: str,
    session_id: str,
    condiciones: Condiciones,
    guarda_video: bool,
    vigia: HandMismatchWatcher,
) -> None:
    """El bucle de captura. Un cuadro por vuelta."""
    import cv2

    buffer = FrameBuffer(capacity=config.capture.static_frames)
    # Imágenes en crudo, solo si se va a guardar video. Sin consentimiento esta
    # lista se queda vacía y no hay ruta por la que un cuadro acabe en disco.
    # Se dimensiona al mayor de los dos modos: una dinámica larga necesita más
    # cuadros que la ventana estática, y recortar aquí dejaría el video de una J
    # empezando a media seña.
    cuadros_maximos = max(
        config.capture.static_frames,
        maximum_frames(SampleKind.DYNAMIC, config) or 0,
    )
    imagenes: list[Any] = []
    ventana = "captura LSM"
    # La tasa real, para que la máquina de estados que delimita las dinámicas
    # convierta sus umbrales como en la demo (Bloque 4).
    marcas: deque[float] = deque(maxlen=31)

    while True:
        frame = camera.read()
        marcas.append(time.perf_counter())
        slot = detector.detect(frame.rgb)
        buffer.push(slot, frame.mean_luminance)

        if guarda_video:
            imagenes.append(frame.bgr.copy())
            del imagenes[: max(0, len(imagenes) - cuadros_maximos)]

        quality = _calidad(buffer, sesion, config)
        aviso_mano = vigia.observe(slot)
        lienzo = _dibujar(
            frame.bgr, slot, quality, sesion, config, guarda_video, aviso_mano
        )
        cv2.imshow(ventana, lienzo)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in _SALIR:
            break
        if tecla in _SIGUIENTE:
            sesion.mover(1)
        elif tecla in _ANTERIOR:
            sesion.mover(-1)
        elif tecla in _MODO:
            sesion.kind = (
                SampleKind.DYNAMIC
                if sesion.kind is SampleKind.STATIC
                else SampleKind.STATIC
            )
            sesion.grabando = None
            sesion.mensaje = f"modo {sesion.kind.value.lower()}"
        elif tecla in _REHACER:
            sesion.grabando = None
            sesion.mensaje = "grabación descartada"
        elif tecla in _GUARDAR and sesion.kind is SampleKind.DYNAMIC:
            seguir = _grabar_dinamica(
                camera=camera,
                detector=detector,
                config=config,
                sesion=sesion,
                vigia=vigia,
                ventana=ventana,
                guarda_video=guarda_video,
                imagenes=imagenes,
                cuadros_maximos=cuadros_maximos,
                fps=_tasa(marcas, config),
                raiz=raiz,
                signer_id=signer_id,
                session_id=session_id,
                condiciones=condiciones,
            )
            buffer.clear()
            if not seguir:
                break
        elif tecla in _GUARDAR:
            _al_pulsar_guardar(
                sesion=sesion,
                buffer=buffer,
                imagenes=imagenes if guarda_video else None,
                config=config,
                raiz=raiz,
                signer_id=signer_id,
                session_id=session_id,
                condiciones=condiciones,
                camera_fps=config.capture.camera_fps,
            )

    cv2.destroyWindow(ventana)


def _calidad(buffer: FrameBuffer, sesion: _Sesion, config: Config) -> WindowQuality:
    """La calidad de lo que se guardaría **ahora mismo**.

    Es lo que hace útil el preview: no describe la última muestra guardada sino la
    que saldría de pulsar la tecla en este instante, que es la información que
    quien graba necesita antes de pulsarla.
    """
    return evaluate_window(
        buffer.tail(minimum_frames(sesion.kind, config)), sesion.kind, config
    )


def _al_pulsar_guardar(
    *,
    sesion: _Sesion,
    buffer: FrameBuffer,
    imagenes: list[Any] | None,
    config: Config,
    raiz: Path,
    signer_id: str,
    session_id: str,
    condiciones: Condiciones,
    camera_fps: int,
) -> None:
    """ESPACIO en estática: guarda lo que la mano acaba de sostener.

    Las dinámicas no pasan por aquí desde el Bloque 4: las delimita la máquina
    de estados (`_grabar_dinamica`).
    """
    frames = buffer.tail(minimum_frames(sesion.kind, config))

    resultado = guardar_muestra(
        frames=frames,
        kind=sesion.kind,
        label=sesion.label,
        config=config,
        raiz=raiz,
        signer_id=signer_id,
        session_id=session_id,
        condiciones=condiciones,
        con_video=imagenes is not None,
    )
    if isinstance(resultado, Rejection):
        sesion.mensaje = f"no se guardo: {explain(resultado)}"
        return

    if imagenes is not None and resultado.video is not None:
        _guardar_video(resultado.video, imagenes[-len(frames) :], camera_fps)

    etiqueta = sesion.label.value
    sesion.guardadas[etiqueta] = sesion.guardadas.get(etiqueta, 0) + 1
    sesion.grabando = None
    sesion.mensaje = (
        f"guardada {resultado.path.name}: {len(frames)} frames, "
        f"sigma {resultado.quality.dispersion:.4f}"
    )
    # El buffer se vacía para que la siguiente repetición no herede frames de la
    # que se acaba de guardar. Sin esto, pulsar ESPACIO dos veces seguidas
    # guardaría dos muestras que comparten casi todos sus frames, y el dataset
    # tendría veinte repeticiones donde en realidad hubo cinco.
    buffer.clear()


def _tasa(marcas: deque[float], config: Config) -> float:
    """Cuadros por segundo de los últimos cuadros, o la nominal si no hay aún."""
    if len(marcas) < 2 or marcas[-1] <= marcas[0]:
        return float(config.capture.camera_fps)
    return (len(marcas) - 1) / (marcas[-1] - marcas[0])


def _grabar_dinamica(
    *,
    camera: Camera,
    detector: HandDetector,
    config: Config,
    sesion: _Sesion,
    vigia: HandMismatchWatcher,
    ventana: str,
    guarda_video: bool,
    imagenes: list[Any],
    cuadros_maximos: int,
    fps: float,
    raiz: Path,
    signer_id: str,
    session_id: str,
    condiciones: Condiciones,
) -> bool:
    """Graba una dinámica delimitada por la máquina de estados (Bloque 4).

    ESPACIO la arma. Desde ahí, la **misma** máquina que usa la demo, con la tasa
    medida, decide dónde empieza el trazo y dónde termina: la muestra es lo que
    el clasificador recibiría en vivo, más el reposo que lo cerró. Si en
    `capture.dynamic_max_ms` no entrega un trazo cerrado, la muestra se rechaza:
    nunca más se guarda un trazo truncado. `r` cancela y `q` sale.

    Devuelve `False` si se pidió salir.
    """
    import cv2

    registro: list[BufferedFrame] = []
    buffer_local = FrameBuffer(capacity=config.capture.static_frames)
    estado = {"salir": False, "cancelada": False}
    inicio = time.perf_counter()
    tope_s = config.capture.dynamic_max_ms / 1000.0
    sesion.grabando = registro
    sesion.mensaje = "grabando: haz la letra y deten la mano al terminar (r cancela)"

    def fuente() -> Iterator[FrameSlot]:
        while time.perf_counter() - inicio < tope_s:
            frame = camera.read()
            slot = detector.detect(frame.rgb)
            registro.append(BufferedFrame(slot=slot, luminance=frame.mean_luminance))
            buffer_local.push(slot, frame.mean_luminance)
            if guarda_video:
                imagenes.append(frame.bgr.copy())
                del imagenes[: max(0, len(imagenes) - cuadros_maximos)]
            calidad = evaluate_window(
                buffer_local.tail(minimum_frames(SampleKind.DYNAMIC, config)),
                SampleKind.DYNAMIC,
                config,
            )
            aviso = vigia.observe(slot)
            cv2.imshow(
                ventana,
                _dibujar(frame.bgr, slot, calidad, sesion, config, guarda_video, aviso),
            )
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in _SALIR:
                estado["salir"] = True
                return
            if tecla in _REHACER:
                estado["cancelada"] = True
                return
            yield slot

    resultado = delimit_dynamic_stroke(fuente(), config, fps)
    sesion.grabando = None
    if estado["salir"]:
        return False
    if estado["cancelada"]:
        sesion.mensaje = "grabacion descartada"
        return True
    if isinstance(resultado, Rejection):
        sesion.mensaje = _rechazo_guardado(
            resultado,
            registro,
            sesion,
            config,
            fps,
            raiz,
            signer_id,
            session_id,
            condiciones,
        )
        return True

    frames = tuple(registro[resultado.start : resultado.end + 1])
    guardada = guardar_muestra(
        frames=frames,
        kind=SampleKind.DYNAMIC,
        label=sesion.label,
        config=config,
        raiz=raiz,
        signer_id=signer_id,
        session_id=session_id,
        condiciones=condiciones,
        con_video=guarda_video,
        stroke_frames=resultado.stroke_frames,
    )
    if isinstance(guardada, Rejection):
        sesion.mensaje = _rechazo_guardado(
            guardada,
            registro,
            sesion,
            config,
            fps,
            raiz,
            signer_id,
            session_id,
            condiciones,
        )
        return True
    if guarda_video and guardada.video is not None:
        _guardar_video(
            guardada.video, imagenes[-len(frames) :], config.capture.camera_fps
        )
    etiqueta = sesion.label.value
    sesion.guardadas[etiqueta] = sesion.guardadas.get(etiqueta, 0) + 1
    sesion.mensaje = (
        f"guardada {guardada.path.name}: trazo de {resultado.stroke_frames} frames "
        f"+ {len(frames) - resultado.stroke_frames} de reposo"
    )
    return True


def _rechazo_guardado(
    motivo: Rejection,
    registro: list[BufferedFrame],
    sesion: _Sesion,
    config: Config,
    fps: float,
    raiz: Path,
    signer_id: str,
    session_id: str,
    condiciones: Condiciones,
) -> str:
    """Guarda el intento rechazado y devuelve el mensaje para el HUD (Paso 0)."""
    ruta = guardar_intento_rechazado(
        motivo=motivo,
        registro=tuple(registro),
        label=sesion.label,
        config=config,
        fps=fps,
        raiz=raiz,
        signer_id=signer_id,
        session_id=session_id,
        condiciones=condiciones,
    )
    guardado = f" (intento en {REJECTED_DIRNAME}/{ruta.name})" if ruta else ""
    return f"no se guardo: {explain(motivo)}{guardado}"


def guardar_intento_rechazado(
    *,
    motivo: Rejection,
    registro: tuple[BufferedFrame, ...],
    label: Label,
    config: Config,
    fps: float,
    raiz: Path,
    signer_id: str,
    session_id: str,
    condiciones: Condiciones,
) -> Path | None:
    """Escribe una grabación dinámica rechazada, entera y en crudo (Paso 0).

    Lo que la cámara entregó desde que se armó la grabación, huecos incluidos, con
    el motivo y la tasa con la que la máquina de estados la delimitó. Queda en
    `<LETRA>/rechazados/`, fuera del alcance del entrenamiento, para volver a
    segmentarla (`lsm-capture rechazados`) cuando la tubería tolere más, sin
    regrabarla. Un intento en el que el detector no vio ninguna mano no tiene
    nada que reprocesar y no se escribe: devuelve `None`.
    """
    mano = next(
        (b.slot.handedness for b in registro if isinstance(b.slot, RawFrame)), None
    )
    if mano is None:
        return None
    return write_rejected_attempt(
        raiz,
        RejectedAttempt(
            label=label.value,
            signer_id=signer_id,
            session_id=session_id,
            timestamp=now(),
            handedness=mano,
            light_level=condiciones.light_level,
            light_direction=condiciones.light_direction,
            distance=condiciones.distance,
            mean_luminance=sum(b.luminance for b in registro) / len(registro),
            reason=motivo.value,
            fps=fps,
            segmentation=config.segmentation.model_dump(mode="json"),
            frames=tuple(b.slot for b in registro),
        ),
    )


@dataclass(frozen=True, slots=True)
class Guardada:
    """Una muestra que sí se escribió, y con qué números."""

    path: Path
    quality: WindowQuality
    #: Ruta donde debe ir el video de esta muestra, si se pidió guardarlo.
    video: Path | None


def guardar_muestra(
    *,
    frames: tuple[BufferedFrame, ...],
    kind: SampleKind,
    label: Label,
    config: Config,
    raiz: Path,
    signer_id: str,
    session_id: str,
    condiciones: Condiciones,
    con_video: bool = False,
    stroke_frames: int | None = None,
) -> Guardada | Rejection:
    """Evalúa una ventana y, si sirve, la escribe. El paso que importa de verdad.

    Es público y no toca OpenCV a propósito. Es **el** camino por el que una
    muestra llega al dataset, así que tiene que poder ejercitarse en la suite sin
    cámara: veinte llamadas a esta función son las veinte repeticiones del
    criterio de aceptación de la Fase 1, y las features que se re-derivan de lo
    que escribe son las que se comparan contra las del pipeline.

    Si lo interactivo —el teclado, el preview, el video— quedara enredado aquí,
    ese test no existiría y la única forma de comprobar la fase sería sentarse
    delante de una webcam.
    """
    # Bloque 4: de una dinámica delimitada por la máquina se evalúa el trazo,
    # que es lo que se entrena; el reposo que lo cerró se guarda sin juzgarlo.
    evaluada = frames if stroke_frames is None else frames[:stroke_frames]
    quality = evaluate_window(evaluada, kind, config)
    if quality.rejection is not None:
        return quality.rejection

    # `accepted` garantiza los tres: una ventana sin lateralidad consistente, sin
    # σ o sin escala no llega hasta aquí. Se comprueba en vez de darse por
    # supuesto porque el que se cuela es el que acaba en el dataset.
    if (
        quality.handedness is None
        or quality.dispersion is None
        or quality.mean_scale_px is None
        or quality.arc_length is None
    ):
        raise DatasetError(
            "ventana aceptada sin metadatos completos: es un error de programación "
            "en lsm.capture.evaluate_window, no algo que quien graba pueda corregir"
        )

    directorio = label_dir(raiz, signer_id, session_id, label.value)
    indice = next_sample_index(directorio)
    nombre_video = f"{indice:03d}.mp4" if con_video else None

    muestra = StoredSample(
        metadata=SampleMetadata(
            label=label.value,
            signer_id=signer_id,
            session_id=session_id,
            timestamp=now(),
            handedness=quality.handedness,
            light_level=condiciones.light_level,
            light_direction=condiciones.light_direction,
            distance=condiciones.distance,
            mean_luminance=quality.mean_luminance,
            mean_scale_px=quality.mean_scale_px,
            kind=kind,
            dispersion=quality.dispersion,
            arc_length=quality.arc_length,
            # El valor efectivo, no el que diga config.yaml cuando alguien lea
            # esta muestra dentro de seis meses. Ver `SampleMetadata`.
            handedness_swapped=config.hands.mediapipe_reports_mirrored_handedness,
            video=nombre_video,
            stroke_frames=stroke_frames,
            feature_spec_version=FEATURE_SPEC_VERSION,
        ),
        frames=tuple(buffered.slot for buffered in frames),
    )
    ruta = write_sample(raiz, muestra, index=indice)
    return Guardada(
        path=ruta,
        quality=quality,
        video=ruta.with_name(nombre_video) if nombre_video else None,
    )


def _guardar_video(destino: Path, imagenes: list[Any], fps: int) -> None:
    """Escribe los cuadros de la muestra recién aceptada.

    Solo se llega aquí con las dos llaves puestas: consentimiento registrado para
    esta persona y `--guardar-video` en la línea de comandos. Se guardan los
    cuadros **sin espejar y sin landmarks encima**, que es el registro de lo que
    pasó y no una captura de pantalla del programa.
    """
    if not imagenes:
        return
    alto, ancho = int(imagenes[0].shape[0]), int(imagenes[0].shape[1])
    with VideoRecorder(path=destino, width=ancho, height=alto, fps=fps) as grabador:
        for imagen in imagenes:
            grabador.write(imagen)


def _dibujar(
    imagen: Any,
    slot: FrameSlot,
    quality: WindowQuality,
    sesion: _Sesion,
    config: Config,
    guarda_video: bool,
    aviso_mano: bool = False,
) -> Any:
    """Arma el cuadro del preview: espejo, landmarks y HUD, en ese orden.

    El espejo va **primero** y sobre una copia. Sobre una copia porque el original
    puede acabar en el archivo de video y ahí tiene que ir tal cual salió de la
    cámara; primero porque `preview_position` dibuja los landmarks ya reflejados
    y encontrarían la mano en el sitio equivocado si la imagen no lo estuviera.
    """
    import cv2

    espejo = config.capture.preview_mirror
    lienzo = cv2.flip(imagen, 1) if espejo else imagen.copy()

    if isinstance(slot, RawFrame):
        draw_landmarks(lienzo, slot, mirrored=espejo)

    letra = "NONE" if sesion.label is Label.NONE else spec(sesion.label).display
    draw_hud(
        lienzo,
        HudState(
            letra=letra,
            label=sesion.label.value,
            kind=sesion.kind,
            guardadas=sesion.guardadas.get(sesion.label.value, 0),
            objetivo=config.capture.target_samples_per_label,
            quality=quality,
            max_dispersion=config.quality.max_dispersion,
            min_trajectory_arc=config.capture.min_trajectory_arc,
            grabando=None if sesion.grabando is None else len(sesion.grabando),
            mensaje=sesion.mensaje,
            guarda_video=guarda_video,
            aviso_mano=aviso_mano,
        ),
    )
    return lienzo


# --------------------------------------------------------------------------- #
# calibrar
# --------------------------------------------------------------------------- #


_GUION_CALIBRACION = """CALIBRACIÓN: QUE LA CÁMARA NO ESPEJE LA IMAGEN

Levanta tu mano DERECHA junto a tu hombro derecho, separada del cuerpo, con la
palma hacia la cámara, y déjala ahí.

El preview va espejado para que te veas como en un espejo, pero lo que se
comprueba es la imagen que recibe el detector, que tiene que llegar SIN espejar
(feature-spec.md §0.3). En ella tu lado derecho queda a la IZQUIERDA.

  arriba dice "entrada: SIN ESPEJAR"  ->  correcto. Pulsa S para confirmar.
  arriba dice "entrada: ESPEJADA"     ->  la cámara o su driver espejan la imagen
                                          por su cuenta. Desactiva el espejo en
                                          la configuración de la webcam y repite.
  no dice nada                         ->  no te detecta: acércate o mejora la luz.

Por qué importa: desde el ADR 0017 el espejo del contrato lo decide la mano que
declaras, no lo que diga MediaPipe. Con la entrada espejada, tu mano derecha se
vería como izquierda y todo el dataset saldría reflejado, sin ningún síntoma."""


def _cmd_calibrar(args: argparse.Namespace) -> int:
    """Comprueba que la entrada del detector no va espejada y lo deja escrito.

    La comprobación es geométrica —de qué lado de la imagen aparece la mano
    derecha levantada— y no depende de la etiqueta de MediaPipe. Una persona
    tiene que levantar la mano; lo que ya no tiene que hacer es leer y juzgar.
    """
    config = load_config(args.config)
    raiz: Path = args.raiz

    print(_GUION_CALIBRACION)
    print()

    detector = build_detector(config)
    try:
        with Camera.from_config(config.capture) as camera, detector:
            clave = _clave_de_camara(camera)
            confirmada = _bucle_calibracion(
                camera=camera, detector=detector, config=config, camara=clave
            )
    except ImportError as error:
        print(MENSAJE_SIN_EXTRAS.format(modulo=error.name))
        return 1
    except CameraError as error:
        print(f"error de cámara: {error}")
        return 1
    except FileNotFoundError as error:
        print(f"error: {error}")
        return 1

    if confirmada is None:
        print("calibración cancelada: no se escribió nada")
        return 1

    ancho, alto = confirmada
    ruta = save_calibration(
        raiz,
        Calibration(
            camera=clave,
            entrada_sin_espejar=True,
            fecha=now(),
            width=ancho,
            height=alto,
            confirmado_por=args.confirmado_por,
        ),
    )
    print(f"{ruta}: {clave} calibrada · la entrada del detector llega sin espejar")
    if not args.confirmado_por:
        print("  aviso: nadie firmó la confirmación. Usa --confirmado-por.")
    return 0


def _bucle_calibracion(
    *, camera: Camera, detector: HandDetector, config: Config, camara: str
) -> tuple[int, int] | None:
    """Mide de qué lado aparece la mano derecha hasta que alguien confirme.

    Solo deja confirmar tras `capture.calibration_frames` cuadros seguidos con la
    mano del lado que corresponde a una entrada sin espejar: un cuadro suelto no
    prueba nada. Devuelve la resolución confirmada, o `None` si se canceló.
    """
    import cv2

    ventana = "calibracion LSM"
    espejo = config.capture.preview_mirror
    necesarios = config.capture.calibration_frames
    racha_bien = 0
    racha_mal = 0

    while True:
        frame = camera.read()
        slot = detector.detect(frame.rgb)
        if isinstance(slot, RawFrame):
            if input_looks_unmirrored(slot, Handedness.RIGHT):
                racha_bien, racha_mal = racha_bien + 1, 0
            else:
                racha_bien, racha_mal = 0, racha_mal + 1
        else:
            racha_bien = racha_mal = 0

        if racha_bien >= necesarios:
            estado = "SIN ESPEJAR (S confirma)"
        elif racha_mal >= necesarios:
            estado = "ESPEJADA: no se puede confirmar"
        elif isinstance(slot, RawFrame):
            estado = "midiendo..."
        else:
            estado = "--"

        lienzo = cv2.flip(frame.bgr, 1) if espejo else frame.bgr.copy()
        if isinstance(slot, RawFrame):
            draw_landmarks(lienzo, slot, mirrored=espejo)
        _dibujar_calibracion(lienzo, estado, camara)
        cv2.imshow(ventana, lienzo)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in _SALIR:
            cv2.destroyWindow(ventana)
            return None
        if tecla in _CONFIRMAR and racha_bien >= necesarios:
            cv2.destroyWindow(ventana)
            return (frame.width, frame.height)


def _dibujar_calibracion(imagen: Any, estado: str, camara: str) -> None:
    """El HUD mínimo de la calibración: el veredicto, en grande."""
    import cv2

    alto, ancho = int(imagen.shape[0]), int(imagen.shape[1])
    cv2.rectangle(imagen, (0, 0), (ancho, 120), (20, 20, 20), -1)
    cv2.putText(
        imagen,
        f"entrada: {estado}",
        (24, 74),
        0,
        1.1,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        imagen,
        "mano DERECHA junto al hombro derecho | S confirma | Q cancela",
        (24, 106),
        0,
        0.6,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        imagen, camara, (24, alto - 20), 0, 0.5, (150, 150, 150), 1, cv2.LINE_AA
    )


# --------------------------------------------------------------------------- #
# consentimiento
# --------------------------------------------------------------------------- #


def _cmd_consentimiento(args: argparse.Namespace) -> int:
    """Registra qué autorizó una persona.

    El registro **no es** el consentimiento. El consentimiento es explícito y por
    escrito (`ARQUITECTURA.md` §4.11); esto anota que existe y dónde está, para
    que la captura pueda comprobarlo sin que nadie tenga que acordarse.
    """
    raiz: Path = args.raiz
    consent = Consent(
        signer_id=check_identifier("signer_id", args.firmante),
        video=args.video,
        fecha=now(),
        referencia=args.referencia,
    )
    ruta = save_consent(raiz, consent)
    permiso = "SÍ" if consent.video else "no"
    print(
        f"{ruta}: {consent.signer_id} · video: {permiso}"
        + (f" · referencia: {consent.referencia}" if consent.referencia else "")
    )
    if consent.video and not consent.referencia:
        print(
            "  aviso: se autorizó video sin anotar dónde está el consentimiento "
            "firmado. Usa --referencia."
        )
    return 0


# --------------------------------------------------------------------------- #
# verificar
# --------------------------------------------------------------------------- #


#: Tolerancia relativa con la que se compara la σ re-derivada contra la anotada.
#:
#: No es la `1e-6` del contrato de features: aquella es para la reimplementación
#: de TypeScript de la Fase 7 y cubre dos implementaciones distintas. Esta cubre
#: la **misma** implementación sobre dos libm distintas, que es un ruido siete
#: órdenes de magnitud menor. Ver `docs/adr/0009-verificacion-entre-plataformas.md`.
SIGMA_REL_TOL: Final = 1e-12


def _cmd_verificar(args: argparse.Namespace) -> int:
    """Relee el dataset y comprueba que las features se re-derivan idénticas.

    Es el criterio de aceptación de la Fase 1 y la razón de guardar landmarks
    crudos: si el archivo re-derivara features distintas a las de la sesión, la
    promesa de "cambiar la normalización cuesta un comando" sería falsa y nadie se
    enteraría hasta la Fase 2.

    **σ se compara con tolerancia relativa, y el motivo es la arquitectura.** La
    captura corre en el host —Windows o macOS, que es donde está la webcam— y la
    verificación en WSL o en el contenedor (`docker/README.md` §b). El paso 3 de
    `feature-spec.md` pasa por `atan2`, `sin` y `cos`, que IEEE 754 **no** obliga
    a redondear correctamente: la libm de MSVC y la de glibc discrepan en el
    último bit y σ hereda la discrepancia.

    Medido sobre las 613 muestras de s01: re-derivadas en Windows, donde se
    grabaron, coinciden las 613 exactamente; en Linux, 100 difieren entre 1 y 10
    ULPs, o sea 1.5e-15 relativo. Truncar los landmarks a seis decimales —una
    pérdida de precisión de las que este comando existe para cazar— mueve σ
    2.3e-5 relativo. `SIGMA_REL_TOL` va mil veces por encima del ruido y diez
    millones por debajo de esa regresión.

    La igualdad **exacta** sigue exigiéndose donde es legítima: en la suite, que
    escribe y relee dentro del mismo proceso y la misma máquina. Ver
    `tests/test_cli_capture.py` y `docs/adr/0009-verificacion-entre-plataformas.md`.

    Se comprueban dos cosas por muestra:

    1. Que la σ re-derivada coincida con la que se anotó al aceptarla. σ es un
       agregado de las 42 componentes sobre todos los frames: si un solo landmark
       hubiera perdido un bit al serializarse, el número cambia.
    2. Que el archivo no tenga huecos que `to_sample()` no acepte: ninguno en
       una estática, y en una dinámica solo los cortos que el Bloque 2 rellena
       (ADR 0021). Una muestra interrumpida no se puede entrenar y conviene
       saberlo ahora y no en mitad de la Fase 2.
    """
    config = load_config(args.config)
    raiz: Path = args.raiz

    revisadas = 0
    problemas: list[str] = []
    #: Muestras cuya σ anotada es de otra versión del contrato de features: la
    #: re-derivada no tiene por qué coincidir (desde FEATURE_SPEC 3 la escala del
    #: paso 4 cambió). Se revisa todo lo demás de ellas.
    otra_version = 0

    for ruta in iter_sample_paths(raiz):
        try:
            almacenada = read_sample(ruta)
            # Una dinámica con huecos cortos se reconstruye (Bloque 2).
            muestra = almacenada.to_sample(
                GapPolicy.from_config(config, config.capture.camera_fps)
            )
        except (DatasetError, ValueError, KeyError) as error:
            problemas.append(f"{ruta}: {error}")
            continue

        extraccion = extract_sequence_features(muestra.sequence, config)
        if isinstance(extraccion, ExtractionRejected):
            problemas.append(
                f"{ruta}: la secuencia guardada ya no se puede procesar "
                f"({extraccion.reason} en el frame {extraccion.frame_index})"
            )
            continue

        revisadas += 1
        if almacenada.metadata.feature_spec_version != FEATURE_SPEC_VERSION:
            otra_version += 1
            continue
        sigma = extraccion.static.dispersion
        anotada = almacenada.metadata.dispersion
        if not math.isclose(sigma, anotada, rel_tol=SIGMA_REL_TOL, abs_tol=0.0):
            desviacion = abs(sigma - anotada) / anotada if anotada else math.inf
            problemas.append(
                f"{ruta}: sigma re-derivada {sigma!r} != anotada {anotada!r} "
                f"(desviación relativa {desviacion:.1e}, sobre una tolerancia de "
                f"{SIGMA_REL_TOL:.0e})"
            )

    print(f"{revisadas} muestras releídas desde {raiz}")
    if otra_version:
        print(
            f"  {otra_version} anotaron su σ con otra versión del contrato de features "
            f"(o no la anotaron): se revisan huecos y extracción, no la σ. La "
            f"actual es la {FEATURE_SPEC_VERSION}."
        )
    if problemas:
        print(f"{len(problemas)} problemas:")
        for problema in problemas:
            print(f"  - {problema}")
        return 1
    if revisadas == 0:
        print("no hay muestras que verificar todavía")
    elif otra_version == revisadas:
        print("ninguna con σ comparable; todas se pueden procesar")
    elif otra_version == 0:
        print("todas re-derivan las mismas features que al grabarse")
    else:
        print(
            f"las {revisadas - otra_version} comparables re-derivan las mismas "
            "features que al grabarse"
        )
    return 0


def _cmd_marcar_truncadas(args: argparse.Namespace) -> int:
    """Marca las dinámicas grabadas antes del Bloque 4 que acaban en movimiento.

    Hasta el Bloque 4 la captura cortaba las dinámicas en un tope de cuadros, y
    muchas quedaron con el trazo en marcha: el modelo entrenaba con trazos que
    el clasificador nunca recibe en vivo. Aquí no se borra nada: se escribe el
    manifiesto `truncadas.json` en la raíz del dataset, y `load_corpus` deja
    fuera lo que liste. Borrar el manifiesto las devuelve.

    Criterio: la velocidad de cierre (§6.1.2) del último frame supera
    `motion_threshold_per_s`. Las dinámicas con `stroke_frames` —grabadas desde
    el Bloque 4— se cerraron por construcción y no se miran.
    """
    config = load_config(args.config)
    raiz: Path = args.raiz
    marcadas: dict[str, float] = {}
    revisadas = 0
    por_letra: dict[str, list[int]] = {}
    for ruta in iter_sample_paths(raiz):
        almacenada = read_sample(ruta)
        meta = almacenada.metadata
        if meta.kind is not SampleKind.DYNAMIC or meta.stroke_frames is not None:
            continue
        revisadas += 1
        fila = por_letra.setdefault(meta.label, [0, 0])
        fila[1] += 1
        if is_truncated(almacenada.frames, config):
            velocidad = final_velocity(almacenada.frames, config)
            marcadas[ruta.relative_to(raiz).as_posix()] = float(velocidad or 0.0)
            fila[0] += 1

    criterio = {
        "rule": "final closing velocity >= segmentation.motion_threshold_per_s",
        "motion_threshold_per_s": config.segmentation.motion_threshold_per_s,
        "closing_window_ms": config.segmentation.closing_window_ms,
        "fps": config.capture.camera_fps,
    }
    destino = write_truncated_manifest(raiz, marcadas, criterio)
    print(
        f"{len(marcadas)} de {revisadas} dinámicas sin delimitar acaban en movimiento"
    )
    for letra, (n, total) in sorted(por_letra.items()):
        print(f"  {letra:8s} {n:4d} de {total:4d}")
    print(f"manifiesto: {destino} (bórralo para volver a incluirlas)")
    return 0


def _cmd_rechazados(args: argparse.Namespace) -> int:
    """Vuelve a segmentar los intentos rechazados con la configuración de hoy.

    No escribe nada: dice, por letra y por el motivo con que se rechazó cada
    intento, cuántos saldrían ahora como un trazo aceptable (Paso 0). Cada
    intento se reprocesa con la tasa con la que se grabó, la que la máquina usó
    para delimitarlo.
    """
    config = load_config(args.config)
    raiz: Path = args.raiz
    #: (letra, motivo original) → [intentos, rescatados]
    tabla: dict[tuple[str, str], list[int]] = {}
    ahora: dict[str, int] = {}
    for ruta in iter_rejected_attempt_paths(raiz):
        intento = read_rejected_attempt(ruta)
        resultado = resegment_attempt(intento.frames, config, intento.fps)
        fila = tabla.setdefault((intento.label, intento.reason), [0, 0])
        fila[0] += 1
        if isinstance(resultado, Rejection):
            ahora[resultado.value] = ahora.get(resultado.value, 0) + 1
        else:
            fila[1] += 1
    if not tabla:
        print(f"no hay intentos rechazados en {raiz}")
        return 0
    total = sum(f[0] for f in tabla.values())
    rescatados = sum(f[1] for f in tabla.values())
    print(f"{total} intentos rechazados; {rescatados} dan hoy un trazo aceptable")
    print(f"  {'letra':8s} {'motivo original':24s} {'intentos':>8s} {'trazo hoy':>9s}")
    for (letra, motivo), (n, ok) in sorted(tabla.items()):
        print(f"  {letra:8s} {motivo:24s} {n:8d} {ok:9d}")
    if ahora:
        print("los que siguen rechazados, por el motivo de hoy:")
        for motivo, n in sorted(ahora.items(), key=lambda kv: -kv[1]):
            print(f"  {motivo:24s} {n:5d}")
    return 0


# --------------------------------------------------------------------------- #
# Argumentos
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Condiciones:
    """Las tres taxonomías anotadas a mano de `docs/adr/0005-...`.

    Se piden una vez por sesión y no por muestra, que es como se graban de verdad:
    la luz y la distancia se eligen al montar la sesión y no cambian a mitad. El
    ADR pide precisamente eso — variar las condiciones **entre** sesiones, no
    dentro de una.
    """

    light_level: LightLevel
    light_direction: LightDirection
    distance: Distance


def _condiciones(args: argparse.Namespace) -> Condiciones:
    return Condiciones(
        light_level=LightLevel(args.luz_nivel),
        light_direction=LightDirection(args.luz_direccion),
        distance=Distance(args.distancia),
    )


def _resolver_letras(pedidas: str | None) -> tuple[Label, ...]:
    """Qué letras se graban en esta sesión.

    Por defecto, el alfabeto entero **más la clase negativa**. `NONE` va incluida a
    propósito y no como extra opcional: sin ella el clasificador asigna una de las
    29 letras aunque la persona se esté rascando la nariz (`ARQUITECTURA.md`
    §4.4), y es la clase que todo el mundo olvida grabar.
    """
    if pedidas is None:
        return (*ALPHABET, Label.NONE)

    letras: list[Label] = []
    for nombre in pedidas.split(","):
        limpio = nombre.strip().upper()
        if not limpio:
            continue
        try:
            letras.append(Label(limpio))
        except ValueError:
            validas = ", ".join(label.value for label in Label)
            raise SystemExit(
                f"letra desconocida: {limpio!r}. Las válidas son: {validas}"
            ) from None
    if not letras:
        raise SystemExit("--letras no seleccionó ninguna letra")
    return tuple(letras)


def _resolver_video(raiz: Path, signer_id: str, *, pedido: bool) -> bool:
    """Dos llaves para guardar video, y ninguna se abre sola.

    La bandera está apagada por defecto y, aun encendida, no basta: hace falta un
    consentimiento registrado para esa persona concreta. Los landmarks no
    identifican a nadie, el video sí, y esa es toda la diferencia
    (`ARQUITECTURA.md` §4.11).
    """
    if not pedido:
        return False
    if not may_store_video(raiz, signer_id):
        raise SystemExit(
            f"--guardar-video pedido, pero {signer_id} no tiene consentimiento de "
            "video registrado. Regístralo primero:\n"
            f"  lsm-capture consentimiento --firmante {signer_id} --video "
            '--referencia "donde esté el consentimiento firmado"'
        )
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lsm-capture",
        description=(
            "Recolección del dataset de deletreo manual de LSM. Guarda landmarks "
            "crudos con sus metadatos; el video solo con consentimiento explícito."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="ruta del config.yaml",
    )
    parser.add_argument(
        "--raiz",
        type=Path,
        default=Path("data/raw"),
        help="raíz del dataset",
    )
    parser.add_argument(
        "--glosario",
        type=Path,
        default=DEFAULT_GLOSSARY,
        help="ruta del glosario, del que sale el registro de validación",
    )
    subcomandos = parser.add_subparsers(dest="comando", required=True)

    calibrar = subcomandos.add_parser(
        "calibrar",
        help="comprueba que el cuadro llega al detector sin espejar",
    )
    calibrar.add_argument(
        "--confirmado-por",
        default="",
        dest="confirmado_por",
        help="quién miró la pantalla y confirmó",
    )
    calibrar.set_defaults(func=_cmd_calibrar)

    grabar = subcomandos.add_parser("grabar", help="sesión de captura con cámara")
    grabar.add_argument("--firmante", required=True, help="signer_id de quien firma")
    grabar.add_argument("--sesion", required=True, help="session_id de la grabación")
    grabar.add_argument("--mano", required=True, choices=sorted(MANOS), help=AYUDA_MANO)
    grabar.add_argument(
        "--letras",
        default=None,
        help=(
            "letras a grabar, separadas por comas (por ejemplo A,B,ENIE). "
            "Por defecto, el alfabeto completo más la clase negativa NONE."
        ),
    )
    grabar.add_argument(
        "--luz-nivel",
        required=True,
        choices=[valor.value for valor in LightLevel],
        help="cuánta luz hay en la sesión",
    )
    grabar.add_argument(
        "--luz-direccion",
        required=True,
        choices=[valor.value for valor in LightDirection],
        help="de dónde viene la luz respecto de quien firma",
    )
    grabar.add_argument(
        "--distancia",
        required=True,
        choices=[valor.value for valor in Distance],
        help="distancia aproximada de la mano a la cámara",
    )
    grabar.add_argument(
        "--guardar-video",
        action="store_true",
        help=(
            "guarda también el video de cada muestra. APAGADO POR DEFECTO y exige "
            "consentimiento registrado para esa persona."
        ),
    )
    grabar.add_argument(
        "--sesion-prueba",
        action="store_true",
        dest="sesion_prueba",
        help=(
            "sesión de prueba: escribe en data/raw/pruebas/, que no entra al "
            "dataset, y no exige el glosario validado. Para encuadrar la cámara y "
            "ensayar antes de citar a nadie."
        ),
    )
    grabar.set_defaults(func=_cmd_grabar)

    consentimiento = subcomandos.add_parser(
        "consentimiento", help="registra qué autorizó una persona"
    )
    consentimiento.add_argument("--firmante", required=True, help="signer_id")
    consentimiento.add_argument(
        "--video",
        action="store_true",
        help="autoriza almacenar video, no solo landmarks",
    )
    consentimiento.add_argument(
        "--referencia",
        default="",
        help="dónde está el consentimiento firmado: folio, expediente, archivo",
    )
    consentimiento.set_defaults(func=_cmd_consentimiento)

    verificar = subcomandos.add_parser(
        "verificar",
        help="relee el dataset y comprueba que las features se re-derivan igual",
    )
    verificar.set_defaults(func=_cmd_verificar)

    truncadas = subcomandos.add_parser(
        "marcar-truncadas",
        help=(
            "marca las dinámicas grabadas antes del Bloque 4 que acaban con la "
            "mano en movimiento, para excluirlas sin borrarlas"
        ),
    )
    truncadas.set_defaults(func=_cmd_marcar_truncadas)

    rechazados = subcomandos.add_parser(
        "rechazados",
        help=(
            "vuelve a segmentar los intentos dinámicos rechazados con la "
            "configuración actual y dice cuántos darían hoy un trazo aceptable"
        ),
    )
    rechazados.set_defaults(func=_cmd_rechazados)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        resultado: int = args.func(args)
    except DatasetError as error:
        print(f"error de dataset: {error}")
        return 1
    return resultado


if __name__ == "__main__":
    raise SystemExit(main())
