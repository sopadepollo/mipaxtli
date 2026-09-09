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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from lsm.capture import (
    BufferedFrame,
    FrameBuffer,
    Rejection,
    WindowQuality,
    evaluate_window,
    explain,
    minimum_frames,
)
from lsm.config import Config, load_config
from lsm.features import ExtractionRejected, extract_sequence_features
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
    Consent,
    DatasetError,
    SampleMetadata,
    StoredSample,
    check_identifier,
    count_samples,
    iter_sample_paths,
    label_dir,
    may_store_video,
    next_sample_index,
    now,
    read_sample,
    save_consent,
    trial_root,
    write_sample,
)
from lsm.io.glossary import DEFAULT_GLOSSARY, is_validated
from lsm.io.hands import HandDetector, build_detector
from lsm.io.preview import HudState, draw_hud, draw_landmarks
from lsm.types import (
    HANDEDNESS_CONVENTION,
    Distance,
    FrameSlot,
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


#: Qué decir cuando faltan las dependencias opcionales. Ocurre siempre la primera
#: vez, porque `make setup` no las instala a propósito: la suite, el entrenamiento
#: y la evaluación corren sin cámara, y arrastrar MediaPipe a todos esos entornos
#: por un comando que solo se usa al grabar sería un mal negocio.
_MENSAJE_SIN_EXTRAS = (
    "faltan las dependencias de captura ({modulo}). Se instalan aparte porque el "
    "resto del proyecto no las necesita:\n"
    "  make setup-capture\n"
    "  make model"
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
    detector = build_detector(config)

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
            calibracion = require_calibration(
                raiz,
                _clave_de_camara(camera),
                swap_handedness=config.hands.mediapipe_reports_mirrored_handedness,
            )
            modo = "PRUEBA (no entra al dataset)" if es_prueba else "formal"
            video = "SÍ (con consentimiento registrado)" if guarda_video else "no"
            print(
                f"Sesión {session_id} de {signer_id} · {len(letras)} letras · "
                f"meta {config.capture.target_samples_per_label} por letra\n"
                f"Modo: {modo} · destino {destino}\n"
                f"Video: {video}\n"
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
                )
    except ImportError as error:
        print(_MENSAJE_SIN_EXTRAS.format(modulo=error.name))
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
        config.capture.static_frames, config.capture.dynamic_max_frames
    )
    imagenes: list[Any] = []
    ventana = "captura LSM"

    while True:
        frame = camera.read()
        slot = detector.detect(frame.rgb)
        buffer.push(slot, frame.mean_luminance)

        if guarda_video:
            imagenes.append(frame.bgr.copy())
            del imagenes[: max(0, len(imagenes) - cuadros_maximos)]

        # Se deja de acumular en el tope en vez de seguir creciendo: una grabación
        # olvidada llegaría a `TOO_MANY_FRAMES` y ya no se podría guardar, así que
        # congelarla deja la muestra utilizable en vez de obligar a repetirla. El
        # mensaje avisa de que lo que venga después ya no entra.
        tope_dinamico = config.capture.dynamic_max_frames
        if sesion.grabando is not None and len(sesion.grabando) < tope_dinamico:
            sesion.grabando.append(
                BufferedFrame(slot=slot, luminance=frame.mean_luminance)
            )
            if len(sesion.grabando) >= tope_dinamico:
                sesion.mensaje = "tope de frames alcanzado: ESPACIO para guardar"

        quality = _calidad(buffer, sesion, config)
        lienzo = _dibujar(frame.bgr, slot, quality, sesion, config, guarda_video)
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
    if sesion.grabando is not None:
        return evaluate_window(tuple(sesion.grabando), SampleKind.DYNAMIC, config)
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
    """ESPACIO: en estática guarda; en dinámica arranca o cierra la grabación.

    La asimetría es la del gesto que se está grabando. Una estática ya está ahí:
    la mano lleva un segundo quieta y lo que se guarda es lo que acaba de pasar.
    Una dinámica hay que delimitarla, porque su principio y su final son parte de
    la seña y solo quien firma sabe dónde están.
    """
    if sesion.kind is SampleKind.DYNAMIC and sesion.grabando is None:
        sesion.grabando = []
        sesion.mensaje = "grabando: ESPACIO para cerrar, r para descartar"
        return

    if sesion.grabando is not None:
        frames = tuple(sesion.grabando)
    else:
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
    quality = evaluate_window(frames, kind, config)
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
        ),
    )
    return lienzo


# --------------------------------------------------------------------------- #
# calibrar
# --------------------------------------------------------------------------- #


_GUION_CALIBRACION = """CALIBRACIÓN DE LATERALIDAD

Levanta tu mano DERECHA delante de la cámara, con la palma hacia ella.

Mira lo que dice el preview arriba a la derecha:

  dice "mano: RIGHT"  ->  correcto. Pulsa S para confirmar.
  dice "mano: LEFT"   ->  el interruptor está al revés. Pulsa Q, cambia
                          hands.mediapipe_reports_mirrored_handedness a {contrario}
                          en config.yaml, y vuelve a ejecutar este comando.
  no dice nada        ->  no te está detectando: acércate o mejora la luz.

Por qué este minuto importa: si la lateralidad está invertida, TODAS las muestras
se canonizan hacia la mano contraria. El modelo entrenará bien, inferirá bien y la
precisión será idéntica — el error solo aparece en la Fase 7, cuando MediaPipe JS
use la convención contraria y la app web confunda cada seña con su espejo. No hay
ninguna prueba automática que lo detecte, por eso hace falta un ojo humano."""


def _cmd_calibrar(args: argparse.Namespace) -> int:
    """Confirma a ojo la lateralidad y deja constancia.

    Es el único punto del proyecto donde una persona aporta información que
    ninguna prueba puede producir. Todo lo demás se comprueba solo; esto no,
    porque el sistema no tiene forma de saber qué mano levantó quien está
    delante.
    """
    config = load_config(args.config)
    raiz: Path = args.raiz
    swap = config.hands.mediapipe_reports_mirrored_handedness

    print(_GUION_CALIBRACION.format(contrario=str(not swap).lower()))
    print()

    detector = build_detector(config)
    try:
        with Camera.from_config(config.capture) as camera, detector:
            clave = _clave_de_camara(camera)
            confirmada = _bucle_calibracion(
                camera=camera, detector=detector, config=config, camara=clave
            )
    except ImportError as error:
        print(_MENSAJE_SIN_EXTRAS.format(modulo=error.name))
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
            swap_handedness=swap,
            convention=HANDEDNESS_CONVENTION,
            fecha=now(),
            width=ancho,
            height=alto,
            confirmado_por=args.confirmado_por,
        ),
    )
    print(
        f"{ruta}: {clave} calibrada\n"
        f"  convención: {HANDEDNESS_CONVENTION} · "
        f"mediapipe_reports_mirrored_handedness = {str(swap).lower()}"
    )
    if not args.confirmado_por:
        print("  aviso: nadie firmó la confirmación. Usa --confirmado-por.")
    return 0


def _bucle_calibracion(
    *, camera: Camera, detector: HandDetector, config: Config, camara: str
) -> tuple[int, int] | None:
    """Muestra la lateralidad resuelta hasta que alguien confirme o cancele.

    Devuelve la resolución confirmada, o `None` si se canceló. **No hay
    confirmación por omisión**: cerrar la ventana o pulsar `q` no calibra nada,
    porque el valor de este registro es exactamente que alguien miró.
    """
    import cv2

    ventana = "calibracion LSM"
    espejo = config.capture.preview_mirror

    while True:
        frame = camera.read()
        slot = detector.detect(frame.rgb)

        lienzo = cv2.flip(frame.bgr, 1) if espejo else frame.bgr.copy()
        if isinstance(slot, RawFrame):
            draw_landmarks(lienzo, slot, mirrored=espejo)
        _dibujar_calibracion(lienzo, slot, camara)
        cv2.imshow(ventana, lienzo)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in _SALIR:
            cv2.destroyWindow(ventana)
            return None
        if tecla in _CONFIRMAR and isinstance(slot, RawFrame):
            cv2.destroyWindow(ventana)
            return (frame.width, frame.height)


def _dibujar_calibracion(imagen: Any, slot: FrameSlot, camara: str) -> None:
    """El HUD mínimo de la calibración: la mano detectada, en grande."""
    import cv2

    alto, ancho = int(imagen.shape[0]), int(imagen.shape[1])
    lateralidad = slot.handedness.value if isinstance(slot, RawFrame) else "--"

    cv2.rectangle(imagen, (0, 0), (ancho, 120), (20, 20, 20), -1)
    cv2.putText(
        imagen,
        f"mano: {lateralidad}",
        (24, 74),
        0,
        1.8,
        (245, 245, 245),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        imagen,
        "levanta la mano DERECHA | S confirma | Q cancela",
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
    2. Que el archivo no tenga huecos, que es lo que `to_sample()` exige. Una
       muestra interrumpida no se puede entrenar y conviene saberlo ahora y no en
       mitad de la Fase 2.
    """
    config = load_config(args.config)
    raiz: Path = args.raiz

    revisadas = 0
    problemas: list[str] = []

    for ruta in iter_sample_paths(raiz):
        try:
            almacenada = read_sample(ruta)
            muestra = almacenada.to_sample()
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
    if problemas:
        print(f"{len(problemas)} problemas:")
        for problema in problemas:
            print(f"  - {problema}")
        return 1
    if revisadas == 0:
        print("no hay muestras que verificar todavía")
    else:
        print("todas re-derivan las mismas features que al grabarse")
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
        help="confirma a ojo que la lateralidad detectada es la mano real",
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
