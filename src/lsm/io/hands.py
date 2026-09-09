"""Frontera con el detector de manos.

`CLAUDE.md` §3: **MediaPipe solo puede importarse en este módulo.** El resto del
proyecto consume la interfaz definida aquí y no sabe qué hay detrás. Eso es lo que
permite correr entrenamiento, evaluación y la suite completa de tests en una
máquina sin webcam y sin MediaPipe instalado.

Hay dos implementaciones del `Protocol`: `MediaPipeHandDetector`, el detector real
y el único sitio del repositorio con `import mediapipe`, y `FakeHandDetector`, que
reproduce secuencias grabadas y es lo que permite ejercitar la tubería entera —
captura incluida — en CI y en Docker.

**El import de MediaPipe es diferido a propósito.** Ocurre dentro de
`MediaPipeHandDetector.open()`, no en la cabecera del módulo, para que importar
`lsm.io.hands` siga sin arrastrar MediaPipe ni sus dependencias nativas. Sin eso,
la suite dejaría de correr en una máquina que solo quiere entrenar o evaluar, y
`tests/test_io_hands.py` lo verifica explícitamente.

**Sobre el espejado.** El detector real tiene que recibir el frame **sin espejar**
(`feature-spec.md` §0.3). El espejado del preview se hace por comodidad de quien
firma y vive en la capa de visualización (`lsm.capture.preview_position`). Si se le
pasara la imagen espejada, la lateralidad reportada se invertiría y el paso 2 de la
especificación corrompería el vector sin que nada falle.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol, TypeAlias, runtime_checkable

from lsm.config import Config
from lsm.types import (
    NUM_LANDMARKS,
    FrameSlot,
    FrameStream,
    Handedness,
    InvalidFrame,
    InvalidReason,
    Landmark,
    RawFrame,
)

#: Versión del formato de los fixtures de secuencias en disco.
FIXTURE_SCHEMA_VERSION: Final = 1

#: Un frame de video crudo, tal como lo entrega `io/camera.py`. Se tipa laxo a
#: propósito: el tipo real es un arreglo de OpenCV, y nombrarlo aquí metería
#: `numpy` y `cv2` en la interfaz que consume el núcleo puro.
#:
#: **Formato esperado por `MediaPipeHandDetector`: RGB, `uint8`, `(alto, ancho, 3)`
#: y contiguo en memoria.** OpenCV entrega BGR, así que la conversión la hace
#: `io/camera.py`, que es quien ya depende de `cv2`. Pasar BGR aquí no falla: el
#: detector encuentra menos manos y nadie sabe por qué.
VideoImage: TypeAlias = Any

#: Nombres de categoría con que MediaPipe reporta la lateralidad.
_MEDIAPIPE_HANDEDNESS: Final = {"Left": Handedness.LEFT, "Right": Handedness.RIGHT}


@runtime_checkable
class HandDetector(Protocol):
    """Convierte imágenes en frames de landmarks.

    Devuelve siempre un `FrameSlot`: o una mano (`RawFrame`) o el marcador
    explícito de por qué no la hay. Nunca `None` y nunca una excepción por el caso
    normal de que en el encuadre no haya nadie.

    Si el detector encuentra varias manos se queda con la de mayor
    `detection_score`: el alfabeto dactilológico de LSM es monomanual.

    ### El ciclo de vida es parte de la interfaz, no un detalle de MediaPipe

    `open()` y el `with` están declarados aquí y no solo en el detector real
    porque **quien consume un detector tiene que abrirlo y cerrarlo**, y si eso no
    viaja en el `Protocol`, el `Protocol` describe menos de lo que sus
    consumidores necesitan: no protege la frontera, la desplaza. Se ve enseguida
    en la práctica —los dos CLI que abren cámara acabarían anotando el tipo
    concreto para poder escribir `with detector:`, y la fábrica dejaría de
    describir lo que produce.

    La consecuencia que importa es que `FakeHandDetector` sea sustituible **en
    todas partes** donde va el real, los CLI incluidos, y no solo en el núcleo
    puro. Que abrir no cueste nada en el fake no es motivo para que no sepa
    hacerlo: la interfaz la fija quien la usa, no la implementación más cara.

    Que el detector real cargue 8 MB de modelo en `open()` y el fake no cargue
    nada sí es un detalle de implementación. Lo que la interfaz promete es más
    modesto y es lo único de lo que dependen los consumidores: hay un momento en
    que el detector queda listo, y otro en que suelta lo que tuviera.
    """

    def detect(self, image: VideoImage) -> FrameSlot:
        """Procesa una imagen y devuelve la mano detectada, o el motivo."""
        ...

    def open(self) -> HandDetector:
        """Deja el detector listo para detectar. Devuelve `self` para encadenar.

        Es un paso aparte del constructor porque construir no debe costar lo que
        cuesta abrir ni fallar por lo que falla abrir.
        """
        ...

    def close(self) -> None:
        """Libera los recursos del detector. Idempotente."""
        ...

    def __enter__(self) -> HandDetector:
        """`with detector:` abre y garantiza el cierre."""
        ...

    def __exit__(self, *exc: object) -> None: ...


@dataclass
class MediaPipeHandDetector:
    """El detector real. **Único sitio del repositorio que importa MediaPipe.**

    Envuelve `HandLandmarker` de MediaPipe Tasks en modo VIDEO y traduce su salida
    a los tipos del proyecto. Todo lo que MediaPipe tiene de particular queda
    encerrado aquí: el resto del código ve `RawFrame` e `InvalidFrame`.

    ### Por qué la API de Tasks y no `mediapipe.solutions.hands`

    Porque la segunda ya no existe. Desde MediaPipe 1.0 el paquete `solutions`
    quedó fuera de la distribución, así que la elección estaba tomada de antemano.
    La consecuencia práctica es que hace falta un **bundle de modelo en disco**
    (`hand_landmarker.task`, ~8 MB): no se versiona en el repositorio y se
    descarga con `make model`. Ver `docs/adr/0006-deteccion-de-manos-y-captura.md`.

    ### Modo VIDEO, no IMAGE

    En modo IMAGE cada cuadro se detecta desde cero y los landmarks bailan entre
    frames aunque la mano esté quieta. Ese jitter va directo a la σ del
    `feature-spec.md` §2, que es el criterio con el que se acepta o se rechaza una
    muestra: con IMAGE, la captura rechazaría manos perfectamente sostenidas. El
    modo VIDEO mantiene el rastreo entre cuadros a cambio de exigir timestamps
    monótonos, que este detector genera con un contador propio (`_elapsed_ms`).

    Se usa un contador y no el reloj porque el reloj no es reproducible: pasar la
    misma grabación dos veces por el detector daría timestamps distintos y, con
    ellos, resultados distintos. El contador hace que dos ejecuciones sobre los
    mismos cuadros sean la misma ejecución.

    ### La lateralidad, y por qué el interruptor no se decide leyendo documentación

    El paso 2 de `feature-spec.md` espeja en X según la lateralidad, así que si la
    etiqueta viene al revés **todas** las muestras se canonizan hacia la mano
    equivocada. Y no falla nada: el vector sale espejado pero coherente consigo
    mismo, y el modelo entrena tan campante hasta que alguien firma con la otra
    mano — o hasta la Fase 7, con MediaPipe JS usando la otra convención.

    Conviene no confundir dos cosas que suenan parecidas:

    - Espejar la **imagen** que entra al detector sería un error: invierte la
      geometría de los landmarks y corrompe el vector. Por eso `feature-spec.md`
      §0.3 obliga a alimentarlo sin espejar.
    - Qué **etiqueta** devuelve el detector ante esa imagen es otra pregunta, y no
      se responde razonando: se mide.

    **Medido el 2026-09-08 con MediaPipe 1.0.1** (Tasks API, `HandLandmarker`,
    modo VIDEO, webcam por el backend `MSMF`): alimentado con el cuadro **sin
    espejar**, MediaPipe devuelve la mano **anatómica** — mano derecha real,
    `category_name == "Right"`. No hace falta invertir nada, y por eso
    `swap_handedness` viene en `False`.

    La documentación histórica de MediaPipe decía lo contrario ("handedness is
    determined assuming the input image is mirrored"), y ese fue el valor por
    defecto hasta que se comprobó contra una cámara real. Se deja el interruptor
    —en vez de borrar la rama— justamente porque la convención ya cambió una vez
    entre versiones de la librería y puede volver a cambiar.

    La verificación vive en `lsm-capture calibrar` y es obligatoria antes de
    grabar: levantar la mano derecha, mirar que el preview diga `RIGHT`, confirmar.
    Queda registrada en `data/raw/calibracion.json` junto con el valor del
    interruptor, de modo que cambiarlo invalida la calibración anterior. El
    interruptor está en `config.hands.mediapipe_reports_mirrored_handedness`.
    """

    model_path: Path
    #: Frames por debajo de este `detection_score` se marcan inválidos. Se
    #: alimenta de `config.segmentation.min_detection_score`.
    min_detection_score: float
    num_hands: int = 2
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    #: Milisegundos que se le suman al timestamp por cuadro. Sale de los fps de
    #: la cámara; solo importa que crezca de forma monótona y estable.
    frame_interval_ms: int = 33
    #: Ver el bloque sobre lateralidad de la docstring de la clase. Medido, no
    #: deducido: con MediaPipe 1.0.1 no hay que invertir nada.
    swap_handedness: bool = False

    _landmarker: Any = field(default=None, init=False, repr=False)
    _elapsed_ms: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.frame_interval_ms < 1:
            msg = (
                f"frame_interval_ms debe ser ≥ 1, no {self.frame_interval_ms}: "
                "el modo VIDEO de MediaPipe exige timestamps estrictamente "
                "crecientes"
            )
            raise ValueError(msg)
        if not 0.0 <= self.min_detection_score <= 1.0:
            raise ValueError(
                f"min_detection_score fuera de [0, 1]: {self.min_detection_score}"
            )

    def open(self) -> MediaPipeHandDetector:
        """Carga el modelo y deja el detector listo.

        Es un paso aparte del constructor porque aquí ocurre el `import mediapipe`
        y la lectura de un archivo de 8 MB: construir el objeto no debe costar eso
        ni fallar por eso. Devuelve `self` para poder encadenar.
        """
        if self._landmarker is not None:
            return self

        if not self.model_path.is_file():
            msg = (
                f"no está el modelo de MediaPipe en {self.model_path}. "
                "Descárgalo con `make model` (son ~8 MB) o ajusta "
                "`hands.model_path` en config.yaml."
            )
            raise FileNotFoundError(msg)

        # Import diferido: `lsm.io.hands` tiene que poder importarse en una
        # máquina sin MediaPipe. Ver la cabecera del módulo.
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_hand_detection_confidence,
            min_hand_presence_confidence=self.min_hand_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        return self

    def detect(self, image: VideoImage) -> FrameSlot:
        """Procesa un cuadro **RGB y sin espejar**. Ver `VideoImage`.

        Devuelve siempre un `FrameSlot`: que no haya nadie en el encuadre es el
        caso normal, no un error, y por eso no lanza ni devuelve `None`.
        """
        if self._landmarker is None:
            raise RuntimeError("el detector no está abierto: llama a open() primero")

        import mediapipe as mp

        height, width = int(image.shape[0]), int(image.shape[1])
        self._elapsed_ms += self.frame_interval_ms
        result = self._landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=image), self._elapsed_ms
        )
        return self._to_slot(result, width=width, height=height)

    def _to_slot(self, result: Any, *, width: int, height: int) -> FrameSlot:
        """Traduce un `HandLandmarkerResult` al tipo del proyecto.

        Si hay varias manos se queda con la de mayor score, como manda
        `feature-spec.md` §0.3: el alfabeto dactilológico de LSM es monomanual.
        Por eso `num_hands` vale 2 y no 1 — para poder elegir hay que ver más de
        una.
        """
        hands = getattr(result, "hand_landmarks", None) or ()
        handedness = getattr(result, "handedness", None) or ()
        if not hands or not handedness:
            return InvalidFrame(
                reason=InvalidReason.NO_HAND, detail="el detector no encontró mano"
            )

        best_index = -1
        best_score = -1.0
        for index, categories in enumerate(handedness):
            if not categories:
                continue
            score = float(categories[0].score)
            if score > best_score:
                best_index, best_score = index, score

        if best_index < 0:
            return InvalidFrame(
                reason=InvalidReason.NO_HAND,
                detail="el detector devolvió una mano sin lateralidad",
            )

        if best_score < self.min_detection_score:
            return InvalidFrame(
                reason=InvalidReason.LOW_DETECTION_SCORE,
                detail=f"score {best_score:.3f} < {self.min_detection_score:.3f}",
            )

        category = handedness[best_index][0]
        reported = _MEDIAPIPE_HANDEDNESS.get(str(category.category_name))
        if reported is None:
            return InvalidFrame(
                reason=InvalidReason.NO_HAND,
                detail=f"lateralidad desconocida: {category.category_name!r}",
            )
        side = _flip(reported) if self.swap_handedness else reported

        points = hands[best_index]
        if len(points) != NUM_LANDMARKS:
            return InvalidFrame(
                reason=InvalidReason.NO_HAND,
                detail=f"el detector devolvió {len(points)} landmarks",
            )

        return RawFrame(
            landmarks=tuple(
                Landmark(x=float(p.x), y=float(p.y), z=float(p.z)) for p in points
            ),
            width=width,
            height=height,
            handedness=side,
            # MediaPipe Tasks expone **un solo** número de confianza por mano: el
            # de la clasificación de lateralidad. La confianza de detección de la
            # palma se consume dentro del grafo, vía
            # `min_hand_detection_confidence`, y no vuelve a salir. Los dos campos
            # del `feature-spec.md` §0.2 se rellenan por tanto con el mismo valor,
            # y se mantienen separados porque son conceptos distintos y otro
            # detector —o MediaPipe JS en la Fase 7— puede darlos por separado.
            handedness_score=best_score,
            detection_score=best_score,
        )

    def close(self) -> None:
        """Libera el modelo. Idempotente: cerrar dos veces no es un error."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        self._elapsed_ms = 0

    def __enter__(self) -> MediaPipeHandDetector:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()


def build_detector(config: Config) -> HandDetector:
    """El detector real, con todo lo que `config.yaml` dice sobre él.

    Vive aquí y no en un CLI porque los CLI que abren cámara son ya dos —captura
    y demo— y el segundo tendría que importar un privado del primero. Construir
    el detector es traducir configuración a la frontera con MediaPipe, y esa
    frontera es este módulo (`CLAUDE.md` §3).

    Devuelve el `Protocol` y no el tipo concreto: es todo lo que sus consumidores
    necesitan —detectar, abrir, cerrar— y anotarlo así es lo que deja la puerta
    abierta a pasarles un `FakeHandDetector` sin tocar sus firmas.
    """
    return MediaPipeHandDetector(
        model_path=config.hands.model_path,
        min_detection_score=config.segmentation.min_detection_score,
        num_hands=config.hands.num_hands,
        min_hand_detection_confidence=config.hands.min_hand_detection_confidence,
        min_hand_presence_confidence=config.hands.min_hand_presence_confidence,
        min_tracking_confidence=config.hands.min_tracking_confidence,
        frame_interval_ms=max(1, round(1000 / config.capture.camera_fps)),
        swap_handedness=config.hands.mediapipe_reports_mirrored_handedness,
    )


def _flip(side: Handedness) -> Handedness:
    return Handedness.LEFT if side is Handedness.RIGHT else Handedness.RIGHT


@dataclass
class FakeHandDetector:
    """Detector de mentira: reproduce una secuencia grabada, ignora la imagen.

    Es lo que permite ejercitar la tubería completa —segmentación incluida— en CI
    y en Docker, sin cámara y sin MediaPipe. Cuando la grabación se agota devuelve
    frames inválidos, que es exactamente lo que devolvería una cámara apuntando a
    una habitación vacía.
    """

    slots: FrameStream
    _position: int = field(default=0, init=False, repr=False)

    @classmethod
    def from_fixture(cls, path: Path | str) -> FakeHandDetector:
        return cls(slots=load_frame_stream(path))

    def detect(self, image: VideoImage = None) -> FrameSlot:  # noqa: ARG002 — grabado
        if self._position >= len(self.slots):
            return InvalidFrame(
                reason=InvalidReason.NO_HAND, detail="la grabación se agotó"
            )
        slot = self.slots[self._position]
        self._position += 1
        return slot

    def stream(self) -> Iterator[FrameSlot]:
        """Recorre la grabación entera desde el principio, sin consumir `detect`."""
        yield from self.slots

    def open(self) -> FakeHandDetector:
        """No hay nada que abrir: la grabación ya está en memoria.

        Existe porque el ciclo de vida es parte del `Protocol`, y sin él el fake
        no sería sustituible donde va el real. Que abrir no cueste nada aquí es
        justo lo que se quiere de un doble.
        """
        return self

    def close(self) -> None:
        self._position = 0

    def __enter__(self) -> FakeHandDetector:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Fixtures en disco
# --------------------------------------------------------------------------- #


def dump_frame_stream(stream: FrameStream, path: Path | str) -> None:
    """Escribe un flujo de frames como JSON.

    Se guardan los **landmarks crudos**, no las features: si cambia la
    normalización, se re-deriva todo sin volver a grabar (`ARQUITECTURA.md` §4.7).

    Los huecos se escriben como huecos. Si se perdieran al guardar, una secuencia
    interrumpida se convertiría en una continua y el dataset mentiría sobre lo que
    ocurrió frente a la cámara.
    """
    payload = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "frames": frames_to_json(stream),
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_frame_stream(path: Path | str) -> FrameStream:
    """Lee un flujo de frames grabado. Rechaza formatos de otra versión."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = payload.get("schema_version")
    if version != FIXTURE_SCHEMA_VERSION:
        msg = (
            f"{path}: schema_version {version} incompatible; "
            f"este código lee la versión {FIXTURE_SCHEMA_VERSION}"
        )
        raise ValueError(msg)
    return frames_from_json(payload["frames"])


def frames_to_json(stream: FrameStream) -> list[dict[str, Any]]:
    """Serializa un flujo de frames, huecos incluidos.

    Es público porque tiene un segundo consumidor: `io/dataset.py` incrusta este
    mismo arreglo dentro del archivo de una muestra. Que las muestras del dataset
    y los fixtures de los tests compartan la representación de un frame no es
    comodidad: significa que una muestra grabada de verdad se puede pegar tal cual
    como fixture, y que un fixture se puede leer con el mismo código que lee el
    dataset.
    """
    return [_slot_to_json(slot) for slot in stream]


def frames_from_json(entries: Any) -> FrameStream:
    """Inverso de `frames_to_json`. Los huecos vuelven como huecos."""
    return tuple(_slot_from_json(entry) for entry in entries)


def _slot_to_json(slot: FrameSlot) -> dict[str, Any]:
    if isinstance(slot, InvalidFrame):
        return {"valid": False, "reason": str(slot.reason), "detail": slot.detail}
    return {
        "valid": True,
        "width": slot.width,
        "height": slot.height,
        "handedness": str(slot.handedness),
        "handedness_score": slot.handedness_score,
        "detection_score": slot.detection_score,
        "landmarks": [[point.x, point.y, point.z] for point in slot.landmarks],
    }


def _slot_from_json(entry: dict[str, Any]) -> FrameSlot:
    if not entry["valid"]:
        return InvalidFrame(
            reason=InvalidReason(entry["reason"]), detail=entry.get("detail", "")
        )
    landmarks = entry["landmarks"]
    if len(landmarks) != NUM_LANDMARKS:
        msg = f"un frame lleva {NUM_LANDMARKS} landmarks, no {len(landmarks)}"
        raise ValueError(msg)
    return RawFrame(
        landmarks=tuple(Landmark(x=x, y=y, z=z) for x, y, z in landmarks),
        width=entry["width"],
        height=entry["height"],
        handedness=Handedness(entry["handedness"]),
        handedness_score=entry["handedness_score"],
        detection_score=entry["detection_score"],
    )
