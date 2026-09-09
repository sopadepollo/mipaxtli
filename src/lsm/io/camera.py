"""Frontera con la cámara. Único sitio del proyecto que abre hardware de video.

`ARQUITECTURA.md` §4.9: todo lo demás —`features`, `segmentation`,
`classifiers`— son funciones puras sobre arreglos. Que la cámara viva detrás de
esta única puerta es lo que permite correr entrenamiento, evaluación y la suite
completa dentro de Docker o en CI, donde no hay `/dev/video0`, y ejecutar la
captura fuera del contenedor en Windows y macOS sin tocar nada más.

Como `io/hands.py`, importa `cv2` de forma diferida: `import lsm.io.camera` no
debe arrastrar OpenCV a una máquina que solo quiere entrenar.

**Sobre el espacio de color.** OpenCV entrega BGR; MediaPipe quiere RGB. La
conversión se hace aquí, que es donde ya vive la dependencia de `cv2`, y cada
cuadro viaja con las dos vistas: `bgr` para dibujar y mostrar, `rgb` para el
detector. Dejar la conversión al consumidor es la receta conocida para acabar
alimentando BGR al detector, que no falla — solo encuentra menos manos, y nadie
sabe por qué.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lsm.config import CaptureConfig


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """Un cuadro de video, en las dos formas en que hace falta.

    `mean_luminance` se calcula aquí y no después porque después la imagen ya no
    existe: el buffer de captura guarda landmarks, no cuadros. Es la medida
    objetiva que acompaña a `light_level` en los metadatos de la muestra
    (`docs/adr/0005-taxonomias-de-metadatos-de-captura.md`).
    """

    #: Arreglo BGR de OpenCV, tal como salió de la cámara. Para dibujar y mostrar.
    bgr: Any
    #: El mismo cuadro en RGB contiguo, que es lo que espera `HandDetector`.
    rgb: Any
    width: int
    height: int
    #: Luminancia media del cuadro, en `[0, 1]`.
    mean_luminance: float


class CameraError(RuntimeError):
    """La cámara no se pudo abrir o dejó de entregar cuadros."""


@dataclass
class Camera:
    """La webcam, envuelta.

    Se pide una resolución concreta, pero es una **petición**: el driver puede
    entregar otra y no avisa. Por eso `CameraFrame` lleva su propio `width` y
    `height` leídos del arreglo, y por eso cada frame guardado en el dataset lleva
    los suyos: el paso 1 de `feature-spec.md` divide por la relación de aspecto, y
    asumir 1280x720 cuando la cámara entregó 640x480 deforma la mano en silencio.
    """

    index: int
    width: int
    height: int
    fps: int

    _capture: Any = field(default=None, init=False, repr=False)

    @classmethod
    def from_config(cls, config: CaptureConfig) -> Camera:
        return cls(
            index=config.camera_index,
            width=config.frame_width,
            height=config.frame_height,
            fps=config.camera_fps,
        )

    def open(self) -> Camera:
        """Abre el dispositivo. Devuelve `self` para poder encadenar."""
        if self._capture is not None:
            return self

        import cv2

        capture = cv2.VideoCapture(self.index)
        if not capture.isOpened():
            capture.release()
            msg = (
                f"no se pudo abrir la cámara {self.index}. En Linux, comprobar "
                "que exista /dev/video*; en Docker, que se haya pasado el "
                "dispositivo (ver docker/README.md); en Windows y macOS, que la "
                "captura se ejecute FUERA del contenedor."
            )
            raise CameraError(msg)

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        self._capture = capture
        return self

    def read(self) -> CameraFrame:
        """Lee el siguiente cuadro.

        Un cuadro que no llega es un error, no un caso normal: significa que la
        cámara se desconectó a media sesión. La ausencia de **mano** sí es normal
        y la reporta el detector, no esto.
        """
        if self._capture is None:
            raise CameraError("la cámara no está abierta: llama a open() primero")

        import cv2
        import numpy as np

        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError("la cámara dejó de entregar cuadros")

        height, width = int(frame.shape[0]), int(frame.shape[1])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return CameraFrame(
            bgr=frame,
            rgb=rgb,
            width=width,
            height=height,
            # /255 y no /256: un cuadro completamente blanco debe dar 1.0 exacto,
            # que es lo que `Sample` valida como extremo del intervalo.
            mean_luminance=float(gray.mean()) / 255.0,
        )

    def backend_name(self) -> str:
        """Qué API de captura está usando OpenCV: `V4L2`, `DSHOW`, `AVFOUNDATION`…

        Forma parte del identificador con el que se registra la calibración: el
        mismo dispositivo a través de dos backends distintos no entrega lo mismo,
        y una calibración hecha con uno no dice nada del otro. Ver
        `lsm.io.calibration.camera_key`.
        """
        if self._capture is None:
            raise CameraError("la cámara no está abierta: llama a open() primero")
        nombre = self._capture.getBackendName()
        # Un backend sin nombre no debe romper la calibración: se degrada a una
        # etiqueta constante, que sigue siendo un identificador utilizable.
        return str(nombre) if nombre else "DESCONOCIDO"

    def frames(self) -> Iterator[CameraFrame]:
        """Cuadros hasta que la cámara se acabe o quien llama pare."""
        while True:
            yield self.read()

    def close(self) -> None:
        """Libera el dispositivo. Idempotente."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> Camera:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass
class VideoRecorder:
    """Escribe cuadros a un archivo de video.

    **Solo se instancia con consentimiento explícito y por escrito de quien
    firma** (`ARQUITECTURA.md` §4.11). Los landmarks no identifican a nadie; el
    video sí, y esa es toda la diferencia. `cli/capture.py` exige dos llaves para
    llegar aquí: el registro de consentimiento del `signer_id` y la bandera
    `--guardar-video`, que está apagada por defecto.

    No hay ruta por la que el video se grabe "por si acaso". Si este objeto no se
    construye, no se escribe nada, y si no se construye es porque nadie pidió
    explícitamente que se escribiera.
    """

    path: Path
    width: int
    height: int
    fps: int

    _writer: Any = field(default=None, init=False, repr=False)

    def open(self) -> VideoRecorder:
        if self._writer is not None:
            return self

        import cv2

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(self.path), fourcc, float(self.fps), (self.width, self.height)
        )
        if not writer.isOpened():
            writer.release()
            raise CameraError(f"no se pudo abrir el archivo de video {self.path}")
        self._writer = writer
        return self

    def write(self, image: Any) -> None:
        """Escribe un cuadro BGR.

        Se guarda el cuadro **sin espejar y sin landmarks dibujados**: es el
        registro de lo que ocurrió frente a la cámara, no una captura de pantalla
        del programa. Si algún día hay que re-derivar landmarks de este video
        —que es la única razón seria para guardarlo— tienen que salir los mismos.
        """
        if self._writer is None:
            raise CameraError("el grabador no está abierto: llama a open() primero")
        self._writer.write(image)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> VideoRecorder:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()
