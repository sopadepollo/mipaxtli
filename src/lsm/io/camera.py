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
from typing import Any, Final

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
    #: Miniatura en escala de grises, un byte por píxel, fila a fila. Vacía si
    #: la cámara se abrió sin `thumbnail_px`. Es la huella con la que el
    #: diagnóstico de tracking reconoce un cuadro que la cámara **repitió**: con
    #: poca luz muchas webcams bajan de 30 a 15 fps y entregan cada cuadro dos
    #: veces (`src/lsm/tracking_diagnostics.py`). Se calcula aquí por lo mismo
    #: que la luminancia: después la imagen ya no existe.
    thumbnail: bytes = b""
    #: Cuadros repetidos que la cámara entregó justo antes de éste y se
    #: descartaron (`Camera.drop_duplicates`). Cero si no se descarta nada.
    skipped_duplicates: int = 0


class CameraError(RuntimeError):
    """La cámara no se pudo abrir o dejó de entregar cuadros."""


#: Nombres de `capture.backend` y la constante de OpenCV que piden. `auto` deja
#: que OpenCV elija (en Windows, MSMF), que es lo que se hacía hasta el Bloque 1.
#: DSHOW está porque en Windows suele ser la única vía por la que una webcam
#: entrega MJPG a 30 fps: MSMF negocia el formato por su cuenta.
_BACKENDS: Final = {
    "auto": "CAP_ANY",
    "MSMF": "CAP_MSMF",
    "DSHOW": "CAP_DSHOW",
    "V4L2": "CAP_V4L2",
}


@dataclass(frozen=True, slots=True)
class Negotiated:
    """Lo que el driver aceptó de verdad. Se lee de vuelta: pedir no es obtener."""

    backend: str
    width: int
    height: int
    fps: float
    #: Código de cuatro letras del formato entregado, o `""` si no se sabe.
    fourcc: str


@dataclass
class Camera:
    """La webcam, envuelta.

    Se pide una resolución concreta, pero es una **petición**: el driver puede
    entregar otra y no avisa. Por eso `CameraFrame` lleva su propio `width` y
    `height` leídos del arreglo, y por eso cada frame guardado en el dataset lleva
    los suyos: el paso 1 de `feature-spec.md` divide por la relación de aspecto, y
    asumir 1280x720 cuando la cámara entregó 640x480 deforma la mano en silencio.

    **Cuadros duplicados** (Fase 5.1, Bloque 1). Con MSMF la webcam de referencia
    entrega 29.5 cuadros por segundo de los que el 44.5% son repeticiones exactas
    del anterior (ADR 0017): 16.4 fps reales. Con `drop_duplicates` se descartan
    aquí, antes de MediaPipe —un cuadro repetido no aporta nada y alternaba la
    velocidad alto/cero—, reconociéndolos por su miniatura en grises. Un cuadro
    repetido es **idéntico** byte a byte en la miniatura; dos cuadros reales de una
    mano quieta no lo son, por el ruido del sensor.
    """

    index: int
    width: int
    height: int
    fps: int
    #: Ancho de la miniatura de `CameraFrame.thumbnail`. `None` no la calcula,
    #: salvo que `drop_duplicates` la necesite.
    thumbnail_px: int | None = None
    #: Formato pedido (`capture.fourcc`), p. ej. `"MJPG"`. `None`: el del driver.
    fourcc: str | None = None
    #: API de captura (`capture.backend`).
    backend: str = "auto"
    #: Descartar los cuadros que repiten al anterior (`capture.drop_duplicate_frames`).
    drop_duplicates: bool = False
    #: Tope de duplicados seguidos que se descartan antes de entregar uno igual.
    #: Sin tope, una imagen congelada colgaría el bucle en vez de verse congelada.
    max_consecutive_duplicates: int = 10

    _capture: Any = field(default=None, init=False, repr=False)
    _last_thumbnail: bytes = field(default=b"", init=False, repr=False)

    @classmethod
    def from_config(
        cls, config: CaptureConfig, *, thumbnail_px: int | None = None
    ) -> Camera:
        """La cámara de `config.yaml`. `thumbnail_px` fuerza la miniatura aunque
        no se descarten duplicados (el diagnóstico la usa para contarlos)."""
        return cls(
            index=config.camera_index,
            width=config.frame_width,
            height=config.frame_height,
            fps=config.camera_fps,
            thumbnail_px=(
                thumbnail_px
                if thumbnail_px is not None
                else (
                    config.duplicate_thumbnail_px
                    if config.drop_duplicate_frames
                    else None
                )
            ),
            fourcc=config.fourcc,
            backend=config.backend,
            drop_duplicates=config.drop_duplicate_frames,
            max_consecutive_duplicates=config.max_consecutive_duplicates,
        )

    def open(self) -> Camera:
        """Abre el dispositivo. Devuelve `self` para poder encadenar."""
        if self._capture is not None:
            return self

        import cv2

        if self.backend not in _BACKENDS:
            msg = (
                f"backend de cámara desconocido: {self.backend!r} ({sorted(_BACKENDS)})"
            )
            raise CameraError(msg)
        api = getattr(cv2, _BACKENDS[self.backend])
        capture = cv2.VideoCapture(self.index, api)
        if not capture.isOpened():
            capture.release()
            msg = (
                f"no se pudo abrir la cámara {self.index} con el backend "
                f"{self.backend}. En Linux, comprobar que exista /dev/video*; en "
                "Docker, que se haya pasado el dispositivo (ver docker/README.md); "
                "en Windows y macOS, que la captura se ejecute FUERA del "
                "contenedor. DSHOW y MSMF solo existen en Windows."
            )
            raise CameraError(msg)

        # El formato va ANTES que la resolución: varios drivers solo ofrecen
        # ciertas resoluciones a 30 fps en MJPG, y si se pide la resolución con
        # el formato por defecto negocian una tasa menor y ya no la cambian.
        if self.fourcc is not None:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*self.fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        self._capture = capture
        if self.drop_duplicates and self.thumbnail_px is None:
            msg = "drop_duplicates necesita thumbnail_px para reconocer duplicados"
            raise CameraError(msg)
        return self

    def negotiated(self) -> Negotiated:
        """Lo que el driver dice haber aceptado. Una petición no es una garantía."""
        if self._capture is None:
            raise CameraError("la cámara no está abierta: llama a open() primero")
        import cv2

        codigo = int(self._capture.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((codigo >> (8 * i)) & 0xFF) for i in range(4)).strip(
            "\x00 "
        )
        return Negotiated(
            backend=self.backend_name(),
            width=int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(self._capture.get(cv2.CAP_PROP_FPS)),
            fourcc=fourcc if fourcc.isprintable() else "",
        )

    def read(self) -> CameraFrame:
        """Lee el siguiente cuadro **nuevo**.

        Con `drop_duplicates`, los cuadros cuya miniatura repite la del último
        entregado se descartan y se lee el siguiente, hasta
        `max_consecutive_duplicates`. Cuántos se descartaron viaja en
        `CameraFrame.skipped_duplicates`.

        Un cuadro que no llega es un error, no un caso normal: significa que la
        cámara se desconectó a media sesión. La ausencia de **mano** sí es normal
        y la reporta el detector, no esto.
        """
        if self._capture is None:
            raise CameraError("la cámara no está abierta: llama a open() primero")

        import cv2
        import numpy as np

        saltados = 0
        while True:
            ok, frame = self._capture.read()
            if not ok or frame is None:
                raise CameraError("la cámara dejó de entregar cuadros")

            height, width = int(frame.shape[0]), int(frame.shape[1])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            thumbnail = b""
            if self.thumbnail_px is not None:
                alto = max(1, round(self.thumbnail_px * height / width))
                # INTER_AREA promedia los píxeles de cada celda: el ruido del
                # sensor se diluye y una repetición exacta da la misma huella.
                pequena = cv2.resize(
                    gray, (self.thumbnail_px, alto), interpolation=cv2.INTER_AREA
                )
                thumbnail = np.ascontiguousarray(pequena, dtype=np.uint8).tobytes()
            repetido = bool(thumbnail) and thumbnail == self._last_thumbnail
            if (
                self.drop_duplicates
                and repetido
                and saltados < self.max_consecutive_duplicates
            ):
                saltados += 1
                continue
            break

        self._last_thumbnail = thumbnail
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return CameraFrame(
            bgr=frame,
            rgb=rgb,
            width=width,
            height=height,
            # /255 y no /256: un cuadro completamente blanco debe dar 1.0 exacto,
            # que es lo que `Sample` valida como extremo del intervalo.
            mean_luminance=float(gray.mean()) / 255.0,
            thumbnail=thumbnail,
            skipped_duplicates=saltados,
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
