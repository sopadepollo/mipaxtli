"""Instrumentación de latencia del bucle en vivo.

## Por qué existe

Todos los umbrales de `segmentation` están expresados en **frames**:
`buffer_size: 24`, `stable_frames: 5`, `emit_cooldown_frames: 12`. Los
comentarios de `config.yaml` los traducen a milisegundos suponiendo 30 fps —«24
frames son unos 800 ms»— y esa suposición no se ha medido nunca. Si la tubería
corre a 12 fps, los mismos 24 frames son 2 segundos y el deletreo se siente
lento sin que ningún número del repositorio lo explique.

Medir la tasa es, por tanto, el paso previo a calibrar cualquiera de esos
umbrales. Ver `docs/adr/0013-la-ventana-mezclada.md`.

## Qué se mide y por qué son dos cosas

- **Entrega**: cuántos cuadros por segundo completa el bucle. Es un techo que la
  cámara impone (`capture.camera_fps`) y es la tasa en la que están expresados
  los umbrales.
- **Procesamiento**: cuántos podría sostener la tubería —MediaPipe, features,
  segmentación— si la cámara entregara infinitamente rápido. Se obtiene
  descontando el bloqueo de `camera.read()` y el dibujo del preview.

La diferencia entre las dos dice quién es el cuello de botella. Si `camara` se
come el tiempo, la cámara manda y la tubería está esperando; si `deteccion` se
lo come, la tubería no llega y el buffer se llena más despacio de lo que dicen
los comentarios de `config.yaml`.

**Se reportan percentiles y no solo la media.** Una tubería que promedia 28 fps
pero cae a 9 durante medio segundo produce exactamente la lentitud que se
percibe, y en la media no se ve: el percentil 5 es el que la denuncia.

## Pureza

Este módulo es **código puro**, como `features.py`, `segmentation.py` y
`capture.py` (`CLAUDE.md` §2): recibe duraciones ya medidas y, cuando necesita
un reloj, lo recibe inyectado. No importa `cv2`, no toca disco y no llama a
`time`. Quien mide el tiempo real es `cli/demo.py`, que es quien ya tiene el
bucle y la cámara. Eso es lo que permite probar el reparto por etapas con un
reloj guionizado, sin webcam y sin dormir en CI.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final


class Stage(StrEnum):
    """Las cuatro etapas en las que se reparte un cuadro del bucle en vivo.

    Cubren el bucle **entero** por construcción: su suma es el tiempo de pared
    entre dos cuadros, y de ahí que el fps de entrega sea el inverso de esa
    suma. Si alguna vez se añade trabajo al bucle sin añadir su etapa aquí, el
    fps reportado saldrá más alto que el real.

    Se escriben sin acentos porque además se dibujan en el preview, y la fuente
    de OpenCV es ASCII.
    """

    #: Lo que tarda `camera.read()`. Es **espera**, no trabajo: si la tubería es
    #: más lenta que la cámara, el cuadro ya está en el buffer del driver y esto
    #: sale casi cero.
    CAMARA = "camara"
    #: MediaPipe: `HandDetector.detect`.
    DETECCION = "deteccion"
    #: Lo que tarda `run_segmentation` en consumir el cuadro cedido: extracción
    #: de features, máquina de estados y, si la ventana es estable, el
    #: clasificador. Se cierra al principio de la iteración siguiente, que es
    #: cuando el consumidor devuelve el control al generador.
    SEGMENTACION = "segmentacion"
    #: Dibujo del HUD, `imshow` y `waitKey`. No es reconocimiento, así que no
    #: entra en el fps de procesamiento, pero sí en el de entrega: el bucle lo
    #: paga igual.
    PREVIEW = "preview"


#: Las etapas que componen el reconocimiento. `CAMARA` es espera y `PREVIEW` es
#: dibujo: ninguna de las dos dice nada sobre la capacidad de la tubería.
_PROCESAMIENTO: Final = (Stage.DETECCION, Stage.SEGMENTACION)


@dataclass(frozen=True, slots=True)
class FrameTiming:
    """Lo que tardó cada etapa en un cuadro, en segundos."""

    camara: float
    deteccion: float
    segmentacion: float
    preview: float

    def __post_init__(self) -> None:
        for etapa, valor in self.por_etapa().items():
            if valor < 0.0 or math.isnan(valor):
                msg = (
                    f"la etapa {etapa.value} midió una duración negativa o no "
                    f"numérica ({valor!r}): el reloj no es monótono y la medida "
                    "no se puede usar"
                )
                raise ValueError(msg)

    def por_etapa(self) -> dict[Stage, float]:
        return {
            Stage.CAMARA: self.camara,
            Stage.DETECCION: self.deteccion,
            Stage.SEGMENTACION: self.segmentacion,
            Stage.PREVIEW: self.preview,
        }

    @property
    def ciclo(self) -> float:
        """El cuadro completo: tiempo de pared entre este cuadro y el anterior."""
        return self.camara + self.deteccion + self.segmentacion + self.preview

    @property
    def procesamiento(self) -> float:
        """Solo el reconocimiento: sin la espera de la cámara ni el dibujo."""
        return self.deteccion + self.segmentacion


@dataclass(frozen=True, slots=True)
class Distribucion:
    """Una serie resumida. La media va acompañada porque sola no basta."""

    media: float
    mediana: float
    p5: float
    p95: float
    minimo: float
    maximo: float
    n: int


def percentil(valores: Sequence[float], q: float) -> float:
    """Percentil `q ∈ [0, 1]` con interpolación lineal entre rangos vecinos.

    La definición se fija aquí en vez de heredarla de una librería: `numpy` es
    una dependencia **opcional** del proyecto (solo entra con `--extra
    capture`), así que el resumen tiene que poder calcularse en la suite sin
    ella. El método es el mismo que el `linear` de `numpy.percentile`: el índice
    es `q · (n − 1)` y se interpola entre los dos vecinos.
    """
    if not valores:
        msg = "no hay percentil de una serie vacía"
        raise ValueError(msg)
    if not 0.0 <= q <= 1.0:
        msg = f"el percentil pedido ({q}) tiene que estar entre 0 y 1"
        raise ValueError(msg)

    ordenados = sorted(valores)
    if len(ordenados) == 1:
        return ordenados[0]

    posicion = q * (len(ordenados) - 1)
    bajo = math.floor(posicion)
    alto = math.ceil(posicion)
    if bajo == alto:
        return ordenados[bajo]
    peso = posicion - bajo
    return ordenados[bajo] * (1.0 - peso) + ordenados[alto] * peso


def resumir(valores: Sequence[float]) -> Distribucion | None:
    """Resume una serie, o `None` si está vacía.

    `None` y no una distribución de ceros: antes del primer cuadro no hay nada
    que reportar, y un 0.0 en la columna de fps se leería como una caída.
    """
    if not valores:
        return None
    return Distribucion(
        media=sum(valores) / len(valores),
        mediana=percentil(valores, 0.5),
        p5=percentil(valores, 0.05),
        p95=percentil(valores, 0.95),
        minimo=min(valores),
        maximo=max(valores),
        n=len(valores),
    )


@dataclass(frozen=True, slots=True)
class Resumen:
    """El reporte completo de una medición."""

    #: fps por cuadro (`1 / ciclo`), resumido. `None` si no se midió nada.
    entrega: Distribucion | None
    #: fps por cuadro descontando cámara y preview.
    procesamiento: Distribucion | None
    #: Latencia por etapa, en segundos. Vacío si no se midió nada.
    etapas: Mapping[Stage, Distribucion]
    cuadros: int
    #: Tiempo de pared cubierto por los cuadros medidos, en segundos.
    segundos: float

    @property
    def fps_sostenido(self) -> float | None:
        """Cuadros entre tiempo de pared: el caudal, no la media de los inversos.

        Es el número que contesta «cuántos cuadros por segundo ve esto de
        verdad». La media de la columna `entrega` pesa igual un cuadro rápido y
        uno lento, así que sale más alta; el caudal es la media armónica y es la
        que se puede comparar contra `capture.camera_fps`.
        """
        if self.cuadros == 0 or self.segundos <= 0.0:
            return None
        return self.cuadros / self.segundos


@dataclass
class Medicion:
    """Acumula cuadros medidos. El único objeto mutable del módulo.

    Con `ventana` se convierte en una media móvil sobre los últimos cuadros, que
    es lo que el HUD necesita para que el número reaccione a lo que se acaba de
    tocar. Sin `ventana` guarda la sesión entera, que es lo que necesita el
    resumen estadístico de `--medir-fps`.
    """

    #: Cuántos cuadros recordar. `None` es «todos».
    ventana: int | None = None

    _timings: deque[FrameTiming] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.ventana is not None and self.ventana < 1:
            msg = f"la ventana de medición ({self.ventana}) tiene que ser >= 1"
            raise ValueError(msg)
        self._timings = deque(maxlen=self.ventana)

    def agregar(self, timing: FrameTiming) -> None:
        self._timings.append(timing)

    @property
    def cuadros(self) -> int:
        return len(self._timings)

    @property
    def segundos(self) -> float:
        """Tiempo de pared de los cuadros recordados."""
        return sum(timing.ciclo for timing in self._timings)

    @property
    def fps_entrega(self) -> float | None:
        return self._caudal(sum(t.ciclo for t in self._timings))

    @property
    def fps_procesamiento(self) -> float | None:
        return self._caudal(sum(t.procesamiento for t in self._timings))

    def _caudal(self, segundos: float) -> float | None:
        if not self._timings or segundos <= 0.0:
            return None
        return len(self._timings) / segundos

    def resumen(self) -> Resumen:
        """Resume lo acumulado.

        Los cuadros con duración cero se caen de las series de fps: aparecen
        cuando la resolución del reloj no alcanza a distinguir dos marcas, y su
        inverso sería infinito, que envenenaría media y percentiles. Se cuentan
        igual en `cuadros` y en `segundos`, así que el caudal sostenido los ve.
        """
        entrega = resumir([1.0 / t.ciclo for t in self._timings if t.ciclo > 0.0])
        procesamiento = resumir(
            [1.0 / t.procesamiento for t in self._timings if t.procesamiento > 0.0]
        )
        etapas = {
            etapa: resumen
            for etapa in Stage
            if (resumen := resumir([t.por_etapa()[etapa] for t in self._timings]))
            is not None
        }
        return Resumen(
            entrega=entrega,
            procesamiento=procesamiento,
            etapas=etapas,
            cuadros=len(self._timings),
            segundos=self.segundos,
        )


def render_resumen(resumen: Resumen) -> str:
    """Vuelca el resumen como texto plano, para pegarlo en la bitácora.

    Sin marca de tiempo y sin adornos: dos mediciones de la misma máquina se
    comparan línea a línea.
    """
    if resumen.cuadros == 0:
        return "medicion de fps — no se midio ningun cuadro"

    lineas = [
        "== medicion de fps ==",
        f"cuadros: {resumen.cuadros}   pared: {resumen.segundos:.2f} s"
        f"   fps sostenido: {_num(resumen.fps_sostenido)}",
        "",
        f"{'fps por cuadro':<16}{'media':>9}{'mediana':>9}"
        f"{'p5':>9}{'p95':>9}{'min':>9}{'max':>9}",
        _fila_fps("entrega", resumen.entrega),
        _fila_fps("procesamiento", resumen.procesamiento),
        "",
        f"{'latencia (ms)':<16}{'media':>9}{'mediana':>9}"
        f"{'p5':>9}{'p95':>9}{'min':>9}{'max':>9}",
    ]
    lineas.extend(
        _fila_latencia(etapa.value, resumen.etapas.get(etapa)) for etapa in Stage
    )
    lineas.extend(("", _cuello_de_botella(resumen)))
    return "\n".join(lineas)


def _cuello_de_botella(resumen: Resumen) -> str:
    """La lectura del reporte, escrita para no tener que hacerla cada vez.

    Es la pregunta que se le hace a estos números: si la tasa es baja, ¿la
    impone la cámara o la tubería? La respuesta cambia por completo qué hacer
    con ella.
    """
    camara = resumen.etapas.get(Stage.CAMARA)
    if camara is None:
        return ""
    reconocimiento = sum(
        distribucion.mediana
        for etapa in _PROCESAMIENTO
        if (distribucion := resumen.etapas.get(etapa)) is not None
    )
    if camara.mediana > reconocimiento:
        return (
            "manda la cámara: el bucle pasa más tiempo esperando cuadros que "
            "reconociéndolos, así que la tasa medida es el techo de la cámara "
            "(capture.camera_fps) y no el de la tubería."
        )
    return (
        "manda la tubería: el reconocimiento tarda más que la espera de la "
        "cámara, así que la tasa medida está por debajo del techo de la cámara "
        "(capture.camera_fps)."
    )


def _fila_fps(nombre: str, distribucion: Distribucion | None) -> str:
    return _fila(nombre, distribucion, escala=1.0, formato=".1f")


def _fila_latencia(nombre: str, distribucion: Distribucion | None) -> str:
    return _fila(nombre, distribucion, escala=1000.0, formato=".1f")


def _fila(
    nombre: str, distribucion: Distribucion | None, *, escala: float, formato: str
) -> str:
    if distribucion is None:
        return f"{nombre:<16}{'--':>9}"
    valores = (
        distribucion.media,
        distribucion.mediana,
        distribucion.p5,
        distribucion.p95,
        distribucion.minimo,
        distribucion.maximo,
    )
    celdas = "".join(f"{valor * escala:>9{formato}}" for valor in valores)
    return f"{nombre:<16}{celdas}"


def _num(valor: float | None) -> str:
    return "--" if valor is None else f"{valor:.1f}"


def render_fps(*, entrega: float | None, procesamiento: float | None) -> str:
    """La línea de tasa del HUD, en vivo.

    Van las **dos** tasas y no solo una: con `fps` bajo y `proc` alto el bucle
    está esperando a la cámara, y con los dos bajos es la tubería la que no
    llega. Son diagnósticos distintos y en pantalla tienen que distinguirse.

    Vive aquí y no en `io/preview.py` porque componer el texto no necesita
    OpenCV, y así se puede fijar en un test que corre en CI.
    """
    return f"fps {_num(entrega)}  proc {_num(procesamiento)}"


@dataclass
class Cronometro:
    """Reparte el tiempo de un cuadro entre sus etapas.

    El reloj entra **inyectado** para que este módulo siga siendo puro y para
    que los tests puedan guionizarlo. `cli/demo.py` le pasa
    `time.perf_counter`.

    Uso, tal como lo hace el bucle en vivo: `iniciar()` al empezar el cuadro y
    un `marcar(etapa)` al terminar cada etapa. `SEGMENTACION` se marca al
    principio de la iteración **siguiente**, porque es entonces cuando
    `run_segmentation` devuelve el control al generador; por eso hace falta
    `abierto`, para saber si hay un cuadro a medio medir que cerrar.

    El cronómetro se reutiliza toda la sesión: `iniciar()` limpia las etapas del
    cuadro anterior.
    """

    reloj: Callable[[], float]

    _marca: float | None = field(default=None, init=False, repr=False)
    _etapas: dict[Stage, float] = field(default_factory=dict, init=False, repr=False)

    @property
    def abierto(self) -> bool:
        """Si hay un cuadro empezado y sin cerrar."""
        return self._marca is not None

    def iniciar(self) -> None:
        self._etapas = {}
        self._marca = self.reloj()

    def marcar(self, etapa: Stage) -> None:
        """Cierra `etapa` con el tiempo transcurrido desde la marca anterior."""
        if self._marca is None:
            msg = f"hay que iniciar el cuadro antes de marcar {etapa.value}"
            raise ValueError(msg)
        if etapa in self._etapas:
            msg = (
                f"la etapa {etapa.value} ya estaba marcada en este cuadro: "
                "sobrescribirla perdería el tiempo intermedio y el ciclo "
                "dejaría de ser el tiempo de pared"
            )
            raise ValueError(msg)
        ahora = self.reloj()
        self._etapas[etapa] = ahora - self._marca
        self._marca = ahora

    def cerrar(self) -> FrameTiming:
        """Entrega el cuadro medido y queda listo para el siguiente."""
        if self._marca is None:
            msg = "no hay ningún cuadro abierto que cerrar"
            raise ValueError(msg)
        faltan = [etapa.value for etapa in Stage if etapa not in self._etapas]
        if faltan:
            msg = (
                f"faltan etapas por marcar ({', '.join(faltan)}): el ciclo "
                "saldría más corto que el real y el fps más alto que el real"
            )
            raise ValueError(msg)
        timing = FrameTiming(
            camara=self._etapas[Stage.CAMARA],
            deteccion=self._etapas[Stage.DETECCION],
            segmentacion=self._etapas[Stage.SEGMENTACION],
            preview=self._etapas[Stage.PREVIEW],
        )
        self._marca = None
        self._etapas = {}
        return timing


def tasa_insuficiente(fps: float | None, *, minimo: float) -> bool:
    """Si la tasa está tan baja que ninguna ventana temporal aguanta.

    Los umbrales de la segmentación se expresan en milisegundos y se convierten a
    cuadros con la tasa: por debajo de cierto punto, la conversión devuelve uno o
    dos cuadros y el criterio de estabilidad deja de decidir nada. Cuando eso
    pasa, quien está delante de la cámara tiene que enterarse de que el problema
    es de rendimiento y no de su seña — si no, corregirá la seña, que es lo único
    que puede hacer, y no servirá de nada.

    `None` —todavía no hay medida— no avisa: durante el primer segundo de cada
    sesión no se sabe nada, y avisar ahí sería avisar siempre.
    """
    return fps is not None and fps < minimo


def segundos_restantes(*, duracion: float, transcurrido: float) -> float:
    """Lo que le queda a una medición de duración fija. Nunca negativo.

    Alimenta la cuenta atrás del HUD y la condición de parada del bucle de
    `--medir-fps`.
    """
    return max(0.0, duracion - transcurrido)
