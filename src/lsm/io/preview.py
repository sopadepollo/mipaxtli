"""Dibujo del preview de captura. Capa de visualización, nada más.

Aquí vive lo único que se espeja en todo el proyecto. El detector recibe siempre
el cuadro tal como sale de la cámara (`feature-spec.md` §0.3); lo que se ve en
pantalla va reflejado porque es lo que espera quien se mira, y el reflejo se
aplica al final, sobre píxeles que ya no van a ningún sitio.

La geometría del espejado no está aquí sino en `lsm.capture.preview_position`, que
es código puro y por tanto testeable sin OpenCV. Este módulo solo pinta.

Se dibujan las conexiones a mano en vez de usar las utilidades de MediaPipe: desde
MediaPipe 1.0 el paquete `solutions` —donde vivía `drawing_utils`— ya no se
distribuye, y de todos modos la topología de la mano es estructura del proyecto y
está en `lsm.types.HAND_CONNECTIONS`, no una cortesía de la librería.

Importa `cv2` de forma diferida, como el resto de `io/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lsm.capture import WindowQuality, explain, preview_position
from lsm.segmentation import State
from lsm.types import HAND_CONNECTIONS, Handedness, Prediction, RawFrame, SampleKind

#: Colores BGR, que es el orden de OpenCV.
_VERDE = (120, 220, 120)
_ROJO = (90, 90, 240)
_AMBAR = (60, 190, 250)
_BLANCO = (245, 245, 245)
_GRIS = (150, 150, 150)
_NEGRO = (20, 20, 20)

_FUENTE = 0  # cv2.FONT_HERSHEY_SIMPLEX, sin importar cv2 para una constante.


@dataclass(frozen=True, slots=True)
class HudState:
    """Todo lo que el preview tiene que decir en un instante dado.

    Es un dato y no un puñado de argumentos sueltos para que el CLI pueda armarlo
    y pasarlo entero, y para que se vea de un vistazo qué información necesita
    quien graba: qué letra toca, cuántas lleva, si la ventana serviría ahora
    mismo, y con qué mano cree el detector que está firmando.
    """

    #: Letra objetivo, tal como se escribe para una persona: `Ñ`, `LL`, `RR`.
    letra: str
    #: Etiqueta interna, la que va al nombre de archivo.
    label: str
    kind: SampleKind
    guardadas: int
    objetivo: int
    quality: WindowQuality
    max_dispersion: float
    min_trajectory_arc: float
    #: Frames grabados hasta ahora, si hay una grabación dinámica en curso.
    grabando: int | None
    #: Último mensaje que mostrar, ya sea de éxito o de rechazo.
    mensaje: str
    guarda_video: bool


@dataclass(frozen=True, slots=True)
class DemoHudState:
    """Todo lo que la demo tiene que decir en un instante.

    `HudState` es de captura —letra objetivo, muestras guardadas, si graba
    video— y no sirve aquí. Lo que la demo necesita enseñar es otra cosa: qué
    lleva escrito y por qué la máquina de estados está donde está.
    """

    #: El texto completo, palabras cerradas incluidas.
    texto: str
    #: La palabra en curso, para verla crecer letra a letra.
    palabra: str
    estado: State
    #: Última predicción, aunque se haya rechazado. `None` antes de la primera.
    ultima: Prediction | None
    #: σ de la ventana actual, si la hay.
    dispersion: float | None
    #: Último mensaje: por qué se rechazó, o qué se acaba de borrar.
    mensaje: str


def draw_landmarks(image: Any, frame: RawFrame, *, mirrored: bool) -> None:
    """Dibuja el esqueleto de la mano sobre la imagen, en su sitio.

    Modifica `image` en el lugar: es un arreglo de OpenCV de varios megabytes por
    cuadro y copiarlo treinta veces por segundo para no mutar nada sería una
    elegancia cara.
    """
    import cv2

    height, width = int(image.shape[0]), int(image.shape[1])
    puntos = [
        preview_position(landmark, width, height, mirrored=mirrored)
        for landmark in frame.landmarks
    ]

    for inicio, fin in HAND_CONNECTIONS:
        cv2.line(image, puntos[inicio], puntos[fin], _VERDE, 2, cv2.LINE_AA)
    for punto in puntos:
        cv2.circle(image, punto, 3, _BLANCO, -1, cv2.LINE_AA)


def draw_hud(image: Any, state: HudState) -> None:
    """Dibuja la información de la sesión sobre el cuadro.

    Lo que se muestra sale directamente de `ARQUITECTURA.md` §4.7 —letra objetivo,
    contador de muestras, preview con landmarks— más el indicador de dispersión σ
    del `feature-spec.md` §2, que es lo que convierte la sesión en algo medible:
    sin él, quien graba se entera de que las veinte repeticiones salieron
    temblorosas cuando ya se fue a su casa.
    """
    import cv2

    height, width = int(image.shape[0]), int(image.shape[1])
    _panel(image, 0, 0, width, 96)

    cv2.putText(
        image,
        f"{state.letra}",
        (18, 68),
        _FUENTE,
        2.0,
        _BLANCO,
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"{state.label}  [{state.kind.value.lower()}]",
        (110, 40),
        _FUENTE,
        0.6,
        _GRIS,
        1,
        cv2.LINE_AA,
    )

    completo = state.guardadas >= state.objetivo
    cv2.putText(
        image,
        f"{state.guardadas}/{state.objetivo}",
        (110, 72),
        _FUENTE,
        0.8,
        _VERDE if completo else _BLANCO,
        2,
        cv2.LINE_AA,
    )

    lateralidad = state.quality.handedness
    cv2.putText(
        image,
        _texto_lateralidad(lateralidad),
        (width - 260, 40),
        _FUENTE,
        0.6,
        _BLANCO if lateralidad is not None else _GRIS,
        1,
        cv2.LINE_AA,
    )
    if state.guarda_video:
        cv2.putText(
            image,
            "REC video (con consentimiento)",
            (width - 260, 68),
            _FUENTE,
            0.5,
            _ROJO,
            1,
            cv2.LINE_AA,
        )

    _draw_calidad(image, state, origen=(width - 260, 84))
    _draw_estado(image, state, alto=height, ancho=width)


def _draw_calidad(image: Any, state: HudState, origen: tuple[int, int]) -> None:
    """La barra que hay que mirar mientras se firma. Cuál es, depende del modo.

    Los dos modos tienen un criterio gradual y son opuestos, así que la barra mide
    cosas distintas según lo que se esté grabando:

    - **Estática**: σ contra `quality.max_dispersion`. Verde mientras la mano esté
      quieta, roja si se movió de más.
    - **Dinámica**: longitud de arco de τ contra `capture.min_trajectory_arc`.
      Verde cuando el trazo ya recorrió lo suficiente, roja mientras no. Aquí la
      barra **se llena hacia el aprobado**, al revés que la de σ.

    Sin la segunda barra, una J en la que la mano apenas se movió se guardaba sin
    que nadie lo notara y quedaba en el dataset como una I con otra etiqueta. Quien
    graba tiene que saber **antes** de pulsar la tecla si la muestra sirve; después
    ya no hay forma de mirar el archivo y saberlo.

    En los dos casos la barra llega hasta el doble del umbral, para que se vea
    *cuánto* falta o cuánto sobra y no solo de qué lado se está: una muestra que
    roza el límite y otra que lo dobla piden correcciones distintas.
    """
    import cv2

    x, y = origen
    ancho, alto = 240, 10
    cv2.rectangle(image, (x, y), (x + ancho, y + alto), _NEGRO, -1)

    if state.kind is SampleKind.DYNAMIC:
        etiqueta, valor, umbral = (
            "arco",
            state.quality.arc_length,
            (state.min_trajectory_arc),
        )
        aprobado = valor is not None and valor >= umbral
        formato = ".2f"
    else:
        etiqueta, valor, umbral = (
            "sigma",
            state.quality.dispersion,
            (state.max_dispersion),
        )
        aprobado = valor is not None and valor <= umbral
        formato = ".4f"

    if valor is None:
        cv2.putText(
            image, f"{etiqueta}: --", (x, y - 2), _FUENTE, 0.45, _GRIS, 1, cv2.LINE_AA
        )
        return

    tope = umbral * 2.0
    lleno = min(1.0, valor / tope) if tope > 0 else 1.0
    color = _VERDE if aprobado else _ROJO
    cv2.rectangle(image, (x, y), (x + int(ancho * lleno), y + alto), color, -1)

    umbral_x = x + ancho // 2  # el umbral cae a la mitad de una barra de 2×.
    cv2.line(image, (umbral_x, y - 3), (umbral_x, y + alto + 3), _BLANCO, 1)
    cv2.putText(
        image,
        f"{etiqueta} {valor:{formato}} / {umbral:{formato}}",
        (x, y - 2),
        _FUENTE,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_estado(image: Any, state: HudState, *, alto: int, ancho: int) -> None:
    import cv2

    _panel(image, 0, alto - 78, ancho, 78)

    if state.grabando is not None:
        linea, color = f"GRABANDO  {state.grabando} frames", _ROJO
    elif state.quality.rejection is not None:
        linea, color = explain(state.quality.rejection), _AMBAR
    else:
        linea, color = "listo: la ventana serviria como muestra", _VERDE

    cv2.putText(image, linea, (18, alto - 48), _FUENTE, 0.65, color, 2, cv2.LINE_AA)
    cv2.putText(
        image,
        state.mensaje or _ATAJOS,
        (18, alto - 20),
        _FUENTE,
        0.5,
        _GRIS,
        1,
        cv2.LINE_AA,
    )


#: Se escriben sin acentos: la fuente de OpenCV es ASCII y una `ó` sale como `?`.
_ATAJOS = "ESPACIO guardar | n/p letra | m estatica/dinamica | r rehacer | q salir"

#: Siempre visible, nunca condicionada a un intento fallido. Sin ella, quien
#: prueba la demo hace una `J`, no ve nada, y concluye que el sistema falla —
#: cuando lo que pasa es que esa letra llega en la Fase 5.
_AVISO_DINAMICAS = "las 8 letras dinamicas no se reconocen aun: llegan en la Fase 5"


def draw_demo_hud(image: Any, state: DemoHudState) -> None:
    """Dibuja el HUD de la demo en vivo. Modifica `image` en el lugar.

    Confianza y estado van **siempre**, no solo cuando hay letra: sin ellos, una
    ventana rechazada por confianza baja y una mano que el detector no encuentra
    se ven exactamente igual en pantalla, y depurar la demo se vuelve adivinar.
    """
    import cv2

    height, width = int(image.shape[0]), int(image.shape[1])
    _panel(image, 0, 0, width, 96)

    cv2.putText(
        image,
        f"texto:   {state.texto}",
        (18, 30),
        _FUENTE,
        0.7,
        _BLANCO,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"palabra: {state.palabra}",
        (18, 58),
        _FUENTE,
        0.6,
        _GRIS,
        1,
        cv2.LINE_AA,
    )

    confianza = "--" if state.ultima is None else f"{state.ultima.confidence:.2f}"
    letra = "--" if state.ultima is None else state.ultima.label
    sigma = "--" if state.dispersion is None else f"{state.dispersion:.3f}"
    cv2.putText(
        image,
        f"{state.estado.value}   ultima: {letra} ({confianza})   sigma: {sigma}",
        (18, 84),
        _FUENTE,
        0.55,
        _BLANCO,
        1,
        cv2.LINE_AA,
    )

    _panel(image, 0, height - 64, width, 64)
    cv2.putText(
        image,
        _AVISO_DINAMICAS,
        (18, height - 40),
        _FUENTE,
        0.45,
        _GRIS,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        state.mensaje,
        (18, height - 14),
        _FUENTE,
        0.5,
        _AMBAR,
        1,
        cv2.LINE_AA,
    )


def _texto_lateralidad(side: Handedness | None) -> str:
    """Se muestra siempre, y por un motivo concreto.

    Es la comprobación de un segundo que zanja
    `hands.mediapipe_reports_mirrored_handedness`: quien graba levanta la mano
    derecha al empezar y mira si aquí dice RIGHT. Equivocarse en esa opción no
    rompe nada visible —canoniza todo el dataset hacia la mano contraria— así que
    la única defensa es tenerlo delante.
    """
    if side is None:
        return "mano: --"
    return f"mano: {side.value}"


def _panel(image: Any, x: int, y: int, ancho: int, alto: int) -> None:
    """Franja oscura translúcida, para que el texto se lea sobre cualquier fondo."""
    import cv2

    region = image[y : y + alto, x : x + ancho]
    if region.size == 0:
        return
    # `region` es una vista sobre `image`: hay que pintar el negro en una copia y
    # mezclarla de vuelta. Rellenar la vista y mezclarla consigo misma borraría el
    # cuadro en vez de oscurecerlo.
    overlay = region.copy()
    cv2.rectangle(overlay, (0, 0), (ancho, alto), _NEGRO, -1)
    cv2.addWeighted(overlay, 0.55, region, 0.45, 0, region)
