"""Manos sintéticas: una mano canónica y transformaciones conocidas.

Vive en `src/` y no en `tests/` porque tiene dos consumidores: la suite de tests y
el generador de golden vectors (`lsm.cli.golden`), que es la referencia normativa
para la implementación de TypeScript. Si cada uno construyera sus propios
landmarks, los golden vectors dejarían de describir lo que los tests verifican.

Es código puro: sin disco, sin cámara, sin aleatoriedad. Dos ejecuciones producen
los mismos bits.

**Sistema de coordenadas.** Las funciones de este módulo trabajan en *píxeles* con
la convención de imagen: `x` hacia la derecha, `y` hacia **abajo**, origen en la
esquina superior izquierda del frame. Es la convención en la que trabaja una
cámara real, y es la que `to_frame` convierte a las coordenadas normalizadas que
entrega MediaPipe. Construir las manos en píxeles es lo que hace que los tests de
invariancia signifiquen algo: mover, rotar o alejar una mano son operaciones en el
mundo, no en el vector normalizado.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Final

from lsm.types import (
    NUM_LANDMARKS,
    Distance,
    Handedness,
    Landmark,
    LandmarkIndex,
    LightDirection,
    LightLevel,
    Point3,
    Points3,
    RawFrame,
    Sample,
    SampleKind,
    Sequence,
)

#: Instante de referencia de las muestras sintéticas. Fijo a propósito: un
#: `now()` haría que dos ejecuciones de `make eval` produjeran archivos distintos.
_SYNTHETIC_EPOCH: Final = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)

#: Mano derecha canónica: palma hacia la cámara, dedos extendidos hacia arriba,
#: muñeca en el origen. En píxeles, con `y` hacia abajo (por eso los dedos tienen
#: `y` negativa). La distancia muñeca → nudillo del dedo medio es ≈ 100 px, que es
#: la unidad de escala del paso 4 de `feature-spec.md`.
#:
#: No pretende ser antropométricamente exacta: pretende ser asimétrica, estable y
#: no degenerada, que es lo que necesitan los tests de invariancia.
CANONICAL_RIGHT_HAND_PX: Points3 = (
    (0.0, 0.0, 0.0),  # 0  WRIST
    (22.0, -14.0, -0.7),  # 1  THUMB_CMC
    (40.0, -34.0, -1.7),  # 2  THUMB_MCP
    (54.0, -52.0, -2.6),  # 3  THUMB_IP
    (66.0, -68.0, -3.4),  # 4  THUMB_TIP
    (30.0, -92.0, -4.6),  # 5  INDEX_MCP
    (36.0, -128.0, -6.4),  # 6  INDEX_PIP
    (39.0, -150.0, -7.5),  # 7  INDEX_DIP
    (41.0, -170.0, -8.5),  # 8  INDEX_TIP
    (6.0, -100.0, -5.0),  # 9  MIDDLE_MCP
    (8.0, -140.0, -7.0),  # 10 MIDDLE_PIP
    (9.0, -164.0, -8.2),  # 11 MIDDLE_DIP
    (10.0, -184.0, -9.2),  # 12 MIDDLE_TIP
    (-18.0, -96.0, -4.8),  # 13 RING_MCP
    (-24.0, -132.0, -6.6),  # 14 RING_PIP
    (-27.0, -154.0, -7.7),  # 15 RING_DIP
    (-29.0, -172.0, -8.6),  # 16 RING_TIP
    (-40.0, -84.0, -4.2),  # 17 PINKY_MCP
    (-50.0, -112.0, -5.6),  # 18 PINKY_PIP
    (-55.0, -130.0, -6.5),  # 19 PINKY_DIP
    (-58.0, -146.0, -7.3),  # 20 PINKY_TIP
)


def canonical_hand() -> Points3:
    """La mano derecha canónica, con la muñeca en el origen."""
    return CANONICAL_RIGHT_HAND_PX


def fist_hand() -> Points3:
    """Una configuración distinta (dedos flexionados), para tests que necesitan
    dos formas que no se confundan entre sí."""
    hand = canonical_hand()
    curled: list[Point3] = []
    for index, (x, y, z) in enumerate(hand):
        if index in {0, 1, 5, 9, 13, 17}:  # muñeca y nudillos: no se mueven
            curled.append((x, y, z))
        else:
            curled.append((x * 0.55, y * 0.45, z * 0.45))
    return tuple(curled)


def translated(points: Points3, dx: float, dy: float) -> Points3:
    """Traslada la mano dentro del encuadre. No cambia la seña."""
    return tuple((x + dx, y + dy, z) for x, y, z in points)


def scaled(
    points: Points3, factor: float, center: tuple[float, float] | None = None
) -> Points3:
    """Acerca o aleja la mano de la cámara. No cambia la seña.

    `factor = 2.0` equivale a la misma mano al doble de tamaño aparente, es decir,
    a la mitad de distancia.
    """
    cx, cy = center if center is not None else (points[0][0], points[0][1])
    return tuple(
        (cx + (x - cx) * factor, cy + (y - cy) * factor, z * factor)
        for x, y, z in points
    )


def rotated(
    points: Points3, degrees: float, center: tuple[float, float] | None = None
) -> Points3:
    """Inclina la muñeca. No cambia la seña.

    Rotación en el plano de la imagen alrededor de `center` (por defecto, la
    muñeca). Como `y` apunta hacia abajo, un ángulo positivo se ve en pantalla
    como un giro en sentido horario; para lo que verifican los tests da igual el
    sentido, lo que importa es que el paso 5 lo deshaga.
    """
    cx, cy = center if center is not None else (points[0][0], points[0][1])
    angle = math.radians(degrees)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotated_points: list[Point3] = []
    for x, y, z in points:
        dx = x - cx
        dy = y - cy
        rotated_points.append(
            (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a, z)
        )
    return tuple(rotated_points)


def mirrored_x(points: Points3, axis_x: float | None = None) -> Points3:
    """Espeja la mano: convierte la mano derecha canónica en su gemela izquierda.

    Se usa junto con `handedness=Handedness.LEFT`, que es lo que reportaría el
    detector ante esta geometría. El paso 2 de `feature-spec.md` debe devolver las
    features al caso derecho.
    """
    axis = axis_x if axis_x is not None else points[0][0]
    return tuple((2.0 * axis - x, y, z) for x, y, z in points)


def collapsed_scale(points: Points3) -> Points3:
    """Coloca el nudillo del dedo medio sobre la muñeca: escala degenerada.

    El paso 4 debe marcar el frame inválido en vez de dividir por ~0.
    """
    mutated = list(points)
    mutated[LandmarkIndex.MIDDLE_MCP] = (
        points[LandmarkIndex.WRIST][0],
        points[LandmarkIndex.WRIST][1],
        points[LandmarkIndex.MIDDLE_MCP][2],
    )
    return tuple(mutated)


def to_frame(
    points_px: Points3,
    *,
    width: int,
    height: int,
    handedness: Handedness = Handedness.RIGHT,
    handedness_score: float = 0.98,
    detection_score: float = 0.95,
) -> RawFrame:
    """Convierte píxeles a un `RawFrame` con las coordenadas que da MediaPipe.

    `x` se normaliza por el ancho, `y` por el alto y `z` por el ancho (MediaPipe
    entrega `z` en escala aproximada a la de `x`). Los valores fuera de `[0, 1]`
    se dejan pasar: MediaPipe extrapola fuera del encuadre y el contrato exige que
    eso no rompa nada.
    """
    if len(points_px) != NUM_LANDMARKS:
        msg = f"se esperaban {NUM_LANDMARKS} puntos, llegaron {len(points_px)}"
        raise ValueError(msg)
    landmarks = tuple(
        Landmark(x=x / width, y=y / height, z=z / width) for x, y, z in points_px
    )
    return RawFrame(
        landmarks=landmarks,
        width=width,
        height=height,
        handedness=handedness,
        handedness_score=handedness_score,
        detection_score=detection_score,
    )


def still_sequence(
    points_px: Points3,
    *,
    length: int,
    width: int = 1280,
    height: int = 720,
    handedness: Handedness = Handedness.RIGHT,
) -> Sequence:
    """Una seña estática: `length` frames idénticos.

    Es la forma que toma una letra sin movimiento en el tipo base `(T, 21, 3)`.
    """
    frame = to_frame(points_px, width=width, height=height, handedness=handedness)
    return Sequence(frames=tuple(frame for _ in range(length)))


def moving_sequence(
    points_px: Points3,
    offsets_px: tuple[tuple[float, float], ...],
    *,
    width: int = 1280,
    height: int = 720,
    handedness: Handedness = Handedness.RIGHT,
) -> Sequence:
    """Una seña dinámica: la misma configuración de mano recorriendo un trazo.

    La forma de la mano no cambia entre frames; lo único que cambia es dónde está.
    Es el caso que separa una J de una I, y el que hace visible si el canal de
    trayectoria se capturó antes de la traslación del paso 3 o después (después,
    saldría idénticamente cero).
    """
    if not offsets_px:
        raise ValueError("una secuencia con movimiento necesita al menos un offset")
    frames = tuple(
        to_frame(
            translated(points_px, dx, dy),
            width=width,
            height=height,
            handedness=handedness,
        )
        for dx, dy in offsets_px
    )
    return Sequence(frames=frames)


def arc_offsets(
    count: int, *, width_px: float = 60.0, depth_px: float = 90.0
) -> tuple[tuple[float, float], ...]:
    """Trazo en forma de gancho, como el de la J: baja y regresa en curva.

    Determinista y sin aleatoriedad: es entrada de golden vectors.
    """
    if count < 2:
        raise ValueError("un trazo necesita al menos dos puntos")
    offsets: list[tuple[float, float]] = []
    for index in range(count):
        t = index / (count - 1)
        offsets.append((-width_px * t * t, depth_px * math.sin(math.pi * t)))
    return tuple(offsets)


# --------------------------------------------------------------------------- #
# Dataset sintético (Fase 2)
# --------------------------------------------------------------------------- #
#
# La Fase 2 entrega el clasificador estático, el barrido de calibración y el
# contraste de hipótesis. Ninguna de esas tres cosas debería quedar inejecutable
# mientras no haya grabaciones: el día que el dataset exista, lo que hay que
# poder hacer es apuntar `lsm-eval` a `data/raw` y leer el reporte, no empezar
# entonces a escribir y depurar el reporte.
#
# De ahí este generador. **No se parece a LSM y no lo pretende**: produce manos
# separables y deterministas para que la tubería tenga sobre qué correr. El
# reporte que sale de él lleva un aviso que lo dice, porque una matriz de
# confusión sin procedencia acaba citada como si fuera una medición.


def _fnv1a(text: str) -> int:
    """Hash determinista de 64 bits (FNV-1a).

    `hash()` de Python va aleatorizado por proceso desde la 3.3, así que usarlo
    aquí haría que dos ejecuciones de `make eval` produjeran datasets distintos —
    y el criterio de aceptación de la fase es que el reporte sea reproducible.
    """
    value = 0xCBF29CE484222325
    for byte in text.encode("utf-8"):
        value = ((value ^ byte) * 0x100000001B3) % (1 << 64)
    return value


class _Noise:
    """Generador congruencial lineal, escrito a mano y a propósito.

    `random.Random` es determinista dentro de una versión de CPython, pero su
    contrato no promete estabilidad entre versiones. Estos números acaban en un
    reporte que se compara entre ejecuciones y entre máquinas, así que el
    generador se escribe aquí, donde se puede leer y no puede cambiar debajo.
    """

    __slots__ = ("_state",)

    _MULTIPLIER = 6364136223846793005
    _INCREMENT = 1442695040888963407
    _MODULUS = 1 << 64

    def __init__(self, seed: str) -> None:
        self._state = _fnv1a(seed)

    def next(self) -> float:
        """Siguiente valor en `[-1, 1)`."""
        self._state = (self._state * self._MULTIPLIER + self._INCREMENT) % self._MODULUS
        return ((self._state >> 11) / float(1 << 53)) * 2.0 - 1.0


#: Ancla y falanges de cada dedo: el nudillo no se mueve al flexionar, las tres
#: articulaciones distales sí. El ancla del pulgar es la CMC.
_FINGERS: Final[tuple[tuple[int, tuple[int, int, int]], ...]] = (
    (LandmarkIndex.THUMB_CMC, (2, 3, 4)),
    (LandmarkIndex.INDEX_MCP, (6, 7, 8)),
    (LandmarkIndex.MIDDLE_MCP, (10, 11, 12)),
    (LandmarkIndex.RING_MCP, (14, 15, 16)),
    (LandmarkIndex.PINKY_MCP, (18, 19, 20)),
)

#: Tres grados de flexión por dedo. 3⁵ = 243 configuraciones distintas, de sobra
#: para las 22 clases de la Fase 2.
_CURLS: Final[tuple[float, ...]] = (1.0, 0.62, 0.34)

#: Separación lateral que acompaña a cada grado de flexión. Sin ella, dos manos
#: que solo difieren en un dedo quedan demasiado cerca en ℝ⁴².
_SPREADS: Final[tuple[float, ...]] = (0.0, 9.0, -9.0)


def _bend(
    points: Points3, anchor: int, joints: tuple[int, int, int], digit: int
) -> Points3:
    """Flexiona un dedo hacia su nudillo y lo abre o cierra lateralmente."""
    curl = _CURLS[digit]
    spread = _SPREADS[digit]
    anchor_x, anchor_y, _ = points[anchor]
    mutated = list(points)
    for joint in joints:
        x, y, z = points[joint]
        mutated[joint] = (
            anchor_x + (x - anchor_x) * curl + spread,
            anchor_y + (y - anchor_y) * curl,
            z * curl,
        )
    return tuple(mutated)


def class_hand(ordinal: int) -> Points3:
    """Una configuración de mano distinta y determinista por ordinal.

    Los dígitos en base 3 del ordinal eligen el grado de flexión de cada dedo, de
    modo que dos ordinales distintos dan manos distintas por construcción y no
    por suerte. El ordinal 0 es la mano canónica.

    Los nudillos no se mueven, y en particular tampoco el 9: la distancia
    muñeca → nudillo del dedo medio es la unidad de escala del paso 4 de
    `feature-spec.md`, y moverla haría que dos clases se distinguieran por su
    tamaño aparente, que es justo lo que la normalización borra.
    """
    if ordinal < 0:
        raise ValueError(f"el ordinal de clase no puede ser negativo: {ordinal}")
    points = canonical_hand()
    resto = ordinal
    for anchor, joints in _FINGERS:
        points = _bend(points, anchor, joints, resto % len(_CURLS))
        resto //= len(_CURLS)
    if resto:
        msg = f"ordinal {ordinal} fuera del alcance del generador (máximo 242)"
        raise ValueError(msg)
    return points


def _signer_hand(points: Points3, signer: int) -> Points3:
    """Anatomía y estilo de una persona: mano más grande o más pequeña, muñeca
    más o menos inclinada, dedos que se cierran un poco más o un poco menos.

    Es lo que hace que leave-one-signer-out mida algo. Sin esta variación el fold
    de prueba sería una copia exacta del de entrenamiento, el accuracy saldría del
    100% y no diría nada.
    """
    ruido = _Noise(f"firmante:{signer}")
    inclinacion = 18.0 * ruido.next()
    tamano = 0.80 + 0.20 * (ruido.next() + 1.0)
    sesgos = tuple(1.0 + 0.18 * ruido.next() for _ in _FINGERS)

    ajustados = points
    for (anchor, joints), factor in zip(_FINGERS, sesgos, strict=True):
        anchor_x, anchor_y, _ = ajustados[anchor]
        mutados = list(ajustados)
        for joint in joints:
            x, y, z = ajustados[joint]
            mutados[joint] = (
                anchor_x + (x - anchor_x) * factor,
                anchor_y + (y - anchor_y) * factor,
                z,
            )
        ajustados = tuple(mutados)
    return scaled(rotated(ajustados, inclinacion), tamano)


def synthetic_samples(
    labels: tuple[str, ...],
    *,
    signers: int = 3,
    sessions: int = 2,
    repetitions: int = 6,
    frames: int = 8,
    jitter_px: float = 0.9,
    style_jitter: float = 0.16,
    width: int = 1280,
    height: int = 720,
) -> tuple[Sample, ...]:
    """Un dataset etiquetado completo, con metadatos y sin tocar una cámara.

    La configuración de mano de cada clase sale de su **posición en `labels`**, no
    de su nombre: dos llamadas con la misma tupla dan las mismas manos, y con
    tuplas distintas no tienen por qué. Quien lo consuma debe pasar siempre el
    mismo vocabulario, ordenado.

    El jitter por frame no es adorno: sin él σ sale exactamente cero y
    `quality.max_dispersion` no se podría calibrar contra nada.
    """
    if not labels:
        raise ValueError("un dataset necesita al menos una etiqueta")
    for nombre, valor in (
        ("signers", signers),
        ("sessions", sessions),
        ("repetitions", repetitions),
    ):
        if valor < 1:
            raise ValueError(f"{nombre} debe ser ≥ 1, no {valor}")
    if frames < 2:
        raise ValueError(f"una muestra estática necesita ≥ 2 frames, no {frames}")

    muestras: list[Sample] = []
    for ordinal, label in enumerate(labels):
        base = class_hand(ordinal)
        for signer in range(signers):
            mano = _signer_hand(base, signer)
            signer_id = f"sint{signer:02d}"
            for session in range(sessions):
                session_id = f"{signer_id}-s{session:02d}"
                for repeticion in range(repetitions):
                    ruido = _Noise(f"{label}|{signer}|{session}|{repeticion}")
                    # Nadie hace dos veces exactamente la misma seña. Sin esta
                    # variación por repetición, la única diferencia dentro de una
                    # clase sería el jitter por landmark, que la normalización
                    # casi borra: cada clase quedaría en un punto y no en una
                    # nube, leave-one-signer-out daría 100% siempre y el barrido
                    # no tendría nada que optimizar.
                    ejecutada = mano
                    for anchor, joints in _FINGERS:
                        factor = 1.0 + style_jitter * ruido.next()
                        anchor_x, anchor_y, _ = ejecutada[anchor]
                        mutados = list(ejecutada)
                        for joint in joints:
                            x, y, z = ejecutada[joint]
                            mutados[joint] = (
                                anchor_x + (x - anchor_x) * factor,
                                anchor_y + (y - anchor_y) * factor,
                                z,
                            )
                        ejecutada = tuple(mutados)
                    colocada = translated(
                        rotated(ejecutada, 6.0 * ruido.next()),
                        480.0 + 120.0 * ruido.next(),
                        420.0 + 90.0 * ruido.next(),
                    )
                    secuencia = Sequence(
                        frames=tuple(
                            to_frame(
                                tuple(
                                    (
                                        x + jitter_px * ruido.next(),
                                        y + jitter_px * ruido.next(),
                                        z,
                                    )
                                    for x, y, z in colocada
                                ),
                                width=width,
                                height=height,
                            )
                            for _ in range(frames)
                        )
                    )
                    muestras.append(
                        Sample(
                            sequence=secuencia,
                            label=label,
                            signer_id=signer_id,
                            session_id=session_id,
                            timestamp=_SYNTHETIC_EPOCH
                            + timedelta(minutes=len(muestras)),
                            handedness=Handedness.RIGHT,
                            light_level=LightLevel.INDOOR,
                            light_direction=LightDirection.FRONTAL,
                            distance=Distance.MEDIUM,
                            mean_luminance=0.35 + 0.05 * signer,
                            mean_scale_px=100.0 + 6.0 * signer,
                            kind=SampleKind.STATIC,
                        )
                    )
    return tuple(muestras)
