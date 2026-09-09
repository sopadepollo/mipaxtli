"""Tipos base del traductor de deletreo manual de LSM.

Regla no negociable (`CLAUDE.md`, `ARQUITECTURA.md` §2): **el tipo base de entrada
es una secuencia temporal `(T, 21, 3)`, nunca un frame suelto.** Una seña estática
es una secuencia corta y estable; una dinámica es una secuencia que varía. Ambas
recorren la misma tubería.

Por eso este módulo no expone ningún tipo pensado para viajar solo por la API
pública en lugar de una secuencia: `RawFrame` existe para poblar una `Sequence`,
no para clasificarse por su cuenta.

Este módulo es código puro: sin OpenCV, sin MediaPipe, sin disco, sin cámara.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum, StrEnum
from typing import Final, TypeAlias

# --------------------------------------------------------------------------- #
# Constantes estructurales
# --------------------------------------------------------------------------- #

#: Landmarks que entrega MediaPipe Hands por mano (`feature-spec.md` §0.1).
NUM_LANDMARKS: Final = 21

#: Dimensión del vector aplanado del paso 7: 21 landmarks × (x, y), sin z.
NUM_FEATURES: Final = 42

#: Etiqueta reservada para la clase de rechazo (`ARQUITECTURA.md` §4.4).
UNKNOWN_LABEL: Final = "UNKNOWN"

#: Etiqueta de la clase negativa explícita del dataset (`glosario-lsm.md` §4).
NEGATIVE_LABEL: Final = "NONE"


class LandmarkIndex(IntEnum):
    """Índices de los 21 landmarks de MediaPipe Hands.

    El orden es normativo: `feature-spec.md` §0.1 y el paso 7 dependen de él, y la
    reimplementación en TypeScript debe usar exactamente la misma numeración.
    """

    WRIST = 0

    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4

    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8

    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12

    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16

    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


class Handedness(StrEnum):
    """Lateralidad reportada por el detector, *antes* de canonizar.

    Vale la de la mano real: el frame que se alimenta al detector nunca va
    espejado (`feature-spec.md` §0.3).
    """

    LEFT = "LEFT"
    RIGHT = "RIGHT"


class HandednessConvention(StrEnum):
    """Qué mano nombra el campo `handedness`. **No es una preferencia: es un
    acuerdo entre implementaciones.**

    El detector real observa una mano y tiene que ponerle nombre, y hay dos formas
    razonables de hacerlo que se diferencian en un espejo:

    - `SIGNER` — la mano **anatómica** de quien firma. Si levanta su derecha, dice
      `RIGHT`.
    - `IMAGE` — la mano tal como aparece en la imagen espejada, que es la
      convención de selfie con la que MediaPipe decide la lateralidad.

    Este proyecto usa `SIGNER` (`HANDEDNESS_CONVENTION`). Traducir de una a otra es
    trabajo del adaptador del detector, no de la especificación de features: ver
    `hands.mediapipe_reports_mirrored_handedness` y
    `docs/adr/0006-deteccion-de-manos-y-captura.md`.

    ### Por qué esto es un tipo y no un comentario

    Elegir la convención equivocada **no rompe nada de forma observable**. El paso 2
    de `feature-spec.md` espeja en X según este valor, así que con la convención
    contraria *todas* las muestras se canonizan hacia la otra mano: las dos
    poblaciones de vectores difieren por un espejo global y nada más. El modelo
    entrena igual de bien, infiere igual de bien y la precisión es idéntica.

    El error solo duele cuando **dos implementaciones no coinciden**, que es
    exactamente lo que pasará en la Fase 7 cuando MediaPipe JS traiga su propia
    convención. Y los golden vectors no lo detectan: reciben la lateralidad ya
    resuelta como entrada, así que el test de paridad pasaría en verde mientras la
    app web confunde cada seña con su espejo.

    De ahí que la convención viaje **pegada al modelo exportado** y se rechace al
    cargar si no coincide, igual que `feature_spec_version`
    (`classifiers/base.py`). Es el único mecanismo que fuerza el acuerdo.
    """

    #: La mano anatómica de quien firma.
    SIGNER = "SIGNER"
    #: La mano tal como se ve en la imagen espejada (convención de selfie).
    IMAGE = "IMAGE"


#: La convención del proyecto. Cambiarla invalida todos los modelos exportados y
#: obliga a re-canonizar el dataset, no a regrabarlo: los landmarks crudos no
#: dependen de esto, solo su etiqueta de lateralidad.
HANDEDNESS_CONVENTION: Final = HandednessConvention.SIGNER


class LightLevel(StrEnum):
    """Cuánta luz hay (`ARQUITECTURA.md` §4.7).

    Es un eje **independiente** de la dirección: una escena a contraluz puede ser
    brillante u oscura, y son problemas distintos para el detector. Mezclar ambos
    ejes en un solo campo —como hacía la taxonomía original, con `BACKLIT` al lado
    de `DIM`— obliga a elegir cuál de los dos se anota y pierde el otro.

    Categoría subjetiva: la anota a mano quien graba. El número objetivo que la
    acompaña es `Sample.mean_luminance`.
    """

    DIM = "DIM"
    INDOOR = "INDOOR"
    BRIGHT = "BRIGHT"


class LightDirection(StrEnum):
    """De dónde viene la luz respecto de quien firma (`ARQUITECTURA.md` §4.7).

    Importa porque el contraluz es el caso que más degrada la detección de manos:
    la silueta se recorta contra el fondo y los landmarks bailan.
    """

    #: Luz principal delante de quien firma, hacia la cámara.
    FRONTAL = "FRONTAL"
    #: Luz principal de costado. Media mano iluminada, media en sombra.
    LATERAL = "LATERAL"
    #: Luz detrás de quien firma: ventana al fondo, lámpara a la espalda.
    BACKLIT = "BACKLIT"
    #: Varias fuentes de direcciones distintas, o cambiante durante la sesión.
    MIXED = "MIXED"


class Distance(StrEnum):
    """Distancia aproximada de la mano a la cámara (`ARQUITECTURA.md` §4.7).

    Etiqueta gruesa para filtrar el dataset a ojo. El número para el análisis serio
    es `Sample.mean_scale_px`: `NEAR/MEDIUM/FAR` no es más que una discretización
    pobre de una magnitud que el paso 4 ya calcula.
    """

    NEAR = "NEAR"
    MEDIUM = "MEDIUM"
    FAR = "FAR"


class InvalidReason(StrEnum):
    """Por qué un frame no puede entrar a la tubería.

    Es un tipo cerrado y no `None`: un hueco silencioso se propaga sin dejar
    rastro. El motivo importa porque cambia lo que hace la máquina de estados y
    lo que se escribe en los golden vectors.
    """

    #: El detector no encontró ninguna mano en el frame.
    NO_HAND = "NO_HAND"
    #: `detection_score` por debajo del mínimo configurado.
    LOW_DETECTION_SCORE = "LOW_DETECTION_SCORE"
    #: Paso 4: la norma de p_9 quedó por debajo de MIN_SCALE.
    SCALE_TOO_SMALL = "SCALE_TOO_SMALL"


# --------------------------------------------------------------------------- #
# Puntos y frames
# --------------------------------------------------------------------------- #

#: Punto 3D intermedio de la tubería. Se usa `tuple` y no una clase porque los
#: pasos de `feature-spec.md` §1 son aritmética pura y el contrato prohíbe
#: reordenar operaciones: cuanta menos indirección, más fácil de auditar.
Point3: TypeAlias = tuple[float, float, float]
Points3: TypeAlias = tuple[Point3, ...]
Point2: TypeAlias = tuple[float, float]
Points2: TypeAlias = tuple[Point2, ...]


@dataclass(frozen=True, slots=True)
class Landmark:
    """Un landmark crudo de MediaPipe, en coordenadas normalizadas al frame.

    `x` va normalizado por el ancho, `y` por el alto y con el eje apuntando hacia
    abajo (convención de imagen), `z` es profundidad relativa a la muñeca. Los
    valores pueden salirse de `[0, 1]`: MediaPipe extrapola fuera del encuadre y
    eso no es un error.
    """

    x: float
    y: float
    z: float

    def as_tuple(self) -> Point3:
        return (self.x, self.y, self.z)


@dataclass(frozen=True, slots=True)
class RawFrame:
    """Un frame válido: exactamente una mano, con sus metadatos de captura.

    Si se detectaron varias manos, quien construye el frame ya se quedó con la de
    mayor `detection_score` (`feature-spec.md` §0.3): el alfabeto dactilológico de
    LSM es monomanual.
    """

    landmarks: tuple[Landmark, ...]
    width: int
    height: int
    handedness: Handedness
    handedness_score: float
    detection_score: float

    def __post_init__(self) -> None:
        if len(self.landmarks) != NUM_LANDMARKS:
            msg = f"un frame lleva {NUM_LANDMARKS} landmarks, no {len(self.landmarks)}"
            raise ValueError(msg)
        if self.width <= 0 or self.height <= 0:
            msg = f"dimensiones de frame inválidas: {self.width}x{self.height}"
            raise ValueError(msg)
        for name, score in (
            ("handedness_score", self.handedness_score),
            ("detection_score", self.detection_score),
        ):
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"{name} fuera de [0, 1]: {score}")

    @property
    def aspect_ratio(self) -> float:
        """`a = width / height`, el factor del paso 1 de `feature-spec.md` §1."""
        return self.width / self.height

    def points(self) -> Points3:
        """Landmarks como tuplas, en el orden de `LandmarkIndex`."""
        return tuple(landmark.as_tuple() for landmark in self.landmarks)


@dataclass(frozen=True, slots=True)
class InvalidFrame:
    """Marcador explícito de frame inválido.

    No se usa `None`: `feature-spec.md` §0.3 exige que los frames inválidos
    **interrumpan** la secuencia en vez de interpolarse, y para eso hay que poder
    verlos en el flujo.
    """

    reason: InvalidReason
    detail: str = ""


#: Lo que produce el detector para un frame de video: mano válida o marcador.
FrameSlot: TypeAlias = RawFrame | InvalidFrame

#: Flujo crudo de frames tal como sale del detector, huecos incluidos.
FrameStream: TypeAlias = tuple[FrameSlot, ...]


@dataclass(frozen=True, slots=True)
class Sequence:
    """El tipo base del sistema: `(T, 21, 3)` frames válidos y consecutivos.

    Una `Sequence` no contiene huecos por construcción. Partir un `FrameStream`
    con marcadores inválidos en sus secuencias válidas máximas es trabajo de
    `lsm.features.split_valid_runs`.
    """

    frames: tuple[RawFrame, ...]

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("una Sequence necesita al menos un frame válido")

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def shape(self) -> tuple[int, int, int]:
        """`(T, 21, 3)`."""
        return (len(self.frames), NUM_LANDMARKS, 3)

    def window(self, start: int, stop: int) -> Sequence:
        """Sub-secuencia `[start, stop)`. Devuelve `Sequence`, nunca un frame."""
        return Sequence(frames=self.frames[start:stop])


# --------------------------------------------------------------------------- #
# Salidas de la tubería
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """Vector aplanado de 42 componentes del paso 7 de `feature-spec.md` §1.

    Lleva pegada su `spec_version`: un modelo entrenado con otra versión de la
    especificación debe rechazarse al cargarse, no ejecutarse en silencio
    (`ARQUITECTURA.md` §4.6).
    """

    values: tuple[float, ...]
    spec_version: int

    def __post_init__(self) -> None:
        if len(self.values) != NUM_FEATURES:
            msg = (
                f"el vector de features lleva {NUM_FEATURES} componentes, "
                f"no {len(self.values)}"
            )
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.values)


@dataclass(frozen=True, slots=True)
class TrajectoryChannel:
    """Canal de trayectoria de `feature-spec.md` §3.1.

    Es la razón de ser del §3: el paso 3 destruye a propósito la posición de la
    mano en el encuadre, y sin este canal una J y una I son indistinguibles.
    `points[0]` es siempre `(0.0, 0.0)`: el origen es la muñeca del primer frame.
    """

    points: Points2
    mean_scale: float

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("el canal de trayectoria necesita al menos un punto")
        if self.mean_scale <= 0.0:
            raise ValueError(f"escala media no positiva: {self.mean_scale}")

    def __len__(self) -> int:
        return len(self.points)


@dataclass(frozen=True, slots=True)
class Prediction:
    """Resultado de un clasificador: etiqueta y confianza.

    Un clasificador de 27 clases siempre devuelve una de las 27, aunque la persona
    se esté rascando la nariz. Por eso `UNKNOWN` es un valor de primera clase y no
    la ausencia de resultado (`ARQUITECTURA.md` §4.4).
    """

    label: str
    confidence: float

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("una Prediction necesita etiqueta")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confianza fuera de [0, 1]: {self.confidence}")

    @property
    def is_unknown(self) -> bool:
        return self.label == UNKNOWN_LABEL

    @classmethod
    def unknown(cls, confidence: float = 0.0) -> Prediction:
        return cls(label=UNKNOWN_LABEL, confidence=confidence)


class SampleKind(StrEnum):
    """Cómo se grabó la muestra.

    No es lo mismo que `LetterSpec.es_dinamica`, aunque casi siempre coincidan:
    aquello dice qué **es** la letra según el glosario, esto dice qué se **hizo**
    frente a la cámara. Los dos casos en que se separan son reales: la clase
    negativa `NONE` se graba de las dos formas —mano relajada y quieta, mano en
    tránsito— y una letra estática grabada por error en modo dinámico tiene que
    poder reconocerse como tal al depurar el dataset.
    """

    #: Configuración sostenida: se exige quietud, medida con la σ del §2.
    STATIC = "STATIC"
    #: Recorrido: el movimiento **es** la seña, así que no se exige quietud.
    DYNAMIC = "DYNAMIC"


@dataclass(frozen=True, slots=True)
class Sample:
    """Una secuencia etiquetada con sus metadatos (`ARQUITECTURA.md` §4.7).

    Los metadatos no son decorativos: la validación es leave-one-signer-out y sin
    `signer_id` ni `session_id` no se puede construir el split. Un split aleatorio
    de frames mezcla frames de la misma grabación entre train y test y da métricas
    infladas.

    Se guarda la secuencia **cruda**, no las features: si cambia la normalización
    se re-deriva sin volver a grabar.

    Las condiciones de captura van por duplicado, categoría y número:
    `light_level`/`mean_luminance` y `distance`/`mean_scale_px`. Las categorías
    dependen del juicio de quien graba y dos personas etiquetarán distinto la misma
    escena; los números no. Se conservan ambas porque sirven para cosas distintas:
    la categoría para filtrar el dataset a ojo, el número para responder si el
    modelo empeora con poca luz o a distancia, que es la pregunta que de verdad se
    hará al mirar la matriz de confusión.
    """

    sequence: Sequence
    label: str
    signer_id: str
    session_id: str
    timestamp: datetime
    handedness: Handedness
    light_level: LightLevel
    light_direction: LightDirection
    distance: Distance
    #: Luminancia media del frame, en `[0, 1]`, promediada sobre la secuencia.
    #: Es la medida **objetiva** que acompaña a `light_level`: dos personas
    #: etiquetan distinto la misma escena, pero el número no opina.
    mean_luminance: float
    #: Escala del paso 4 en píxeles, promediada sobre la secuencia. Es la medida
    #: objetiva que acompaña a `distance`, y sale gratis: la tubería ya la calcula
    #: para normalizar. Ver `lsm.features.scale_to_pixels`.
    mean_scale_px: float
    #: Qué se **hizo** frente a la cámara, que no siempre es lo que el glosario
    #: dice que la letra **es**. `to_sample()` lo arrastraba hasta que la Fase 2
    #: lo necesitó: `NONE` se graba de las dos formas y un clasificador estático
    #: no puede entrenar con las dinámicas. Ver
    #: `docs/adr/0010-la-clase-negativa-y-el-modo-de-grabacion.md`.
    kind: SampleKind

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("una Sample necesita etiqueta")
        for name, value in (
            ("signer_id", self.signer_id),
            ("session_id", self.session_id),
        ):
            if not value:
                raise ValueError(f"{name} vacío: rompe leave-one-signer-out")
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp debe llevar zona horaria (ISO-8601 con offset)")
        if not 0.0 <= self.mean_luminance <= 1.0:
            raise ValueError(f"mean_luminance fuera de [0, 1]: {self.mean_luminance}")
        if self.mean_scale_px <= 0.0:
            raise ValueError(f"mean_scale_px no positiva: {self.mean_scale_px}")


# --------------------------------------------------------------------------- #
# Topología de la mano
# --------------------------------------------------------------------------- #

#: Pares de landmarks unidos por un hueso, para dibujar el esqueleto de la mano.
#:
#: Vive aquí y no en la capa de dibujo porque es **estructura**, no presentación:
#: es la misma topología que consumirá el preview de escritorio, el de la app web
#: y cualquier figura de la documentación. Duplicarla en cada consumidor es la
#: forma conocida de acabar con tres esqueletos ligeramente distintos.
#:
#: El orden dentro de cada par va de la articulación proximal a la distal, y los
#: grupos van en el orden de `LandmarkIndex`: palma primero, luego cada dedo.
HAND_CONNECTIONS: Final[tuple[tuple[LandmarkIndex, LandmarkIndex], ...]] = (
    # Palma: muñeca a los nudillos, y los nudillos entre sí.
    (LandmarkIndex.WRIST, LandmarkIndex.THUMB_CMC),
    (LandmarkIndex.WRIST, LandmarkIndex.INDEX_MCP),
    (LandmarkIndex.WRIST, LandmarkIndex.PINKY_MCP),
    (LandmarkIndex.INDEX_MCP, LandmarkIndex.MIDDLE_MCP),
    (LandmarkIndex.MIDDLE_MCP, LandmarkIndex.RING_MCP),
    (LandmarkIndex.RING_MCP, LandmarkIndex.PINKY_MCP),
    # Pulgar.
    (LandmarkIndex.THUMB_CMC, LandmarkIndex.THUMB_MCP),
    (LandmarkIndex.THUMB_MCP, LandmarkIndex.THUMB_IP),
    (LandmarkIndex.THUMB_IP, LandmarkIndex.THUMB_TIP),
    # Índice.
    (LandmarkIndex.INDEX_MCP, LandmarkIndex.INDEX_PIP),
    (LandmarkIndex.INDEX_PIP, LandmarkIndex.INDEX_DIP),
    (LandmarkIndex.INDEX_DIP, LandmarkIndex.INDEX_TIP),
    # Medio.
    (LandmarkIndex.MIDDLE_MCP, LandmarkIndex.MIDDLE_PIP),
    (LandmarkIndex.MIDDLE_PIP, LandmarkIndex.MIDDLE_DIP),
    (LandmarkIndex.MIDDLE_DIP, LandmarkIndex.MIDDLE_TIP),
    # Anular.
    (LandmarkIndex.RING_MCP, LandmarkIndex.RING_PIP),
    (LandmarkIndex.RING_PIP, LandmarkIndex.RING_DIP),
    (LandmarkIndex.RING_DIP, LandmarkIndex.RING_TIP),
    # Meñique.
    (LandmarkIndex.PINKY_MCP, LandmarkIndex.PINKY_PIP),
    (LandmarkIndex.PINKY_PIP, LandmarkIndex.PINKY_DIP),
    (LandmarkIndex.PINKY_DIP, LandmarkIndex.PINKY_TIP),
)
