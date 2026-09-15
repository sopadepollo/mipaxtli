"""Dirección texto → señas (`ARQUITECTURA.md` §4.10, Fase 4).

Aquí no hay IA. Hay un manifest que dice qué archivo muestra cada letra y de
dónde salió, una conversión de texto a símbolos del glosario, y un reproductor
que avanza por ticks. Todo puro: sin disco, sin Pillow, sin OpenCV, sin `time`
(`CLAUDE.md` regla 2). El reloj entra por `Tick(dt_ms)`, como en `telemetry.py`.

**Ningún asset viene de internet.** `glosario-lsm.md` §2 lo dice sin rodeos:
casi todo lo que circula como "abecedario en lengua de señas" es ASL, y un
proyecto que presente ASL como LSM queda descalificado ante cualquier persona
usuaria. Cada asset se dibuja a partir de una muestra de `data/raw`, grabada
siguiendo una página concreta de *Manos con voz*, y el manifest apunta a esa
muestra. `manifest_drift` exige además que lo que el manifest dice de cada letra
sea lo que dice `vocabulary.py`, que a su vez transcribe el glosario.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, replace
from datetime import date
from enum import StrEnum
from typing import Any, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lsm.config import Config
from lsm.features import (
    DynamicUnavailable,
    ExtractionRejected,
    extract_sequence_features,
)
from lsm.types import Handedness, Point2, RawFrame, Sample, SampleKind
from lsm.vocabulary import LETTERS, Label

# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #

#: Versión del esquema de `assets/signs/manifest.json`. Un manifest de otra
#: versión se rechaza al cargar, como un modelo con otra `feature_spec_version`.
MANIFEST_SCHEMA_VERSION: Final = 1

#: La fuente contra la que se grabó el dataset y, por tanto, cada asset.
FUENTE_NORMATIVA: Final = (
    "Serafín de Fleischmann, M. E. y González Pérez, R. Manos con voz. "
    "Diccionario de Lengua de Señas Mexicana. CONAPRED / Libre Acceso A.C. "
    "Abecedario: pp. 15-19."
)

#: Único origen hoy. Una foto verificada sería otro valor, no otro esquema.
SourceKind: TypeAlias = Literal["esqueleto_desde_dataset"]

#: Resultado de mirar el asset junto a la descripción del glosario.
ReviewResult: TypeAlias = Literal["coincide", "difiere", "pendiente"]

#: `<firmante>/<sesion>/<LABEL>/<NNN>.json`, relativo a la raíz del dataset.
_SAMPLE_PATH: Final = re.compile(
    r"^(?P<signer>[A-Za-z0-9_-]+)/(?P<session>[A-Za-z0-9_-]+)/"
    r"(?P<label>[A-Z_]+)/[0-9]{3}\.json$"
)


class _Model(BaseModel):
    """Inmutable y sin campos desconocidos, como las secciones de `config.py`."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class AssetSource(_Model):
    """De dónde salió el dibujo: la trazabilidad "LSM y no ASL"."""

    tipo: SourceKind
    #: Ruta de la muestra relativa a la raíz del dataset.
    muestra: str
    signer_id: str
    #: Mano con la que se grabó la muestra.
    lateralidad_original: Handedness
    #: `True` si se espejó en X para mostrar mano derecha, como el diccionario.
    espejada: bool

    @field_validator("muestra")
    @classmethod
    def _con_la_forma_del_dataset(cls, value: str) -> str:
        if _SAMPLE_PATH.match(value) is None:
            msg = f"muestra {value!r} no tiene la forma firmante/sesion/LETRA/NNN.json"
            raise ValueError(msg)
        return value

    @property
    def label_de_la_muestra(self) -> str:
        match = _SAMPLE_PATH.match(self.muestra)
        assert match is not None  # el validador ya lo garantizó
        return match.group("label")


class AssetReview(_Model):
    """La revisión humana del asset contra la descripción del glosario.

    No es la validación por persona usuaria de LSM que pide `ARQUITECTURA.md`
    §4.11 —esa sigue siendo el PENDIENTE-HUMANO G del glosario—; es la anotación
    de que alguien miró el dibujo con la descripción al lado y dijo si coincide.
    """

    fecha: date
    revisor: str
    resultado: ReviewResult
    nota: str = ""


class SignAsset(_Model):
    """Una letra del manifest."""

    #: Cómo se escribe: `Ñ`, `LL`, `RR`. Lo que se muestra en pantalla.
    letra: str
    #: `<LABEL>.png` o `<LABEL>.gif`. Lo fija la etiqueta; ver `expected_filename`.
    archivo: str
    es_dinamica: bool
    descripcion: str
    #: `NO_TRAJECTORY` si es estática.
    trayectoria: str
    #: Página de *Manos con voz* donde se verificó la letra.
    pagina: int = Field(ge=1)
    #: Duración de una vuelta del GIF. Obligatoria en dinámicas; `None` en
    #: estáticas, cuya duración en pantalla la decide `config.signs`.
    duracion_ms: int | None
    fuente: AssetSource
    revision: AssetReview

    @model_validator(mode="after")
    def _el_movimiento_va_en_gif(self) -> SignAsset:
        extension = ".gif" if self.es_dinamica else ".png"
        if not self.archivo.endswith(extension):
            clase = "dinámica" if self.es_dinamica else "estática"
            msg = (
                f"{self.letra}: una letra {clase} va en {extension[1:]}, "
                f"no {self.archivo!r}"
            )
            raise ValueError(msg)
        if self.es_dinamica and (self.duracion_ms is None or self.duracion_ms <= 0):
            msg = f"{self.letra}: una dinámica necesita duracion_ms > 0"
            raise ValueError(msg)
        if not self.es_dinamica and self.duracion_ms is not None:
            msg = (
                f"{self.letra}: una estática no lleva duracion_ms; la fija config.signs"
            )
            raise ValueError(msg)
        return self


class Manifest(_Model):
    """`assets/signs/manifest.json` entero."""

    schema_version: int
    fuente_normativa: str
    letras: dict[Label, SignAsset]

    @field_validator("schema_version")
    @classmethod
    def _de_esta_version(cls, value: int) -> int:
        if value != MANIFEST_SCHEMA_VERSION:
            msg = (
                f"schema_version {value} incompatible; este código lee la "
                f"versión {MANIFEST_SCHEMA_VERSION}"
            )
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _cada_entrada_es_de_su_letra(self) -> Manifest:
        if Label.NONE in self.letras:
            raise ValueError("NONE no es una letra y no tiene asset")
        for label, asset in self.letras.items():
            esperado = expected_filename(label)
            if asset.archivo != esperado:
                msg = (
                    f"{label}: el archivo tiene que ser {esperado}, "
                    f"no {asset.archivo!r}"
                )
                raise ValueError(msg)
            if asset.fuente.label_de_la_muestra != label:
                msg = (
                    f"{label}: la muestra de origen {asset.fuente.muestra!r} "
                    "es de otra letra"
                )
                raise ValueError(msg)
        return self


def expected_filename(label: Label) -> str:
    """El nombre del asset lo fija la etiqueta: `A.png`, `J.gif`, `DOBLE_L.gif`."""
    extension = "gif" if LETTERS[label].es_dinamica else "png"
    return f"{label}.{extension}"


def manifest_drift(manifest: Manifest) -> list[str]:
    """Discrepancias entre el manifest y `vocabulary.py`. Vacía si coinciden.

    Los campos descriptivos se **copian** al manifest y no se referencian, para
    que la Fase 7 pueda consumirlo desde JavaScript sin `vocabulary.py`. Dos
    copias son dos verdades salvo que algo las compare; esto es ese algo, y el
    test del manifest real lo exige vacío.
    """
    deriva: list[str] = []
    for label, letra in LETTERS.items():
        asset = manifest.letras.get(label)
        if asset is None:
            deriva.append(f"{label}: falta en el manifest")
            continue
        comparaciones = (
            ("letra", asset.letra, letra.display),
            ("es_dinamica", asset.es_dinamica, letra.es_dinamica),
            ("descripcion", asset.descripcion, letra.descripcion),
            ("trayectoria", asset.trayectoria, letra.trayectoria),
            ("pagina", asset.pagina, letra.pagina),
        )
        for campo, en_manifest, en_glosario in comparaciones:
            if en_manifest != en_glosario:
                deriva.append(
                    f"{label}: {campo} difiere del glosario "
                    f"({en_manifest!r} frente a {en_glosario!r})"
                )
    for label in manifest.letras:
        if label not in LETTERS:
            deriva.append(f"{label}: sobra, no está en el glosario")
    return deriva


# --------------------------------------------------------------------------- #
# Texto → símbolos
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class WordGap:
    """Un espacio del texto: pausa entre palabras."""


Token: TypeAlias = Label | WordGap

#: Dígrafos del glosario, en el orden en que se prueban. Voraz y de izquierda a
#: derecha: en ortografía española `ll` y `rr` son siempre dígrafos, y es lo
#: simétrico a `spelling.py`, donde `DOBLE_L` es un símbolo y no dos.
_DIGRAPHS: Final[dict[str, Label]] = {"LL": Label.DOBLE_L, "RR": Label.DOBLE_R}

#: Cómo se lee cada símbolo en pantalla.
_GAP_DISPLAY: Final = "·"


class UnsupportedCharacters(ValueError):  # noqa: N818
    """El texto tiene caracteres sin seña en el glosario. Nada se salta en silencio."""

    def __init__(self, chars: tuple[str, ...]) -> None:
        self.chars = chars
        listado = ", ".join(repr(char) for char in chars)
        super().__init__(f"caracteres sin seña en el glosario: {listado}")


def _base_letter(char: str) -> str:
    """`á` → `A`, `ü` → `U`, `ñ` → `Ñ`; lo demás, tal cual en mayúsculas."""
    if char in "ñÑ":
        return "Ñ"
    decomposed = unicodedata.normalize("NFD", char)
    return decomposed[0].upper()


def text_to_symbols(text: str) -> tuple[Token, ...]:
    """Convierte texto escrito en símbolos del glosario.

    Normaliza (`§6.1` del spec): `ñ` se protege antes de quitar acentos, los
    dígrafos se toman vorazmente, `ch` son dos letras porque el glosario no
    tiene CH, los espacios separan palabras y cualquier otro carácter es un
    error con la lista completa de ofensores. Un carácter que se expande al
    pasar a mayúsculas (e.g., ß → SS) se rechaza como unsupported.
    """
    tokens: list[Token] = []
    unsupported: list[str] = []
    for palabra in unicodedata.normalize("NFC", text).split():
        if tokens:
            tokens.append(WordGap())
        # Map each character and track its origin in case it expands (e.g., ß → SS).
        mapped_chars: list[str] = []
        char_origins: list[str] = []  # original character for each mapped character
        for char in palabra:
            mapped = _base_letter(char)
            # Reject characters that expand to multiple characters or are not valid
            # letters.
            if len(mapped) != 1:
                if char not in unsupported:
                    unsupported.append(char)
                continue
            m = mapped[0]
            if m in "ABCDEFGHIJKLMNOPQRSTUVWXYZÑ":
                mapped_chars.append(m)
                char_origins.append(char)
            else:
                if char not in unsupported:
                    unsupported.append(char)
        letras = "".join(mapped_chars)
        i = 0
        while i < len(letras):
            digrafo = _DIGRAPHS.get(letras[i : i + 2])
            if digrafo is not None:
                tokens.append(digrafo)
                i += 2
                continue
            char = letras[i]
            if char == "Ñ":
                tokens.append(Label.ENIE)
            elif "A" <= char <= "Z":
                tokens.append(Label(char))
            else:
                # This shouldn't happen as we already filter above.
                original = char_origins[i]
                if original not in unsupported:
                    unsupported.append(original)
            i += 1
    if unsupported:
        raise UnsupportedCharacters(tuple(unsupported))
    return tuple(tokens)


def render_tokens(tokens: SequenceABC[Token]) -> str:
    """`A Ñ O · LL`: cada símbolo como se escribe, los espacios como `·`."""
    partes = [
        _GAP_DISPLAY if isinstance(token, WordGap) else LETTERS[token].display
        for token in tokens
    ]
    return " ".join(partes)


# --------------------------------------------------------------------------- #
# Lista de pasos
# --------------------------------------------------------------------------- #


class StepKind(StrEnum):
    LETTER = "LETTER"
    GAP = "GAP"


@dataclass(frozen=True, slots=True)
class Step:
    """Un paso de la reproducción: una letra sostenida o una pausa."""

    kind: StepKind
    #: `None` si es una pausa.
    label: Label | None
    #: A velocidad 1.0. El reproductor la divide por la velocidad al comparar.
    duration_ms: float


def build_playlist(
    tokens: SequenceABC[Token], manifest: Manifest, config: Config
) -> tuple[Step, ...]:
    """Símbolos → pasos con duración.

    Estática: `signs.static_hold_ms`. Dinámica: la vuelta del GIF que dice el
    manifest por `signs.dynamic_loops`. Pausa: `signs.word_gap_ms`.
    """
    pasos: list[Step] = []
    for token in tokens:
        if isinstance(token, WordGap):
            pasos.append(Step(StepKind.GAP, None, config.signs.word_gap_ms))
            continue
        asset = manifest.letras.get(token)
        if asset is None:
            raise ValueError(f"el manifest no tiene asset para {token}")
        if asset.es_dinamica:
            assert asset.duracion_ms is not None  # lo garantiza el esquema
            duracion = float(asset.duracion_ms * config.signs.dynamic_loops)
        else:
            duracion = config.signs.static_hold_ms
        pasos.append(Step(StepKind.LETTER, token, duracion))
    return tuple(pasos)


# --------------------------------------------------------------------------- #
# Reproductor
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Tick:
    """Ha pasado tiempo de pared. Es la única forma en que entra el reloj."""

    dt_ms: float


@dataclass(frozen=True, slots=True)
class TogglePause: ...


@dataclass(frozen=True, slots=True)
class Next: ...


@dataclass(frozen=True, slots=True)
class Prev: ...


@dataclass(frozen=True, slots=True)
class Restart: ...


@dataclass(frozen=True, slots=True)
class Faster: ...


@dataclass(frozen=True, slots=True)
class Slower: ...


PlayerInput: TypeAlias = Tick | TogglePause | Next | Prev | Restart | Faster | Slower


@dataclass(frozen=True, slots=True)
class PlayerState:
    index: int = 0
    #: Tiempo ya reproducido del paso actual, **a velocidad 1.0**: los ticks se
    #: multiplican por `speed` antes de sumarse.
    elapsed_ms: float = 0.0
    paused: bool = False
    speed: float = 1.0
    #: Se agotó el último paso. El estado se queda en él hasta `Restart`/`Prev`.
    finished: bool = False


@dataclass(frozen=True, slots=True)
class StepStarted:
    index: int


@dataclass(frozen=True, slots=True)
class Finished: ...


PlayerEvent: TypeAlias = StepStarted | Finished


def start(playlist: SequenceABC[Step]) -> PlayerState:
    """El estado inicial. Una lista vacía es un error de quien la construyó."""
    if not playlist:
        raise ValueError("la lista de pasos está vacía: no hay nada que reproducir")
    return PlayerState()


def player_step(
    state: PlayerState,
    event: PlayerInput,
    playlist: SequenceABC[Step],
    config: Config,
) -> tuple[PlayerState, tuple[PlayerEvent, ...]]:
    """Un evento → el estado siguiente y lo que pasó. Puro e inmutable."""
    ultimo = len(playlist) - 1
    match event:
        case Tick(dt_ms=dt):
            if state.paused or state.finished:
                return state, ()
            elapsed = state.elapsed_ms + dt * state.speed
            if elapsed < playlist[state.index].duration_ms:
                return replace(state, elapsed_ms=elapsed), ()
            return _advance(state, ultimo)
        case Next():
            return _advance(state, ultimo)
        case Prev():
            index = max(state.index - 1, 0)
            return (
                replace(state, index=index, elapsed_ms=0.0, finished=False),
                (StepStarted(index),),
            )
        case Restart():
            return PlayerState(speed=state.speed), (StepStarted(0),)
        case TogglePause():
            return replace(state, paused=not state.paused), ()
        case Faster():
            velocidad = min(
                state.speed + config.signs.speed_step, config.signs.speed_max
            )
            return replace(state, speed=velocidad), ()
        case Slower():
            velocidad = max(
                state.speed - config.signs.speed_step, config.signs.speed_min
            )
            return replace(state, speed=velocidad), ()
        case _:
            raise AssertionError(f"evento desconocido: {event!r}")


def _advance(
    state: PlayerState, ultimo: int
) -> tuple[PlayerState, tuple[PlayerEvent, ...]]:
    if state.index >= ultimo:
        if state.finished:
            return state, ()
        return replace(state, elapsed_ms=0.0, finished=True), (Finished(),)
    index = state.index + 1
    return replace(state, index=index, elapsed_ms=0.0, finished=False), (
        StepStarted(index),
    )


def asset_frame(state: PlayerState, n_frames: int, duracion_ms: int) -> int:
    """Qué frame del GIF mostrar. Da vueltas: con `dynamic_loops = 2`, dos."""
    posicion = int(state.elapsed_ms / duracion_ms * n_frames)
    return posicion % n_frames


# --------------------------------------------------------------------------- #
# Muestra de referencia
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Candidate:
    """Una muestra del dataset y su ruta relativa, que es lo que irá al manifest."""

    sample: Sample
    path: str


@dataclass(frozen=True, slots=True)
class ReferenceChoice:
    sample: Sample
    path: str
    #: Se dibujará espejada en X para mostrar mano derecha.
    mirrored: bool
    #: Cuántas muestras se consideraron. Va al log de `render`.
    candidates: int


def _comparison_vector(
    sample: Sample, dynamic: bool, config: Config
) -> tuple[float, ...] | None:
    """Vector en el que se mide "típica": forma para estáticas, matriz DTW
    aplanada para dinámicas. `None` si la extracción no da para comparar."""
    outcome = extract_sequence_features(sample.sequence, config)
    if isinstance(outcome, ExtractionRejected):
        return None
    if not dynamic:
        return outcome.static.shape.values
    if isinstance(outcome.dynamic, DynamicUnavailable):
        return None
    return tuple(valor for fila in outcome.dynamic.rows for valor in fila)


def choose_reference(
    candidates: SequenceABC[Candidate], label: Label, config: Config
) -> ReferenceChoice:
    """La muestra más típica de la letra: la medoide del grupo.

    1. Solo muestras de la letra y del `kind` que le corresponde.
    2. Mano derecha si la hay (el diccionario muestra mano derecha); si no, las
       izquierdas y se anota que hay que espejar.
    3. La más cercana al centroide, en distancia euclidiana. Empate → ruta
       menor, para que `render` sea determinista.

    No es un promedio inventado: es una grabación concreta que se puede ir a ver.
    """
    dynamic = LETTERS[label].es_dinamica
    kind = SampleKind.DYNAMIC if dynamic else SampleKind.STATIC
    pool = [c for c in candidates if c.sample.label == label and c.sample.kind == kind]
    derechas = [c for c in pool if c.sample.handedness == Handedness.RIGHT]
    mirrored = not derechas
    pool = derechas or pool

    vectores: list[tuple[Candidate, tuple[float, ...]]] = []
    for candidate in pool:
        vector = _comparison_vector(candidate.sample, dynamic, config)
        if vector is not None:
            vectores.append((candidate, vector))
    if not vectores:
        raise ValueError(f"{label}: sin muestras de referencia válidas en el dataset")

    dimension = len(vectores[0][1])
    centroide = [
        sum(vector[i] for _, vector in vectores) / len(vectores)
        for i in range(dimension)
    ]

    def distancia(entrada: tuple[Candidate, tuple[float, ...]]) -> tuple[float, str]:
        _, vector = entrada
        return (
            math.sqrt(
                sum((v - c) ** 2 for v, c in zip(vector, centroide, strict=True))
            ),
            entrada[0].path,
        )

    elegida, _ = min(vectores, key=distancia)
    return ReferenceChoice(
        sample=elegida.sample,
        path=elegida.path,
        mirrored=mirrored,
        candidates=len(vectores),
    )


# --------------------------------------------------------------------------- #
# Proyección al lienzo
# --------------------------------------------------------------------------- #


def project_frames(
    frames: SequenceABC[RawFrame], *, mirrored: bool, canvas_px: int, margin: float
) -> tuple[tuple[Point2, ...], ...]:
    """Landmarks normalizados → píxeles de un lienzo cuadrado.

    Corrige la relación de aspecto (`x · a`, con `a = ancho / alto`), espeja en
    X si se pide, y encuadra **una** caja envolvente sobre toda la secuencia:
    en una dinámica el desplazamiento es la seña, y encuadrar frame a frame lo
    borraría. Vista de cámara, no espejo: como te ve quien te mira.
    """
    if not frames:
        raise ValueError("project_frames necesita al menos un frame")
    crudos = [
        [
            (
                -lm.x * frame.aspect_ratio if mirrored else lm.x * frame.aspect_ratio,
                lm.y,
            )
            for lm in frame.landmarks
        ]
        for frame in frames
    ]
    xs = [x for puntos in crudos for x, _ in puntos]
    ys = [y for puntos in crudos for _, y in puntos]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    lado = max(x1 - x0, y1 - y0)
    utilizable = canvas_px * (1.0 - 2.0 * margin)
    escala = utilizable / lado if lado > 0.0 else 1.0
    centro = canvas_px / 2.0
    dx = centro - escala * (x0 + x1) / 2.0
    dy = centro - escala * (y0 + y1) / 2.0
    return tuple(
        tuple((x * escala + dx, y * escala + dy) for x, y in puntos)
        for puntos in crudos
    )


# --------------------------------------------------------------------------- #
# Lo que la ventana dibuja
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Scene:
    """Todo lo que hace falta para componer un cuadro. Datos, no dibujo.

    `frame` es la imagen del asset ya elegida para este instante (o `None` en
    una pausa). Es `Any` porque este módulo no importa Pillow: quien compone
    sabe qué es.
    """

    tokens: tuple[Token, ...]
    playlist: tuple[Step, ...]
    state: PlayerState
    asset: SignAsset | None
    frame: Any | None

    @property
    def step(self) -> Step:
        return self.playlist[self.state.index]

    @property
    def progress(self) -> float:
        """Fracción del paso actual ya reproducida, en `[0, 1]`."""
        return min(self.state.elapsed_ms / self.step.duration_ms, 1.0)

    @property
    def token_index(self) -> int:
        """Índice del token que corresponde al paso actual. Coinciden uno a uno
        porque `build_playlist` produce un paso por token."""
        return self.state.index
