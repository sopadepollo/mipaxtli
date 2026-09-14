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

import re
import unicodedata
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from datetime import date
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lsm.types import Handedness
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
    error con la lista completa de ofensores.
    """
    tokens: list[Token] = []
    unsupported: list[str] = []
    for palabra in unicodedata.normalize("NFC", text).split():
        if tokens:
            tokens.append(WordGap())
        letras = "".join(_base_letter(char) for char in palabra)
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
            elif palabra[i] not in unsupported:
                unsupported.append(palabra[i])
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
