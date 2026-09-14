# Fase 4 — Texto a señas: plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que `lsm-signs reproducir "casa"` muestre, sin cámara, la secuencia de señas LSM de un texto, con temporizador y control manual, a partir de 29 assets renderizados desde el dataset propio y descritos en `assets/signs/manifest.json`.

**Architecture:** `src/lsm/signs.py` es código **puro**: esquema del manifest, texto → símbolos, lista de pasos, elección de la muestra de referencia, geometría del dibujo y la máquina de estados del reproductor (el reloj entra por `Tick`). `src/lsm/io/signs.py` toca disco y Pillow: lee/escribe el manifest, carga candidatas de `data/raw`, dibuja PNG/GIF y compone el cuadro de la ventana. `src/lsm/cli/signs.py` cablea los tres subcomandos y solo ahí se importa OpenCV, para `imshow` y `waitKey`.

**Tech Stack:** Python 3.11+, uv, Pydantic v2, pytest, mypy strict, ruff. Pillow es dependencia del grupo `dev` (la suite la necesita) y del extra `capture` (el reproductor también). OpenCV sigue siendo opcional y solo se importa dentro de funciones.

**Spec:** `docs/superpowers/specs/2026-09-14-fase-4-texto-a-senas-design.md`

## Global Constraints

Copiadas de `CLAUDE.md` y del spec. Aplican a **todas** las tareas.

- **`src/lsm/signs.py` es código puro.** Sin OpenCV, sin Pillow, sin MediaPipe, sin disco, sin cámara, sin `time`. Toda la I/O vive en `src/lsm/io/signs.py` y `src/lsm/cli/signs.py`.
- **MediaPipe solo se importa en `src/lsm/io/hands.py`.** OpenCV, solo dentro de funciones de `cli/signs.py`. Pillow, solo en `io/signs.py` (y en tests).
- **Cero umbrales hardcodeados.** Duraciones, fps, tamaño del lienzo y límites de velocidad viven en `config.yaml`, sección `signs`, validados con Pydantic. Colores y grosores de dibujo son constantes de módulo, no umbrales.
- **El tipo base es una secuencia `(T, 21, 3)`.** `project_frames` recibe frames, en plural, aunque el PNG use uno.
- **Ningún asset viene de internet.** Cada uno se dibuja desde una muestra de `data/raw` y el manifest apunta a ella. Lo que dice el manifest de cada letra se copia de `vocabulary.py`; `manifest_drift` exige que no diverja.
- **Terminología:** el proyecto traduce **deletreo manual**, no lengua de señas. `signer` = quien firma, `label` = letra, `sample` = secuencia etiquetada. En el manifest y la UI, `letra` es cómo se escribe (`Ñ`, `LL`) y `label` el identificador (`ENIE`, `DOBLE_L`).
- **Los defaults de Pydantic y los valores de `config.yaml` tienen que coincidir.** `tests/test_config.py::test_el_config_yaml_de_ejemplo_es_valido_y_coincide_con_los_defaults` lo exige.
- **Comandos:** `uv run pytest`, `uv run mypy`, `uv run ruff check .`, `uv run ruff format --check .`. En WSL con el repositorio en Linux: `wsl -d Ubuntu -- bash -lc 'cd ~/mipaxtli && ...'`. No hay `make` en ese entorno: cada receta del Makefile es una línea de `uv run`.
- **Antes de cada commit:** `uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest`. Los cuatro en verde.
- **Los mensajes de commit terminan con** las dos líneas de atribución de los commits de la rama (`Co-Authored-By:` y `Claude-Session:`). Cópialas de `git log -1 --format=%B`.

## Mapa de archivos

| Archivo | Responsabilidad | Tarea |
|---|---|---|
| `src/lsm/config.py`, `config.yaml` | Sección `signs` | 1 |
| `tests/test_config.py` | La sección y su validación cruzada | 1 |
| `src/lsm/signs.py` | **Nuevo.** Esquema del manifest y `manifest_drift` | 2 |
| `src/lsm/signs.py` | `text_to_symbols`, `render_tokens` | 3 |
| `src/lsm/signs.py` | `build_playlist`, reproductor, `asset_frame` | 4 |
| `src/lsm/signs.py` | `choose_reference`, `project_frames` | 5 |
| `tests/test_signs.py` | **Nuevo.** Todo lo puro | 2–5 |
| `pyproject.toml` | Pillow en `dev` y en `capture`; script `lsm-signs` | 6, 7 |
| `src/lsm/io/signs.py` | **Nuevo.** Manifest en disco, candidatas, PNG/GIF, cuadro de la ventana | 6, 8 |
| `tests/test_io_signs.py` | **Nuevo.** Render y manifest en `tmp_path` | 6, 8 |
| `src/lsm/cli/signs.py` | **Nuevo.** `render`, `verificar`, `reproducir` | 7, 8 |
| `tests/test_cli_signs.py` | **Nuevo.** Los tres subcomandos sin OpenCV | 7, 8 |
| `assets/signs/manifest.json` + 29 archivos | **Nuevos.** El entregable de la fase | 9 |
| `tests/test_signs_manifest.py` | **Nuevo.** El criterio de la fase | 9 |
| `.gitignore` | Los assets se versionan | 9 |
| `docs/adr/0014-*.md`, `ARQUITECTURA.md`, `COMO-PROBAR.md`, `README.md`, `CLAUDE.md`, `Makefile` | Cierre | 10 |

---

### Task 1: La sección `signs` de la configuración

**Files:**
- Modify: `src/lsm/config.py` (nueva clase antes de `class Config`; nuevo campo en `Config`)
- Modify: `config.yaml` (nueva sección al final)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `_Section`, `Field`, `model_validator` de `src/lsm/config.py`
- Produces: `config.signs` con `static_hold_ms: float`, `dynamic_loops: int`, `word_gap_ms: float`, `render_fps: int`, `canvas_px: int`, `canvas_margin: float`, `speed_min: float`, `speed_max: float`, `speed_step: float`, `tick_ms: int`

- [ ] **Step 1: Escribe los tests que fallan**

Al final de `tests/test_config.py`:

```python
def test_texto_a_senas_tiene_su_seccion() -> None:
    """Duraciones, fps del render y límites de velocidad son umbrales: viven en
    `config.yaml` como los demás (`CLAUDE.md` regla 5)."""
    signs = Config().signs

    assert signs.static_hold_ms > 0.0
    assert signs.dynamic_loops >= 1
    assert signs.word_gap_ms > 0.0
    assert signs.render_fps >= 1
    assert signs.canvas_px >= 64
    assert 0.0 <= signs.canvas_margin < 0.5
    assert signs.speed_min < 1.0 <= signs.speed_max
    assert signs.tick_ms >= 1


def test_un_rango_de_velocidad_vacio_se_rechaza() -> None:
    """`Faster`/`Slower` acotan entre `speed_min` y `speed_max`; con el rango al
    revés no hay velocidad válida y el reproductor no debería descubrirlo."""
    with pytest.raises(ValidationError):
        Config.model_validate({"signs": {"speed_min": 2.0, "speed_max": 1.0}})
```

- [ ] **Step 2: Corre los tests para verificar que fallan**

Run: `uv run pytest tests/test_config.py -k "senas or velocidad_vacio" -v`
Expected: FAIL con `AttributeError: 'Config' object has no attribute 'signs'` y el segundo con `DID NOT RAISE`.

- [ ] **Step 3: Añade `SignsConfig` y el campo**

En `src/lsm/config.py`, justo antes de `class Config(_Section):`:

```python
class SignsConfig(_Section):
    """Dirección texto → señas (`src/lsm/signs.py`, `ARQUITECTURA.md` §4.10).

    No hay clasificador en esta dirección; lo que hay son tiempos. Todos en
    milisegundos a velocidad 1.0: el reproductor los multiplica por la velocidad
    que la persona elija con `+`/`-`.
    """

    #: Cuánto se sostiene en pantalla una letra estática.
    static_hold_ms: float = Field(default=1500.0, gt=0.0, le=60000.0)

    #: Vueltas completas del GIF de una letra dinámica. La duración del paso es
    #: `duracion_ms` del manifest por este número: el movimiento se ve entero
    #: tantas veces como diga.
    dynamic_loops: int = Field(default=2, ge=1, le=20)

    #: Pausa entre palabras: lo que ocupa un espacio del texto.
    word_gap_ms: float = Field(default=800.0, gt=0.0, le=60000.0)

    #: Cuadros por segundo del GIF al renderizar. Fija `duracion_ms` de cada
    #: dinámica: `round(1000 · frames / render_fps)`.
    render_fps: int = Field(default=12, ge=1, le=60)

    #: Lado, en píxeles, del lienzo cuadrado de cada asset.
    canvas_px: int = Field(default=320, ge=64, le=2048)

    #: Aire alrededor de la mano, como fracción del lienzo por cada lado.
    canvas_margin: float = Field(default=0.12, ge=0.0, lt=0.5)

    #: Límites y paso de la velocidad de reproducción.
    speed_min: float = Field(default=0.25, gt=0.0, le=10.0)
    speed_max: float = Field(default=4.0, gt=0.0, le=10.0)
    speed_step: float = Field(default=0.25, gt=0.0, le=10.0)

    #: Espera de `waitKey` en cada vuelta del bucle de la ventana. No es el
    #: `dt` del reproductor: ese se mide con el reloj, este solo decide cada
    #: cuánto se mira el teclado.
    tick_ms: int = Field(default=33, ge=1, le=1000)

    @model_validator(mode="after")
    def _la_velocidad_tiene_rango(self) -> SignsConfig:
        if self.speed_min >= self.speed_max:
            msg = (
                f"signs.speed_min ({self.speed_min}) no es menor que "
                f"signs.speed_max ({self.speed_max}): no hay velocidad válida"
            )
            raise ValueError(msg)
        return self
```

En `class Config(_Section)`, después de `telemetry: ...`:

```python
    signs: SignsConfig = Field(default_factory=SignsConfig)
```

Al final de `config.yaml`:

```yaml
# Dirección texto → señas (src/lsm/signs.py, docs/adr/0014-...).
#
# Aquí no hay clasificador: hay tiempos. Todos en milisegundos a velocidad 1.0;
# el reproductor los multiplica por la velocidad elegida con + y -.
signs:
  # Cuánto se sostiene en pantalla una letra estática.
  static_hold_ms: 1500.0

  # Vueltas completas del GIF de una dinámica. La duración del paso es
  # duracion_ms del manifest por este número.
  dynamic_loops: 2

  # Pausa entre palabras: lo que ocupa un espacio del texto.
  word_gap_ms: 800.0

  # Cuadros por segundo del GIF al renderizar. Fija duracion_ms de cada
  # dinámica en el manifest: round(1000 * frames / render_fps).
  render_fps: 12

  # Lado del lienzo cuadrado de cada asset, en píxeles, y aire alrededor de la
  # mano como fracción del lienzo por cada lado.
  canvas_px: 320
  canvas_margin: 0.12

  # Límites y paso de la velocidad de reproducción. config.py valida que
  # speed_min < speed_max.
  speed_min: 0.25
  speed_max: 4.0
  speed_step: 0.25

  # Espera de waitKey en cada vuelta del bucle de la ventana. No es el dt del
  # reproductor —ese se mide con el reloj—, solo cada cuánto se lee el teclado.
  tick_ms: 33
```

- [ ] **Step 4: Corre la suite de configuración**

Run: `uv run pytest tests/test_config.py -v`
Expected: todo PASS, incluido `test_el_config_yaml_de_ejemplo_es_valido_y_coincide_con_los_defaults`.

- [ ] **Step 5: Commit**

```bash
git add src/lsm/config.py config.yaml tests/test_config.py
git commit -m "feat(config): seccion signs para la direccion texto a senas"
```

---

### Task 2: El esquema del manifest y `manifest_drift`

**Files:**
- Create: `src/lsm/signs.py`
- Create: `tests/test_signs.py`

**Interfaces:**
- Consumes: `lsm.vocabulary.LETTERS`, `Label`, `NO_TRAJECTORY`; `lsm.types.Handedness`
- Produces: `MANIFEST_SCHEMA_VERSION`, `FUENTE_NORMATIVA`, `AssetSource`, `AssetReview`, `SignAsset`, `Manifest`, `expected_filename(label: Label) -> str`, `manifest_drift(manifest: Manifest) -> list[str]`

- [ ] **Step 1: Escribe los tests que fallan**

`tests/test_signs.py`:

```python
"""La dirección texto → señas, sin disco ni ventana.

`lsm.signs` es código puro por la regla 2 de `CLAUDE.md`: recibe manifests ya
cargados, texto y ticks de reloj, y devuelve estados. Todo lo que hay aquí se
ejercita con datos construidos a mano.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from lsm.signs import (
    FUENTE_NORMATIVA,
    MANIFEST_SCHEMA_VERSION,
    AssetReview,
    AssetSource,
    Manifest,
    SignAsset,
    expected_filename,
    manifest_drift,
)
from lsm.types import Handedness
from lsm.vocabulary import LETTERS, Label, spec

HOY = date(2026, 9, 14)


def asset(label: Label, **cambios: object) -> SignAsset:
    """Una entrada válida del manifest, copiada de `vocabulary.py`."""
    letra = spec(label)
    base: dict[str, object] = {
        "letra": letra.display,
        "archivo": expected_filename(label),
        "es_dinamica": letra.es_dinamica,
        "descripcion": letra.descripcion,
        "trayectoria": letra.trayectoria,
        "pagina": letra.pagina,
        "duracion_ms": 2000 if letra.es_dinamica else None,
        "fuente": AssetSource(
            tipo="esqueleto_desde_dataset",
            muestra=f"s01/2026-09-09-manana/{label}/007.json",
            signer_id="s01",
            lateralidad_original=Handedness.RIGHT,
            espejada=False,
        ),
        "revision": AssetReview(fecha=HOY, revisor="tests", resultado="coincide"),
    }
    base.update(cambios)
    return SignAsset.model_validate(base)


def manifiesto(**cambios: SignAsset) -> Manifest:
    letras = {label: asset(label) for label in LETTERS}
    letras.update(cambios)
    return Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        fuente_normativa=FUENTE_NORMATIVA,
        letras=letras,
    )


# --------------------------------------------------------------------------- #
# Esquema
# --------------------------------------------------------------------------- #


def test_el_nombre_del_archivo_lo_fija_la_etiqueta() -> None:
    assert expected_filename(Label.A) == "A.png"
    assert expected_filename(Label.J) == "J.gif"
    assert expected_filename(Label.DOBLE_L) == "DOBLE_L.gif"


def test_un_manifest_completo_no_deriva_del_glosario() -> None:
    assert manifest_drift(manifiesto()) == []


def test_una_dinamica_con_imagen_fija_se_rechaza() -> None:
    """Las dinámicas necesitan movimiento: GIF, no PNG. Es la regla del spec
    hecha tipo, y falla al cargar en vez de al reproducir."""
    with pytest.raises(ValidationError, match="gif"):
        asset(Label.J, archivo="J.png")


def test_una_estatica_con_gif_se_rechaza() -> None:
    with pytest.raises(ValidationError, match="png"):
        asset(Label.A, archivo="A.gif")


def test_una_dinamica_sin_duracion_se_rechaza() -> None:
    with pytest.raises(ValidationError, match="duracion_ms"):
        asset(Label.J, duracion_ms=None)


def test_una_estatica_con_duracion_se_rechaza() -> None:
    with pytest.raises(ValidationError, match="duracion_ms"):
        asset(Label.A, duracion_ms=1500)


def test_la_muestra_de_origen_tiene_la_forma_del_dataset() -> None:
    with pytest.raises(ValidationError, match="firmante/sesion"):
        AssetSource(
            tipo="esqueleto_desde_dataset",
            muestra="J.json",
            signer_id="s01",
            lateralidad_original=Handedness.RIGHT,
            espejada=False,
        )


def test_el_archivo_tiene_que_llamarse_como_la_etiqueta() -> None:
    with pytest.raises(ValidationError, match="A.png"):
        manifiesto(A=asset(Label.A, archivo="B.png"))


def test_la_muestra_de_origen_tiene_que_ser_de_la_misma_letra() -> None:
    fuente = AssetSource(
        tipo="esqueleto_desde_dataset",
        muestra="s01/2026-09-09-manana/B/001.json",
        signer_id="s01",
        lateralidad_original=Handedness.RIGHT,
        espejada=False,
    )
    with pytest.raises(ValidationError, match="muestra"):
        manifiesto(A=asset(Label.A, fuente=fuente))


def test_la_clase_negativa_no_es_una_letra() -> None:
    letras = {label: asset(label) for label in LETTERS}
    letras[Label.NONE] = asset(Label.A)
    with pytest.raises(ValidationError, match="NONE"):
        Manifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            fuente_normativa=FUENTE_NORMATIVA,
            letras=letras,
        )


def test_otra_version_de_esquema_se_rechaza() -> None:
    with pytest.raises(ValidationError, match="schema_version"):
        Manifest(
            schema_version=MANIFEST_SCHEMA_VERSION + 1,
            fuente_normativa=FUENTE_NORMATIVA,
            letras={label: asset(label) for label in LETTERS},
        )


def test_campos_desconocidos_se_rechazan() -> None:
    with pytest.raises(ValidationError):
        AssetReview.model_validate(
            {"fecha": HOY, "revisor": "x", "resultado": "coincide", "extra": 1}
        )


# --------------------------------------------------------------------------- #
# Deriva contra vocabulary.py
# --------------------------------------------------------------------------- #


def test_una_letra_que_falta_es_deriva() -> None:
    letras = {label: asset(label) for label in LETTERS}
    del letras[Label.Q]
    manifest = Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        fuente_normativa=FUENTE_NORMATIVA,
        letras=letras,
    )

    deriva = manifest_drift(manifest)

    assert len(deriva) == 1
    assert "Q" in deriva[0] and "falta" in deriva[0]


def test_una_descripcion_distinta_es_deriva() -> None:
    deriva = manifest_drift(manifiesto(M=asset(Label.M, descripcion="otra cosa")))

    assert len(deriva) == 1
    assert "M" in deriva[0] and "descripcion" in deriva[0]


def test_una_pagina_distinta_es_deriva() -> None:
    deriva = manifest_drift(manifiesto(Z=asset(Label.Z, pagina=1)))

    assert deriva and "pagina" in deriva[0]


def test_la_forma_de_escribir_la_letra_tambien_se_compara() -> None:
    deriva = manifest_drift(manifiesto(ENIE=asset(Label.ENIE, letra="N")))

    assert deriva and "letra" in deriva[0]
```

- [ ] **Step 2: Corre los tests para verificar que fallan**

Run: `uv run pytest tests/test_signs.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'lsm.signs'`.

- [ ] **Step 3: Escribe `src/lsm/signs.py` con el esquema**

```python
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
from datetime import date
from typing import Final, Literal

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
SourceKind = Literal["esqueleto_desde_dataset"]

#: Resultado de mirar el asset junto a la descripción del glosario.
ReviewResult = Literal["coincide", "difiere", "pendiente"]

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
            msg = f"{self.letra}: una letra {clase} va en {extension[1:]}, no {self.archivo!r}"
            raise ValueError(msg)
        if self.es_dinamica and (self.duracion_ms is None or self.duracion_ms <= 0):
            msg = f"{self.letra}: una dinámica necesita duracion_ms > 0"
            raise ValueError(msg)
        if not self.es_dinamica and self.duracion_ms is not None:
            msg = f"{self.letra}: una estática no lleva duracion_ms; la fija config.signs"
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
                msg = f"{label}: el archivo tiene que ser {esperado}, no {asset.archivo!r}"
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
```

- [ ] **Step 4: Corre los tests**

Run: `uv run pytest tests/test_signs.py -v && uv run mypy`
Expected: todo PASS, mypy limpio. Si mypy se queja de `Literal` en `SourceKind`/`ReviewResult` como alias, cámbialos a `TypeAlias` explícito: `SourceKind: TypeAlias = Literal[...]`.

- [ ] **Step 5: Commit**

```bash
git add src/lsm/signs.py tests/test_signs.py
git commit -m "feat(signs): esquema del manifest de assets y deriva contra el glosario"
```

---

### Task 3: Texto → símbolos

**Files:**
- Modify: `src/lsm/signs.py`
- Modify: `tests/test_signs.py`

**Interfaces:**
- Produces: `WordGap`, `Token = Label | WordGap`, `UnsupportedCharacters(ValueError)` con `.chars: tuple[str, ...]`, `text_to_symbols(text: str) -> tuple[Token, ...]`, `render_tokens(tokens: Sequence[Token]) -> str`

- [ ] **Step 1: Escribe los tests que fallan**

Añade a los imports de `tests/test_signs.py`: `UnsupportedCharacters, WordGap, render_tokens, text_to_symbols`. Al final del archivo:

```python
# --------------------------------------------------------------------------- #
# Texto → símbolos
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("llave", (Label.DOBLE_L, Label.A, Label.V, Label.E)),
        ("carro", (Label.C, Label.A, Label.DOBLE_R, Label.O)),
        ("año", (Label.A, Label.ENIE, Label.O)),
        ("chico", (Label.C, Label.H, Label.I, Label.C, Label.O)),
        ("ll", (Label.DOBLE_L,)),
        ("lll", (Label.DOBLE_L, Label.L)),
        ("Ñ", (Label.ENIE,)),
        ("pingüino", (Label.P, Label.I, Label.N, Label.G, Label.U, Label.I, Label.N, Label.O)),
    ],
)
def test_texto_a_simbolos(texto: str, esperado: tuple[Label, ...]) -> None:
    assert text_to_symbols(texto) == esperado


def test_los_espacios_separan_palabras_y_no_se_acumulan() -> None:
    tokens = text_to_symbols("  Árbol  verde ")

    assert tokens == (
        Label.A, Label.R, Label.B, Label.O, Label.L,
        WordGap(),
        Label.V, Label.E, Label.R, Label.D, Label.E,
    )  # fmt: skip


def test_la_enie_sobrevive_a_quitar_los_acentos() -> None:
    """En NFD la ñ es `n` + tilde: quitar marcas combinantes sin cuidado la
    convertiría en N y "año" se deletrearía como "ano"."""
    assert text_to_symbols("ñandú") == (Label.ENIE, Label.A, Label.N, Label.D, Label.U)


def test_un_caracter_sin_sena_se_rechaza_con_la_lista_exacta() -> None:
    with pytest.raises(UnsupportedCharacters) as excinfo:
        text_to_symbols("hola2! 2")

    assert excinfo.value.chars == ("2", "!")


def test_un_texto_sin_letras_da_una_tupla_vacia() -> None:
    assert text_to_symbols("   ") == ()
    assert text_to_symbols("") == ()


def test_los_simbolos_se_muestran_como_se_escriben() -> None:
    tokens = text_to_symbols("año ll")

    assert render_tokens(tokens) == "A Ñ O · LL"
```

- [ ] **Step 2: Corre los tests para verificar que fallan**

Run: `uv run pytest tests/test_signs.py -k "simbolos or espacios or enie or rechaza_con or sin_letras or se_muestran" -v`
Expected: FAIL con `ImportError: cannot import name 'text_to_symbols'`.

- [ ] **Step 3: Implementa**

En `src/lsm/signs.py`, añade a los imports `import unicodedata`, `from collections.abc import Sequence as SequenceABC`, `from dataclasses import dataclass`, `from typing import TypeAlias`. Después de `manifest_drift`:

```python
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


class UnsupportedCharacters(ValueError):
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
```

Nota sobre `palabra[i]` en el `elif`: `letras` y `palabra` tienen la misma longitud porque `_base_letter` devuelve exactamente un carácter por entrada, así que el índice `i` sirve para recuperar el carácter original y reportarlo tal como lo escribió la persona (`!` y no `!` normalizado).

- [ ] **Step 4: Corre los tests**

Run: `uv run pytest tests/test_signs.py -v && uv run mypy && uv run ruff check .`
Expected: PASS. Si ruff protesta por `E501` en la línea larga de `pingüino`, parte la tupla en varias líneas.

- [ ] **Step 5: Commit**

```bash
git add src/lsm/signs.py tests/test_signs.py
git commit -m "feat(signs): texto a simbolos del glosario con digrafos y acentos"
```

---

### Task 4: Lista de pasos y reproductor

**Files:**
- Modify: `src/lsm/signs.py`
- Modify: `tests/test_signs.py`

**Interfaces:**
- Consumes: `Manifest`, `Token`, `config.signs`
- Produces: `StepKind`, `Step(kind, label, duration_ms)`, `build_playlist(tokens, manifest, config) -> tuple[Step, ...]`, eventos `Tick(dt_ms)`, `TogglePause`, `Next`, `Prev`, `Restart`, `Faster`, `Slower`; `PlayerState`; `StepStarted(index)`, `Finished`; `start(playlist) -> PlayerState`; `player_step(state, event, playlist, config) -> tuple[PlayerState, tuple[PlayerEvent, ...]]`; `asset_frame(state, n_frames, duracion_ms) -> int`

- [ ] **Step 1: Escribe los tests que fallan**

Añade a los imports de `tests/test_signs.py`: `from lsm.config import Config` y de `lsm.signs`: `Faster, Finished, Next, PlayerEvent, PlayerInput, PlayerState, Prev, Restart, Slower, Step, StepKind, StepStarted, Tick, TogglePause, asset_frame, build_playlist, player_step, start`. Al final:

```python
# --------------------------------------------------------------------------- #
# Lista de pasos
# --------------------------------------------------------------------------- #

CONFIG = Config()


def test_las_duraciones_salen_de_la_configuracion_y_del_manifest() -> None:
    pasos = build_playlist(text_to_symbols("aj a"), manifiesto(), CONFIG)

    assert [paso.kind for paso in pasos] == [
        StepKind.LETTER, StepKind.LETTER, StepKind.GAP, StepKind.LETTER
    ]  # fmt: skip
    assert pasos[0].duration_ms == CONFIG.signs.static_hold_ms
    assert pasos[1].duration_ms == 2000 * CONFIG.signs.dynamic_loops
    assert pasos[2].duration_ms == CONFIG.signs.word_gap_ms
    assert pasos[2].label is None


def test_una_letra_sin_asset_no_entra_en_la_lista_en_silencio() -> None:
    letras = {label: asset(label) for label in LETTERS}
    del letras[Label.J]
    incompleto = Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        fuente_normativa=FUENTE_NORMATIVA,
        letras=letras,
    )
    with pytest.raises(ValueError, match="J"):
        build_playlist((Label.J,), incompleto, CONFIG)


def test_una_lista_vacia_no_arranca() -> None:
    with pytest.raises(ValueError, match="vac"):
        start(())


# --------------------------------------------------------------------------- #
# Reproductor
# --------------------------------------------------------------------------- #

DOS_PASOS = (
    Step(kind=StepKind.LETTER, label=Label.A, duration_ms=1000.0),
    Step(kind=StepKind.LETTER, label=Label.B, duration_ms=500.0),
)


def avanzar(
    estado: PlayerState, *eventos: PlayerInput
) -> tuple[PlayerState, list[PlayerEvent]]:
    emitidos: list[PlayerEvent] = []
    for evento in eventos:
        estado, nuevos = player_step(estado, evento, DOS_PASOS, CONFIG)
        emitidos.extend(nuevos)
    return estado, emitidos


def test_los_ticks_acumulan_y_al_agotar_el_paso_pasan_al_siguiente() -> None:
    estado, eventos = avanzar(start(DOS_PASOS), Tick(400.0), Tick(400.0))
    assert estado.index == 0
    assert estado.elapsed_ms == 800.0
    assert eventos == []

    estado, eventos = avanzar(estado, Tick(200.0))
    assert estado.index == 1
    assert estado.elapsed_ms == 0.0
    assert eventos == [StepStarted(index=1)]


def test_al_agotar_el_ultimo_paso_termina_y_se_queda_en_el() -> None:
    estado = PlayerState(index=1, elapsed_ms=400.0)

    estado, eventos = avanzar(estado, Tick(100.0))
    assert estado.finished
    assert estado.index == 1
    assert eventos == [Finished()]

    otra, mas = avanzar(estado, Tick(5000.0))
    assert otra == estado
    assert mas == []


def test_en_pausa_los_ticks_no_avanzan() -> None:
    estado, _ = avanzar(start(DOS_PASOS), TogglePause(), Tick(5000.0))
    assert estado.paused
    assert estado.elapsed_ms == 0.0

    estado, _ = avanzar(estado, TogglePause(), Tick(100.0))
    assert not estado.paused
    assert estado.elapsed_ms == 100.0


def test_la_velocidad_multiplica_el_tiempo() -> None:
    estado, _ = avanzar(start(DOS_PASOS), Faster(), Tick(100.0))

    assert estado.speed == 1.0 + CONFIG.signs.speed_step
    assert estado.elapsed_ms == pytest.approx(100.0 * estado.speed)


def test_la_velocidad_queda_acotada() -> None:
    muchas = [Faster()] * 100
    estado, _ = avanzar(start(DOS_PASOS), *muchas)
    assert estado.speed == CONFIG.signs.speed_max

    pocas = [Slower()] * 100
    estado, _ = avanzar(estado, *pocas)
    assert estado.speed == CONFIG.signs.speed_min


def test_siguiente_y_anterior_quedan_acotados() -> None:
    estado, eventos = avanzar(start(DOS_PASOS), Tick(300.0), Next())
    assert estado.index == 1
    assert estado.elapsed_ms == 0.0
    assert eventos == [StepStarted(index=1)]

    estado, eventos = avanzar(estado, Next())
    assert estado.finished
    assert estado.index == 1
    assert eventos == [Finished()]

    estado, eventos = avanzar(estado, Prev())
    assert not estado.finished
    assert estado.index == 0
    assert eventos == [StepStarted(index=0)]

    estado, eventos = avanzar(estado, Tick(300.0), Prev())
    assert estado.index == 0
    assert estado.elapsed_ms == 0.0


def test_reiniciar_conserva_la_velocidad() -> None:
    estado, _ = avanzar(start(DOS_PASOS), Faster(), Next(), Next())
    assert estado.finished

    estado, eventos = avanzar(estado, Restart())
    assert estado == PlayerState(speed=1.0 + CONFIG.signs.speed_step)
    assert eventos == [StepStarted(index=0)]


def test_el_frame_del_gif_da_vueltas_con_los_loops() -> None:
    n_frames, duracion = 24, 2000
    assert asset_frame(PlayerState(elapsed_ms=0.0), n_frames, duracion) == 0
    assert asset_frame(PlayerState(elapsed_ms=1000.0), n_frames, duracion) == 12
    assert asset_frame(PlayerState(elapsed_ms=2000.0), n_frames, duracion) == 0
    assert asset_frame(PlayerState(elapsed_ms=3999.0), n_frames, duracion) == 23
```

- [ ] **Step 2: Corre los tests para verificar que fallan**

Run: `uv run pytest tests/test_signs.py -v`
Expected: FAIL con `ImportError` de `build_playlist`.

- [ ] **Step 3: Implementa**

En `src/lsm/signs.py`, añade a los imports `from dataclasses import dataclass, replace`, `from enum import StrEnum`, `from lsm.config import Config`. Después de `render_tokens`:

```python
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
            velocidad = min(state.speed + config.signs.speed_step, config.signs.speed_max)
            return replace(state, speed=velocidad), ()
        case Slower():
            velocidad = max(state.speed - config.signs.speed_step, config.signs.speed_min)
            return replace(state, speed=velocidad), ()


def _advance(state: PlayerState, ultimo: int) -> tuple[PlayerState, tuple[PlayerEvent, ...]]:
    if state.index >= ultimo:
        if state.finished:
            return state, ()
        return replace(state, elapsed_ms=0.0, finished=True), (Finished(),)
    index = state.index + 1
    return replace(state, index=index, elapsed_ms=0.0, finished=False), (StepStarted(index),)


def asset_frame(state: PlayerState, n_frames: int, duracion_ms: int) -> int:
    """Qué frame del GIF mostrar. Da vueltas: con `dynamic_loops = 2`, dos."""
    posicion = int(state.elapsed_ms / duracion_ms * n_frames)
    return posicion % n_frames
```

Ojo con el `match`: mypy strict con `warn_unreachable` exige que los brazos cubran `PlayerInput` entero; si se queja de "missing return", añade al final del `match` un `case _:` con `raise AssertionError(f"evento desconocido: {event!r}")`.

- [ ] **Step 4: Corre los tests**

Run: `uv run pytest tests/test_signs.py -v && uv run mypy && uv run ruff check . && uv run ruff format --check .`
Expected: PASS y limpio. Ajusta el formato con `uv run ruff format .` si hace falta.

- [ ] **Step 5: Commit**

```bash
git add src/lsm/signs.py tests/test_signs.py
git commit -m "feat(signs): lista de pasos y reproductor por ticks con reloj inyectado"
```

---

### Task 5: Elegir la muestra de referencia y proyectar el dibujo

**Files:**
- Modify: `src/lsm/signs.py`
- Modify: `tests/test_signs.py`

**Interfaces:**
- Consumes: `lsm.features.extract_sequence_features`, `ExtractionRejected`, `DynamicUnavailable`; `lsm.types.Sample`, `SampleKind`, `RawFrame`, `Point2`; `lsm.vocabulary.spec`
- Produces: `Candidate(sample, path)`, `ReferenceChoice(sample, path, mirrored, candidates)`, `choose_reference(candidates, label, config) -> ReferenceChoice`, `project_frames(frames, *, mirrored, canvas_px, margin) -> tuple[tuple[Point2, ...], ...]`

- [ ] **Step 1: Escribe los tests que fallan**

Añade a los imports de `tests/test_signs.py`: `from dataclasses import replace`, `from lsm.signs import Candidate, choose_reference, project_frames`, `from lsm.synthetic import arc_offsets, canonical_hand, class_hand, moving_sequence, still_sequence, synthetic_samples, to_frame, translated`, `from lsm.types import Point2, Sample, SampleKind`. Al final:

```python
# --------------------------------------------------------------------------- #
# Muestra de referencia
# --------------------------------------------------------------------------- #


def candidatas(muestras: tuple[Sample, ...]) -> list[Candidate]:
    return [
        Candidate(
            sample=muestra,
            path=f"{muestra.signer_id}/{muestra.session_id}/{muestra.label}/{i:03d}.json",
        )
        for i, muestra in enumerate(muestras)
    ]


def test_la_referencia_es_la_medoide_y_no_una_rareza() -> None:
    """Tres muestras parecidas y una hecha con otra mano: la elegida es una de
    las tres. Y siempre la misma: `render` tiene que ser determinista."""
    tipicas = synthetic_samples(("A",), signers=1, sessions=1, repetitions=3)
    rara = replace(tipicas[0], sequence=still_sequence(class_hand(40), length=8))
    todas = candidatas((*tipicas, rara))

    eleccion = choose_reference(todas, Label.A, CONFIG)

    assert eleccion.sample in tipicas
    assert eleccion.candidates == 4
    assert eleccion.mirrored is False
    assert choose_reference(todas, Label.A, CONFIG) == eleccion


def test_se_prefiere_la_mano_derecha_y_si_no_hay_se_espeja() -> None:
    muestras = synthetic_samples(("A",), signers=1, sessions=1, repetitions=2)
    izquierda = replace(muestras[0], handedness=Handedness.LEFT)
    derecha = muestras[1]

    eleccion = choose_reference(candidatas((izquierda, derecha)), Label.A, CONFIG)
    assert eleccion.sample is derecha
    assert eleccion.mirrored is False

    solo_izquierdas = candidatas((izquierda, replace(muestras[1], handedness=Handedness.LEFT)))
    eleccion = choose_reference(solo_izquierdas, Label.A, CONFIG)
    assert eleccion.mirrored is True


def test_una_dinamica_solo_toma_grabaciones_dinamicas() -> None:
    """`NONE` y las grabaciones estáticas de una letra dinámica no sirven de
    referencia: la seña **es** el movimiento (ADR 0010)."""
    estatica = replace(
        synthetic_samples(("J",), signers=1, sessions=1, repetitions=1)[0],
        label="J",
    )
    trazo = moving_sequence(class_hand(3), arc_offsets(16))
    dinamica = replace(estatica, sequence=trazo, kind=SampleKind.DYNAMIC)

    eleccion = choose_reference(candidatas((estatica, dinamica)), Label.J, CONFIG)

    assert eleccion.sample is dinamica
    assert eleccion.candidates == 1


def test_sin_candidatas_validas_se_dice_cual_letra() -> None:
    with pytest.raises(ValueError, match="Q"):
        choose_reference([], Label.Q, CONFIG)


# --------------------------------------------------------------------------- #
# Proyección al lienzo
# --------------------------------------------------------------------------- #

LIENZO, MARGEN = 320, 0.1


def caja(puntos: tuple[Point2, ...]) -> tuple[float, float, float, float]:
    xs = [x for x, _ in puntos]
    ys = [y for _, y in puntos]
    return min(xs), min(ys), max(xs), max(ys)


def test_la_mano_llena_el_lienzo_respetando_el_margen() -> None:
    frame = to_frame(canonical_hand(), width=1280, height=720)

    (puntos,) = project_frames((frame,), mirrored=False, canvas_px=LIENZO, margin=MARGEN)

    x0, y0, x1, y1 = caja(puntos)
    utilizable = LIENZO * (1 - 2 * MARGEN)
    assert x0 >= LIENZO * MARGEN - 1e-6 and y0 >= LIENZO * MARGEN - 1e-6
    assert x1 <= LIENZO * (1 - MARGEN) + 1e-6 and y1 <= LIENZO * (1 - MARGEN) + 1e-6
    assert max(x1 - x0, y1 - y0) == pytest.approx(utilizable)


def test_espejar_invierte_el_eje_x_y_deja_el_y() -> None:
    frame = to_frame(canonical_hand(), width=1280, height=720)

    (normal,) = project_frames((frame,), mirrored=False, canvas_px=LIENZO, margin=MARGEN)
    (espejo,) = project_frames((frame,), mirrored=True, canvas_px=LIENZO, margin=MARGEN)

    for (x, y), (mx, my) in zip(normal, espejo, strict=True):
        assert mx == pytest.approx(LIENZO - x)
        assert my == pytest.approx(y)


def test_la_caja_es_una_sola_para_toda_la_secuencia() -> None:
    """En una dinámica el desplazamiento es la seña: encuadrar frame a frame lo
    borraría. Con dos frames, el segundo desplazado, el primero no queda centrado."""
    quieta = to_frame(canonical_hand(), width=1280, height=720)
    movida = to_frame(translated(canonical_hand(), 200.0, 0.0), width=1280, height=720)

    primero, segundo = project_frames(
        (quieta, movida), mirrored=False, canvas_px=LIENZO, margin=MARGEN
    )

    x0_primero, _, _, _ = caja(primero)
    _, _, x1_segundo, _ = caja(segundo)
    assert x0_primero == pytest.approx(LIENZO * MARGEN)
    assert x1_segundo == pytest.approx(LIENZO * (1 - MARGEN))
    assert caja(primero)[2] < caja(segundo)[2]


def test_la_relacion_de_aspecto_no_achata_la_mano() -> None:
    """Los landmarks vienen normalizados por ancho y alto por separado. Sin
    corregirlo, una mano en 16:9 saldría estirada a lo alto."""
    ancha = to_frame(canonical_hand(), width=1280, height=720)
    cuadrada = to_frame(canonical_hand(), width=720, height=720)

    (en_ancha,) = project_frames((ancha,), mirrored=False, canvas_px=LIENZO, margin=MARGEN)
    (en_cuadrada,) = project_frames((cuadrada,), mirrored=False, canvas_px=LIENZO, margin=MARGEN)

    for (x, y), (cx, cy) in zip(en_ancha, en_cuadrada, strict=True):
        assert x == pytest.approx(cx, abs=1e-6)
        assert y == pytest.approx(cy, abs=1e-6)
```

- [ ] **Step 2: Corre los tests para verificar que fallan**

Run: `uv run pytest tests/test_signs.py -v`
Expected: FAIL con `ImportError` de `Candidate`.

- [ ] **Step 3: Implementa**

En `src/lsm/signs.py`, añade a los imports: `import math`, `from lsm.features import DynamicUnavailable, ExtractionRejected, extract_sequence_features`, `from lsm.types import Handedness, Point2, RawFrame, Sample, SampleKind`. Al final del módulo:

```python
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
        sum(vector[i] for _, vector in vectores) / len(vectores) for i in range(dimension)
    ]

    def distancia(entrada: tuple[Candidate, tuple[float, ...]]) -> tuple[float, str]:
        _, vector = entrada
        return (
            math.sqrt(sum((v - c) ** 2 for v, c in zip(vector, centroide, strict=True))),
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
            (-lm.x * frame.aspect_ratio if mirrored else lm.x * frame.aspect_ratio, lm.y)
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
        tuple((x * escala + dx, y * escala + dy) for x, y in puntos) for puntos in crudos
    )
```

- [ ] **Step 4: Corre los tests y el gate completo**

Run: `uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest`
Expected: todo en verde.

- [ ] **Step 5: Commit**

```bash
git add src/lsm/signs.py tests/test_signs.py
git commit -m "feat(signs): medoide como muestra de referencia y proyeccion al lienzo"
```

---

### Task 6: `io/signs.py` — manifest en disco, candidatas y PNG/GIF con Pillow

**Files:**
- Modify: `pyproject.toml` (Pillow en `dev` y en `capture`)
- Create: `src/lsm/io/signs.py`
- Create: `tests/test_io_signs.py`

**Interfaces:**
- Consumes: `Manifest`, `Candidate`, `project_frames`; `lsm.io.dataset.iter_sample_paths`, `read_sample`, `DatasetError`; `lsm.types.HAND_CONNECTIONS`, `LandmarkIndex`
- Produces: `MANIFEST_FILENAME`, `DEFAULT_ASSETS_DIR`, `DEFAULT_RAW_DIR`, `load_manifest(path) -> Manifest`, `save_manifest(manifest, path) -> None`, `load_candidates(root) -> dict[Label, list[Candidate]]`, `write_static_asset(path, frames_points, canvas_px) -> None`, `write_dynamic_asset(path, frames_points, canvas_px, fps) -> int`, `gif_frame_count(path) -> int`, `load_asset_frames(path) -> list[Image.Image]`

- [ ] **Step 1: Añade Pillow**

En `pyproject.toml`:

```toml
[project.optional-dependencies]
capture = [
    "mediapipe>=1.0",
    "opencv-python>=4.10",
    "numpy>=1.26",
    # Escribe los GIF de las letras dinamicas y compone el cuadro del reproductor
    # (la fuente de OpenCV es ASCII y las descripciones llevan acentos).
    "pillow>=10.1",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "mypy>=1.10",
    "ruff>=0.6",
    "types-PyYAML>=6.0",
    # La suite renderiza assets en tmp_path y lee los GIF del manifest real.
    "pillow>=10.1",
]
```

Run: `uv sync && uv run python -c "import PIL; print(PIL.__version__)"`
Expected: imprime una versión ≥ 10.1.

- [ ] **Step 2: Escribe los tests que fallan**

`tests/test_io_signs.py`:

```python
"""Manifest en disco, candidatas del dataset y render PNG/GIF. Todo en tmp_path."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from PIL import Image

from lsm.io.dataset import SampleMetadata, StoredSample, write_sample
from lsm.io.signs import (
    MANIFEST_FILENAME,
    gif_frame_count,
    load_asset_frames,
    load_candidates,
    load_manifest,
    save_manifest,
    write_dynamic_asset,
    write_static_asset,
)
from lsm.signs import (
    FUENTE_NORMATIVA,
    MANIFEST_SCHEMA_VERSION,
    AssetReview,
    AssetSource,
    Manifest,
    SignAsset,
    expected_filename,
    project_frames,
)
from lsm.synthetic import arc_offsets, canonical_hand, moving_sequence, still_sequence
from lsm.types import (
    Distance,
    Handedness,
    LightDirection,
    LightLevel,
    SampleKind,
    Sequence,
)
from lsm.vocabulary import LETTERS, Label

FECHA = datetime(2026, 9, 9, 10, 0, tzinfo=UTC)
HOY = date(2026, 9, 14)


def manifiesto() -> Manifest:
    """Un manifest completo y válido, copiado de `vocabulary.py`. Es el mismo
    helper que en `tests/test_signs.py`; los tests no se importan entre sí."""
    letras: dict[Label, SignAsset] = {}
    for label, letra in LETTERS.items():
        letras[label] = SignAsset(
            letra=letra.display,
            archivo=expected_filename(label),
            es_dinamica=letra.es_dinamica,
            descripcion=letra.descripcion,
            trayectoria=letra.trayectoria,
            pagina=letra.pagina,
            duracion_ms=2000 if letra.es_dinamica else None,
            fuente=AssetSource(
                tipo="esqueleto_desde_dataset",
                muestra=f"s01/2026-09-09-manana/{label}/007.json",
                signer_id="s01",
                lateralidad_original=Handedness.RIGHT,
                espejada=False,
            ),
            revision=AssetReview(fecha=HOY, revisor="tests", resultado="coincide"),
        )
    return Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        fuente_normativa=FUENTE_NORMATIVA,
        letras=letras,
    )


def guardada(label: str, secuencia: Sequence, kind: SampleKind) -> StoredSample:
    metadata = SampleMetadata(
        label=label,
        signer_id="s01",
        session_id="2026-09-09-manana",
        timestamp=FECHA,
        handedness=Handedness.RIGHT,
        light_level=LightLevel.INDOOR,
        light_direction=LightDirection.FRONTAL,
        distance=Distance.MEDIUM,
        mean_luminance=0.5,
        mean_scale_px=90.0,
        kind=kind,
        dispersion=0.01,
        arc_length=0.0 if kind is SampleKind.STATIC else 1.2,
        handedness_swapped=False,
    )
    return StoredSample(metadata=metadata, frames=secuencia.frames)


def test_el_manifest_va_y_vuelve_y_siempre_con_los_mismos_bytes(tmp_path: Path) -> None:
    ruta = tmp_path / MANIFEST_FILENAME
    original = manifiesto()

    save_manifest(original, ruta)
    primera = ruta.read_bytes()
    save_manifest(load_manifest(ruta), ruta)

    assert load_manifest(ruta) == original
    assert ruta.read_bytes() == primera
    assert primera.endswith(b"\n")


def test_las_candidatas_se_agrupan_por_letra_con_su_ruta_relativa(tmp_path: Path) -> None:
    write_sample(tmp_path, guardada("A", still_sequence(canonical_hand(), length=8), SampleKind.STATIC))
    write_sample(tmp_path, guardada("A", still_sequence(canonical_hand(), length=8), SampleKind.STATIC))
    write_sample(
        tmp_path,
        guardada("J", moving_sequence(canonical_hand(), arc_offsets(12)), SampleKind.DYNAMIC),
    )

    candidatas = load_candidates(tmp_path)

    assert sorted(candidatas) == [Label.A, Label.J]
    assert len(candidatas[Label.A]) == 2
    assert candidatas[Label.A][0].path == "s01/2026-09-09-manana/A/001.json"
    assert candidatas[Label.J][0].sample.kind is SampleKind.DYNAMIC


def test_el_png_es_cuadrado_y_del_tamano_pedido(tmp_path: Path) -> None:
    frames = still_sequence(canonical_hand(), length=5).frames
    puntos = project_frames(frames, mirrored=False, canvas_px=200, margin=0.1)
    ruta = tmp_path / "A.png"

    write_static_asset(ruta, puntos, canvas_px=200)

    with Image.open(ruta) as imagen:
        assert imagen.size == (200, 200)
        assert imagen.format == "PNG"


def test_el_gif_lleva_un_frame_por_frame_y_devuelve_su_duracion(tmp_path: Path) -> None:
    frames = moving_sequence(canonical_hand(), arc_offsets(18)).frames
    puntos = project_frames(frames, mirrored=False, canvas_px=160, margin=0.1)
    ruta = tmp_path / "J.gif"

    duracion = write_dynamic_asset(ruta, puntos, canvas_px=160, fps=12)

    assert gif_frame_count(ruta) == 18
    assert duracion == round(1000 * 18 / 12)
    cuadros = load_asset_frames(ruta)
    assert len(cuadros) == 18
    assert cuadros[0].size == (160, 160)
    assert cuadros[0].mode == "RGB"


def test_un_png_se_carga_como_un_solo_cuadro(tmp_path: Path) -> None:
    frames = still_sequence(canonical_hand(), length=2).frames
    puntos = project_frames(frames, mirrored=False, canvas_px=64, margin=0.1)
    ruta = tmp_path / "A.png"
    write_static_asset(ruta, puntos, canvas_px=64)

    assert gif_frame_count(ruta) == 1
    assert len(load_asset_frames(ruta)) == 1
```

- [ ] **Step 3: Corre los tests para verificar que fallan**

Run: `uv run pytest tests/test_io_signs.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'lsm.io.signs'`.

- [ ] **Step 4: Escribe `src/lsm/io/signs.py`**

```python
"""Texto → señas: lo que toca disco y Pillow.

`lsm.signs` decide; esto lee el manifest, carga candidatas de `data/raw`, dibuja
los esqueletos y los guarda. Pillow y no OpenCV porque OpenCV no escribe GIF y
su fuente es ASCII: las descripciones del glosario llevan acentos y una Ñ.

Pillow es dependencia del extra `capture` y del grupo `dev`, no del núcleo.
Se importa arriba porque este módulo entero es I/O; nadie del núcleo lo importa.
"""

from __future__ import annotations

import json
from collections.abc import Sequence as SequenceABC
from pathlib import Path
from typing import Any, Final

from PIL import Image, ImageDraw

from lsm.io.dataset import DatasetError, iter_sample_paths, read_sample
from lsm.signs import Candidate, Manifest
from lsm.types import HAND_CONNECTIONS, LandmarkIndex, Point2
from lsm.vocabulary import Label

MANIFEST_FILENAME: Final = "manifest.json"
DEFAULT_ASSETS_DIR: Final = Path("assets/signs")
DEFAULT_RAW_DIR: Final = Path("data/raw")

# Constantes de dibujo, no umbrales: nada de esto cambia lo que el sistema
# decide, solo cómo se ve.
_FONDO: Final = (250, 250, 248)
_HUESO: Final = (40, 110, 60)
_ARTICULACION: Final = (25, 25, 25)
_YEMA: Final = (200, 60, 40)
_MUNECA: Final = (30, 80, 200)
_GROSOR_HUESO: Final = 4
_RADIO_ARTICULACION: Final = 4
_RADIO_YEMA: Final = 6
_RADIO_MUNECA: Final = 7
_YEMAS: Final = frozenset(
    {
        LandmarkIndex.THUMB_TIP,
        LandmarkIndex.INDEX_TIP,
        LandmarkIndex.MIDDLE_TIP,
        LandmarkIndex.RING_TIP,
        LandmarkIndex.PINKY_TIP,
    }
)


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def load_manifest(path: Path) -> Manifest:
    """Lee y valida. Los errores de esquema salen como `ValidationError`."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    return Manifest.model_validate(payload)


def save_manifest(manifest: Manifest, path: Path) -> None:
    """Escribe con claves ordenadas y salto final: mismos datos, mismos bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Candidatas
# --------------------------------------------------------------------------- #


def load_candidates(root: Path) -> dict[Label, list[Candidate]]:
    """Todas las muestras utilizables del dataset, agrupadas por letra.

    Una muestra con huecos (`to_sample` la rechaza) o con una etiqueta que el
    vocabulario no conoce se salta: no sirve de referencia y no es motivo para
    no renderizar las demás.
    """
    por_letra: dict[Label, list[Candidate]] = {}
    for ruta in iter_sample_paths(root):
        stored = read_sample(ruta)
        try:
            label = Label(stored.metadata.label)
            sample = stored.to_sample()
        except (ValueError, DatasetError):
            continue
        por_letra.setdefault(label, []).append(
            Candidate(sample=sample, path=ruta.relative_to(root).as_posix())
        )
    return por_letra


# --------------------------------------------------------------------------- #
# Dibujo
# --------------------------------------------------------------------------- #


def draw_skeleton(points: SequenceABC[Point2], canvas_px: int) -> Image.Image:
    """Un frame: huesos, articulaciones, yemas y muñeca destacadas."""
    imagen = Image.new("RGB", (canvas_px, canvas_px), _FONDO)
    lapiz = ImageDraw.Draw(imagen)
    for inicio, fin in HAND_CONNECTIONS:
        lapiz.line([points[inicio], points[fin]], fill=_HUESO, width=_GROSOR_HUESO)
    for indice, (x, y) in enumerate(points):
        if indice == LandmarkIndex.WRIST:
            radio, color = _RADIO_MUNECA, _MUNECA
        elif indice in _YEMAS:
            radio, color = _RADIO_YEMA, _YEMA
        else:
            radio, color = _RADIO_ARTICULACION, _ARTICULACION
        lapiz.ellipse([x - radio, y - radio, x + radio, y + radio], fill=color)
    return imagen


def write_static_asset(
    path: Path, frames_points: SequenceABC[SequenceABC[Point2]], canvas_px: int
) -> None:
    """PNG del frame central de la secuencia."""
    central = frames_points[len(frames_points) // 2]
    path.parent.mkdir(parents=True, exist_ok=True)
    draw_skeleton(central, canvas_px).save(path, format="PNG", optimize=True)


def write_dynamic_asset(
    path: Path,
    frames_points: SequenceABC[SequenceABC[Point2]],
    canvas_px: int,
    fps: int,
) -> int:
    """GIF con todos los frames, en bucle. Devuelve `duracion_ms` de una vuelta."""
    cuadros = [draw_skeleton(puntos, canvas_px) for puntos in frames_points]
    path.parent.mkdir(parents=True, exist_ok=True)
    cuadros[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=cuadros[1:],
        duration=round(1000 / fps),
        loop=0,
        disposal=2,
        optimize=False,
    )
    return round(1000 * len(cuadros) / fps)


def gif_frame_count(path: Path) -> int:
    with Image.open(path) as imagen:
        return int(getattr(imagen, "n_frames", 1))


def load_asset_frames(path: Path) -> list[Image.Image]:
    """Todos los cuadros de un asset como imágenes RGB independientes."""
    cuadros: list[Image.Image] = []
    with Image.open(path) as imagen:
        for indice in range(int(getattr(imagen, "n_frames", 1))):
            imagen.seek(indice)
            cuadros.append(imagen.convert("RGB").copy())
    return cuadros
```

- [ ] **Step 5: Corre los tests y mypy**

Run: `uv run pytest tests/test_io_signs.py -v && uv run mypy`
Expected: PASS. Si mypy no encuentra stubs de `PIL`, añade `"PIL", "PIL.*"` a la lista `module` del override de `[[tool.mypy.overrides]]` en `pyproject.toml` (Pillow ≥ 10 lleva `py.typed`; no debería hacer falta).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/lsm/io/signs.py tests/test_io_signs.py
git commit -m "feat(io): manifest en disco, candidatas del dataset y render PNG/GIF con Pillow"
```

---

### Task 7: `lsm-signs render` y `lsm-signs verificar`

**Files:**
- Create: `src/lsm/cli/signs.py`
- Create: `tests/test_cli_signs.py`
- Modify: `pyproject.toml` (`[project.scripts]`)

**Interfaces:**
- Consumes: todo lo de las tareas 2–6
- Produces: `main(argv) -> int`; `render(raw, assets, config, revisor, hoy) -> int`; `verificar(assets, config) -> list[str]`; `SIN_MANIFEST: Final[str]`

- [ ] **Step 1: Registra el script**

En `pyproject.toml`, `[project.scripts]`:

```toml
lsm-signs = "lsm.cli.signs:main"
```

Run: `uv sync`

- [ ] **Step 2: Escribe los tests que fallan**

`tests/test_cli_signs.py`:

```python
"""Los tres subcomandos de `lsm-signs`, sin cámara ni ventana.

`render` y `verificar` corren de verdad sobre un dataset sintético en tmp_path.
`reproducir` se prueba hasta justo antes de abrir la ventana, y su bucle con una
ventana falsa (Task 8).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lsm.cli.signs import main
from lsm.io.dataset import SampleMetadata, StoredSample, write_sample
from lsm.io.signs import MANIFEST_FILENAME, load_manifest
from lsm.signs import manifest_drift
from lsm.synthetic import arc_offsets, class_hand, moving_sequence, synthetic_samples
from lsm.types import (
    Distance,
    Handedness,
    LightDirection,
    LightLevel,
    SampleKind,
    Sequence,
)
from lsm.vocabulary import ALPHABET, DYNAMIC_LABELS, LETTERS, Label

RAIZ_REPO = Path(__file__).resolve().parent.parent
CONFIG_YAML = RAIZ_REPO / "config.yaml"
FECHA = datetime(2026, 9, 9, 10, 0, tzinfo=UTC)


def _stored(label: str, secuencia: Sequence, kind: SampleKind, signer: str) -> StoredSample:
    metadata = SampleMetadata(
        label=label,
        signer_id=signer,
        session_id="2026-09-09-manana",
        timestamp=FECHA,
        handedness=Handedness.RIGHT,
        light_level=LightLevel.INDOOR,
        light_direction=LightDirection.FRONTAL,
        distance=Distance.MEDIUM,
        mean_luminance=0.5,
        mean_scale_px=90.0,
        kind=kind,
        dispersion=0.01,
        arc_length=0.0 if kind is SampleKind.STATIC else 1.2,
        handedness_swapped=False,
    )
    return StoredSample(metadata=metadata, frames=secuencia.frames)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Un data/raw con las 29 letras: estáticas de `synthetic_samples`,
    dinámicas como un trazo en gancho sobre una mano distinta por letra."""
    raiz = tmp_path / "raw"
    estaticas = tuple(sorted(str(label) for label in ALPHABET if label not in DYNAMIC_LABELS))
    for muestra in synthetic_samples(estaticas, signers=1, sessions=1, repetitions=2, frames=6):
        write_sample(raiz, _stored(muestra.label, muestra.sequence, SampleKind.STATIC, "s01"))
    for ordinal, label in enumerate(sorted(DYNAMIC_LABELS)):
        trazo = moving_sequence(class_hand(ordinal + 1), arc_offsets(12))
        write_sample(raiz, _stored(str(label), trazo, SampleKind.DYNAMIC, "s01"))
    return raiz


def render(dataset: Path, assets: Path) -> int:
    return main(
        [
            "render",
            "--raw", str(dataset),
            "--assets", str(assets),
            "--config", str(CONFIG_YAML),
            "--revisor", "tests",
            "--hoy", "2026-09-14",
        ]
    )  # fmt: skip


def test_render_produce_29_assets_y_un_manifest_sin_deriva(
    dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assets = tmp_path / "signs"

    assert render(dataset, assets) == 0

    manifest = load_manifest(assets / MANIFEST_FILENAME)
    assert set(manifest.letras) == set(LETTERS)
    assert manifest_drift(manifest) == []
    for label, entrada in manifest.letras.items():
        assert (assets / entrada.archivo).is_file(), label
        assert entrada.revision.resultado == "pendiente"
        assert entrada.revision.revisor == "tests"
        assert entrada.fuente.muestra.startswith("s01/2026-09-09-manana/")
    assert "29" in capsys.readouterr().out


def test_render_es_determinista(dataset: Path, tmp_path: Path) -> None:
    assets = tmp_path / "signs"
    render(dataset, assets)
    primera = (assets / MANIFEST_FILENAME).read_bytes()

    render(dataset, assets)

    assert (assets / MANIFEST_FILENAME).read_bytes() == primera


def test_un_re_render_conserva_la_revision_si_la_muestra_no_cambio(
    dataset: Path, tmp_path: Path
) -> None:
    """Regenerar los assets no puede borrar trabajo humano sin avisar."""
    assets = tmp_path / "signs"
    render(dataset, assets)
    ruta = assets / MANIFEST_FILENAME
    payload = json.loads(ruta.read_text(encoding="utf-8"))
    payload["letras"]["A"]["revision"] = {
        "fecha": "2026-09-13",
        "revisor": "persona",
        "resultado": "coincide",
        "nota": "pulgar visible",
    }
    ruta.write_text(json.dumps(payload), encoding="utf-8")

    render(dataset, assets)

    manifest = load_manifest(ruta)
    assert manifest.letras[Label.A].revision.resultado == "coincide"
    assert manifest.letras[Label.A].revision.nota == "pulgar visible"
    assert manifest.letras[Label.B].revision.resultado == "pendiente"


def test_render_sin_una_letra_no_escribe_nada_y_dice_cual(
    dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for ruta in (dataset / "s01" / "2026-09-09-manana" / "Q").glob("*.json"):
        ruta.unlink()
    assets = tmp_path / "signs"

    assert render(dataset, assets) == 1

    assert not (assets / MANIFEST_FILENAME).exists()
    assert "Q" in capsys.readouterr().out


def test_verificar_exige_archivos_presentes_y_revisiones_cerradas(
    dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assets = tmp_path / "signs"
    render(dataset, assets)
    (assets / "J.gif").unlink()

    codigo = main(["verificar", "--assets", str(assets), "--config", str(CONFIG_YAML)])

    salida = capsys.readouterr().out
    assert codigo == 1
    assert "J.gif" in salida
    assert "pendiente" in salida


def test_verificar_sin_manifest_lo_dice(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    codigo = main(["verificar", "--assets", str(tmp_path), "--config", str(CONFIG_YAML)])

    assert codigo == 1
    assert "lsm-signs render" in capsys.readouterr().out
```

- [ ] **Step 3: Corre los tests para verificar que fallan**

Run: `uv run pytest tests/test_cli_signs.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'lsm.cli.signs'`.

- [ ] **Step 4: Escribe `src/lsm/cli/signs.py` con `render` y `verificar`**

```python
"""`lsm-signs`: la dirección texto → señas (`ARQUITECTURA.md` §4.10, Fase 4).

Tres subcomandos:

- `render`: dibuja los 29 assets desde `data/raw` y escribe el manifest.
- `verificar`: el manifest y los archivos están completos y revisados.
- `reproducir "texto"`: la ventana, con temporizador y control manual.

No hay cámara en ninguno. OpenCV solo hace falta en `reproducir`, para la
ventana, y se importa ahí dentro.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Final

from lsm.config import Config, load_config
from lsm.io.signs import (
    DEFAULT_ASSETS_DIR,
    DEFAULT_RAW_DIR,
    MANIFEST_FILENAME,
    gif_frame_count,
    load_candidates,
    load_manifest,
    save_manifest,
    write_dynamic_asset,
    write_static_asset,
)
from lsm.signs import (
    FUENTE_NORMATIVA,
    MANIFEST_SCHEMA_VERSION,
    AssetReview,
    AssetSource,
    Manifest,
    SignAsset,
    choose_reference,
    expected_filename,
    manifest_drift,
    project_frames,
)
from lsm.vocabulary import ALPHABET, LETTERS, Label

SIN_MANIFEST: Final = (
    "no existe {ruta}.\n"
    "Los assets se generan desde el dataset propio:\n"
    "  lsm-signs render --revisor <tu nombre>"
)


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #


def render(raw: Path, assets: Path, config: Config, revisor: str, hoy: date) -> int:
    """Dibuja un asset por letra y escribe el manifest. Falla antes de escribir
    nada si a alguna letra le faltan candidatas."""
    candidatas = load_candidates(raw)
    faltan = [str(label) for label in ALPHABET if not candidatas.get(label)]
    if faltan:
        print(f"sin muestras en {raw} para: {', '.join(faltan)}. No se escribe nada.")
        return 1

    ruta_manifest = assets / MANIFEST_FILENAME
    previo = load_manifest(ruta_manifest) if ruta_manifest.is_file() else None

    letras: dict[Label, SignAsset] = {}
    for label in ALPHABET:
        letra = LETTERS[label]
        eleccion = choose_reference(candidatas[label], label, config)
        puntos = project_frames(
            eleccion.sample.sequence.frames,
            mirrored=eleccion.mirrored,
            canvas_px=config.signs.canvas_px,
            margin=config.signs.canvas_margin,
        )
        archivo = assets / expected_filename(label)
        duracion: int | None = None
        if letra.es_dinamica:
            duracion = write_dynamic_asset(
                archivo, puntos, config.signs.canvas_px, config.signs.render_fps
            )
        else:
            write_static_asset(archivo, puntos, config.signs.canvas_px)

        anterior = previo.letras.get(label) if previo is not None else None
        if anterior is not None and anterior.fuente.muestra == eleccion.path:
            revision = anterior.revision
        else:
            revision = AssetReview(fecha=hoy, revisor=revisor, resultado="pendiente")

        letras[label] = SignAsset(
            letra=letra.display,
            archivo=archivo.name,
            es_dinamica=letra.es_dinamica,
            descripcion=letra.descripcion,
            trayectoria=letra.trayectoria,
            pagina=letra.pagina,
            duracion_ms=duracion,
            fuente=AssetSource(
                tipo="esqueleto_desde_dataset",
                muestra=eleccion.path,
                signer_id=eleccion.sample.signer_id,
                lateralidad_original=eleccion.sample.handedness,
                espejada=eleccion.mirrored,
            ),
            revision=revision,
        )
        print(
            f"{letra.display:>3}  {archivo.name:<14} {eleccion.path}  "
            f"({eleccion.candidates} candidatas{', espejada' if eleccion.mirrored else ''})"
        )

    manifest = Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        fuente_normativa=FUENTE_NORMATIVA,
        letras=letras,
    )
    save_manifest(manifest, ruta_manifest)
    pendientes = sum(1 for a in letras.values() if a.revision.resultado != "coincide")
    print(f"\n{len(letras)} assets en {assets}; {pendientes} revisiones pendientes.")
    return 0


# --------------------------------------------------------------------------- #
# verificar
# --------------------------------------------------------------------------- #


def verificar(assets: Path, config: Config) -> list[str]:
    """Todo lo que tiene que cumplirse para dar la fase por cerrada."""
    ruta_manifest = assets / MANIFEST_FILENAME
    if not ruta_manifest.is_file():
        return [SIN_MANIFEST.format(ruta=ruta_manifest)]
    manifest = load_manifest(ruta_manifest)
    problemas = manifest_drift(manifest)
    for label, asset in manifest.letras.items():
        archivo = assets / asset.archivo
        if not archivo.is_file():
            problemas.append(f"{label}: falta el archivo {asset.archivo}")
        elif asset.es_dinamica:
            assert asset.duracion_ms is not None
            esperados = round(asset.duracion_ms * config.signs.render_fps / 1000)
            reales = gif_frame_count(archivo)
            if reales != esperados:
                problemas.append(
                    f"{label}: {asset.archivo} tiene {reales} frames y duracion_ms "
                    f"dice {esperados}"
                )
        if asset.revision.resultado != "coincide":
            problemas.append(
                f"{label}: revisión {asset.revision.resultado}"
                + (f" — {asset.revision.nota}" if asset.revision.nota else "")
            )
    return problemas


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lsm-signs",
        description=(
            "Deletreo manual, dirección texto → señas: muestra la secuencia de "
            "señas del abecedario LSM para un texto. Sin cámara."
        ),
    )
    # Opciones comunes como `parents`: argparse solo reconoce las opciones del
    # subparser que está parseando, así que `lsm-signs render --assets X` no
    # funcionaría con `--assets` definido solo en el parser raíz.
    comun = argparse.ArgumentParser(add_help=False)
    comun.add_argument("--config", type=Path, default=Path("config.yaml"))
    comun.add_argument(
        "--assets",
        type=Path,
        default=DEFAULT_ASSETS_DIR,
        help="carpeta con manifest.json y los PNG/GIF",
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    p_render = sub.add_parser(
        "render", parents=[comun], help="dibuja los assets desde el dataset propio"
    )
    p_render.add_argument("--raw", type=Path, default=DEFAULT_RAW_DIR)
    p_render.add_argument(
        "--revisor",
        required=True,
        help="quién queda como revisor de las entradas nuevas (resultado: pendiente)",
    )
    p_render.add_argument(
        "--hoy",
        type=date.fromisoformat,
        default=None,
        help="fecha de las revisiones nuevas; por defecto, hoy",
    )

    sub.add_parser(
        "verificar",
        parents=[comun],
        help="manifest completo, archivos presentes, revisiones cerradas",
    )

    p_play = sub.add_parser(
        "reproducir", parents=[comun], help="muestra la secuencia de señas de un texto"
    )
    p_play.add_argument("texto", help='el texto a deletrear, por ejemplo "casa"')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)

    if args.comando == "render":
        hoy = args.hoy if args.hoy is not None else date.today()
        return render(args.raw, args.assets, config, args.revisor, hoy)

    if args.comando == "verificar":
        problemas = verificar(args.assets, config)
        for problema in problemas:
            print(problema)
        if problemas:
            print(f"\n{len(problemas)} problema(s).")
            return 1
        print("manifest completo: 29 letras con archivo, fuente y revisión.")
        return 0

    return reproducir(args.texto, args.assets, config)


def reproducir(texto: str, assets: Path, config: Config) -> int:
    raise NotImplementedError("Task 8")
```

- [ ] **Step 5: Corre los tests**

Run: `uv run pytest tests/test_cli_signs.py -v && uv run mypy && uv run ruff check .`
Expected: PASS. (`reproducir` aún no se ejercita.) Si `synthetic_samples` tarda: son 21 letras × 2 repeticiones; `render` extrae features de 50 secuencias cortas, debería quedar por debajo de 5 s.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/lsm/cli/signs.py tests/test_cli_signs.py
git commit -m "feat(cli): lsm-signs render y verificar"
```

---

### Task 8: `lsm-signs reproducir` — el cuadro y la ventana

**Files:**
- Modify: `src/lsm/signs.py` (añadir `Scene`)
- Modify: `src/lsm/io/signs.py` (añadir `draw_scene`)
- Modify: `src/lsm/cli/signs.py` (`reproducir`, `Ventana`, `bucle`, `VentanaOpenCV`)
- Modify: `tests/test_io_signs.py`, `tests/test_cli_signs.py`

**Interfaces:**
- Produces en `signs.py`: `Scene(tokens, playlist, state, asset: SignAsset | None, frame: Any | None)`
- Produces en `io/signs.py`: `draw_scene(scene: Scene, config: Config) -> Image.Image`
- Produces en `cli/signs.py`: `Ventana` (Protocol con `mostrar(imagen)`, `tecla(espera_ms) -> int`, `cerrar()`), `bucle(playlist, tokens, manifest, cuadros, config, ventana, reloj, componer) -> PlayerState`, `reproducir(texto, assets, config, ventana=None, reloj=None) -> int`

- [ ] **Step 1: Escribe los tests que fallan**

En `tests/test_io_signs.py`, añade a los imports `from lsm.config import Config`, `from lsm.io.signs import draw_scene`, `from lsm.signs import PlayerState, Scene, build_playlist, text_to_symbols` y `from lsm.vocabulary import Label` (ya está). Al final:

```python
def test_el_cuadro_tiene_asset_a_la_izquierda_y_texto_a_la_derecha() -> None:
    config = Config()
    manifest = manifiesto()
    tokens = text_to_symbols("año")
    playlist = build_playlist(tokens, manifest, config)
    lienzo = Image.new("RGB", (config.signs.canvas_px, config.signs.canvas_px), (255, 0, 0))
    escena = Scene(
        tokens=tokens,
        playlist=playlist,
        state=PlayerState(index=1, elapsed_ms=300.0),
        asset=manifest.letras[Label.ENIE],
        frame=lienzo,
    )

    cuadro = draw_scene(escena, config)

    assert cuadro.width > config.signs.canvas_px
    assert cuadro.height > config.signs.canvas_px
    # El asset se pega tal cual: un píxel del centro del panel izquierdo es rojo.
    centro = config.signs.canvas_px // 2
    assert cuadro.getpixel((centro, centro)) == (255, 0, 0)


def test_una_pausa_se_dibuja_sin_asset() -> None:
    config = Config()
    manifest = manifiesto()
    tokens = text_to_symbols("a b")
    escena = Scene(
        tokens=tokens,
        playlist=build_playlist(tokens, manifest, config),
        state=PlayerState(index=1),
        asset=None,
        frame=None,
    )

    cuadro = draw_scene(escena, config)

    assert cuadro.width > 0
```

En `tests/test_cli_signs.py`, añade a los imports `from dataclasses import dataclass, field`, `from lsm.cli.signs import bucle, reproducir`, `from lsm.config import Config`, `from lsm.signs import build_playlist, text_to_symbols` y de `lsm.io.signs`: `load_asset_frames` (ya se importa `load_manifest`). Al final:

```python
# --------------------------------------------------------------------------- #
# reproducir
# --------------------------------------------------------------------------- #


@dataclass
class VentanaFalsa:
    """Devuelve las teclas programadas, luego -1, y cuenta cuadros mostrados."""

    teclas: list[int]
    mostrados: int = 0
    cerrada: bool = False
    esperas: list[int] = field(default_factory=list)

    def mostrar(self, imagen: object) -> None:
        self.mostrados += 1

    def tecla(self, espera_ms: int) -> int:
        self.esperas.append(espera_ms)
        return self.teclas.pop(0) if self.teclas else -1

    def cerrar(self) -> None:
        self.cerrada = True


class RelojFalso:
    """Avanza `paso_s` en cada lectura."""

    def __init__(self, paso_s: float) -> None:
        self.ahora = 0.0
        self.paso = paso_s

    def __call__(self) -> float:
        self.ahora += self.paso
        return self.ahora


def test_el_bucle_termina_con_q_y_los_ticks_salen_del_reloj(
    dataset: Path, tmp_path: Path
) -> None:
    assets = tmp_path / "signs"
    render(dataset, assets)
    config = Config()
    manifest = load_manifest(assets / MANIFEST_FILENAME)
    tokens = text_to_symbols("ab")
    playlist = build_playlist(tokens, manifest, config)
    cuadros = {label: load_asset_frames(assets / manifest.letras[label].archivo) for label in (Label.A, Label.B)}
    # 100 ms por vuelta; 20 vueltas son 2 s: la A (1.5 s) ya pasó a la B.
    ventana = VentanaFalsa(teclas=[-1] * 20 + [ord("q")])

    estado = bucle(playlist, tokens, manifest, cuadros, config, ventana, RelojFalso(0.1), lambda escena: None)

    assert estado.index == 1
    assert ventana.mostrados == 21
    assert ventana.cerrada
    assert set(ventana.esperas) == {config.signs.tick_ms}


def test_las_teclas_controlan_el_reproductor(dataset: Path, tmp_path: Path) -> None:
    assets = tmp_path / "signs"
    render(dataset, assets)
    config = Config()
    manifest = load_manifest(assets / MANIFEST_FILENAME)
    tokens = text_to_symbols("abc")
    playlist = build_playlist(tokens, manifest, config)
    cuadros = {label: load_asset_frames(assets / manifest.letras[label].archivo) for label in (Label.A, Label.B, Label.C)}
    ventana = VentanaFalsa(teclas=[ord("n"), ord("n"), ord("p"), ord(" "), ord("+"), ord("q")])

    estado = bucle(playlist, tokens, manifest, cuadros, config, ventana, RelojFalso(0.001), lambda escena: None)

    assert estado.index == 1
    assert estado.paused
    assert estado.speed == 1.0 + config.signs.speed_step


def test_reproducir_rechaza_caracteres_sin_sena_antes_de_abrir_nada(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    codigo = reproducir("hola2", tmp_path, Config())

    salida = capsys.readouterr().out
    assert codigo == 1
    assert "'2'" in salida


def test_reproducir_sin_letras_lo_dice(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert reproducir("   ", tmp_path, Config()) == 1
    assert "ninguna letra" in capsys.readouterr().out


def test_reproducir_sin_manifest_apunta_a_render(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert reproducir("casa", tmp_path, Config()) == 1
    assert "lsm-signs render" in capsys.readouterr().out


def test_reproducir_con_un_archivo_que_falta_da_la_ruta(
    dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assets = tmp_path / "signs"
    render(dataset, assets)
    (assets / "S.png").unlink()

    codigo = reproducir("casa", assets, Config())

    assert codigo == 1
    assert "S.png" in capsys.readouterr().out


def test_reproducir_con_ventana_inyectada_no_necesita_opencv(
    dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assets = tmp_path / "signs"
    render(dataset, assets)
    ventana = VentanaFalsa(teclas=[ord("q")])

    codigo = reproducir("casa", assets, Config(), ventana=ventana, reloj=RelojFalso(0.01))

    assert codigo == 0
    assert ventana.mostrados == 1
    assert "C A S A" in capsys.readouterr().out
```

- [ ] **Step 2: Corre los tests para verificar que fallan**

Run: `uv run pytest tests/test_io_signs.py tests/test_cli_signs.py -v`
Expected: FAIL con `ImportError` de `Scene` / `bucle`.

- [ ] **Step 3: `Scene` en `signs.py`**

Al final de `src/lsm/signs.py` (añade `from typing import Any` a los imports):

```python
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
```

- [ ] **Step 4: `draw_scene` en `io/signs.py`**

Añade a los imports de `src/lsm/io/signs.py`: `from PIL import Image, ImageDraw, ImageFont`, `from lsm.config import Config`, `from lsm.signs import Candidate, Manifest, Scene, StepKind, WordGap, render_tokens` y `from lsm.vocabulary import LETTERS, Label`. Constantes y función, al final del módulo:

```python
# --------------------------------------------------------------------------- #
# El cuadro de la ventana
# --------------------------------------------------------------------------- #

_PANEL_ANCHO: Final = 520
_PIE_ALTO: Final = 96
_MARGEN_TEXTO: Final = 24
_TINTA: Final = (30, 30, 30)
_TINTA_SUAVE: Final = (110, 110, 110)
_ACENTO: Final = (30, 80, 200)
_BARRA_FONDO: Final = (225, 225, 222)
_AVISO: Final = "Deletreo manual, no LSM como lengua. Procesamiento local."
_ATAJOS: Final = "ESPACIO pausa | n/p siguiente/anterior | r reinicio | +/- velocidad | q salir"


def _fuente(tamano: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # La fuente por defecto de Pillow ≥ 10.1 es escalable y cubre latín con
    # acentos y Ñ, que es lo que la Hershey de OpenCV no hace.
    return ImageFont.load_default(size=tamano)


def _envolver(texto: str, ancho_max: int, fuente: ImageFont.FreeTypeFont | ImageFont.ImageFont, lapiz: ImageDraw.ImageDraw) -> list[str]:
    lineas: list[str] = []
    actual = ""
    for palabra in texto.split():
        prueba = f"{actual} {palabra}".strip()
        if lapiz.textlength(prueba, font=fuente) <= ancho_max or not actual:
            actual = prueba
        else:
            lineas.append(actual)
            actual = palabra
    if actual:
        lineas.append(actual)
    return lineas


def draw_scene(scene: Scene, config: Config) -> Image.Image:
    """Asset a la izquierda; letra, descripción, progreso y estado a la derecha;
    el texto completo con el símbolo actual resaltado abajo."""
    lado = config.signs.canvas_px
    ancho, alto = lado + _PANEL_ANCHO, lado + _PIE_ALTO
    cuadro = Image.new("RGB", (ancho, alto), _FONDO)
    lapiz = ImageDraw.Draw(cuadro)

    # Panel izquierdo: el asset, o "espacio" en una pausa.
    if scene.frame is not None:
        cuadro.paste(scene.frame.resize((lado, lado)), (0, 0))
    else:
        lapiz.rectangle([0, 0, lado, lado], fill=_BARRA_FONDO)
        lapiz.text((lado // 2, lado // 2), "espacio", fill=_TINTA_SUAVE, font=_fuente(28), anchor="mm")

    # Panel derecho.
    x = lado + _MARGEN_TEXTO
    y = _MARGEN_TEXTO
    posicion = f"{scene.state.index + 1} / {len(scene.playlist)}"
    lapiz.text((ancho - _MARGEN_TEXTO, y), posicion, fill=_TINTA_SUAVE, font=_fuente(20), anchor="ra")
    if scene.asset is not None:
        lapiz.text((x, y), scene.asset.letra, fill=_TINTA, font=_fuente(72))
        y += 96
        for linea in _envolver(scene.asset.descripcion, _PANEL_ANCHO - 2 * _MARGEN_TEXTO, _fuente(18), lapiz):
            lapiz.text((x, y), linea, fill=_TINTA, font=_fuente(18))
            y += 24
        if scene.asset.es_dinamica:
            y += 8
            lapiz.text((x, y), f"Trayectoria: {scene.asset.trayectoria}", fill=_ACENTO, font=_fuente(18))
            y += 24
    else:
        lapiz.text((x, y), "pausa entre palabras", fill=_TINTA_SUAVE, font=_fuente(28))

    # Estado, velocidad y barra de progreso, pegados al borde inferior del panel.
    estado = "PAUSA" if scene.state.paused else ("FIN" if scene.state.finished else "REPRODUCIENDO")
    y_barra = lado - _MARGEN_TEXTO - 32
    lapiz.text((x, y_barra - 28), f"{estado}   {scene.state.speed:.2f}x", fill=_TINTA, font=_fuente(18))
    ancho_barra = _PANEL_ANCHO - 2 * _MARGEN_TEXTO
    lapiz.rectangle([x, y_barra, x + ancho_barra, y_barra + 12], fill=_BARRA_FONDO)
    lapiz.rectangle([x, y_barra, x + int(ancho_barra * scene.progress), y_barra + 12], fill=_ACENTO)
    transcurrido = scene.state.elapsed_ms / 1000.0
    total = scene.step.duration_ms / 1000.0
    lapiz.text((x + ancho_barra, y_barra + 16), f"{transcurrido:.1f} s / {total:.1f} s", fill=_TINTA_SUAVE, font=_fuente(16), anchor="ra")

    # Pie: el texto completo, símbolo a símbolo, con el actual resaltado.
    y_pie = lado + 20
    x_pie = _MARGEN_TEXTO
    fuente_pie = _fuente(26)
    for indice, token in enumerate(scene.tokens):
        simbolo = "·" if isinstance(token, WordGap) else LETTERS[token].display
        color = _ACENTO if indice == scene.token_index else _TINTA
        lapiz.text((x_pie, y_pie), simbolo, fill=color, font=fuente_pie)
        x_pie += int(lapiz.textlength(simbolo + "  ", font=fuente_pie))
    lapiz.text((_MARGEN_TEXTO, alto - 40), _ATAJOS, fill=_TINTA_SUAVE, font=_fuente(14))
    lapiz.text((_MARGEN_TEXTO, alto - 22), _AVISO, fill=_TINTA_SUAVE, font=_fuente(14))
    return cuadro
```

Si `render_tokens` no se usa aquí, quítalo del import. Si mypy se queja del tipo de retorno de `ImageFont.load_default`, anota `_fuente` como `-> Any`.

- [ ] **Step 5: `reproducir`, `Ventana` y `bucle` en `cli/signs.py`**

Reemplaza el `reproducir` provisional. Añade a los imports: `import time`, `from collections.abc import Callable, Mapping`, `from typing import Any, Protocol`, `from lsm.cli import MENSAJE_SIN_EXTRAS`, `from lsm.io.signs import draw_scene, load_asset_frames`, y de `lsm.signs`: `Faster, Next, PlayerInput, PlayerState, Prev, Restart, Scene, Slower, StepKind, Tick, TogglePause, Token, UnsupportedCharacters, asset_frame, build_playlist, player_step, render_tokens, start, text_to_symbols`.

```python
# --------------------------------------------------------------------------- #
# reproducir
# --------------------------------------------------------------------------- #

_SALIR: Final = frozenset({ord("q"), 27})
_TECLAS: Final[dict[int, PlayerInput]] = {
    ord(" "): TogglePause(),
    ord("n"): Next(),
    ord("p"): Prev(),
    ord("r"): Restart(),
    ord("+"): Faster(),
    ord("="): Faster(),
    ord("-"): Slower(),
}


class Ventana(Protocol):
    """Lo que el bucle necesita de una ventana. La real es OpenCV; los tests
    inyectan una falsa con teclas programadas."""

    def mostrar(self, imagen: Any) -> None: ...
    def tecla(self, espera_ms: int) -> int: ...
    def cerrar(self) -> None: ...


class VentanaOpenCV:
    def __init__(self, titulo: str) -> None:
        self.titulo = titulo

    def mostrar(self, imagen: Any) -> None:
        import cv2
        import numpy as np

        # Pillow entrega RGB; OpenCV espera BGR.
        cv2.imshow(self.titulo, np.asarray(imagen)[:, :, ::-1])

    def tecla(self, espera_ms: int) -> int:
        import cv2

        return int(cv2.waitKey(espera_ms) & 0xFF)

    def cerrar(self) -> None:
        import cv2

        cv2.destroyAllWindows()


def bucle(
    playlist: tuple[Step, ...],
    tokens: tuple[Token, ...],
    manifest: Manifest,
    cuadros: Mapping[Label, list[Any]],
    config: Config,
    ventana: Ventana,
    reloj: Callable[[], float],
    componer: Callable[[Scene], Any],
) -> PlayerState:
    """Mostrar, leer tecla, medir el tiempo, avanzar. Hasta `q`.

    El `dt` de cada `Tick` es tiempo de pared entre vueltas, medido con `reloj`:
    `config.signs.tick_ms` solo dice cuánto espera `waitKey`. Una tecla sustituye
    al tick de esa vuelta; el par de milisegundos que se pierden no se notan.
    """
    estado = start(playlist)
    anterior = reloj()
    try:
        while True:
            paso = playlist[estado.index]
            asset = manifest.letras[paso.label] if paso.label is not None else None
            frame: Any | None = None
            if paso.label is not None and asset is not None:
                imagenes = cuadros[paso.label]
                if asset.es_dinamica and asset.duracion_ms is not None:
                    frame = imagenes[asset_frame(estado, len(imagenes), asset.duracion_ms)]
                else:
                    frame = imagenes[0]
            ventana.mostrar(componer(Scene(tokens, playlist, estado, asset, frame)))

            tecla = ventana.tecla(config.signs.tick_ms)
            ahora = reloj()
            dt_ms = (ahora - anterior) * 1000.0
            anterior = ahora
            if tecla in _SALIR:
                return estado
            evento = _TECLAS.get(tecla, Tick(dt_ms))
            estado, _ = player_step(estado, evento, playlist, config)
    finally:
        ventana.cerrar()


def reproducir(
    texto: str,
    assets: Path,
    config: Config,
    ventana: Ventana | None = None,
    reloj: Callable[[], float] | None = None,
) -> int:
    """Texto → símbolos → pasos → ventana. Todo lo que puede fallar, falla
    antes de abrir la ventana y con un mensaje concreto."""
    try:
        tokens = text_to_symbols(texto)
    except UnsupportedCharacters as error:
        print(error)
        return 1
    if not any(not isinstance(token, WordGap) for token in tokens):
        print("el texto no tiene ninguna letra que deletrear")
        return 1

    ruta_manifest = assets / MANIFEST_FILENAME
    if not ruta_manifest.is_file():
        print(SIN_MANIFEST.format(ruta=ruta_manifest))
        return 1
    manifest = load_manifest(ruta_manifest)

    try:
        playlist = build_playlist(tokens, manifest, config)
    except ValueError as error:
        print(error)
        return 1

    letras = {paso.label for paso in playlist if paso.label is not None}
    faltan = [manifest.letras[label].archivo for label in sorted(letras) if not (assets / manifest.letras[label].archivo).is_file()]
    if faltan:
        print(f"faltan assets en {assets}: {', '.join(faltan)}. Corre lsm-signs render.")
        return 1

    if ventana is None:
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
        except ImportError as error:
            print(MENSAJE_SIN_EXTRAS.format(modulo=error.name))
            return 1
        ventana = VentanaOpenCV("lsm-signs — texto a señas (deletreo manual)")
    if reloj is None:
        reloj = time.perf_counter

    cuadros = {label: load_asset_frames(assets / manifest.letras[label].archivo) for label in letras}
    print(render_tokens(tokens))
    bucle(playlist, tokens, manifest, cuadros, config, ventana, reloj, lambda escena: draw_scene(escena, config))
    return 0
```

Añade `WordGap` y `Step` a los imports de `lsm.signs` en `cli/signs.py`.

- [ ] **Step 6: Corre el gate completo**

Run: `uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest`
Expected: todo en verde. Si `test_el_bucle_termina_con_q...` da `index == 0`, revisa que el reloj se lea **después** de `tecla()` y que `RelojFalso` avance también en la primera lectura (la del `anterior` inicial): 21 lecturas × 100 ms; la primera fija `anterior`, las 20 siguientes suman 2000 ms > 1500 ms.

- [ ] **Step 7: Prueba manual (solo si hay OpenCV instalado)**

Run: `uv run lsm-signs render --revisor "prueba" --assets /tmp/signs-prueba && uv run lsm-signs reproducir --assets /tmp/signs-prueba "hola mundo"`
Expected: se abre una ventana; la H se sostiene 1.5 s, el espacio pausa 0.8 s; `ESPACIO` congela la barra; `+` acelera; `q` cierra y la terminal muestra `H O L A · M U N D O`.

- [ ] **Step 8: Commit**

```bash
git add src/lsm/signs.py src/lsm/io/signs.py src/lsm/cli/signs.py tests/test_io_signs.py tests/test_cli_signs.py
git commit -m "feat(cli): lsm-signs reproducir con temporizador y control manual"
```

---

### Task 9: Los 29 assets reales, la revisión y el criterio de la fase

**Files:**
- Create: `tests/test_signs_manifest.py`
- Create: `assets/signs/manifest.json` y 21 PNG + 8 GIF
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `lsm-signs render`, `lsm-signs verificar`, `load_manifest`, `manifest_drift`, `gif_frame_count`

- [ ] **Step 1: Escribe el test del criterio de la fase**

`tests/test_signs_manifest.py`:

```python
"""El criterio de la Fase 4: 29 letras con asset, fuente documentada y revisión.

Lee disco a propósito, como los tests del glosario: el entregable de la fase es
un archivo concreto del repositorio y este test es lo que impide cerrarla con
una letra sin dibujar o sin revisar.
"""

from __future__ import annotations

from pathlib import Path

from lsm.config import Config
from lsm.io.signs import DEFAULT_ASSETS_DIR, MANIFEST_FILENAME, gif_frame_count, load_manifest
from lsm.signs import FUENTE_NORMATIVA, manifest_drift
from lsm.vocabulary import DYNAMIC_LABELS, LETTERS, Label

RAIZ_REPO = Path(__file__).resolve().parent.parent
ASSETS = RAIZ_REPO / DEFAULT_ASSETS_DIR


def test_el_manifest_existe_y_carga() -> None:
    manifest = load_manifest(ASSETS / MANIFEST_FILENAME)

    assert manifest.fuente_normativa == FUENTE_NORMATIVA


def test_las_29_letras_estan_y_ninguna_sobra() -> None:
    manifest = load_manifest(ASSETS / MANIFEST_FILENAME)

    assert set(manifest.letras) == set(LETTERS)
    assert Label.NONE not in manifest.letras


def test_el_manifest_dice_lo_mismo_que_el_glosario() -> None:
    assert manifest_drift(load_manifest(ASSETS / MANIFEST_FILENAME)) == []


def test_cada_asset_existe_y_las_dinamicas_se_mueven() -> None:
    manifest = load_manifest(ASSETS / MANIFEST_FILENAME)
    fps = Config().signs.render_fps

    for label, asset in manifest.letras.items():
        archivo = ASSETS / asset.archivo
        assert archivo.is_file(), f"{label}: falta {asset.archivo}"
        frames = gif_frame_count(archivo)
        if label in DYNAMIC_LABELS:
            assert asset.duracion_ms is not None
            assert frames == round(asset.duracion_ms * fps / 1000), label
            assert frames >= 2, f"{label}: una dinámica con un solo frame no se mueve"
        else:
            assert frames == 1, label


def test_cada_asset_remite_a_una_muestra_del_dataset() -> None:
    manifest = load_manifest(ASSETS / MANIFEST_FILENAME)

    for label, asset in manifest.letras.items():
        assert asset.fuente.tipo == "esqueleto_desde_dataset"
        assert asset.fuente.muestra.split("/")[2] == label
        assert asset.fuente.signer_id == asset.fuente.muestra.split("/")[0]


def test_todas_las_revisiones_coinciden_con_la_descripcion() -> None:
    manifest = load_manifest(ASSETS / MANIFEST_FILENAME)

    pendientes = {
        str(label): asset.revision.resultado
        for label, asset in manifest.letras.items()
        if asset.revision.resultado != "coincide"
    }
    assert pendientes == {}, f"revisiones sin cerrar: {pendientes}"
```

Run: `uv run pytest tests/test_signs_manifest.py -v`
Expected: FAIL con `FileNotFoundError` del manifest.

- [ ] **Step 2: Deja de ignorar los assets**

En `.gitignore`, elimina las cuatro líneas:

```
# Assets binarios grandes de la dirección texto → señas (Fase 4).
assets/signs/*.webp
assets/signs/*.gif
assets/signs/*.mp4
```

y en su lugar deja un comentario:

```
# assets/signs/ SE VERSIONA: son esqueletos de pocos KB renderizados desde el
# dataset propio, y el test del manifest (tests/test_signs_manifest.py) los
# necesita para exigir el criterio de la Fase 4 en CI.
```

- [ ] **Step 3: Renderiza desde el dataset real**

Run: `uv run lsm-signs render --revisor "Claude Opus 5 — revisión contra la descripción del glosario, no validación por persona usuaria de LSM"`
Expected: 29 líneas, una por letra, con la muestra elegida y el número de candidatas; ninguna `espejada` (hay mano derecha para todas). `assets/signs/` con `manifest.json`, 21 PNG y 8 GIF. Comprueba el tamaño: `du -sh assets/signs` debería quedar por debajo de 1 MB.

Si alguna letra sale `sin muestras de referencia válidas`, mira `data/raw` para esa letra: `to_sample()` rechaza muestras con huecos y `choose_reference` exige `kind` correcto; el mensaje dice la letra. No relajes el criterio: regraba o elige otra sesión con `--raw`.

- [ ] **Step 4: Revisa cada asset contra el glosario**

Para las dinámicas, vuelca una tira de frames a PNG para poder mirarla en un visor que no anime GIF (el scratchpad de la sesión, fuera del repositorio):

```bash
uv run python - <<'EOF'
from pathlib import Path
from PIL import Image
from lsm.io.signs import load_asset_frames
from lsm.vocabulary import DYNAMIC_LABELS
salida = Path("/tmp/tiras"); salida.mkdir(exist_ok=True)
for label in sorted(DYNAMIC_LABELS):
    cuadros = load_asset_frames(Path(f"assets/signs/{label}.gif"))
    paso = max(1, len(cuadros) // 6)
    muestra = cuadros[::paso][:6]
    tira = Image.new("RGB", (sum(c.width for c in muestra), muestra[0].height), (255, 255, 255))
    x = 0
    for c in muestra:
        tira.paste(c, (x, 0)); x += c.width
    tira.save(salida / f"{label}.png")
    print(label, len(cuadros), "frames ->", salida / f"{label}.png")
EOF
```

Después, letra por letra, abre el PNG (o la tira) junto a la fila de `docs/glosario-lsm.md` §3 y responde tres preguntas:

1. ¿Los dedos que la descripción dice estirados se ven estirados, y los doblados, doblados? (En un esqueleto 2D un dedo doblado hacia la palma queda **corto**: la yema cae cerca del nudillo.)
2. En dinámicas: ¿la trayectoria de la tira es la del glosario? (J: baja y vuelve en gancho; Z: tres tramos en zigzag; LL y RR: desplazamiento horizontal; Ñ, Q: giro de muñeca a los lados; K: sube; X: va al frente y regresa — en 2D se ve como un cambio de escala.)
3. Las cuatro parejas idénticas salvo movimiento (L/LL, R/RR, N/Ñ, I/J): ¿el PNG de la estática y el primer frame del GIF muestran la misma mano, y el GIF se mueve?

Edita `assets/signs/manifest.json` y, por cada letra, pon `revision.resultado` en `"coincide"` con una `nota` corta y concreta cuando algo merezca decirse (por ejemplo `"palma de lado: no se aprecia en 2D, la descripción lo cubre"`). Si una letra **no** coincide con la descripción, ponla en `"difiere"` con la nota, no la fuerces: el test de la fase quedará en rojo y eso es lo correcto — la salida es elegir otra muestra (edita `fuente.muestra` a mano y re-renderiza esa letra, o regraba). Registra en la nota del manifest que el dibujo pierde la orientación de la palma y que la descripción la suple: es lo que el ADR 0014 (Task 10) explica.

- [ ] **Step 5: Verifica**

Run: `uv run lsm-signs verificar && uv run pytest tests/test_signs_manifest.py -v`
Expected: `manifest completo: 29 letras con archivo, fuente y revisión.` y los seis tests PASS.

- [ ] **Step 6: Gate completo y commit**

Run: `uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest`

```bash
git add .gitignore assets/signs tests/test_signs_manifest.py
git commit -m "feat(assets): 29 senas renderizadas desde el dataset propio, con fuente y revision"
```

---

### Task 10: Documentación y cierre de la fase

**Files:**
- Create: `docs/adr/0014-assets-como-esqueleto-del-dataset-propio.md`
- Modify: `docs/ARQUITECTURA.md` (§3 árbol, §4.10, §5 tabla)
- Modify: `docs/COMO-PROBAR.md` (sección nueva 7; renumerar 7→8 y 8→9)
- Modify: `README.md` (estado de la fase)
- Modify: `CLAUDE.md` (regla 2)
- Modify: `Makefile` (`signs`, `signs-render`, `signs-verificar`, `help`)

- [ ] **Step 1: El ADR**

`docs/adr/0014-assets-como-esqueleto-del-dataset-propio.md`:

```markdown
# ADR 0014 — Los assets de texto → señas son esqueletos del dataset propio

- **Estado:** aceptada e implementada el 2026-09-14
- **Fecha:** 2026-09-14
- **Fase:** 4
- **Implementa:** `src/lsm/signs.py`, `src/lsm/io/signs.py`, `src/lsm/cli/signs.py`,
  `assets/signs/manifest.json`
- **Spec:** `docs/superpowers/specs/2026-09-14-fase-4-texto-a-senas-design.md`

## Contexto

`ARQUITECTURA.md` §4.10 avisa que la dirección texto → señas no tiene riesgo
técnico sino de **corrección del material**: casi todo lo que circula en
internet como "abecedario en lengua de señas" es ASL, y un proyecto que
presente ASL como LSM queda descalificado ante cualquier persona usuaria
(`glosario-lsm.md` §2). A eso se suma que el material fotográfico ajeno tiene
derechos, y que las ocho letras dinámicas necesitan movimiento, no una foto
con flecha.

## Decisión

**Ningún asset viene de internet.** Cada letra se dibuja como esqueleto de 21
landmarks a partir de una muestra de `data/raw`: la **medoide** de su clase (la
grabación real más cercana al centroide en el espacio de features), con
preferencia por mano derecha, que es la que muestra el diccionario. Estáticas
en PNG del frame central; dinámicas en GIF con todos los frames a
`signs.render_fps`.

`assets/signs/manifest.json` lleva por letra: archivo, `es_dinamica`,
descripción y trayectoria copiadas de `vocabulary.py`, página de *Manos con
voz*, y una `fuente` que apunta a la muestra concreta (`signer/sesion/LETRA/NNN.json`).
`manifest_drift` exige que las copias no diverjan del glosario;
`tests/test_signs_manifest.py` exige las 29 letras con archivo, fuente y
revisión `coincide`.

Los assets **se versionan**: son esqueletos de pocos KB, y CI solo puede exigir
el criterio de la fase si los archivos están en el repositorio.

## Qué se gana

- **LSM por construcción.** Las muestras se grabaron siguiendo las páginas 15-19
  de la fuente primaria; no hay ningún paso por el que entre una imagen ajena.
- Sin derechos de terceros; reproducible con `lsm-signs render`.
- Las dinámicas tienen movimiento real, el de la grabación, no una flecha.

## Qué se pierde y cómo se compensa

- **La orientación de la palma.** Un esqueleto en 2D no distingue bien "palma
  al frente" de "palma de lado", y en LSM eso separa letras. El reproductor
  muestra **siempre** la descripción y la trayectoria del glosario junto al
  dibujo; el dibujo es el apoyo, el texto es la norma.
- **Legibilidad frente a una foto.** El manifest admite otros orígenes por
  `fuente.tipo`: una letra puede sustituirse por una foto verificada cambiando
  su entrada y su archivo, sin tocar código.

## Qué NO cierra este ADR

La revisión registrada en `revision` de cada letra es **contra la descripción
del glosario**, hecha por quien ejecutó la fase mirando el dibujo con la fila
del glosario al lado. No es la validación por persona usuaria de LSM o
intérprete que pide `ARQUITECTURA.md` §4.11: el PENDIENTE-HUMANO G del glosario
sigue abierto y aplica también a estos assets.

## Alternativas descartadas

- **Fotos propias.** Más legibles; 29 assets a mano y sin ventaja de
  trazabilidad sobre el dataset, que ya existe.
- **Recortes del PDF de CONAPRED.** Fidelidad máxima, pero es material con
  derechos y no da movimiento para las dinámicas.
- **No versionar los assets.** Habría dejado el criterio de la fase fuera de CI.
```

- [ ] **Step 2: `ARQUITECTURA.md`**

En el árbol del §3, añade tras `spelling.py`:

```
│   ├── signs.py                  # texto → símbolos, lista de pasos, reproductor — SIN I/O
```

tras `corpus.py`:

```
│   │   └── signs.py              # manifest en disco, render PNG/GIF, cuadro de la ventana
```

y tras `demo.py`:

```
│       └── signs.py              # lsm-signs: render / verificar / reproducir
```

Cambia el comentario de `demo.py` de `# demo en vivo, ambas direcciones` a `# demo en vivo, señas → texto`. Al final del §4.10, añade:

```markdown
Resuelto en la Fase 4 (`docs/adr/0014-assets-como-esqueleto-del-dataset-propio.md`):
los assets son esqueletos renderizados desde `data/raw`, el manifest apunta a la
muestra de origen y a la página del diccionario, y el reproductor muestra siempre
la descripción del glosario junto al dibujo. `signs.py` es puro; el reloj entra
por ticks como en `telemetry.py`.
```

En la tabla del §5, fila 4: `27 letras` → `29 letras (26 sin CH, más Ñ, LL y RR)`.

- [ ] **Step 3: `COMO-PROBAR.md`**

Inserta antes de `## 7. Cuando algo falla` y renumera esa a `## 8.` y `## 8. Dónde seguir leyendo` a `## 9.`:

```markdown
## 7. Texto a señas 🖼️

La dirección inversa: escribes y ves las señas. No necesita cámara ni MediaPipe,
pero sí Pillow y OpenCV (`make setup-capture`) para la ventana.

```bash
uv run lsm-signs reproducir "hola mundo"
# o: make signs TEXTO="hola mundo"
```

Se abre una ventana: la seña a la izquierda; la letra, su descripción del
glosario y la barra de progreso a la derecha; el texto completo abajo con el
símbolo actual resaltado. `"ll"` y `"rr"` son un solo símbolo; los acentos se
quitan; `ñ` es `Ñ`. Dígitos y puntuación se rechazan con la lista exacta.

**Controles:** `ESPACIO` pausa, `n`/`p` siguiente/anterior, `r` reinicia,
`+`/`-` velocidad, `q` sale. Las duraciones viven en `config.yaml`, sección
`signs`.

### 7.1 Regenerar los assets

Los 29 assets de `assets/signs/` están versionados y se dibujan desde el
dataset propio, nunca de internet (`docs/adr/0014-...`). Para regenerarlos tras
regrabar:

```bash
uv run lsm-signs render --revisor "tu nombre"
# o: make signs-render REVISOR="tu nombre"
```

Elige por letra la muestra más típica de `data/raw` (mano derecha si la hay) y
escribe `manifest.json`. Las revisiones ya hechas se conservan si la muestra
elegida no cambió; si cambió, la letra vuelve a `pendiente` y hay que mirarla
otra vez contra `docs/glosario-lsm.md` §3 y ponerla en `coincide` a mano.

### 7.2 Verificar

```bash
uv run lsm-signs verificar
# o: make signs-verificar
```

Exige 29 letras, archivos presentes, dinámicas en GIF con los frames que dice
`duracion_ms`, cero deriva contra el glosario y todas las revisiones en
`coincide`. Es lo mismo que exige `tests/test_signs_manifest.py` en CI.
```

- [ ] **Step 4: `README.md`, `CLAUDE.md`, `Makefile`**

`README.md`: en el estado de fases, añade que la Fase 4 está cerrada, con una frase: *"Texto → señas: `lsm-signs reproducir "casa"`. Los 29 assets son esqueletos renderizados desde el dataset propio, con fuente y revisión por letra en `assets/signs/manifest.json`; ver ADR 0014. La revisión es contra la descripción del glosario: la validación por persona usuaria de LSM sigue pendiente."* Sigue el formato de los párrafos de las fases anteriores.

`CLAUDE.md`, regla 2: `telemetry.py` y `classifiers/` → `telemetry.py`, `signs.py` y `classifiers/`.

`Makefile`: en `.PHONY` añade `signs signs-render signs-verificar`; en `help`, tras `medir-fps`:

```make
	@echo ""
	@echo "Texto a senas (sin camara; necesita Pillow y OpenCV: setup-capture):"
	@echo "signs          reproduce un texto; pasa TEXTO=\"hola mundo\""
	@echo "signs-render   regenera assets/signs desde data/raw; pasa REVISOR=..."
	@echo "signs-verificar  manifest completo y revisado"
```

y al final del archivo:

```make
# --------------------------------------------------------------------------- #
# Texto -> senas (Fase 4). Sin camara. Ver docs/adr/0014-...
# --------------------------------------------------------------------------- #

TEXTO ?= hola
REVISOR ?= $(USER)

signs:
	$(UV) run lsm-signs reproducir "$(TEXTO)"

# Regenera los 29 assets desde el dataset propio. Conserva las revisiones cuya
# muestra de origen no cambio; las demas vuelven a "pendiente".
signs-render:
	$(UV) run lsm-signs render --revisor "$(REVISOR)"

signs-verificar:
	$(UV) run lsm-signs verificar
```

- [ ] **Step 5: Gate completo y commit**

Run: `uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest && uv run lsm-signs verificar`
Expected: todo en verde.

```bash
git add docs/adr/0014-assets-como-esqueleto-del-dataset-propio.md docs/ARQUITECTURA.md docs/COMO-PROBAR.md README.md CLAUDE.md Makefile
git commit -m "docs: cierra la Fase 4 y registra el ADR 0014 sobre los assets"
```

---

## Auto-revisión del plan

**Cobertura del spec.** §3 esquema → Task 2. §4.1 medoide → Task 5. §4.2 proyección → Task 5. §4.3 Pillow → Task 6. §4.4 `render` con conservación de revisiones y salida determinista → Task 7. §5 tres capas de verificación → Tasks 2 (deriva), 7 (`verificar`) y 9 (revisión humana). §6.1–6.3 → Tasks 3 y 4. §7 ventana → Task 8 (con el cambio a Pillow para el texto, por la fuente ASCII de OpenCV). §8 config → Task 1. §9 tests → cada tarea; el criterio de la fase → Task 9. §10 docs → Task 10. `.gitignore` → Task 9.

**Desviaciones del spec, deliberadas.** (1) El cuadro se compone con Pillow y OpenCV solo lo muestra: la Hershey de OpenCV no dibuja Ñ ni acentos. (2) La función del reproductor se llama `player_step` y no `step`, para no chocar con la clase `Step` del mismo módulo. (3) `asset_frame` no recibe el `Step`: no lo necesitaba.

**Consistencia de nombres.** `Candidate.path` es lo que va a `fuente.muestra` (Tasks 5, 6, 7). `expected_filename` se usa en Tasks 2, 7 y 9. `gif_frame_count` devuelve 1 para PNG (Task 6) y `verificar` y el test de la fase lo usan así (Tasks 7, 9). `Scene.token_index == state.index` porque `build_playlist` produce un paso por token (Task 4 → Task 8).
