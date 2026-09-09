# Fase 3 — Deletreo en vivo: plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Conectar cámara → MediaPipe → features → segmentación → clasificador → texto, de modo que se pueda deletrear una palabra de cinco letras sin errores de segmentación.

**Architecture:** `src/lsm/spelling.py` es un reductor **puro** e inmutable: `step(estado, señal, config) -> StepResult`. La demo le pasa lo que ve en cada frame y guarda el estado devuelto para dibujarlo. `src/lsm/cli/demo.py` conecta el hardware; `run_segmentation` no se toca, porque su contrato está versionado. El truco del bucle es que el flujo que se le pasa a `run_segmentation` es un generador que captura, dibuja y lee el teclado en cada `next()`.

**Tech Stack:** Python 3.11+, uv, Pydantic v2, pytest, mypy strict, ruff. OpenCV y MediaPipe son dependencias **opcionales** (extra `capture`) y solo se importan dentro de funciones.

**Spec:** `docs/superpowers/specs/2026-09-09-fase-3-deletreo-en-vivo-design.md`

## Global Constraints

Copiadas de `CLAUDE.md` y del spec. Aplican a **todas** las tareas.

- **`src/lsm/spelling.py` es código puro.** Sin OpenCV, sin MediaPipe, sin acceso a disco, sin cámara. Toda la I/O vive en `src/lsm/io/` y `src/lsm/cli/`. Hay un test que lo comprueba.
- **MediaPipe solo se importa en `src/lsm/io/hands.py`.** OpenCV, solo en `io/camera.py`, `io/preview.py` y dentro de funciones de `cli/`.
- **Cero umbrales hardcodeados.** Todo valor ajustable vive en `config.yaml`, validado con Pydantic.
- **El tipo base es una secuencia `(T, 21, 3)`**, nunca un frame suelto. No introducir APIs que acepten un frame.
- **No se toca `src/lsm/segmentation.py`.** Su contrato está versionado con `SEGMENTATION_SPEC_VERSION` (`docs/adr/0004-contrato-de-segmentacion.md`). Si el trabajo revelara que no funciona, hay que parar y proponerlo, no parchear desde el CLI.
- **Terminología:** el proyecto traduce **deletreo manual**, no lengua de señas. `signer` = persona que ejecuta la seña, `label` = letra, `sample` = secuencia etiquetada.
- **Los defaults de Pydantic y los valores de `config.yaml` tienen que coincidir.** `tests/test_config.py::test_el_config_yaml_de_ejemplo_es_valido_y_coincide_con_los_defaults` lo exige.
- **Comandos:** `uv run pytest`, `uv run mypy`, `uv run ruff check .`, `uv run ruff format --check .`. Si trabajas en WSL con el repositorio en Linux, prefija todo con `wsl -d Ubuntu -- bash -lc 'cd ~/mipaxtli && ...'`.
- **Los mensajes de commit terminan con** las dos líneas de atribución que ya usan los commits de la rama (`Co-Authored-By:` y `Claude-Session:`). Cópialas de `git log -1`.

## Mapa de archivos

| Archivo | Responsabilidad | Tarea |
|---|---|---|
| `src/lsm/config.py` | Añade `SpellingConfig` y la validación cruzada contra `segmentation` | 1 |
| `config.yaml` | Documenta la sección `spelling` | 1 |
| `src/lsm/spelling.py` | **Nuevo.** Señales, estado, eventos, `step`, renderizado | 2, 3, 4 |
| `tests/test_spelling.py` | **Nuevo.** El reductor, con listas de señales | 2, 3, 4 |
| `src/lsm/io/preview.py` | Añade `DemoHudState` y `draw_demo_hud` | 5 |
| `src/lsm/cli/demo.py` | **Nuevo.** Cablea todo; `--desde-dataset` y sesión en vivo | 6, 7 |
| `tests/test_cli_demo.py` | **Nuevo.** El criterio de la fase, sin cámara | 7 |
| `pyproject.toml` | Registra el script `lsm-demo` | 6 |
| `Makefile` | `make demo` deja de ser un `exit 1` | 8 |
| `docs/COMO-PROBAR.md` | Completa la sección 6 | 8 |
| `README.md` | Fase 3 cerrada | 8 |

---

### Task 1: La configuración del espacio

**Files:**
- Modify: `src/lsm/config.py` (añadir `SpellingConfig` antes de `class Config`, y el campo en `Config`)
- Modify: `config.yaml` (nueva sección al final)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `_Section`, `Field`, `model_validator` de `src/lsm/config.py`
- Produces: `config.spelling.space_after_absent_frames: int`

- [ ] **Step 1: Escribe el test que falla**

En `tests/test_config.py`, al final del archivo:

```python
def test_el_espacio_exige_mas_ausencia_que_la_vuelta_a_idle() -> None:
    """Si bastara con lo que la maquina de estados considera "mano perdida", un
    parpadeo del detector escribiria un espacio. El espacio es una intencion de
    quien firma, no un fallo de deteccion."""
    with pytest.raises(ValidationError):
        Config.model_validate(
            {
                "segmentation": {"missing_frames_to_idle": 8},
                "spelling": {"space_after_absent_frames": 8},
            }
        )


def test_el_espacio_por_defecto_es_un_segundo_a_treinta_fps() -> None:
    assert Config().spelling.space_after_absent_frames == 30
```

- [ ] **Step 2: Corre el test y comprueba que falla**

```bash
uv run pytest tests/test_config.py -k espacio -v
```

Esperado: FAIL. `Config` no tiene atributo `spelling` y `extra="forbid"` rechaza la clave.

- [ ] **Step 3: Implementa la sección**

En `src/lsm/config.py`, justo antes de `class Config(_Section):`:

```python
class SpellingConfig(_Section):
    """Acumulación de letras en palabras (`ARQUITECTURA.md` §4.2)."""

    #: Frames consecutivos sin mano antes de cerrar la palabra en curso. A 30 fps,
    #: 30 frames es un segundo. Es el único gesto de control del proyecto: bajar
    #: la mano entre palabras es lo que se hace de todos modos, y el clasificador
    #: no tiene clases libres para un gesto dedicado.
    space_after_absent_frames: int = Field(default=30, ge=1, le=600)
```

Añade el campo a `Config`, después de `capture`:

```python
    spelling: SpellingConfig = Field(default_factory=SpellingConfig)
```

Y la validación cruzada, como método de `Config` junto a las que ya hay:

```python
    @model_validator(mode="after")
    def _el_espacio_no_lo_dispara_un_parpadeo(self) -> Config:
        """El espacio tiene que costar más ausencia que volver a IDLE.

        `missing_frames_to_idle` es cuánto tarda la máquina de estados en dar la
        mano por perdida, y se cruza con cualquier oclusión momentánea. Si el
        espacio se disparara ahí, un parpadeo del detector partiría una palabra
        en dos y quien firma no tendría forma de evitarlo.
        """
        espacio = self.spelling.space_after_absent_frames
        idle = self.segmentation.missing_frames_to_idle
        if espacio <= idle:
            msg = (
                f"spelling.space_after_absent_frames ({espacio}) no supera "
                f"segmentation.missing_frames_to_idle ({idle}): un parpadeo "
                "del detector escribiría un espacio"
            )
            raise ValueError(msg)
        return self
```

- [ ] **Step 4: Documenta la sección en `config.yaml`**

Al final del archivo:

```yaml
# Acumulación de letras en palabras (ARQUITECTURA.md §4.2, docs/adr/0012-...).
spelling:
  # Frames consecutivos sin mano antes de cerrar la palabra en curso y empezar
  # otra. A 30 fps, 30 frames es un segundo.
  #
  # Es el único gesto de control del proyecto. Bajar la mano entre palabras es lo
  # que se hace de todos modos, y el clasificador no tiene clases libres: conoce
  # 22 y las 21 letras están ocupadas, así que un gesto dedicado sería una clase
  # nueva, y una clase nueva son tres personas citadas otra vez.
  #
  # config.py valida que supere a segmentation.missing_frames_to_idle. Si no, un
  # parpadeo del detector partiría una palabra en dos.
  space_after_absent_frames: 30
```

- [ ] **Step 5: Corre los tests y comprueba que pasan**

```bash
uv run pytest tests/test_config.py -v
```

Esperado: PASS, incluido `test_el_config_yaml_de_ejemplo_es_valido_y_coincide_con_los_defaults`.

- [ ] **Step 6: Commit**

```bash
git add src/lsm/config.py config.yaml tests/test_config.py
git commit -m "feat(config): el umbral del espacio, validado contra la vuelta a IDLE"
```

---

### Task 2: El estado y las letras

**Files:**
- Create: `src/lsm/spelling.py`
- Test: `tests/test_spelling.py`

**Interfaces:**
- Consumes: `Config` de `lsm.config`; `Label` y `spec` de `lsm.vocabulary` (**`Label` NO está en `types.py`**)
- Produces: `SpellingState`, `LetterSignal`, `StepResult`, `LetterWritten`, `step()`, `render_word()`, `render_text()`

- [ ] **Step 1: Escribe los tests que fallan**

Crea `tests/test_spelling.py`:

```python
"""El reductor de deletreo, sin camara y sin modelo.

Cada test es una lista de señales y una afirmacion sobre el texto resultante.
Eso es todo lo que hace falta: `spelling.py` es puro por la regla 2 de
`CLAUDE.md`, y esa pureza es lo que permite ejercitar aqui el criterio de la
fase en vez de delante de una webcam.
"""

from __future__ import annotations

from lsm.config import Config
from lsm.spelling import (
    LetterSignal,
    Signal,
    SpellingState,
    render_text,
    render_word,
    step,
)
from lsm.vocabulary import Label

CONFIG = Config()


def aplicar(señales: list[Signal], config: Config = CONFIG) -> SpellingState:
    """Corre una lista de señales desde el estado vacio."""
    state = SpellingState()
    for señal in señales:
        state = step(state, señal, config).state
    return state


def letras(*labels: Label) -> list[Signal]:
    return [LetterSignal(label=label) for label in labels]


def test_cinco_letras_dan_cinco_simbolos() -> None:
    state = aplicar(letras(Label.C, Label.A, Label.S, Label.A, Label.S))

    assert state.word == (Label.C, Label.A, Label.S, Label.A, Label.S)
    assert render_word(state) == "casas"


def test_la_clase_negativa_no_escribe_nada() -> None:
    """`segmentation.py` filtra UNKNOWN y la confianza baja, pero NO filtra NONE:
    con la clase negativa acertando bien, llegaran LetterEmitted con label NONE
    cada vez que la persona baje la mano. Escribirlos seria poner una letra en
    cada transicion, que es justo lo que la clase negativa existe para evitar."""
    state = aplicar(letras(Label.C, Label.NONE, Label.A))

    assert state.word == (Label.C, Label.A)
    assert render_word(state) == "ca"


def test_el_digrafo_es_un_simbolo_y_dos_erres_son_dos() -> None:
    """El buffer guarda simbolos del glosario, no caracteres. Las dos formas se
    leen igual y se deshacen distinto, que es la razon de la decision."""
    una = aplicar(letras(Label.DOBLE_R))
    dos = aplicar(letras(Label.R, Label.R))

    assert render_word(una) == render_word(dos) == "rr"
    assert len(una.word) == 1
    assert len(dos.word) == 2


def test_la_enie_se_escribe_con_su_letra() -> None:
    assert render_word(aplicar(letras(Label.ENIE))) == "ñ"


def test_el_texto_vacio_es_una_cadena_vacia() -> None:
    assert render_text(SpellingState()) == ""
```

- [ ] **Step 2: Corre los tests y comprueba que fallan**

```bash
uv run pytest tests/test_spelling.py -v
```

Esperado: FAIL con `ModuleNotFoundError: No module named 'lsm.spelling'`.

- [ ] **Step 3: Implementa el mínimo**

Crea `src/lsm/spelling.py`:

```python
"""Acumulación de letras en palabras: el buffer de deletreo.

`ARQUITECTURA.md` §4.2 dejó esto para la Fase 3, y con él la pregunta de qué se
hace con los dígrafos. La respuesta está en el tipo: **el buffer guarda símbolos
del glosario, no caracteres.**

`DOBLE_R` es un símbolo que se lee `rr`; dos `R` seguidas son dos símbolos que
también se leen `rr`. Se ven igual y se deshacen distinto, que es exactamente lo
que se quiere: `BACKSPACE` borra un símbolo, así que deshace lo que se señó. Con
un buffer de caracteres, borrar sobre una `RR` dejaría una `r` que nadie ejecutó.

## Puro por la regla 2 de `CLAUDE.md`

Sin OpenCV, sin MediaPipe, sin disco, sin cámara. Un reductor: `step` recibe el
estado y una señal, y devuelve el estado nuevo. Todo el criterio de esta fase se
ejercita con una lista de señales, que es lo que permite comprobar el criterio de
aceptación —cinco letras, cinco símbolos— en CI y no delante de una webcam.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypeAlias

from lsm.config import Config
from lsm.vocabulary import Label, spec


@dataclass(frozen=True, slots=True)
class LetterSignal:
    """Una letra que la segmentación dio por buena."""

    label: Label


#: Las señales que la demo produce. Se irá ampliando en las tareas siguientes.
Signal: TypeAlias = LetterSignal


@dataclass(frozen=True, slots=True)
class LetterWritten:
    """Se añadió un símbolo a la palabra en curso."""

    label: Label


#: Qué acaba de pasar. `None` significa que la señal no cambió nada.
SpellingEvent: TypeAlias = LetterWritten


@dataclass(frozen=True, slots=True)
class SpellingState:
    """El buffer de deletreo. Inmutable.

    Guarda **símbolos** y no texto: el texto se deriva con `render_word` y
    `render_text`, y así el estado conserva qué se señó de verdad.
    """

    #: La palabra en curso.
    word: tuple[Label, ...] = ()
    #: Palabras ya cerradas, cada una con sus símbolos.
    finished: tuple[tuple[Label, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class StepResult:
    """El estado nuevo y qué pasó al llegar ahí."""

    state: SpellingState
    event: SpellingEvent | None = None


def step(  # noqa: ARG001 — `config` lo usa el espacio, que llega en la tarea 3
    state: SpellingState, signal: Signal, config: Config
) -> StepResult:
    """Aplica una señal. Nunca muta `state`.

    `config` entra aunque esta versión todavía no lo use: el umbral del espacio
    llega en la tarea siguiente y la firma no debe cambiar entonces.
    """
    match signal:
        case LetterSignal(label=label):
            return _escribir_letra(state, label)


def _escribir_letra(state: SpellingState, label: Label) -> StepResult:
    """La clase negativa no escribe.

    `segmentation.py` filtra `UNKNOWN` y la confianza por debajo del umbral, pero
    no filtra `NONE`. Con la clase negativa acertando bien, llegará una predicción
    de `NONE` cada vez que quien firma baje la mano o transite entre letras.
    Escribirla sería poner una letra en cada transición, que es justo lo que la
    clase negativa existe para evitar.

    Se descarta aquí y no en `cli/demo.py` porque es un criterio, y los criterios
    viven en código puro; y no en `segmentation.py` porque su contrato está
    versionado y no hay motivo para tocarlo.
    """
    if label is Label.NONE:
        return StepResult(state=state)
    return StepResult(
        state=replace(state, word=(*state.word, label)),
        event=LetterWritten(label=label),
    )


def render_word(state: SpellingState) -> str:
    """La palabra en curso, tal como se escribe para una persona."""
    return _render(state.word)


def render_text(state: SpellingState) -> str:
    """Todo el texto: las palabras cerradas y la que está en curso."""
    palabras = [_render(word) for word in state.finished]
    if state.word:
        palabras.append(_render(state.word))
    return " ".join(palabras)


def _render(word: tuple[Label, ...]) -> str:
    """`LetterSpec.display` existe justo para esto: `Ñ`, `LL`, `RR`."""
    return "".join(spec(label).display.lower() for label in word)
```

- [ ] **Step 4: Corre los tests y comprueba que pasan**

```bash
uv run pytest tests/test_spelling.py -v
uv run mypy && uv run ruff check .
```

Esperado: PASS. Si mypy se queja de que el `match` de `step` no es exhaustivo, es porque `Signal` tiene un solo miembro: se resuelve solo en la tarea 3.

- [ ] **Step 5: Commit**

```bash
git add src/lsm/spelling.py tests/test_spelling.py
git commit -m "feat(spelling): el buffer guarda simbolos del glosario, no caracteres"
```

---

### Task 3: El espacio, con sus dos cerrojos

**Files:**
- Modify: `src/lsm/spelling.py`
- Test: `tests/test_spelling.py`

**Interfaces:**
- Consumes: todo lo de la tarea 2
- Produces: `HandPresent`, `HandAbsent`, `SpaceWritten`; `SpellingState.absent_frames`, `SpellingState.space_emitted`

- [ ] **Step 1: Escribe los tests que fallan**

Añade a `tests/test_spelling.py`:

```python
from lsm.spelling import HandAbsent, HandPresent, SpaceWritten


def ausencia(frames: int) -> list[Signal]:
    return [HandAbsent()] * frames


UMBRAL = CONFIG.spelling.space_after_absent_frames


def test_la_ausencia_corta_no_pone_espacio() -> None:
    """Un parpadeo del detector no es una intencion de quien firma."""
    state = aplicar([*letras(Label.C, Label.A), *ausencia(UMBRAL - 1)])

    assert state.finished == ()
    assert render_text(state) == "ca"


def test_al_alcanzar_el_umbral_pone_un_espacio_y_solo_uno() -> None:
    """La mano abajo no es un evento, es un estado que dura: sin cerrojo pondria
    un espacio en cada frame. Es el mismo problema que `pending_repeat` resuelve
    en segmentation.py, un nivel mas arriba."""
    state = aplicar([*letras(Label.C, Label.A), *ausencia(UMBRAL * 3)])

    assert state.finished == ((Label.C, Label.A),)
    assert state.word == ()
    assert render_text(state) == "ca"


def test_la_mano_de_vuelta_libera_el_cerrojo() -> None:
    señales = [
        *letras(Label.C, Label.A),
        *ausencia(UMBRAL),
        HandPresent(),
        *letras(Label.S, Label.A),
        *ausencia(UMBRAL),
    ]

    state = aplicar(señales)

    assert state.finished == ((Label.C, Label.A), (Label.S, Label.A))
    assert render_text(state) == "ca sa"


def test_la_ausencia_sobre_una_palabra_vacia_no_pone_espacio() -> None:
    """Ni al principio ni entre dos ausencias seguidas: espacios sueltos o
    dobles serian texto que nadie seño."""
    state = aplicar([*ausencia(UMBRAL * 2), HandPresent(), *ausencia(UMBRAL * 2)])

    assert state.finished == ()
    assert render_text(state) == ""


def test_el_espacio_emite_su_evento() -> None:
    state = aplicar([*letras(Label.A), *ausencia(UMBRAL - 1)])

    resultado = step(state, HandAbsent(), CONFIG)

    assert isinstance(resultado.event, SpaceWritten)
```

- [ ] **Step 2: Corre los tests y comprueba que fallan**

```bash
uv run pytest tests/test_spelling.py -k "ausencia or espacio or cerrojo" -v
```

Esperado: FAIL con `ImportError: cannot import name 'HandAbsent'`.

- [ ] **Step 3: Implementa**

En `src/lsm/spelling.py`, añade los tipos junto a `LetterSignal`:

```python
@dataclass(frozen=True, slots=True)
class HandPresent:
    """Un frame con mano. Libera el cerrojo del espacio."""


@dataclass(frozen=True, slots=True)
class HandAbsent:
    """Un frame sin mano."""


@dataclass(frozen=True, slots=True)
class SpaceWritten:
    """Se cerró la palabra en curso y se abrió otra."""
```

Amplía los alias:

```python
Signal: TypeAlias = LetterSignal | HandPresent | HandAbsent
SpellingEvent: TypeAlias = LetterWritten | SpaceWritten
```

Añade los dos campos a `SpellingState`:

```python
    #: Frames consecutivos sin mano. Vive en el estado y no en el CLI para que el
    #: criterio del espacio sea puro y se pueda testear con una lista de señales.
    absent_frames: int = 0
    #: Ya se puso espacio por esta ausencia. Sin esto, la mano quieta abajo
    #: escribiría un espacio por frame: la mano abajo no es un evento, es un
    #: estado que dura. Es el mismo cerrojo que `pending_repeat` en
    #: `segmentation.py`, un nivel más arriba.
    space_emitted: bool = False
```

Amplía el `match` de `step`:

```python
match signal:
    case LetterSignal(label=label):
        return _escribir_letra(state, label)
    case HandPresent():
        return StepResult(state=replace(state, absent_frames=0, space_emitted=False))
    case HandAbsent():
        return _mano_ausente(state, config)
```

Y la función nueva:

```python
def _mano_ausente(state: SpellingState, config: Config) -> StepResult:
    """Cierra la palabra cuando la ausencia deja de ser un parpadeo."""
    absent = state.absent_frames + 1
    alcanzado = absent >= config.spelling.space_after_absent_frames

    if not alcanzado or state.space_emitted:
        return StepResult(state=replace(state, absent_frames=absent))

    # El cerrojo se pone aunque no haya nada que cerrar: si no, cada frame
    # siguiente volvería a evaluar el umbral sobre una palabra vacía.
    if not state.word:
        return StepResult(
            state=replace(state, absent_frames=absent, space_emitted=True)
        )

    return StepResult(
        state=replace(
            state,
            word=(),
            finished=(*state.finished, state.word),
            absent_frames=absent,
            space_emitted=True,
        ),
        event=SpaceWritten(),
    )
```

- [ ] **Step 4: Corre los tests y comprueba que pasan**

```bash
uv run pytest tests/test_spelling.py -v && uv run mypy && uv run ruff check .
```

Esperado: PASS, todos.

- [ ] **Step 5: Commit**

```bash
git add src/lsm/spelling.py tests/test_spelling.py
git commit -m "feat(spelling): la mano abajo cierra la palabra, con su cerrojo"
```

---

### Task 4: Borrado y cierre de frase

**Files:**
- Modify: `src/lsm/spelling.py`
- Test: `tests/test_spelling.py`

**Interfaces:**
- Consumes: todo lo anterior
- Produces: `Backspace`, `CommitText`, `SymbolDeleted`, `NothingToDelete`, `TextCommitted`

- [ ] **Step 1: Escribe los tests que fallan**

Añade a `tests/test_spelling.py`:

```python
from lsm.spelling import (
    Backspace,
    CommitText,
    NothingToDelete,
    SymbolDeleted,
    TextCommitted,
)


def test_backspace_borra_un_simbolo() -> None:
    state = aplicar([*letras(Label.C, Label.A, Label.S), Backspace()])

    assert render_word(state) == "ca"


def test_backspace_deshace_exactamente_lo_que_se_seño() -> None:
    """La razon de guardar simbolos: sobre una RR el borrado quita la seña
    entera, y sobre dos R quita una R. Con un buffer de caracteres, lo primero
    dejaria una `r` que nadie ejecuto."""
    digrafo = aplicar([*letras(Label.DOBLE_R), Backspace()])
    dos_erres = aplicar([*letras(Label.R, Label.R), Backspace()])

    assert render_word(digrafo) == ""
    assert render_word(dos_erres) == "r"


def test_backspace_sobre_una_palabra_vacia_no_hace_nada_y_lo_dice() -> None:
    """No recupera la palabra anterior: reabrir algo ya cerrado no vale la
    complejidad en esta fase."""
    state = aplicar([*letras(Label.A), *ausencia(UMBRAL)])

    resultado = step(state, Backspace(), CONFIG)

    assert resultado.state == state
    assert isinstance(resultado.event, NothingToDelete)


def test_backspace_emite_el_simbolo_que_quito() -> None:
    state = aplicar(letras(Label.C, Label.DOBLE_L))

    resultado = step(state, Backspace(), CONFIG)

    assert resultado.event == SymbolDeleted(label=Label.DOBLE_L)


def test_enter_cierra_la_frase_y_deja_el_estado_vacio() -> None:
    señales = [
        *letras(Label.C, Label.A),
        *ausencia(UMBRAL),
        HandPresent(),
        *letras(Label.S, Label.A),
    ]
    state = aplicar(señales)

    resultado = step(state, CommitText(), CONFIG)

    assert resultado.event == TextCommitted(text="ca sa")
    assert resultado.state == SpellingState()


def test_enter_sobre_un_texto_vacio_no_emite_nada() -> None:
    resultado = step(SpellingState(), CommitText(), CONFIG)

    assert resultado.event is None
    assert resultado.state == SpellingState()
```

- [ ] **Step 2: Corre los tests y comprueba que fallan**

```bash
uv run pytest tests/test_spelling.py -k "backspace or enter" -v
```

Esperado: FAIL con `ImportError: cannot import name 'Backspace'`.

- [ ] **Step 3: Implementa**

Añade los tipos:

```python
@dataclass(frozen=True, slots=True)
class Backspace:
    """Tecla: borra el último símbolo de la palabra en curso."""


@dataclass(frozen=True, slots=True)
class CommitText:
    """Tecla: cierra la frase entera."""


@dataclass(frozen=True, slots=True)
class SymbolDeleted:
    """Se quitó un símbolo. Lleva cuál, para poder decirlo en el HUD."""

    label: Label


@dataclass(frozen=True, slots=True)
class NothingToDelete:
    """`BACKSPACE` con la palabra en curso vacía."""


@dataclass(frozen=True, slots=True)
class TextCommitted:
    """La frase se cerró. `text` es lo que hay que imprimir."""

    text: str
```

Amplía los alias:

```python
Signal: TypeAlias = LetterSignal | HandPresent | HandAbsent | Backspace | CommitText
SpellingEvent: TypeAlias = (
    LetterWritten | SpaceWritten | SymbolDeleted | NothingToDelete | TextCommitted
)
```

Amplía el `match`:

```python
        case Backspace():
            return _borrar(state)
        case CommitText():
            return _cerrar_frase(state)
```

Y las dos funciones:

```python
def _borrar(state: SpellingState) -> StepResult:
    """Quita el último símbolo de la palabra en curso.

    Con la palabra vacía no hace nada. Recuperar la palabra anterior significaría
    reabrir algo ya cerrado, y no vale la complejidad en esta fase: un borrado que
    hace más de lo que se espera destruye trabajo.
    """
    if not state.word:
        return StepResult(state=state, event=NothingToDelete())
    return StepResult(
        state=replace(state, word=state.word[:-1]),
        event=SymbolDeleted(label=state.word[-1]),
    )


def _cerrar_frase(state: SpellingState) -> StepResult:
    """Vacía el buffer y entrega el texto para imprimirlo."""
    texto = render_text(state)
    if not texto:
        return StepResult(state=state)
    return StepResult(state=SpellingState(), event=TextCommitted(text=texto))
```

- [ ] **Step 4: Corre los tests y comprueba que pasan**

```bash
uv run pytest tests/test_spelling.py -v && uv run mypy && uv run ruff check .
```

Esperado: PASS. mypy ya no debería quejarse del `match`: `Signal` está completo.

- [ ] **Step 5: Añade el test de pureza**

`CLAUDE.md` regla 2 tiene que ser ejecutable. Busca en `tests/` el test que ya comprueba que los módulos puros no importan OpenCV ni MediaPipe (`grep -rn "mediapipe" tests/ | grep -i import`) y añade `lsm.spelling` a su lista de módulos. Si no encuentras uno, añade este a `tests/test_spelling.py`:

```python
def test_el_modulo_es_puro() -> None:
    """`CLAUDE.md` regla 2: sin OpenCV, sin MediaPipe, sin disco, sin camara.
    Es lo que permite que esta suite corra en CI sin hardware."""
    import lsm.spelling

    fuente = Path(lsm.spelling.__file__).read_text(encoding="utf-8")
    for prohibido in ("import cv2", "import mediapipe", "open(", "Path("):
        assert prohibido not in fuente, prohibido
```

Con `from pathlib import Path` arriba.

- [ ] **Step 6: Commit**

```bash
git add src/lsm/spelling.py tests/test_spelling.py
git commit -m "feat(spelling): borrado por simbolo y cierre de frase"
```

---

### Task 5: El HUD de la demo

**Files:**
- Modify: `src/lsm/io/preview.py`
- Test: `tests/test_io_preview.py` (créalo si no existe)

**Interfaces:**
- Consumes: `_panel` y el estilo de `draw_hud` en `io/preview.py`; `State` de `lsm.segmentation`; `Prediction` de `lsm.types`
- Produces: `DemoHudState`, `draw_demo_hud(image, state)`

- [ ] **Step 1: Lee cómo dibuja el HUD de captura**

```bash
sed -n '86,160p' src/lsm/io/preview.py
```

Fíjate en cómo usa `_panel`, de dónde saca los colores y cómo posiciona el texto. El HUD nuevo debe parecerse, no inventar un estilo.

- [ ] **Step 2: Escribe el test que falla**

`draw_demo_hud` necesita OpenCV, así que el test se salta si no está — igual que hacen los tests de `io/` que ya existen. Lo que sí se puede comprobar sin OpenCV es el dato:

```python
def test_el_hud_de_la_demo_dice_confianza_y_estado() -> None:
    """Confianza y estado siempre visibles: es lo que hace depurable la demo, y
    sin ellos un rechazo y un fallo de deteccion se ven igual."""
    state = DemoHudState(
        texto="casa me",
        palabra="me",
        estado=State.STABLE,
        ultima=Prediction(label="E", confidence=0.91),
        dispersion=0.019,
        mensaje="",
    )

    assert state.estado is State.STABLE
    assert state.ultima is not None
    assert state.ultima.confidence == 0.91
```

- [ ] **Step 3: Corre el test y comprueba que falla**

```bash
uv run pytest tests/test_io_preview.py -v
```

Esperado: FAIL con `ImportError: cannot import name 'DemoHudState'`.

- [ ] **Step 4: Implementa**

En `src/lsm/io/preview.py`:

```python
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


def draw_demo_hud(image: Any, state: DemoHudState) -> None:
    """Dibuja el HUD de la demo. Modifica `image` en el lugar.

    Confianza y estado van **siempre**, no solo cuando hay letra: sin ellos, una
    ventana rechazada por confianza baja y una mano que el detector no encuentra
    se ven exactamente igual, y depurar la demo se vuelve adivinar.
    """
    import cv2

    alto, ancho = image.shape[:2]
    _panel(image, 0, 0, ancho, 96)

    cv2.putText(
        image,
        f"texto:   {state.texto}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"palabra: {state.palabra}",
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    confianza = "—" if state.ultima is None else f"{state.ultima.confidence:.2f}"
    letra = "—" if state.ultima is None else state.ultima.label
    sigma = "—" if state.dispersion is None else f"{state.dispersion:.3f}"
    cv2.putText(
        image,
        f"{state.estado}   ultima: {letra} ({confianza})   sigma: {sigma}",
        (12, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 220, 180),
        1,
        cv2.LINE_AA,
    )

    if state.mensaje:
        _panel(image, 0, alto - 40, ancho, 40)
        cv2.putText(
            image,
            state.mensaje,
            (12, alto - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (120, 200, 255),
            1,
            cv2.LINE_AA,
        )

    aviso = "las 8 letras dinamicas llegan en la Fase 5"
    cv2.putText(
        image,
        aviso,
        (12, alto - 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (140, 140, 140),
        1,
        cv2.LINE_AA,
    )
```

Los imports que hagan falta arriba: `from lsm.segmentation import State` y `from lsm.types import Prediction`.

> El aviso de las dinámicas no es decoración: sin él, quien pruebe la demo hará una `J`, no pasará nada, y concluirá que el sistema falla cuando lo que ocurre es que esa letra no está implementada.

- [ ] **Step 5: Corre los tests y comprueba que pasan**

```bash
uv run pytest tests/test_io_preview.py -v && uv run mypy && uv run ruff check .
```

- [ ] **Step 6: Commit**

```bash
git add src/lsm/io/preview.py tests/test_io_preview.py
git commit -m "feat(preview): HUD de la demo con confianza y estado siempre visibles"
```

---

### Task 6: El CLI y la sesión, sin cámara todavía

**Files:**
- Create: `src/lsm/cli/demo.py`
- Modify: `pyproject.toml` (sección `[project.scripts]`)

**Interfaces:**
- Consumes: `run_segmentation`, `LetterEmitted`, `WindowStable`, `WindowRejected`, `StateChanged`, `State` de `lsm.segmentation`; `StaticKnnClassifier` de `lsm.classifiers.static_knn`; todo `lsm.spelling`
- Produces: `Sesion` (clase con `.aplicar(signal)`, `.hud()`, `.state`), `cargar_clasificador(path)`, `flujo_desde_dataset(raiz, config)`, `main(argv)`

- [ ] **Step 1: Escribe el CLI**

Crea `src/lsm/cli/demo.py`. Empieza por lo que **no** necesita cámara: la sesión, la carga del modelo y el flujo desde dataset.

```python
"""Demo en vivo: señas → texto (`ARQUITECTURA.md` §5, Fase 3).

Cablea cámara → MediaPipe → features → segmentación → clasificador → spelling.
Todas las piezas ya existían; esto es el cable.

## Quién mueve el bucle

`run_segmentation` **consume** el flujo, así que la demo no puede iterar frame a
frame por fuera. En vez de cambiar la segmentación —cuyo contrato está versionado
(`docs/adr/0004-contrato-de-segmentacion.md`)—, el flujo que se le pasa es un
generador que en cada `next()` captura el cuadro, dibuja el HUD con el estado que
dejó el frame anterior, lee el teclado y cede el `FrameSlot`. Iterar los eventos
mueve el bucle entero.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lsm.classifiers.static_knn import StaticKnnClassifier
from lsm.config import Config, load_config
from lsm.io.dataset import iter_sample_paths, read_sample
from lsm.io.preview import DemoHudState
from lsm.segmentation import (
    LetterEmitted,
    SegmentationEvent,
    State,
    StateChanged,
    WindowRejected,
    WindowStable,
    run_segmentation,
)
from lsm.spelling import (
    Backspace,
    CommitText,
    HandAbsent,
    HandPresent,
    LetterSignal,
    LetterWritten,
    NothingToDelete,
    Signal,
    SpaceWritten,
    SpellingState,
    SymbolDeleted,
    TextCommitted,
    render_text,
    render_word,
    step,
)
from lsm.types import InvalidFrame, InvalidReason, Prediction
from lsm.vocabulary import Label, spec

DEFAULT_MODEL = Path("data/models/static_knn.json")


@dataclass
class Sesion:
    """El estado mutable de una sesión de demo. Solo aquí hay mutación.

    `spelling.py` es puro y devuelve estados nuevos; esto es lo que los sostiene
    entre frames y lo que el HUD lee para dibujar.
    """

    config: Config
    state: SpellingState = field(default_factory=SpellingState)
    estado_maquina: State = State.IDLE
    ultima: Prediction | None = None
    dispersion: float | None = None
    mensaje: str = ""

    def aplicar(self, signal: Signal) -> None:
        resultado = step(self.state, signal, self.config)
        self.state = resultado.state
        match resultado.event:
            case LetterWritten(label=label):
                self.mensaje = f"letra {spec(label).display}"
            case SpaceWritten():
                self.mensaje = "palabra cerrada"
            case SymbolDeleted(label=label):
                self.mensaje = f"borrada {spec(label).display}"
            case NothingToDelete():
                self.mensaje = "nada que borrar"
            case TextCommitted(text=texto):
                print(texto)
                self.mensaje = f"frase cerrada: {texto}"
            case None:
                pass

    def hud(self) -> DemoHudState:
        return DemoHudState(
            texto=render_text(self.state),
            palabra=render_word(self.state),
            estado=self.estado_maquina,
            ultima=self.ultima,
            dispersion=self.dispersion,
            mensaje=self.mensaje,
        )


def aplicar_evento(sesion: Sesion, evento: SegmentationEvent) -> None:
    """Traduce un evento de la máquina de estados a lo que la demo hace con él.

    `WindowRejected` lleva el motivo pero **no** la predicción: la ventana pudo
    rechazarse antes de clasificarla. Por eso el motivo va al mensaje y la última
    predicción se deja como estaba.
    """
    match evento:
        case LetterEmitted(prediction=prediction):
            sesion.ultima = prediction
            sesion.aplicar(LetterSignal(label=Label(prediction.label)))
        case WindowStable(dispersion=dispersion):
            sesion.dispersion = dispersion
        case WindowRejected(reason=reason):
            sesion.mensaje = f"rechazo: {reason}"
        case StateChanged(current=current):
            sesion.estado_maquina = current
        case _:
            pass


def cargar_clasificador(path: Path) -> StaticKnnClassifier:
    """Carga el modelo exportado.

    `from_export` rechaza un `feature_spec_version` o un
    `handedness_convention` que no coincidan, que es la defensa contra predecir
    en silencio con una normalización distinta a la del entrenamiento.
    """
    if not path.exists():
        raise SystemExit(
            f"no hay modelo en {path}. Entrenalo primero:\n"
            "  uv run lsm-train --sin-sintetico"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return StaticKnnClassifier.from_export(payload)


def flujo_desde_dataset(raiz: Path, config: Config) -> Iterator[Any]:
    """Las muestras de `raiz`, en orden, separadas por ausencia de mano.

    El hueco entre muestras no es adorno: sin él las señas se fundirían en una
    sola ventana y la segmentación no vería dónde acaba una y empieza la
    siguiente. Su longitud es la del espacio, para que además cierre la palabra
    igual que lo haría en vivo.
    """
    hueco = config.spelling.space_after_absent_frames
    for ruta in sorted(iter_sample_paths(raiz)):
        for slot in read_sample(ruta).frames:
            yield slot
        for _ in range(hueco):
            yield InvalidFrame(reason=InvalidReason.NO_HAND, detail="entre muestras")
```

> Comprueba las firmas reales de `WindowStable` y `WindowRejected` antes de
> escribir el `match`: `grep -n "class WindowStable" -A 8 src/lsm/segmentation.py`.
> Si sus campos no se llaman `dispersion`, `reason` o `prediction`, usa los
> nombres que haya — no inventes.

- [ ] **Step 2: Añade el `main` y el parser**

Al final de `src/lsm/cli/demo.py`:

```python
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lsm-demo",
        description=(
            "Demo de deletreo manual: reconoce las señas del abecedario por "
            "cámara y las convierte en texto. El procesamiento es local."
        ),
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--modelo",
        type=Path,
        default=DEFAULT_MODEL,
        help="modelo exportado por lsm-train",
    )
    parser.add_argument(
        "--desde-dataset",
        type=Path,
        default=None,
        dest="desde_dataset",
        help=(
            "reproduce las muestras de esa ruta en vez de abrir la cámara. "
            "No necesita webcam ni MediaPipe."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    classifier = cargar_clasificador(args.modelo)
    sesion = Sesion(config=config)

    if args.desde_dataset is not None:
        flujo = flujo_desde_dataset(args.desde_dataset, config)
        for evento in run_segmentation(flujo, config, classifier.predict):
            aplicar_evento(sesion, evento)
        print(render_text(sesion.state))
        return 0

    return _sesion_en_vivo(config, classifier, sesion)
```

`_sesion_en_vivo` llega en la tarea 7. Por ahora:

```python
def _sesion_en_vivo(
    config: Config, classifier: StaticKnnClassifier, sesion: Sesion
) -> int:
    raise SystemExit("la sesión en vivo llega en la tarea 7")
```

- [ ] **Step 3: Registra el script**

En `pyproject.toml`, dentro de `[project.scripts]`:

```toml
lsm-demo = "lsm.cli.demo:main"
```

- [ ] **Step 4: Comprueba que importa sin OpenCV ni MediaPipe**

```bash
uv run python -c "import lsm.cli.demo; print('importa sin arrastrar hardware')"
uv run mypy && uv run ruff check .
```

Esperado: no falla. Si falla por un import de OpenCV, muévelo dentro de la función que lo use — es lo que hace `cli/capture.py`.

- [ ] **Step 5: Commit**

```bash
git add src/lsm/cli/demo.py pyproject.toml
git commit -m "feat(demo): sesion, carga de modelo y reproduccion desde dataset"
```

---

### Task 7: El criterio de la fase y la sesión en vivo

**Files:**
- Modify: `src/lsm/cli/demo.py`
- Test: `tests/test_cli_demo.py`

**Interfaces:**
- Consumes: todo lo de la tarea 6; `Camera`, `MediaPipeHandDetector`, `draw_landmarks`, `draw_demo_hud`
- Produces: `_sesion_en_vivo` funcional; `lsm-demo` completo

- [ ] **Step 1: Escribe el test del criterio**

Crea `tests/test_cli_demo.py`:

```python
"""El criterio de aceptacion de la Fase 3, sin camara.

> Deletrear una palabra de cinco letras sin errores de segmentacion.

Las secuencias son **sinteticas y deterministas** a proposito. Con el accuracy
de la Fase 2 (0.9261) una palabra de cinco letras sale entera el 68% de las
veces, asi que un test sobre datos reales seria intermitente — y un test
intermitente se acaba ignorando, que es exactamente como paso desapercibida la
deriva del glosario que cuenta el README.

Lo que el criterio pide es que cinco señas produzcan cinco simbolos: que la
maquina de estados no parta una seña en dos, no funda dos en una y no repita la
que sigue sostenida. Eso si es determinista.
"""

from __future__ import annotations

from lsm.cli.demo import Sesion, aplicar_evento
from lsm.config import Config
from lsm.segmentation import run_segmentation
from lsm.spelling import render_text
from lsm.types import InvalidFrame, InvalidReason
from lsm.vocabulary import Label

CONFIG = Config()
PALABRA = (Label.C, Label.A, Label.S, Label.A, Label.S)
```

Los ayudantes salen de `tests/test_segmentation.py`, que ya los tiene resueltos.
Cópialos tal cual —`still_frames`, `moving_frames`, `responses`— en vez de
inventar otros:

```python
from collections.abc import Callable

from lsm.segmentation import Classify
from lsm.synthetic import canonical_hand, to_frame, translated
from lsm.types import FrameSlot, Prediction, Sequence
from lsm.vocabulary import Label


def still_frames(
    count: int, *, at: tuple[float, float] = (640.0, 400.0)
) -> list[FrameSlot]:
    """La mano quieta en el mismo sitio: velocidad cero, la ventana se estabiliza."""
    frame = to_frame(translated(canonical_hand(), *at), width=1280, height=720)
    return [frame for _ in range(count)]


def moving_frames(count: int, *, step: float = 25.0) -> list[FrameSlot]:
    """La mano viajando: velocidad muy por encima del umbral. Es el rebote."""
    return [
        to_frame(
            translated(canonical_hand(), 300.0 + step * index, 400.0),
            width=1280,
            height=720,
        )
        for index in range(count)
    ]


def _clasificador(labels: tuple[Label, ...], confianza: float = 0.95) -> Classify:
    """Devuelve `labels` en orden, una por ventana estable.

    Determinista a proposito: lo que se testea es la segmentacion, no el
    acierto del clasificador. Recibe las etiquetas en vez de leer una global
    para que el test de la letra doble pueda pedir la suya.
    """
    pendientes = [Prediction(label=l.value, confidence=confianza) for l in labels]

    def classify(sequence: Sequence) -> Prediction:  # noqa: ARG001 — doble de pruebas
        return pendientes.pop(0) if len(pendientes) > 1 else pendientes[0]

    return classify


def _frames(labels: tuple[Label, ...]) -> list[FrameSlot]:
    """Una ventana estable por letra, con movimiento entre ellas.

    El movimiento intermedio no es adorno: sin el, la maquina se queda en STABLE
    sobre la misma ventana y el cerrojo de letras dobles bloquea la siguiente.
    """
    flujo: list[FrameSlot] = []
    for indice, _ in enumerate(labels):
        if indice:
            flujo += moving_frames(4)
        flujo += still_frames(CONFIG.segmentation.buffer_size + 2)
    return flujo


def _flujo(labels: tuple[Label, ...]) -> Iterator[FrameSlot]:
    return iter(_frames(labels))


def _deletrear(labels: tuple[Label, ...], *, rebote: bool) -> SpellingState:
    """Corre la tuberia y devuelve el estado final."""
    flujo: list[FrameSlot] = []
    for indice, _ in enumerate(labels):
        if indice and rebote:
            flujo += moving_frames(4)
        flujo += still_frames(CONFIG.segmentation.buffer_size + 2)

    sesion = Sesion(config=CONFIG)
    for evento in run_segmentation(iter(flujo), CONFIG, _clasificador(labels)):
        aplicar_evento(sesion, evento)
    return sesion.state


def _con_sesion(frames: list[FrameSlot], sesion: Sesion) -> Iterator[FrameSlot]:
    """Cede los frames aplicando presencia y ausencia, como hace la demo real.

    `run_segmentation` no emite un evento por frame, asi que la sesion no ve las
    ausencias por esa via: en la demo se aplican dentro del generador del flujo,
    un frame por vuelta. Esto lo replica, y testearlo aqui es lo que impide que
    ese mecanismo se rompa sin avisar.
    """
    for slot in frames:
        sesion.aplicar(
            HandAbsent() if isinstance(slot, InvalidFrame) else HandPresent()
        )
        yield slot
```

El test central:

```python
def test_una_palabra_de_cinco_letras_produce_cinco_simbolos() -> None:
    """EL CRITERIO DE LA FASE."""
    sesion = Sesion(config=CONFIG)

    for evento in run_segmentation(_flujo(PALABRA), CONFIG, _clasificador()):
        aplicar_evento(sesion, evento)

    assert len(sesion.state.word) == 5
    assert render_text(sesion.state) == "casas"
```

Y los tres que lo acompañan:

```python
def test_la_letra_doble_exige_el_rebote() -> None:
    """`segmentation.py` no repite la misma letra sin que la mano se mueva por
    encima de velocity_threshold. Sin rebote entre las dos N se emite una sola;
    con rebote, dos."""
    sin_rebote = _deletrear((Label.N, Label.N), rebote=False)
    con_rebote = _deletrear((Label.N, Label.N), rebote=True)

    assert render_text(sin_rebote) == "n"
    assert render_text(con_rebote) == "nn"


def test_nada_por_debajo_del_umbral_llega_al_buffer() -> None:
    exigente = Config.model_validate({"segmentation": {"min_confidence": 0.99}})
    sesion = Sesion(config=exigente)

    for evento in run_segmentation(_flujo(PALABRA), exigente, _clasificador(0.7)):
        aplicar_evento(sesion, evento)

    assert sesion.state.word == ()


def test_la_mano_abajo_entre_dos_palabras_pone_un_espacio_y_uno_solo() -> None:
    sesion = Sesion(config=CONFIG)
    palabras = (Label.C, Label.A, Label.S, Label.A)
    hueco: list[FrameSlot] = [InvalidFrame(reason=InvalidReason.NO_HAND)] * (
        CONFIG.spelling.space_after_absent_frames * 2
    )
    frames = [*_frames(palabras[:2]), *hueco, *_frames(palabras[2:])]

    for evento in run_segmentation(
        _con_sesion(frames, sesion), CONFIG, _clasificador(palabras)
    ):
        aplicar_evento(sesion, evento)

    assert render_text(sesion.state) == "ca sa"
```

- [ ] **Step 2: Corre los tests y comprueba que fallan**

```bash
uv run pytest tests/test_cli_demo.py -v
```

Esperado: FAIL. Los ayudantes `_flujo`, `_frames`, `_clasificador` y `_deletrear`
no existen todavía: escríbelos en el propio archivo de test.

- [ ] **Step 3: Haz que pasen**

Los ayudantes son de test, no de producción. Si al escribirlos descubres que
`aplicar_evento` o `Sesion` necesitan un cambio, hazlo — pero **no toques
`segmentation.py`**: si la máquina de estados no permitiera cumplir el criterio,
para y díselo a tu interlocutor humano, que es lo que manda `CLAUDE.md`.

- [ ] **Step 4: Corre los tests y comprueba que pasan**

```bash
uv run pytest tests/test_cli_demo.py -v
```

- [ ] **Step 5: Implementa la sesión en vivo**

Sustituye el `_sesion_en_vivo` provisional. OpenCV se importa **dentro** de la
función, como en `cli/capture.py`:

```python
def _sesion_en_vivo(
    config: Config, classifier: StaticKnnClassifier, sesion: Sesion
) -> int:
    """Abre la cámara y deletrea.

    El generador de frames es quien mueve el bucle: en cada `next()` captura,
    dibuja y lee el teclado. `run_segmentation` lo consume y sus eventos
    actualizan el estado que el siguiente frame dibujará.
    """
    import cv2

    from lsm.io.camera import Camera
    from lsm.io.preview import draw_demo_hud, draw_landmarks

    from lsm.io.hands import build_detector

    ventana = "demo LSM — deletreo manual"
    salir = False

    def flujo(camera: Camera, detector: Any) -> Iterator[Any]:
        nonlocal salir
        while not salir:
            frame = camera.read()
            slot = detector.detect(frame.rgb)

            imagen = frame.bgr
            if config.capture.preview_mirror:
                imagen = cv2.flip(imagen, 1)
            if not isinstance(slot, InvalidFrame):
                draw_landmarks(imagen, slot, mirrored=config.capture.preview_mirror)
            draw_demo_hud(imagen, sesion.hud())
            cv2.imshow(ventana, imagen)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                salir = True
                return
            if tecla == 8:  # BACKSPACE
                sesion.aplicar(Backspace())
            elif tecla in (13, 10):  # ENTER
                sesion.aplicar(CommitText())

            sesion.aplicar(
                HandAbsent() if isinstance(slot, InvalidFrame) else HandPresent()
            )
            yield slot

    with Camera.from_config(config.capture).open() as camera:
        detector = build_detector(config)
        try:
            for evento in run_segmentation(
                flujo(camera, detector), config, classifier.predict
            ):
                aplicar_evento(sesion, evento)
        finally:
            detector.close()
            cv2.destroyAllWindows()

    texto = render_text(sesion.state)
    if texto:
        print(texto)
    return 0
```

> **Antes de escribir esto**, mueve `_construir_detector` de `src/lsm/cli/capture.py`
> a `src/lsm/io/hands.py` como `build_detector(config: Config) -> HandDetector`,
> y haz que `cli/capture.py` la importe de ahí. Importar un privado de otro CLI es
> exactamente lo que el revisor va a marcar, y el detector le corresponde a `io/`:
> es la frontera con MediaPipe. El cuerpo de la función no cambia.

- [ ] **Step 6: Corre todo**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest -q
```

Esperado: todo verde. La suite completa debe pasar **sin cámara y sin MediaPipe**.

- [ ] **Step 7: Commit**

```bash
git add src/lsm/cli/demo.py tests/test_cli_demo.py
git commit -m "feat(demo): sesion en vivo y el criterio de cinco letras en CI"
```

---

### Task 8: Cerrar la fase

**Files:**
- Modify: `Makefile`, `docs/COMO-PROBAR.md`, `README.md`
- Create: `docs/adr/0012-controles-del-deletreo.md`

**Interfaces:**
- Consumes: todo lo anterior
- Produces: nada de código

- [ ] **Step 1: `make demo` deja de ser un `exit 1`**

En `Makefile`, sustituye la receta actual:

```makefile
# Demo en vivo: señas -> texto. Necesita camara, MediaPipe y un modelo
# entrenado. Para probarla sin webcam:
#   make demo ARGS="--desde-dataset data/raw/s01/2026-09-09-manana"
demo:
	$(UV) run lsm-demo $(ARGS)
```

- [ ] **Step 2: Escribe el ADR de los controles**

`CLAUDE.md` lo pide para decisiones reversibles con costo, y esta lo es: los
gestos de control son una interfaz que quien use la app tiene que aprender.

Crea `docs/adr/0012-controles-del-deletreo.md` con: el contexto (el clasificador
conoce 22 clases y las 21 letras están ocupadas), la decisión (mano abajo cierra
palabra; `BACKSPACE` y `ENTER` por teclado), las alternativas descartadas (un
gesto de control clasificado exigiría una clase nueva y tres personas citadas otra
vez; un borrado por gesto destruye trabajo cuando se dispara solo), y la
consecuencia (la demo no es señable de extremo a extremo, y eso hay que decirlo).

Copia el formato de cabecera de `docs/adr/0011-calibracion-de-la-fase-2.md`.

- [ ] **Step 3: Completa la sección 6 de `docs/COMO-PROBAR.md`**

Quita el aviso de «Fase 3, en construcción» y escribe los comandos reales, las
teclas (`BACKSPACE`, `ENTER`, `q`), el gesto del espacio, y que las ocho letras
dinámicas no se reconocen todavía.

- [ ] **Step 4: Actualiza el estado en `README.md`**

La Fase 3 pasa a cerrada, con el criterio y cómo se comprueba. Menciona que la
demo avisa en pantalla de las dinámicas pendientes.

- [ ] **Step 5: Corre todo por última vez**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add Makefile docs/ README.md
git commit -m "docs: cierra la Fase 3 y documenta los controles del deletreo"
```

---

## Notas para quien ejecute esto

**Lo que no puede pasar sin avisar.** Si en la tarea 7 el criterio de cinco letras
no se cumple y la causa está en `segmentation.py`, **para**. `CLAUDE.md` dice que
cuando una tarea revela que la arquitectura documentada no funciona, se propone el
cambio en vez de improvisar una excepción local. Un parche desde el CLI para
esquivar la máquina de estados es exactamente esa excepción local.

**Lo que este plan no cubre y hay que mirar a ojo:** que la demo se vea bien y que
la latencia sea tolerable. Ningún test lo dice.

**Las ocho letras dinámicas no se reconocen.** No es un fallo de esta fase: llegan
en la Fase 5 con `dynamic_dtw`. El HUD lo avisa para que nadie lo confunda con un
error.
