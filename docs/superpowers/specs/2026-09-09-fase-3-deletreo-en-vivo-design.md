# Fase 3 — Deletreo en vivo: `spelling.py` y `cli/demo.py`

- **Fecha:** 2026-09-09
- **Fase:** 3
- **Criterio de aceptación:** deletrear una palabra de cinco letras sin errores de
  segmentación (`docs/ARQUITECTURA.md` §5)
- **Depende de:** Fase 2 cerrada (`docs/adr/0011-calibracion-de-la-fase-2.md`)

## 1. Qué se construye

La dirección **señas → texto** de extremo a extremo:

```
cámara → MediaPipe → features → segmentación → clasificador → spelling → texto
```

Todas las piezas existen menos las dos últimas. `spelling.py` acumula las letras
en palabras; `cli/demo.py` conecta el resto y dibuja lo que está pasando.

## 2. El criterio dice «segmentación», no «clasificación»

La distinción decide cómo se testea la fase, así que va primero.

Con el accuracy de la Fase 2 (0.9261), una palabra de cinco letras sale entera el
**68%** de las veces por pura aritmética. Si el test de la fase dependiera del
acierto del clasificador sería intermitente, y un test intermitente se acaba
ignorando — que es exactamente cómo pasó desapercibida la deriva del glosario que
cuenta el README.

Lo que el criterio pide es que **cinco señas produzcan cinco símbolos**: que la
máquina de estados no parta una seña en dos, no funda dos en una, no repita la que
sigue sostenida y no escriba nada mientras la mano viaja. Eso es determinista y se
puede exigir siempre.

De ahí el reparto:

| Qué | Cómo | Dónde |
|---|---|---|
| El criterio de la fase | secuencias sintéticas deterministas | `tests/test_spelling.py`, `tests/test_cli_demo.py` |
| Acierto extremo a extremo con datos reales | `lsm-demo --desde-dataset` | a mano, para depurar |
| La demo de verdad | `lsm-demo` | delante de una cámara |

## 3. `spelling.py` — reductor puro sobre señales

`CLAUDE.md` regla 2: sin OpenCV, sin MediaPipe, sin disco, sin cámara.

### Tipos

```python
# Señales: lo que la demo ve en cada frame. Tipos cerrados, ni cadenas ni bools.
LetterSignal(label: Label)   # de un LetterEmitted de la segmentación
HandPresent()                # un frame con mano
HandAbsent()                 # un frame sin mano
Backspace()                  # tecla
CommitText()                 # tecla

Signal: TypeAlias = LetterSignal | HandPresent | HandAbsent | Backspace | CommitText

@dataclass(frozen=True, slots=True)
class SpellingState:
    #: La palabra en curso, en SÍMBOLOS del glosario y no en caracteres.
    word: tuple[Label, ...] = ()
    #: Palabras ya cerradas, cada una con sus símbolos.
    finished: tuple[tuple[Label, ...], ...] = ()
    #: Frames consecutivos sin mano. Vive en el estado para que el criterio del
    #: espacio sea puro y no un contador suelto en el CLI.
    absent_frames: int = 0
    #: Ya se puso espacio por esta ausencia. Sin esto, la mano quieta abajo
    #: escribiría un espacio por frame.
    space_emitted: bool = False

# Eventos: qué acaba de pasar. Alimentan el HUD y el log.
LetterWritten(label) | SpaceWritten() | SymbolDeleted(label)
| TextCommitted(text: str) | NothingToDelete()

def step(state: SpellingState, signal: Signal, config: Config) -> StepResult
def render_word(state: SpellingState) -> str
def render_text(state: SpellingState) -> str
```

`StepResult` es un frozen dataclass con `state` y `event: SpellingEvent | None`.

### Por qué símbolos y no caracteres

El buffer guarda `Label`. Se dibuja con `LetterSpec.display`, que ya existe en
`vocabulary.py` con ese propósito exacto (`Ñ`, `LL`, `RR`).

Así `DOBLE_R` es **un** símbolo que se lee `rr`, y dos `R` seguidas son **dos**
símbolos que también se leen `rr`. `BACKSPACE` borra un símbolo, de modo que
deshace exactamente lo que se señó. Con un buffer de caracteres, borrar sobre una
`RR` dejaría una `r` que nadie ejecutó.

Esto resuelve la pregunta que `ARQUITECTURA.md` §4.2 dejó abierta para esta fase.

### `space_emitted` es el mismo cerrojo que `pending_repeat`

`segmentation.py` usa `pending_repeat` para no reemitir la letra que sigue
sostenida. Aquí el problema es idéntico un nivel más arriba: la mano abajo no es un
evento, es un estado que dura, y sin cerrojo pondría un espacio en cada frame. Se
libera con el primer `HandPresent`.

## 4. Los controles

| Control | Señal | Qué hace |
|---|---|---|
| mano abajo ≥ `space_after_absent_frames` | `HandAbsent` repetida | cierra la palabra en curso y abre otra — **el commit de palabra** |
| `BACKSPACE` | `Backspace` | borra el último símbolo de la palabra en curso |
| `ENTER` | `CommitText` | cierra la **frase**: la imprime y vacía el buffer |
| `q` | — | salir |

**Por qué la mano abajo y no un gesto clasificado.** El clasificador conoce 22
clases y las 21 letras están ocupadas; un gesto de control tendría que ser una
clase nueva, y una clase nueva son tres personas citadas otra vez. Bajar la mano
entre palabras es además lo que se hace de todos modos, y la máquina de estados ya
lo detecta.

**Por qué el borrado va por teclado.** Un borrado que se dispara solo es peor que
no tener borrado: destruye trabajo y quien firma no siempre mira la pantalla. En
una demo, la tecla es la opción honesta.

`BACKSPACE` con la palabra vacía **no hace nada** y lo dice (`NothingToDelete`). No
recupera la palabra anterior: recuperar significaría reabrir algo ya cerrado y no
vale la complejidad en esta fase.

## 5. `cli/demo.py` — quién mueve el bucle

`run_segmentation(stream, config, classify)` **consume** el flujo, así que no puede
ser la demo la que itere frame a frame por fuera.

La solución no toca la segmentación, cuyo contrato está versionado con
`SEGMENTATION_SPEC_VERSION` (`docs/adr/0004-contrato-de-segmentacion.md`): **el
flujo que se le pasa es un generador que, en cada `next()`, captura el cuadro,
dibuja el HUD con el estado actual, lee el teclado y cede el `FrameSlot`.** Iterar
los eventos mueve el bucle entero.

```python
def _frames(camara, sesion, ...) -> Iterator[FrameSlot]:
    while True:
        imagen = camara.read()
        slot = detector.detect(imagen)
        draw_landmarks(imagen, slot, mirrored=...)
        draw_demo_hud(imagen, sesion.hud())   # el estado que dejó el frame anterior
        tecla = mostrar_y_leer(imagen)
        sesion.aplicar(_senal_de_tecla(tecla))
        sesion.aplicar(HandPresent() if slot.valid else HandAbsent())
        yield slot

for evento in run_segmentation(_frames(...), config, classify):
    if isinstance(evento, LetterEmitted):
        sesion.aplicar(LetterSignal(Label(evento.prediction.label)))
    ...
```

El modelo se carga de `data/models/static_knn.json` con
`StaticKnnClassifier.from_export`, que ya rechaza un `feature_spec_version` o un
`handedness_convention` que no coincidan.

`--desde-dataset RUTA` sustituye la cámara por las muestras de esa ruta,
concatenadas con huecos entre ellas. Misma tubería, sin webcam.

## 6. El HUD

`io/preview.py` recibe un `DemoHudState` y `draw_demo_hud`. El `HudState` actual es
de captura —letra objetivo, muestras guardadas, si graba video— y no sirve.

```
texto:    casa mes_
palabra:  c a s
estado:   STABLE      conf 0.87     sigma 0.019
ultimo:   S (0.91)              rechazo: LOW_CONFIDENCE
```

Confianza y estado de la máquina **siempre visibles**: es lo que hace depurable la
demo, y sin ellos un rechazo y un fallo de detección se ven igual.

El HUD avisa además de que las ocho letras dinámicas no se reconocen todavía, o
quien pruebe la demo creerá que la `J` falla cuando en realidad no está.

## 7. Configuración

Sección `spelling` en `config.yaml`, validada con Pydantic (`CLAUDE.md` regla 5):

```yaml
spelling:
  # Frames consecutivos sin mano antes de cerrar la palabra en curso. A 30 fps,
  # 30 frames es un segundo.
  space_after_absent_frames: 30
```

`config.py` valida que sea **mayor** que `segmentation.missing_frames_to_idle` (8).
Si no, un parpadeo del detector escribiría un espacio. Es la misma validación
cruzada que ya existe entre `reject_cooldown_frames` y `emit_cooldown_frames`.

Los defaults de Pydantic y `config.yaml` tienen que coincidir: hay un test que lo
comprueba.

## 8. Qué se testea

**`tests/test_spelling.py`** — el reductor, con listas de señales:

- cinco letras dan cinco símbolos y `render_word` los une bien
- `HandAbsent` repetida por debajo del umbral **no** pone espacio; al alcanzarlo
  pone exactamente uno, por mucho que siga la ausencia
- `HandPresent` libera el cerrojo y permite el siguiente espacio
- `BACKSPACE` borra un símbolo; con la palabra vacía no hace nada y lo dice
- `DOBLE_R` renderiza `rr` y se borra de una; dos `R` renderizan `rr` y se borran
  de dos. Es la prueba de la decisión de §3
- `ENTER` cierra la frase y deja el estado vacío

**`tests/test_cli_demo.py`** — el criterio de la fase, sin cámara:

- una palabra de cinco letras construida con secuencias sintéticas produce
  exactamente esas cinco letras: ni una de más por la ventana que sigue estable, ni
  una de menos por el cooldown
- una palabra con letra doble ejercita la regla de letras dobles: sin el rebote no
  se emite la segunda
- nada por debajo de `segmentation.min_confidence` llega al buffer
- la mano abajo entre dos palabras produce un espacio y uno solo

**Lo que no se testea automáticamente** es que la demo se vea bien y que la
latencia sea tolerable. Eso se mira.

## 9. Fuera de alcance

- Dirección texto → señas: Fase 4.
- Las ocho letras dinámicas: Fase 5. La demo lo dice en pantalla.
- Diccionario, autocorrección o predicción de palabra. El proyecto traduce
  deletreo manual; adivinar la palabra sería otra cosa y taparía justo los errores
  que esta fase existe para enseñar.
- Cambiar `segmentation.py`. Si la demo revelara que el contrato de segmentación no
  funciona, `CLAUDE.md` manda parar y proponerlo, no parchear desde el CLI.
