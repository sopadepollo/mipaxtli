# Fase 4 — Texto a señas: `signs.py`, `assets/signs/manifest.json` y `lsm-signs`

- **Fecha:** 2026-09-14
- **Fase:** 4
- **Criterio de aceptación:** las 29 letras del glosario con asset y fuente
  documentada (`docs/ARQUITECTURA.md` §5 dice 27; el glosario cerrado tiene 29:
  26 del alfabeto sin CH, más Ñ, LL y RR)
- **Depende de:** Fase 3 cerrada (`docs/adr/0013-la-ventana-mezclada.md`,
  `SEGMENTATION_SPEC_VERSION = 2`)

## 1. Qué se construye

La dirección **texto → señas**, de extremo a extremo y sin cámara:

```
texto ──► símbolos del glosario ──► lista de pasos con duración ──► reproductor ──► ventana
                                                                        ▲
                         assets/signs/manifest.json + PNG/GIF por letra ┘
```

No hay IA. El riesgo de esta fase no es técnico sino de **corrección del
material**: `ARQUITECTURA.md` §4.10 y `glosario-lsm.md` §2 avisan que casi todo
lo que circula en internet como "abecedario en lengua de señas" es ASL, y que un
proyecto que presente ASL como LSM queda descalificado ante cualquier persona
usuaria. Esta fase resuelve ese riesgo por construcción: **ningún asset viene de
internet**. Cada uno se dibuja a partir de una muestra del dataset propio, que se
grabó siguiendo una página concreta de *Manos con voz*, y el manifest apunta a
esa muestra.

Tres piezas nuevas, con el reparto puro / I-O del resto del repositorio:

| Archivo | Responsabilidad | Puro |
|---|---|---|
| `src/lsm/signs.py` | esquema del manifest, texto → símbolos, lista de pasos, elección de la muestra de referencia, geometría del dibujo, máquina de estados del reproductor | sí — sin disco, sin cámara, sin `time` |
| `src/lsm/io/signs.py` | leer/escribir `manifest.json`, cargar muestras de `data/raw`, dibujar PNG/GIF con Pillow, decodificar GIF para la ventana | no |
| `src/lsm/cli/signs.py` | `lsm-signs render` / `verificar` / `reproducir` y la ventana OpenCV | no |

`lsm-signs` es un comando propio y no `lsm-demo --texto`: `cli/demo.py` ya pasa
de 400 líneas y la dirección inversa no comparte nada con la cámara.
`ARQUITECTURA.md` §3 decía "demo.py, ambas direcciones"; se actualiza.

## 2. Decisiones ya tomadas (brainstorming del 2026-09-14)

| Decisión | Elegido | Alternativas descartadas |
|---|---|---|
| Origen de los assets | esqueleto de 21 landmarks renderizado desde `data/raw` | fotos propias (29 assets a mano), recortes del PDF de CONAPRED (derechos, sin GIF), híbrido |
| Dónde se reproduce | ventana OpenCV, como la demo | HTML generado (lógica en JS sin tests), ambos |
| `ll` y `rr` del texto | siempre el dígrafo: "llave" → `DOBLE_L,A,V,E` | dos letras, flag en config |
| Assets en git | **sí**, se versionan (esqueletos de pocos KB) | dejarlos fuera y fiarse de `render` |
| Caracteres no soportados | error con la lista exacta | saltarlos con aviso |

Lo que se pierde con el esqueleto y cómo se compensa: un dibujo 2D de landmarks
no comunica bien la **orientación de la palma** ("mira al frente" vs "de lado"),
que en LSM distingue letras. Por eso el reproductor muestra siempre la
`descripcion` y la `trayectoria` del glosario junto al dibujo, y por eso el
manifest lleva `fuente.tipo`: una letra puede sustituirse después por una foto
verificada cambiando la entrada y el archivo, sin tocar código.

## 3. `assets/signs/manifest.json`

```json
{
  "schema_version": 1,
  "fuente_normativa": "Serafín de Fleischmann y González Pérez, Manos con voz. Diccionario de Lengua de Señas Mexicana. CONAPRED / Libre Acceso A.C. Abecedario: pp. 15-19.",
  "letras": {
    "J": {
      "letra": "J",
      "archivo": "J.gif",
      "es_dinamica": true,
      "descripcion": "Mano cerrada, el dedo meñique bien estirado señalando hacia arriba y la palma a un lado dibuja una j en el aire",
      "trayectoria": "dibuja una j en el aire",
      "pagina": 16,
      "duracion_ms": 2000,
      "fuente": {
        "tipo": "esqueleto_desde_dataset",
        "muestra": "s01/2026-09-09-manana/J/020.json",
        "signer_id": "s01",
        "lateralidad_original": "RIGHT",
        "espejada": false
      },
      "revision": {
        "fecha": "2026-09-14",
        "revisor": "…",
        "resultado": "coincide",
        "nota": ""
      }
    }
  }
}
```

### Esquema (Pydantic, en `signs.py`)

```python
MANIFEST_SCHEMA_VERSION: Final = 1

class AssetSource(BaseModel):
    tipo: Literal["esqueleto_desde_dataset"]   # único tipo hoy; una foto sería otro
    muestra: str          # ruta relativa a data/raw: signer/sesion/LABEL/NNN.json
    signer_id: str
    lateralidad_original: Handedness
    espejada: bool        # True si se espejó en X para mostrar mano derecha

class AssetReview(BaseModel):
    fecha: date
    revisor: str
    resultado: Literal["coincide", "difiere", "pendiente"]
    nota: str = ""

class SignAsset(BaseModel):
    letra: str            # cómo se escribe: "Ñ", "LL", "RR"
    archivo: str          # "<LABEL>.png" | "<LABEL>.gif"
    es_dinamica: bool
    descripcion: str
    trayectoria: str      # NO_TRAJECTORY ("—") si es estática
    pagina: int
    duracion_ms: int | None   # obligatorio y > 0 si es dinámica; None si es estática
    fuente: AssetSource
    revision: AssetReview

class Manifest(BaseModel):
    schema_version: int
    fuente_normativa: str
    letras: dict[Label, SignAsset]
```

Reglas validadas por el modelo (fallan al cargar, no en silencio):

- `schema_version == MANIFEST_SCHEMA_VERSION`; una versión distinta se rechaza,
  igual que un modelo exportado con otra `feature_spec_version`.
- `es_dinamica` ⇔ `archivo` termina en `.gif`; estática ⇔ `.png`. Es la regla
  "las dinámicas necesitan movimiento, no imagen fija", hecha tipo.
- `archivo == f"{label}.{ext}"`: el nombre lo fija la etiqueta, no se elige.
- `duracion_ms` es `int > 0` si dinámica y `None` si estática.
- `Label.NONE` no puede aparecer: no es una letra.
- `fuente.muestra` tiene forma `<signer>/<sesion>/<LABEL>/<NNN>.json` y el
  `<LABEL>` coincide con la clave.

### Deriva contra `vocabulary.py`

`descripcion`, `trayectoria`, `pagina`, `es_dinamica` y `letra` se **copian** de
`vocabulary.py` al renderizar. Copiarlas y no referenciarlas es deliberado: el
manifest tiene que poder consumirse desde JavaScript en la Fase 7 sin arrastrar
`vocabulary.py`. Para que dos copias no sean dos verdades:

```python
def manifest_drift(manifest: Manifest) -> list[str]
```

Devuelve una lista de discrepancias legibles (letra que falta, letra sobrante,
campo distinto) contra `vocabulary.LETTERS`. Vacía si coinciden. Es el mismo
patrón que `io/glossary.vocabulary_drift` para glosario ↔ código.

## 4. Render de assets — `lsm-signs render`

### 4.1 Elegir la muestra de referencia (puro, `signs.py`)

```python
@dataclass(frozen=True, slots=True)
class ReferenceChoice:
    sample: Sample
    path: str            # fuente.muestra
    mirrored: bool       # se dibujará espejada
    candidates: int      # cuántas se consideraron; va al log

def choose_reference(samples: Sequence[Sample], config: Config) -> ReferenceChoice
```

1. Filtra las muestras del `kind` que corresponde a la letra (`STATIC` para
   estáticas, `DYNAMIC` para dinámicas): `NONE` y las grabaciones "estáticas" de
   una letra dinámica no sirven de referencia.
2. Prefiere mano derecha: si hay muestras con `handedness == RIGHT`, se queda
   solo con ellas y `mirrored = False`. Si no hay ninguna, usa las izquierdas y
   `mirrored = True`. El diccionario muestra mano derecha; el dataset tiene mitad
   y mitad por firmante, así que en la práctica no habrá que espejar, pero el
   camino existe y queda anotado en `fuente.espejada`.
3. Extrae features con `extract_sequence_features`. El vector de comparación es
   `static.shape` (42 componentes) para estáticas y `dynamic.rows` aplanado
   (24×44) para dinámicas; una dinámica cuya extracción devuelva
   `DynamicUnavailable` o `ExtractionRejected` se descarta como candidata.
4. Calcula el centroide y devuelve la **medoide**: la muestra real cuya distancia
   euclidiana al centroide es mínima. Empate → la de ruta lexicográficamente
   menor, para que `render` sea determinista.

No es un promedio inventado: es una grabación concreta, la más típica de cómo
ese grupo hizo la letra, y se puede ir a ver.

### 4.2 Geometría del dibujo (puro, `signs.py`)

```python
Point2: TypeAlias = tuple[float, float]

def project_frames(
    frames: Sequence[RawFrame], *, mirrored: bool, canvas_px: int, margin: float
) -> tuple[tuple[Point2, ...], ...]
```

- Toma `x, y` normalizados de cada landmark, corrige la relación de aspecto
  (`x · aspect_ratio`) para que la mano no salga achatada, y espeja en X si
  `mirrored`.
- Calcula **una** caja envolvente sobre **toda** la secuencia, no por frame: en
  una dinámica el desplazamiento es la seña, y encuadrar frame a frame lo
  borraría.
- Escala y centra esa caja en un lienzo cuadrado de `canvas_px` con `margin`
  (fracción) de aire, conservando proporciones.
- Vista de cámara, no espejo: el dibujo es como te ve quien te mira, igual que
  las fotos del diccionario.

Sin Pillow ni OpenCV: devuelve puntos. Se testea con secuencias sintéticas.

### 4.3 Dibujar y escribir (`io/signs.py`)

- Pillow (`ImageDraw`): fondo claro, conexiones de `HAND_CONNECTIONS` como
  líneas, landmarks como círculos, la muñeca (`LandmarkIndex.WRIST`) y las yemas
  algo más grandes para orientar la vista. Colores y grosores son constantes de
  dibujo, no umbrales: viven en el módulo.
- **Estática:** un PNG del frame central de la secuencia.
- **Dinámica:** un GIF con todos los frames de la secuencia a
  `config.signs.render_fps`, en bucle. `duracion_ms = round(1000 · n / fps)`.
- Pillow entra como dependencia del extra `capture` en `pyproject.toml` (ya está
  en `uv.lock` de forma transitiva; OpenCV no escribe GIF). No es dependencia del
  núcleo: la suite sigue corriendo sin ella.
- Nota de implementación: el plan puso además Pillow en el grupo `dev` de
  `pyproject.toml`, porque la suite renderiza assets en `tmp_path` y lee los
  GIF del manifest real (`tests/test_io_signs.py`, `tests/test_signs_manifest.py`).
  Por eso `uv sync` (sin extras) sí instala Pillow y la suite la necesita; lo
  que sigue siendo opcional es OpenCV, para la ventana de `reproducir`.

### 4.4 El comando

```
lsm-signs render [--raw data/raw] [--assets assets/signs] [--revisor NOMBRE]
```

1. Carga las muestras de `data/raw` con `io.dataset.iter_samples` y agrupa por
   etiqueta. Si a alguna letra del glosario le faltan candidatas válidas, **falla
   antes de escribir nada** y dice cuáles.
2. Para cada letra: `choose_reference` → `project_frames` → PNG/GIF.
3. Escribe `manifest.json`. **Las revisiones sobreviven al re-render** si
   `fuente.muestra` no cambió: se copian de la entrada anterior. Si la muestra
   elegida es otra, la revisión vuelve a `pendiente`. Sin esto, regenerar los
   assets borraría trabajo humano sin avisar.
4. Escribe con `indent=2`, claves ordenadas y salto final: dos ejecuciones sobre
   el mismo dataset dan los mismos bytes (como `lsm-eval`).

## 5. Verificación LSM-no-ASL

Tres capas, de la más automática a la más humana:

1. **Trazabilidad (automática, en CI).** Cada asset apunta a una muestra de
   `data/raw`, y cada muestra se grabó contra una página de *Manos con voz* que
   el manifest repite. No existe camino por el que entre una imagen ajena.
2. **Coherencia (automática, en CI).** `manifest_drift` vacía: lo que dice el
   manifest de cada letra es exactamente lo que dice el glosario transcrito.
3. **Revisión contra la descripción (humana, registrada).** Se mira cada render
   junto a `descripcion` y `trayectoria` y se anota `revision.resultado`.
   La primera pasada la hace quien ejecuta esta fase y se registra como revisión
   **contra descripción** — no es la validación por persona usuaria de LSM del
   PENDIENTE-HUMANO G del glosario, que sigue abierto y se dice así en el ADR.

Qué mirar en la revisión, letra por letra: los dedos estirados/doblados que la
descripción nombra, la dirección de la trayectoria en las dinámicas (una J traza
una j, no una i con gancho invertido), y que las cuatro parejas
"idénticas salvo movimiento" (L/LL, R/RR, N/Ñ, I/J) se distingan **en el GIF**.

`lsm-signs verificar` repite 1 y 2, comprueba que cada archivo existe, que el
número de frames de cada GIF cuadra con `duracion_ms` y `render_fps`, y que
todas las revisiones dicen `coincide`. Sale con 1 y lista los problemas si no.

## 6. Texto → símbolos → pasos (puro, `signs.py`)

### 6.1 Símbolos

```python
class WordGap: ...                              # separador de palabras
Token: TypeAlias = Label | WordGap

class UnsupportedCharacters(ValueError):
    chars: tuple[str, ...]                      # únicos, en orden de aparición

def text_to_symbols(text: str) -> tuple[Token, ...]
```

Normalización, en este orden:

1. `ñ`/`Ñ` → marcador interno antes de tocar los acentos (en NFD la ñ es
   `n` + tilde y se perdería).
2. NFD y descarte de marcas combinantes: `á→a`, `ü→u`.
3. Mayúsculas.
4. Dígrafos, vorazmente y de izquierda a derecha: `LL→DOBLE_L`, `RR→DOBLE_R`.
   `CH` son dos letras (`C`, `H`): el glosario no tiene CH.
5. Espacios: una o más → un `WordGap`; los del principio y del final se descartan.
6. Cualquier otro carácter → `UnsupportedCharacters` con la lista completa.
   Nada se descarta en silencio.

Ejemplos que fijan el contrato:

| Entrada | Salida |
|---|---|
| `"llave"` | `DOBLE_L, A, V, E` |
| `"carro"` | `C, A, DOBLE_R, O` |
| `"año"` | `A, ENIE, O` |
| `"Árbol  verde "` | `A, R, B, O, L, WordGap, V, E, R, D, E` |
| `"chico"` | `C, H, I, C, O` |
| `"hola2!"` | `UnsupportedCharacters(("2", "!"))` |

### 6.2 Pasos

```python
class StepKind(StrEnum): LETTER, GAP

@dataclass(frozen=True, slots=True)
class Step:
    kind: StepKind
    label: Label | None       # None si GAP
    duration_ms: float        # a velocidad 1.0

def build_playlist(tokens, manifest: Manifest, config: Config) -> tuple[Step, ...]
```

- Estática: `config.signs.static_hold_ms`.
- Dinámica: `manifest.letras[label].duracion_ms × config.signs.dynamic_loops` —
  el GIF se ve completo N veces.
- `WordGap`: `config.signs.word_gap_ms`.
- Texto sin ninguna letra → lista vacía; el CLI lo rechaza con mensaje.

### 6.3 Reproductor

Máquina de estados sobre eventos cerrados, como `spelling.py`. El tiempo entra
por `Tick(dt_ms)`: el módulo no llama a `time` (`CLAUDE.md` regla 2, como
`telemetry.py`).

```python
# Entradas
Tick(dt_ms: float) | TogglePause | Next | Prev | Restart | Faster | Slower

@dataclass(frozen=True, slots=True)
class PlayerState:
    index: int = 0
    elapsed_ms: float = 0.0
    paused: bool = False
    speed: float = 1.0
    finished: bool = False

# Salidas
StepStarted(index) | Finished

def step(state, event, playlist, config) -> tuple[PlayerState, tuple[PlayerEvent, ...]]
def asset_frame(state, step_: Step, n_frames: int, duracion_ms: int) -> int
```

Reglas:

- `Tick` en pausa o terminado no cambia nada. Si no, `elapsed += dt · speed`;
  al alcanzar `duration_ms` pasa al siguiente paso con `elapsed = 0` y emite
  `StepStarted`. Al agotar el último, `finished = True`, `index` se queda en el
  último y emite `Finished`.
- `Next` en el último paso termina; `Prev` en el primero reinicia el paso;
  ambos ponen `elapsed = 0` y quitan `finished`.
- `Restart` vuelve a `PlayerState(speed=state.speed)`: la velocidad se conserva.
- `Faster`/`Slower` suman/restan `config.signs.speed_step` acotado a
  `[speed_min, speed_max]`.
- `asset_frame` da qué frame del GIF mostrar: `int(elapsed / duracion_ms ·
  n_frames) % n_frames`. Con `dynamic_loops = 2` el GIF da dos vueltas.

Una lista vacía es error del constructor del reproductor, no un estado.

## 7. La ventana — `lsm-signs reproducir "texto"`

Patrón de `io/preview.py`: OpenCV solo se importa dentro de la función que
dibuja; la suite no lo necesita.

```
┌────────────────────────────┬────────────────────────────────────────┐
│                            │  LL                          3 / 5     │
│      [asset PNG/GIF]       │  Mano cerrada y los dedos índice y     │
│       canvas_px            │  pulgar alargados, se forma una l…     │
│                            │  Trayectoria: movimientos horizontales │
│                            │  ▶ 1.0×   ████████░░░░  1.2 s / 2.0 s  │
├────────────────────────────┴────────────────────────────────────────┤
│  L L A V E   ·   espacio: pausa  n/p: sig/ant  r: reinicio  +/-  q  │
└─────────────────────────────────────────────────────────────────────┘
```

- Panel izquierdo: el asset actual; en un `GAP`, lienzo vacío con "espacio".
- Panel derecho: letra grande (`letra`, no `label`: se ve `Ñ`, no `ENIE`),
  `descripcion` con salto de línea, `trayectoria` si es dinámica, estado
  (▶/⏸), velocidad, barra de progreso y posición.
- Abajo: el texto completo en símbolos, con el actual resaltado.
- Teclas: `ESPACIO` pausa/reanuda, `n`/`p` siguiente/anterior, `r` reinicia,
  `+`/`-` velocidad, `q` sale. Letras y no flechas porque los códigos de flecha
  cambian por plataforma en `cv2.waitKey`.
- El `dt_ms` de cada `Tick` es tiempo real medido en el CLI con
  `time.monotonic` entre iteraciones; `config.signs.tick_ms` es solo la espera
  de `waitKey`.
- Al terminar se queda en el último paso hasta `q` o `r`.
- El pie de la ventana lleva el aviso de siempre: deletreo manual, no LSM como
  lengua; procesamiento local.

Antes de abrir la ventana el CLI carga el manifest, comprueba que existen los
archivos de las letras que **este** texto necesita y decodifica sus frames una
vez (Pillow → arreglos BGR). Un archivo que falta es un error con la ruta, no un
recuadro negro.

## 8. `config.yaml` — sección `signs`

```yaml
signs:
  static_hold_ms: 1500     # cuánto se sostiene una letra estática
  dynamic_loops: 2         # vueltas completas del GIF de una dinámica
  word_gap_ms: 800         # pausa entre palabras
  render_fps: 12           # fps del GIF al renderizar; fija duracion_ms
  canvas_px: 320           # lado del lienzo de cada asset
  canvas_margin: 0.12      # aire alrededor de la mano, fracción del lienzo
  speed_min: 0.25
  speed_max: 4.0
  speed_step: 0.25
  tick_ms: 33              # espera de waitKey en la ventana
```

`SignsConfig(_Section)` en `config.py` con rangos, y `speed_min < speed_max`
validado. Cero umbrales en código.

## 9. Tests

| Archivo | Qué exige |
|---|---|
| `tests/test_signs.py` | `text_to_symbols`: la tabla del §6.1 completa, dígrafos al inicio/fin/solos, `ñ` con y sin acentos alrededor, rechazo con lista exacta. `build_playlist`: duraciones desde config y manifest. Reproductor: avance por ticks, pausa congela, `Next`/`Prev` acotados, `Restart` conserva velocidad, límites de velocidad, `Finished` una sola vez, `asset_frame` con `dynamic_loops`. `choose_reference`: prefiere derecha, medoide determinista, descarta `kind` ajeno. `project_frames`: caja única sobre la secuencia, aspecto, espejo, margen. |
| `tests/test_signs_manifest.py` | Carga el `manifest.json` real. Exige: 29 claves = `ALPHABET` sin `NONE`; validadores del §3 (dinámica ⇔ `.gif`); `manifest_drift` vacía; cada `archivo` existe en `assets/signs/`; cada GIF tiene `round(duracion_ms · render_fps / 1000)` frames; `revision.resultado == "coincide"` en todas. **Este test es el criterio de la fase.** |
| `tests/test_config.py` | `signs` carga con los valores de `config.yaml`; `speed_min >= speed_max` se rechaza. |
| `tests/test_cli_signs.py` | `render` sobre un `data/raw` sintético en `tmp_path` produce 29 archivos y un manifest válido; re-render conserva revisiones; `verificar` sale 1 con un archivo borrado; `reproducir` con texto vacío o caracteres no soportados sale 1 sin abrir ventana. Sin OpenCV: la ventana se inyecta como en `test_cli_demo.py`. |

Los tests del manifest leen disco a propósito, como los del glosario: el
criterio de la fase es un archivo concreto en el repositorio.

## 10. Documentación y cierre

- **ADR 0014 — Los assets son esqueletos del dataset propio.** Por qué (LSM por
  construcción, sin derechos, reproducible), qué se pierde (orientación de la
  palma; legibilidad frente a una foto) y cómo se compensa (descripción siempre
  visible; `fuente.tipo` extensible). Deja explícito que la revisión del manifest
  es contra descripción y que el PENDIENTE-HUMANO G sigue abierto.
- `docs/ARQUITECTURA.md`: árbol del §3 (`signs.py`, `io/signs.py`,
  `cli/signs.py`), §4.10 apunta al ADR, §5 corrige "27" por "29".
- `docs/COMO-PROBAR.md`: sección nueva "7. Texto a señas" con los tres comandos
  y qué esperar; las secciones siguientes se renumeran.
- `README.md`: estado de la fase.
- `CLAUDE.md` regla 2: `signs.py` entra en la lista de módulos puros.
- `Makefile`: `signs TEXTO="..."`, `signs-render`, `signs-verificar`; `help`.
- `pyproject.toml`: `lsm-signs = "lsm.cli.signs:main"`; `pillow` en el extra
  `capture`.
- `.gitignore`: se quitan las tres líneas de `assets/signs/*.{webp,gif,mp4}`.
  Los assets se versionan.

## 11. Fuera de alcance de esta fase

- Fotos reales de manos. El manifest lo permite (`fuente.tipo`), pero no se
  hacen ahora.
- Validación por persona usuaria de LSM (PENDIENTE-HUMANO G).
- Reproductor en JavaScript (Fase 7). El manifest ya está pensado para que lo
  consuma: JSON plano, sin referencias a código Python.
- Reconocer letras dinámicas en la dirección señas → texto (Fase 5).
