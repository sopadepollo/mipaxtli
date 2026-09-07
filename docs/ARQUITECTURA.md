# Traductor LSM — Especificación de arquitectura

Proyecto: traductor bidireccional del alfabeto dactilológico de la Lengua de Señas
Mexicana (LSM).

- **Dirección A (señas → texto):** la cámara captura al usuario, el sistema reconoce
  las letras y las acumula en palabras.
- **Dirección B (texto → señas):** el usuario escribe y el sistema muestra la
  secuencia de señas correspondiente.

Este documento define la estructura de código, las decisiones de diseño y las
complicaciones previstas. Es el insumo para la implementación.

---

## 1. Alcance y no-alcance

**Dentro del alcance**

- Alfabeto dactilológico completo de LSM (27 letras + dígrafos LL y RR si aplica).
- Señas estáticas y señas con movimiento (J, K, Ñ, Q, RR, X, Z y las que apliquen).
- Ejecución en escritorio (Fase 1) y en navegador móvil (Fase 2).

**Fuera del alcance (declararlo explícitamente en el README y en la UI)**

- No traduce LSM como lengua. LSM tiene gramática, señas léxicas y componentes no
  manuales (expresión facial, movimiento corporal). Esto es únicamente deletreo
  manual. Presentarlo como "traductor de lengua de señas" es incorrecto y la
  comunidad sorda lo señala con razón.
- No hay reconocimiento de rostro ni identificación de personas.

---

## 2. Decisión de diseño central

> **Todo dato de entrada es una secuencia temporal de landmarks, nunca un frame
> suelto.**

Esta es la decisión que hay que tomar el primer día y no cambiar. Si el tipo base
del sistema es `Sequence[Frame[Landmark]]` con forma `(T, 21, 3)`:

- Una seña estática es una secuencia donde `T` frames son aproximadamente iguales.
- Una seña dinámica es una secuencia donde `T` frames varían.

Ambas usan la misma tubería, el mismo formato de dataset, el mismo código de
captura y la misma interfaz de clasificador. Lo único que cambia es la
implementación del clasificador.

El error costoso sería empezar con `predict(frame) -> letra` y descubrir a mitad
del proyecto que hay que rehacer captura, dataset, entrenamiento y evaluación para
soportar movimiento.

---

## 3. Estructura de repositorio

```
lsm-translator/
├── CLAUDE.md                     # instrucciones permanentes para Claude Code
├── README.md
├── Makefile                      # atajos: make capture / train / eval / dev
├── pyproject.toml                # un solo proyecto Python, uv o poetry
├── .env.example
│
├── docs/
│   ├── adr/                      # Architecture Decision Records numerados
│   │   ├── 0001-secuencias-como-tipo-base.md
│   │   ├── 0002-formato-de-features.md
│   │   └── 0003-formato-de-export-de-modelo.md
│   ├── feature-spec.md           # CONTRATO: landmarks crudos → vector de features
│   ├── dataset-schema.md         # estructura y metadatos del dataset
│   └── glosario-lsm.md           # letras, cuáles son dinámicas, referencias
│
├── src/lsm/
│   ├── types.py                  # LandmarkFrame, Sequence, Prediction, Handedness
│   ├── features.py               # normalización y extracción — SIN I/O
│   ├── segmentation.py           # máquina de estados: cuándo empieza/termina seña
│   ├── classifiers/
│   │   ├── base.py               # Protocol común
│   │   ├── static_knn.py
│   │   ├── dynamic_dtw.py
│   │   └── registry.py           # selección por configuración
│   ├── export.py                 # modelo entrenado → JSON portable
│   ├── spelling.py               # letras → palabras (buffer, espacio, borrado)
│   ├── io/
│   │   ├── camera.py             # OpenCV: única fuente de frames
│   │   ├── hands.py              # wrapper de MediaPipe, aislado tras interfaz
│   │   └── dataset.py            # lectura/escritura de muestras
│   └── cli/
│       ├── capture.py            # recolección de dataset
│       ├── train.py
│       ├── evaluate.py           # métricas + matriz de confusión
│       └── demo.py               # demo en vivo, ambas direcciones
│
├── tests/
│   ├── fixtures/
│   │   ├── golden_features.json  # vectores de referencia Python ↔ JS
│   │   └── sequences/            # secuencias grabadas para tests sin cámara
│   ├── test_features.py
│   ├── test_segmentation.py
│   └── test_classifiers.py
│
├── web/                          # Fase 2 — no tocar hasta cerrar Fase 1
│   ├── src/
│   │   ├── features.ts           # REIMPLEMENTACIÓN del contrato de features
│   │   ├── segmentation.ts
│   │   └── classifier.ts         # consume el JSON exportado
│   └── tests/
│       └── features.test.ts      # valida contra golden_features.json
│
├── assets/signs/                 # imágenes/GIFs por letra (dirección texto→señas)
│   └── manifest.json             # letra → archivo, es_dinamica, descripción
│
├── data/                         # gitignored
│   ├── raw/
│   └── models/
│
└── docker/
    ├── Dockerfile
    ├── docker-compose.yml
    └── README.md                 # cómo pasar la cámara al contenedor
```

---

## 4. Complicaciones previstas y cómo se abordan desde el inicio

### 4.1 Invariancia: distancia, posición y mano

**Problema.** MediaPipe entrega coordenadas normalizadas al frame. Si entrenas a
50 cm de la cámara y luego te paras a 1.5 m, el vector de entrada es distinto y el
modelo falla. Igual si mueves la mano a otra esquina del encuadre, o si el usuario
es zurdo.

**Solución (va en `features.py` y se documenta en `docs/feature-spec.md`):**

1. Trasladar: restar la coordenada de la muñeca (landmark 0) a todos los puntos.
2. Escalar: dividir por una distancia de referencia estable, por ejemplo
   muñeca → nudillo del dedo medio (landmark 0 → 9).
3. Rotar: alinear ese mismo vector con un eje fijo, para tolerar inclinación de
   muñeca.
4. Canonizar lateralidad: si MediaPipe reporta mano izquierda, espejar en X. Así
   el dataset no necesita duplicarse por mano.
5. Aplanar a vector de 63 valores (o 42 si se descarta la profundidad Z, que en
   MediaPipe es poco confiable — evaluarlo y registrarlo en un ADR).

**No omitir este paso.** Es la diferencia entre un modelo que funciona solo para
quien lo entrenó y uno que funciona en general.

### 4.2 Segmentación: ¿cuándo empieza y termina una seña?

**Problema.** El video es continuo, las letras son discretas. Sin resolver esto, el
sistema emite 30 letras por segundo o emite letras basura mientras la mano viaja de
una posición a otra.

**Solución: máquina de estados explícita en `segmentation.py`.**

```
IDLE ──(mano detectada)──> TRACKING
TRACKING ──(velocidad < umbral por N frames)──> STABLE
STABLE ──(clasificación con confianza > umbral)──> EMIT
EMIT ──(cooldown)──> TRACKING
TRACKING ──(sin mano por M frames)──> IDLE
```

Detalles a implementar:

- Velocidad = media del desplazamiento de landmarks entre frames consecutivos.
- Buffer circular de los últimos `T` frames (T configurable, ~15-30).
- Cooldown tras emitir, para no repetir la misma letra mientras la mano sigue quieta.
- Para señas dinámicas: si se detecta movimiento sostenido con un patrón (no ruido),
  la ventana completa se manda al clasificador dinámico en vez de esperar
  estabilidad.

Los umbrales van en un archivo de configuración, no hardcodeados. Se van a ajustar
mucho durante las pruebas.

#### Aristas que el diagrama no dibuja (resueltas en la Fase 0)

El diagrama describe el camino feliz. Al implementarlo aparecieron tres huecos que
cualquier implementación tiene que cubrir; se resolvieron así y quedan
documentados en `src/lsm/segmentation.py`:

1. **`STABLE → TRACKING` cuando la mano vuelve a moverse.** Sin esta arista, una
   ventana que se estabilizó y no llegó a emitir se quedaría estable para siempre.
2. **`→ IDLE` desde cualquier estado al perder la mano**, no solo desde TRACKING:
   la mano puede desaparecer con la ventana ya estable o durante el cooldown.
3. **Qué hacer cuando la clasificación devuelve `UNKNOWN` o baja confianza.** Es
   la decisión con más consecuencias. Si un rechazo no cuesta nada, la máquina se
   queda en STABLE reclasificando la misma ventana en cada frame: treinta llamadas
   por segundo al clasificador para volver a rechazarla. Si cuesta lo mismo que
   una emisión, una letra que quedó apenas bajo el umbral obliga a rehacer la seña
   completa.

   **Se resuelve con dos cooldowns distintos**: `emit_cooldown_frames` tras emitir
   y `reject_cooldown_frames`, más corto, tras rechazar. El rechazo no cambia de
   estado —la mano sigue quieta y la ventana sigue siendo estable—, solo suspende
   la clasificación unos frames. `config.py` valida que el cooldown de rechazo no
   supere al de emisión.

Comportamiento conocido y aceptado: con la mano quieta, la letra se re-emite cada
`emit_cooldown_frames + stable_frames` frames. El cooldown acota la repetición
pero no la elimina. Eliminarla exigiría pedir movimiento explícito entre letras,
que es una arista nueva en este diagrama; si las pruebas en vivo de la Fase 3
muestran que molesta, se propone aquí y se registra en un ADR.

### 4.3 Estáticas vs dinámicas: dos clasificadores, una interfaz

**Interfaz común (`classifiers/base.py`):**

```python
class Classifier(Protocol):
    def fit(self, samples: list[Sample]) -> None: ...
    def predict(self, seq: Sequence) -> Prediction: ...  # incluye confianza
    def export(self) -> dict: ...  # JSON portable a JS
```

**Implementaciones:**

- `static_knn.py` — promedia los frames de la secuencia, compara contra centroides
  por clase con distancia euclidiana o coseno. Simple, interpretable, exportable a
  JSON trivialmente (solo son vectores).
- `dynamic_dtw.py` — Dynamic Time Warping sobre la secuencia de features contra
  plantillas de referencia. Tolera que la seña se haga más rápido o más lento.
  También exportable como JSON (las plantillas) y reimplementable en JS en ~60
  líneas.

**Por qué DTW y no una LSTM.** Con un dataset pequeño hecho por 3-5 personas, una
red recurrente sobreajusta y además complica el export a navegador. DTW no requiere
entrenamiento, funciona con pocas muestras y es depurable. Si al evaluar resulta
insuficiente, se agrega una tercera implementación detrás de la misma interfaz sin
tocar nada más. Registrar la decisión en un ADR.

**Enrutamiento:** un `registry.py` decide qué clasificador usar según la energía de
movimiento de la ventana, o simplemente corre ambos y toma la mayor confianza.

### 4.4 Clase de rechazo

**Problema.** Un clasificador de 27 clases siempre devuelve una de las 27, aunque
la mano esté saludando o rascándose la nariz.

**Solución:**

- Umbral de confianza; por debajo, se emite `UNKNOWN` y no se agrega nada al texto.
- Incluir en el dataset una clase negativa explícita: mano relajada, mano en
  movimiento de transición, gestos no-seña.
- La UI debe mostrar la confianza. Ayuda muchísimo a depurar en demos.

### 4.5 Paridad Python ↔ JavaScript

**Problema.** En Fase 2 se reimplementa la extracción de features en TypeScript. Si
hay la menor diferencia (orden de operaciones, qué landmark se usa como referencia,
si se normaliza antes o después de espejar), el modelo entrenado en Python da
resultados distintos en el navegador y la depuración es infernal.

**Solución, desde Fase 1:**

1. `docs/feature-spec.md` describe la transformación paso a paso, sin ambigüedad.
2. `tests/fixtures/golden_features.json` contiene ~20 pares
   `(landmarks_de_entrada, vector_de_features_esperado)` generados por Python.
3. El test de TypeScript carga ese mismo archivo y verifica coincidencia dentro de
   una tolerancia de punto flotante.

Estos golden vectors se generan en Fase 1 aunque el código web no exista todavía.

### 4.6 Formato de export del modelo

Un solo archivo JSON versionado:

```json
{
  "schema_version": 1,
  "feature_spec_version": 1,
  "classifier": "static_knn",
  "labels": ["A", "B", "..."],
  "params": { "...": "..." },
  "data": { "...": "..." }
}
```

`feature_spec_version` es obligatorio: si cambia la normalización, los modelos
viejos deben rechazarse en carga en vez de dar predicciones silenciosamente malas.

### 4.7 Dataset: el cuello de botella real

**Problema.** El modelo no será mejor que los datos. Un dataset hecho por una sola
persona, en una sola sesión, con una sola iluminación, da 98% en validación y 40%
en la demo frente al profesor.

**Solución:**

- Cada muestra guarda metadatos: `signer_id`, `session_id`, `timestamp`,
  `handedness`, `lighting`, `distance`, `label`.
- **Validación por persona (leave-one-signer-out), no aleatoria.** Un split
  aleatorio mezcla frames de la misma grabación entre train y test y da métricas
  falsamente optimistas.
- Mínimo 3 personas distintas, 2 sesiones cada una, ~20 repeticiones por letra.
  Es menos trabajo del que suena: son unos 40 minutos por persona.
- Guardar los landmarks crudos, no solo las features. Si cambia la normalización,
  se re-deriva sin volver a grabar. Guardar video opcionalmente y solo con
  consentimiento explícito.
- El CLI de captura debe mostrar en pantalla qué letra se está grabando, contador de
  muestras y preview con landmarks dibujados.

### 4.8 Letras confundibles

En LSM varias letras comparten configuración de puño con diferencias sutiles de
pulgar. Se abordará con:

- Matriz de confusión obligatoria en el reporte de `evaluate.py`.
- Si un par concentra el error, agregar features específicas (ángulos entre falanges,
  distancias pulgar-dedos) antes que cambiar de modelo.

### 4.9 Cámara y Docker

**Problema.** En Linux se pasa `--device=/dev/video0`. En Windows/macOS, Docker
corre en una VM y no hay acceso directo a la webcam.

**Solución arquitectónica:** el módulo `io/camera.py` es la única frontera con el
hardware. Todo lo demás (`features`, `segmentation`, `classifiers`) son funciones
puras sobre arreglos. Esto permite:

- Correr entrenamiento, evaluación y tests dentro de Docker sin cámara.
- Correr la captura y la demo fuera de Docker en Windows/macOS si hace falta.
- Ejecutar la suite de tests en CI sin hardware.

Documentar ambos caminos en `docker/README.md`. No pelear con Docker Desktop por
la webcam: no vale el tiempo.

### 4.10 Dirección texto → señas

Esta dirección no requiere IA y por lo tanto es la más rápida de terminar, pero
tiene su propio riesgo: **la corrección de los materiales**.

- `assets/signs/manifest.json` mapea letra → archivo, `es_dinamica`, descripción
  textual de la configuración manual.
- Las señas dinámicas necesitan GIF o video corto, no imagen fija.
- Las imágenes deben verificarse contra una fuente confiable de LSM (no ASL — se
  confunden constantemente en internet) y, si es posible, revisarse con una persona
  usuaria de LSM o un intérprete. Anotar la fuente de cada asset en el manifest.

### 4.11 Ética, privacidad y validación

- El procesamiento ocurre en el dispositivo. Ningún frame sale a un servidor.
  Declararlo en la UI.
- No almacenar video sin consentimiento explícito y por escrito de quien firma.
- Buscar retroalimentación de la comunidad sorda o de un intérprete de LSM antes de
  presentar el proyecto. Un traductor hecho sin consultar a sus usuarios es un
  patrón conocido y criticado.
- El README debe indicar con claridad que la herramienta es un apoyo puntual, no un
  sustituto de un intérprete.

---

## 5. Plan de fases

| Fase | Entregable | Criterio de aceptación |
|---|---|---|
| 0 | Andamiaje: repo, tipos, `features.py`, tests con fixtures sintéticos, Docker | `make test` pasa en CI sin cámara |
| 1 | `capture.py` + dataset de 1 persona, letras estáticas | ≥100 muestras por letra estática con metadatos |
| 2 | `static_knn` + `train`/`evaluate` + matriz de confusión | ≥90% con validación leave-one-signer-out |
| 3 | `segmentation.py` + `demo.py` (señas → texto en vivo) | deletrear una palabra de 5 letras sin errores de segmentación |
| 4 | Dirección texto → señas + assets verificados | 27 letras con asset y fuente documentada |
| 5 | `dynamic_dtw` + señas con movimiento | letras dinámicas reconocidas en vivo |
| 6 | Dataset multi-persona, ajuste de umbrales | métricas estables entre firmantes |
| 7 | Web app (MediaPipe JS) + paridad de features | test de golden vectors pasa en TS; funciona en celular |

Las fases 0-3 son el núcleo. Si el tiempo se acorta, 4 y 5 se recortan antes que
comprometer la calidad del núcleo.

---

## 6. Convenciones técnicas

- Python 3.11+, tipado estático con anotaciones, `mypy` en CI.
- `ruff` para lint y formato.
- `pytest` con fixtures; el núcleo debe testearse sin cámara ni MediaPipe.
- Configuración en `config.yaml` validado con Pydantic. Cero umbrales hardcodeados.
- Logging estructurado, no `print`.
- Sin dependencias de MediaPipe fuera de `src/lsm/io/hands.py`.
- Commits convencionales, un ADR por cada decisión de arquitectura reversible con
  costo.
