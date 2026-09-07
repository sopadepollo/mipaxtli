# Especificación de Features — v1

**`FEATURE_SPEC_VERSION = 1`**

Este documento es un **contrato**. Define la transformación exacta desde los
landmarks crudos de MediaPipe hasta el vector de características que consumen los
clasificadores.

Cualquier implementación (Python, TypeScript, o futura) debe producir resultados
idénticos dentro de una tolerancia de `1e-6`, verificado contra
`tests/fixtures/golden_features.json`.

**Regla de cambio:** modificar cualquier paso de este documento obliga a incrementar
`FEATURE_SPEC_VERSION`, regenerar los golden vectors, reentrenar todos los modelos y
escribir un ADR. Los modelos exportados con una versión distinta a la del runtime
deben rechazarse al cargarse, nunca ejecutarse.

---

## 0. Entrada

### 0.1 Landmarks

MediaPipe Hands entrega 21 landmarks por mano, cada uno con `(x, y, z)`:

- `x` normalizado por el **ancho** del frame, rango aproximado `[0, 1]`
- `y` normalizado por el **alto** del frame, rango aproximado `[0, 1]`, con el eje
  **apuntando hacia abajo** (convención de imagen)
- `z` profundidad relativa a la muñeca, en escala aproximada a la de `x`

Índices (constantes en `src/lsm/types.py`):

```
 0  WRIST
 1  THUMB_CMC     2  THUMB_MCP     3  THUMB_IP     4  THUMB_TIP
 5  INDEX_MCP     6  INDEX_PIP     7  INDEX_DIP    8  INDEX_TIP
 9  MIDDLE_MCP   10  MIDDLE_PIP   11  MIDDLE_DIP  12  MIDDLE_TIP
13  RING_MCP     14  RING_PIP     15  RING_DIP    16  RING_TIP
17  PINKY_MCP    18  PINKY_PIP    19  PINKY_DIP   20  PINKY_TIP
```

### 0.2 Metadatos requeridos por frame

- `width`, `height` del frame en píxeles
- `handedness` ∈ `{LEFT, RIGHT}` y su score
- `detection_score`

### 0.3 Convenciones obligatorias de captura

> **MediaPipe recibe siempre el frame sin espejar.** El espejado del preview (que se
> hace por comodidad del usuario) ocurre únicamente en la capa de visualización. Si
> se alimenta a MediaPipe la imagen espejada, la lateralidad reportada se invierte y
> el paso 2 de esta especificación corrompe el vector de forma silenciosa.

- Si se detectan varias manos, se usa la de mayor `detection_score`. El alfabeto
  dactilológico de LSM es monomanual.
- Si no se detecta ninguna mano, el frame se marca **inválido**. Los frames
  inválidos no se interpolan: interrumpen la secuencia.

---

## 1. Pipeline por frame

Se aplica en este orden exacto. `p_i` denota el landmark `i`.

### Paso 1 — Corrección de relación de aspecto y orientación

MediaPipe normaliza `x` e `y` por dimensiones distintas. En un frame 16:9 esto
deforma la mano horizontalmente. Además se invierte `y` para que el eje apunte hacia
arriba, que es la convención del resto de la especificación.

```
a  = width / height

p_i.x ←  p_i.x * a
p_i.y ← -p_i.y
p_i.z ←  p_i.z * a
```

### Paso 2 — Canonicalización de lateralidad

Todas las muestras se llevan a una mano derecha canónica. Esto evita duplicar el
dataset y permite que una persona zurda use un modelo entrenado por diestros.

```
si handedness == LEFT:
    p_i.x ← -p_i.x    para todo i
```

### Paso 3 — Traslación al origen

```
w = p_0
p_i ← p_i - w    para todo i
```

Tras este paso `p_0 = (0, 0, 0)`.

> El desplazamiento de la mano en el encuadre se destruye aquí de forma
> **intencional**: la identidad de una seña no depende de dónde esté la mano. La
> trayectoria, que sí importa para las señas dinámicas, se preserva en un canal
> aparte (§3).

### Paso 4 — Escala

Se usa la distancia muñeca → nudillo del dedo medio como unidad. Es estable frente
a la flexión de los dedos, a diferencia de cualquier distancia que involucre una
punta.

```
s = sqrt(p_9.x² + p_9.y²)          # norma 2D, ignora z

si s < 1e-6:
    frame INVÁLIDO — descartar

p_i ← p_i / s    para todo i
```

Tras este paso `‖p_9‖₂ = 1` en el plano XY.

### Paso 5 — Rotación en el plano XY

Alinea el eje de la palma con el eje +Y. Da tolerancia a la inclinación de la
muñeca.

```
θ = atan2(p_9.y, p_9.x)
φ = π/2 - θ

para todo i:
    x' = p_i.x * cos(φ) - p_i.y * sin(φ)
    y' = p_i.x * sin(φ) + p_i.y * cos(φ)
    p_i.x ← x'
    p_i.y ← y'
    # p_i.z no se modifica
```

Tras este paso `p_0 = (0, 0)` y `p_9 = (0, 1)` exactamente.

### Paso 6 — Descarte de Z

**Decisión v1: `z` se descarta.** Ver `docs/adr/0002-formato-de-features.md`.

Razón: la coordenada `z` de MediaPipe es una profundidad relativa estimada, ruidosa
e inconsistente entre frames y entre cámaras. Introduce varianza sin aportar señal
confiable con datasets pequeños. El código conserva el canal calculado para poder
reactivarlo como spec v2 si la matriz de confusión lo justifica.

### Paso 7 — Aplanado

```
f = [p_0.x, p_0.y, p_1.x, p_1.y, ..., p_20.x, p_20.y]    ∈ ℝ⁴²
```

Orden estricto: por índice de landmark ascendente, `x` antes que `y`.

**Sobre los componentes constantes.** Por construcción `p_0 = (0,0)` y `p_9 = (0,1)`
en todas las muestras. Se conservan deliberadamente: aportan distancia cero en
cualquier métrica, no afectan al clasificador, y mantener los 21 índices alineados
con la numeración de MediaPipe elimina una fuente crónica de errores off-by-one al
depurar y al reimplementar en TypeScript.

---

## 2. Agregación de secuencia — señas estáticas

Entrada: ventana de `T` frames válidos consecutivos.

```
F = mean_t( f_t )        ∈ ℝ⁴²     # vector de forma
σ = mean_j( std_t( f_t[j] ) )      # escalar de dispersión
```

- `F` es el vector que recibe el clasificador estático.
- `σ` **no es una feature**. Es un indicador de calidad: si supera
  `config.quality.max_dispersion`, la ventana se considera inestable y se rechaza
  antes de clasificar. Sirve además como criterio de aceptación durante la captura
  del dataset.

**`std` es la desviación poblacional (ddof = 0).** Explícitamente:

```
media_j = ( Σ_t f_t[j] ) / T                    # suma en orden temporal ascendente
var_j   = ( Σ_t (f_t[j] - media_j)² ) / T       # divide entre T, no entre T-1
σ       = ( Σ_j sqrt(var_j) ) / 42              # suma en orden de índice ascendente
```

Fijarlo importa más de lo que parece: σ no viaja en los golden vectors por frame,
así que si Python usara poblacional y TypeScript muestral, ningún test lo
detectaría y la app web rechazaría ventanas que en escritorio pasaban. Por eso los
`sequence_cases` de `golden_features.json` incluyen σ como valor esperado.

---

## 3. Agregación de secuencia — señas dinámicas

Las letras con movimiento (J, Ñ, Q, RR, X, Z, según el glosario) no se distinguen por
la configuración de la mano sino por el recorrido que traza.

> **Problema crítico.** El paso 3 traslada cada frame por su propia muñeca, lo que
> **elimina el movimiento de la mano en el espacio**. Un clasificador alimentado solo
> con `f_t` no puede distinguir una J de una I: la configuración de dedos es la misma
> y solo cambia el trazo. La solución no es omitir la traslación —eso reintroduciría
> la dependencia de la posición en el encuadre— sino recuperar la trayectoria en un
> canal separado.

### 3.1 Canal de trayectoria

Se computa a nivel de secuencia, no de frame. Sea `w_t` la muñeca cruda tras el
**paso 2** (con aspecto corregido, `y` invertida y lateralidad canonizada, pero
**antes** de trasladar), y `s_t` la escala del paso 4:

```
s̄ = mean_t( s_t )
τ_t = (w_t - w_0) / s̄        ∈ ℝ²
```

- Origen en la muñeca del primer frame → invariante a la posición en el encuadre.
- Escala en unidades de mano → invariante a la distancia a la cámara.

### 3.2 Remuestreo temporal

Toda secuencia se remuestrea a `T_ref = 24` frames por interpolación lineal sobre el
índice temporal normalizado a `[0, 1]`. Esto normaliza la duración y acota el costo
del DTW.

Sean T_src frames válidos. Tiempos origen s_i = i / (T_src − 1) para i ∈ [0, T_src−1]. Tiempos destino u_j = j / 23 para j ∈ [0, 23]. Interpolación lineal componente a componente sobre f_t y τ_t por separado. Los extremos se preservan exactamente: u_0 → s_0 y u_23 → s_{T_src−1}. Si T_src == 1, se replica el frame. Si T_src < config.dtw.min_source_frames, la secuencia se rechaza en vez de interpolarse.

**Orden de operaciones del mapeo**, obligatorio para la paridad numérica:

```
pos_j = ( j · (T_src − 1) ) / (T_ref − 1)        # el producto ANTES que la división
i     = floor(pos_j)
frac  = pos_j - i
out_j = rows[i] + frac · (rows[i+1] - rows[i])   # si i ≥ T_src-1, out_j = rows[T_src-1]
```

Escrito como `(j / (T_ref − 1)) · (T_src − 1)` el resultado es matemáticamente el
mismo pero arrastra el redondeo del cociente intermedio: con `T_src = T_ref = 24`,
`(7/23)·23` no da exactamente `7`, y filas que deberían quedarse quietas se
desplazan una fracción de índice. La forma de arriba sí devuelve el índice exacto
en ese caso.

El peso `w_τ` del §3.3 se aplica **después** de interpolar, nunca antes.

La ponderación y el remuestreo se aplican a los canales por separado: primero se
remuestrea `f_t`, luego `τ_t`, y solo entonces se concatenan.

### 3.3 Vector por frame para el clasificador dinámico

```
g_t = concat( f_t , w_τ · τ_t )        ∈ ℝ⁴⁴
```

`w_τ = config.features.trajectory_weight`, por defecto `4.0`.

Justificación del peso: el canal de forma aporta 42 componentes y el de trayectoria
solo 2. Sin ponderación, la distancia euclidiana queda dominada por la forma y el
trazo —que es precisamente lo que distingue estas letras— resulta invisible. El valor
por defecto es un punto de partida; debe ajustarse empíricamente y el ajuste
registrarse.

### 3.4 Distancia

DTW sobre las secuencias `(24, 44)` con distancia euclidiana local, ventana de
Sakoe-Chiba de radio `config.dtw.band_radius` (por defecto `6`), y costo normalizado
por la longitud del camino de alineación.

---

## 4. Suavizado temporal (opcional)

Media móvil exponencial sobre los landmarks crudos, **antes del paso 1**:

```
p̃_t = α · p_t + (1 - α) · p̃_{t-1}
```

con `p̃_0 = p_0`: la serie arranca en el primer frame observado de la secuencia.
Cualquier otra inicialización —arrancar en cero, por ejemplo— inventaría un
desplazamiento desde el origen del encuadre que nadie ejecutó, y contaminaría el
canal de trayectoria del §3.1. El estado se reinicia en cada secuencia: un frame
inválido interrumpe la secuencia (§0.3) y con ella la media móvil.

`α = config.smoothing.alpha`, por defecto `1.0` (desactivado). Valores menores
reducen el jitter de MediaPipe a costa de latencia y de emborronar los movimientos
rápidos, lo que perjudica a las señas dinámicas. Si se activa, debe activarse
idénticamente en la implementación web.

---

## 5. Determinismo entre implementaciones

Requisitos para que Python y TypeScript coincidan:

1. **Aritmética en float64** en ambas. En TypeScript, `number` ya es float64; evitar
   `Float32Array` en la ruta de features.
2. **Clamp obligatorio antes de `acos`** al rango `[-1, 1]`. El error de punto
   flotante produce argumentos como `1.0000000000000002` que devuelven `NaN`. Aplica
   al bloque derivado del apéndice A.
3. **`atan2(y, x)`** con ese orden de argumentos en ambos lenguajes.
4. **Sin reordenar operaciones**: la suma en punto flotante no es asociativa. El
   promedio del §2 se calcula sumando en orden temporal ascendente y dividiendo al
   final.
5. **Sin optimizaciones que salten pasos**, aunque sean matemáticamente equivalentes
   (por ejemplo, fusionar traslación y rotación en una matriz). La equivalencia
   algebraica no implica equivalencia numérica.

### 5.1 Golden vectors

`tests/fixtures/golden_features.json`:

```json
{
  "feature_spec_version": 1,
  "tolerance": 1e-6,
  "cases": [
    {
      "id": "right_hand_upright_open",
      "input": {
        "width": 1280,
        "height": 720,
        "handedness": "RIGHT",
        "landmarks": [[0.51, 0.62, 0.0], "... 21 tripletas ..."]
      },
      "expected_features": ["... 42 valores float64 ..."]
    }
  ]
}
```

Cobertura mínima obligatoria — 20 casos que incluyan:

| Caso | Qué valida |
|---|---|
| Mano derecha vertical | Camino base |
| Mano izquierda, misma seña | Paso 2; debe dar features casi idénticas al caso derecho |
| Mano rotada 45° y 90° | Paso 5 |
| Mano cerca y lejos (escala 2×) | Paso 4; features casi idénticas |
| Mano en las 4 esquinas del encuadre | Paso 3; features casi idénticas |
| Frame 16:9 y frame 1:1 | Paso 1 |
| `p_9 == p_0` (escala degenerada) | Debe marcar frame inválido |
| Landmarks con `x` fuera de `[0,1]` | MediaPipe extrapola fuera del frame; no debe romper |

Los pares "debe dar features casi idénticas" son los tests más valiosos del
proyecto: verifican que las invariancias realmente se cumplen, no solo que el código
corre.

El archivo lo genera `make golden` desde la implementación de Python, que es la
referencia normativa.

### 5.2 Campos del archivo

El esquema del §5.1 se mantiene y se completa con lo que un frame suelto no puede
expresar. Cada caso de `cases` lleva:

| Campo | Significado |
|---|---|
| `id` | Identificador estable del caso. |
| `description` | Qué representa la entrada. |
| `validates` | Qué paso del contrato verifica. Un golden vector sin explicación es un número mágico con formato JSON. |
| `input` | `width`, `height`, `handedness` y los 21 landmarks. |
| `expected_features` | Los 42 valores, o `null` si el frame debe rechazarse. |
| `expected_invalid_reason` | Motivo tipado del rechazo (`SCALE_TOO_SMALL`), o `null`. |
| `same_features_as` | Opcional: `id` de otro caso cuyas features deben coincidir dentro de la tolerancia. Codifica los pares de invariancia. |

**Los frames inválidos se representan con un centinela explícito**
(`expected_features: null` más `expected_invalid_reason`), nunca por ausencia del
campo: un caso al que le falta una clave se confunde con un archivo truncado.

### 5.3 `sequence_cases`

Bloque hermano de `cases`. Cubre lo que el §5.1 no alcanza: el canal de
trayectoria (§3.1), el remuestreo (§3.2), la ponderación de `g_t` (§3.3), la
dispersión σ (§2) y las secuencias interrumpidas (§0.3).

Cada caso lleva la secuencia fuente completa —de longitud arbitraria y con los
huecos marcados con `{"valid": false, "reason": ...}`— y, en `expected.runs`, una
entrada por cada secuencia válida máxima con:

| Campo | Significado |
|---|---|
| `length` | Frames de la secuencia válida. |
| `frame_features` | `f_t` de cada frame. |
| `static_features`, `dispersion` | `F` y σ del §2. |
| `mean_scale`, `trajectory` | `s̄` y `τ_t` del §3.1, sin remuestrear. |
| `resampled_trajectory` | `τ` tras el remuestreo del §3.2, sin ponderar. |
| `dynamic_rows` | `g_t` final: 24 filas de 44 componentes. |
| `dynamic_unavailable_reason` | `TOO_FEW_SOURCE_FRAMES` si la secuencia se rechazó para el canal dinámico; en ese caso los dos campos anteriores van en `null`. |

La cobertura incluye **la misma interrupción al inicio, en medio y al final**,
porque los tres casos se comportan distinto y no basta con probar uno:

- al inicio, el origen de `τ` se mueve al primer frame válido y el trazo restante
  se mide desde otro punto;
- al final, la secuencia simplemente se corta antes y `τ` conserva su origen;
- en medio quedan **dos** secuencias, no una con un salto.

---

## Apéndice A — Bloque derivado (v2 planificada, no activa)

Especificado por adelantado para poder activarse sin rediseño si la matriz de
confusión muestra que las letras con configuración de puño (A, E, M, N, S y
similares) no se separan con las 42 componentes geométricas.

Se concatena a `f`, produciendo ℝ⁵⁶. Todas las distancias se calculan sobre los
landmarks ya normalizados (tras el paso 5), en 2D.

**Distancias punta–muñeca (5):**
`‖p_4‖`, `‖p_8‖`, `‖p_12‖`, `‖p_16‖`, `‖p_20‖`

**Distancias entre puntas adyacentes (4):**
`‖p_4 - p_8‖`, `‖p_8 - p_12‖`, `‖p_12 - p_16‖`, `‖p_16 - p_20‖`

**Ángulos de flexión (5):** ángulo interno en la articulación media de cada dedo.

| Dedo | Cadena `(a, j, b)` |
|---|---|
| Pulgar | (2, 3, 4) |
| Índice | (5, 6, 7) |
| Medio | (9, 10, 11) |
| Anular | (13, 14, 15) |
| Meñique | (17, 18, 19) |

```
u = p_a - p_j
v = p_b - p_j
c = (u · v) / (‖u‖ ‖v‖)
c = clamp(c, -1, 1)
ángulo = acos(c)          # radianes, [0, π]
```

Al activarse: `FEATURE_SPEC_VERSION = 2`, nuevos golden vectors, reentrenamiento
completo y ADR justificando el cambio con la matriz de confusión que lo motivó.
