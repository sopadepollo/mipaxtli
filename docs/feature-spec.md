# Especificación de Features — v4

**`FEATURE_SPEC_VERSION = 4`**

> **v4** (2026-09-29, bloque de tolerancia a MediaPipe,
> `docs/adr/0034-bloque-de-tolerancia-v4.md`): tres cambios en una sola versión,
> con los golden regenerados una vez. La **plausibilidad anatómica** (§0.4)
> invalida los frames imposibles antes del relleno; el **filtro One Euro** (§4)
> suaviza los landmarks crudos con el dt real; y **δ_t = ln(m_t / m_0)**, la
> profundidad relativa, es la componente 45 de `g_t` (§3.1, §3.3). Cada uno
> tiene su interruptor en `config.yaml` para medirlo por separado; el modelo
> exportado declara el preprocesado con que se entrenó y el runtime rechaza uno
> distinto. Modelos v3 rechazados al cargar; hay que reentrenar.

> **v3** (2026-09-28, `docs/adr/0020-escala-y-rotacion-con-la-palma-de-canto.md`):
> la escala del paso 4 es el **tamaño de palma** y no la distancia muñeca →
> nudillo 9, que se colapsa con la palma de canto (en la X quedaba en 0.17 del
> tamaño real). El ángulo del paso 5 se **sostiene** desde el último frame
> fiable mientras la palma está de canto, con histéresis, y una secuencia sin
> ningún frame fiable se rechaza (`PALM_EDGE_ON`). Las estáticas no se colapsan
> nunca (s/palma ≥ 0.702): para ellas solo cambia el divisor del paso 4.
> Modelos v2 rechazados al cargar; hay que reentrenar.

> **v2** (2026-09-26, `docs/adr/0017-diagnostico-de-tracking.md`): la lateralidad
> que entra al paso 2 es la **mano declarada** de la sesión, no la etiqueta que el
> detector pone a cada frame. Las cuentas de los §1 a §6 no cambian; cambia de
> dónde sale la entrada `handedness`, y un modelo v1 se entrenó con la otra. El
> modelo exportado deja de llevar `handedness_convention` y lleva
> `detector_input = "UNMIRRORED"` (§0.3).

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
- `handedness` ∈ `{LEFT, RIGHT}`: **la mano declarada** de la sesión (v2). La
  declara quien firma al abrir la captura o la demo (`--mano`) y es la misma en
  todos los frames. La etiqueta del detector viaja aparte, como
  `detected_handedness` con su score, y **no entra en ningún paso**: con la palma
  de lado MediaPipe cambia de opinión a mitad de un trazo —217 veces en una
  sesión de diagnóstico—, y cada cambio espejaba la mano de un frame al
  siguiente (ADR 0017).
- `detection_score`
- `timestamp_ms` (opcional, ADR 0018): milisegundos de un reloj monótono con
  origen arbitrario, el mismo que recibe el detector. Lo llevan todos los frames
  de un flujo o ninguno, y crece estrictamente. **No entra en los §1 a §3.** Un
  frame sin marca vale `índice · 1000 / fps` a la tasa nominal
  (`lsm.timing.frame_times_ms`).

### 0.3 Convenciones obligatorias de captura

> **MediaPipe recibe siempre el frame sin espejar.** El espejado del preview (que se
> hace por comodidad del usuario) ocurre únicamente en la capa de visualización. Si
> se alimenta a MediaPipe la imagen espejada, la mano declarada derecha llega con la
> geometría de una izquierda, el paso 2 la canoniza hacia el reflejo y el vector se
> corrompe de forma silenciosa. Desde la v2 es **esto** lo que se verifica
> (`lsm-capture calibrar`, geométricamente: la mano derecha levantada junto al
> hombro derecho tiene que aparecer en la mitad izquierda de la imagen) y lo que
> el modelo exportado declara en `detector_input`.

- Si se detectan varias manos, se usa la de mayor `detection_score`. El alfabeto
  dactilológico de LSM es monomanual.
- Si no se detecta ninguna mano, el frame se marca **inválido**. Los frames
  inválidos no se interpolan: interrumpen la secuencia.
- **Excepción: los huecos cortos se rellenan** (SEGMENTATION_SPEC_VERSION 7, ADR
  0021; ampliada en el Paso 4, ADR 0029, que entra con el bloque de tolerancia en
  la versión siguiente). Una racha de frames inválidos —sin mano, score bajo o
  `IMPLAUSIBLE` (§0.4)— entre dos frames válidos se rellena interpolando los
  **landmarks crudos** —antes del paso 1— si dura como mucho el límite del estado
  en que empieza: `segmentation.dynamic_max_gap_ms` en DYNAMIC_CANDIDATE,
  `tracking_max_gap_ms` en TRACKING y `stable_max_gap_ms` en STABLE (0 = no se
  rellena); en IDLE y EMIT, nunca. En las muestras guardadas, el límite es el del
  trazo en una dinámica y el de STABLE en una estática. En vivo, mientras el hueco
  está abierto la máquina **sostiene** el último frame válido: no cuenta ausencia
  ni decide nada, y va hasta ese límite por detrás:

  ```
  n       = frames del hueco
  t_k     = k / (n + 1),  k = 1 … n
  p_{k,i} = a_i + (b_i − a_i) · t_k         # x, y, z de cada landmark, en ese orden de operaciones
  ```

  con `a` el último frame válido antes del hueco y `b` el primero después. Los
  frames rellenados heredan la resolución y la mano de `a`, toman como scores el
  menor de los dos extremos y no tienen `detected_handedness`. Condiciones:

  - la mano declarada y la resolución coinciden en `a` y `b`; si no, no se
    rellena;
  - un hueco más largo que el máximo no se rellena y corta la secuencia, como
    siempre;
  - en una muestra, los inválidos del principio y del final no se rellenan: se
    recortan;
  - si los frames interpolados superan `segmentation.dynamic_max_interpolated_fraction`
    de la secuencia resultante (con `>`), la secuencia se rechaza; en vivo, un
    trazo así se descarta y una ventana estable así no se clasifica todavía
    (espera a crecer, sin cooldown).

  La implementación es `lsm.gaps` (relleno) y `segmentation.run_segmentation`
  (sostén); los casos normativos son los `gap_cases` de `golden_features.json`,
  que se amplían con el sostén al regenerarse al cerrar el bloque.

> **Nota — desde la v2 solo afecta a `detected_handedness`**, que es diagnóstico:
> la etiqueta ya no llega al paso 2. Se conserva el texto porque la etiqueta sigue
> sirviendo para el aviso «¿cambiaste de mano?» y para el diagnóstico de tracking.
>
> Lo anterior dice qué
> imagen recibe el detector. Qué etiqueta **devuelve** es otra cosa, y en MediaPipe
> las dos no coinciden: determina la lateralidad *asumiendo que la imagen está
> espejada*, que es como se ve una persona en una cámara frontal. Alimentado sin
> espejar, como aquí se exige, reporta la mano contraria a la real y hay que
> invertir la etiqueta antes de que llegue al paso 2.
>
> Es responsabilidad del adaptador del detector, no de esta especificación, y por
> eso vive en `src/lsm/io/hands.py` tras el interruptor
> `hands.mediapipe_reports_mirrored_handedness`. Se anota aquí porque **la
> implementación de TypeScript de la Fase 7 se va a encontrar exactamente lo
> mismo** con MediaPipe JS, y porque equivocarse no produce ningún síntoma: el
> paso 2 canoniza todas las muestras hacia la mano equivocada, el vector queda
> coherente consigo mismo y el modelo entrena sin quejarse. Ver
> `docs/adr/0006-deteccion-de-manos-y-captura.md`.

### 0.4 Plausibilidad anatómica (v4, ADR 0027)

> **Versión.** Forma parte del bloque de tolerancia a MediaPipe (plausibilidad,
> One Euro del §4 y δ), `FEATURE_SPEC_VERSION` 4. Los golden la cubren en
> `plausibility_cases`, con los parámetros en cada caso.

Antes que nada —antes del relleno de huecos del §0.3 y del filtro del §4—, cada
frame válido se juzga contra los anteriores del mismo flujo. Los que no pasan se
convierten en **inválidos** con motivo `IMPLAUSIBLE` y se tratan como un frame
sin mano: no se corrigen.

Sea `q` el frame tras los pasos 1 y 2 **conservando z** (`z · a`), `t` su tiempo
en ms (`timestamp_ms`, o `índice · 1000 / fps`), y

```
P       = los segmentos de palma del §1, paso 4
palma3  = max_{(a,b) ∈ P} ‖q_a − q_b‖                      # 3D
L_k     = ‖q_hijo − q_padre‖ / palma3,  k = 1 … 20          # BONES, en su orden
c       = media de q_0, q_5, q_9, q_13, q_17, en 2D          # centro de la palma
palma2  = la palma del paso 4 (2D)
```

Estado: `R`, los `L` de los últimos `bone_reference_frames` frames **aceptados**;
`U = (c, palma2, t)` del último aceptado; `r0`, el tiempo del primer rechazo de la
racha en curso. Si `palma3 < 1e-6` el frame no se juzga ni entra en `R`.

```
motivo = ninguno
si |R| = bone_reference_frames:
    para k = 1 … 20:  si |L_k − mediana(R_k)| > bone_max_deviation → BONE
si motivo = ninguno y max_mcp_dorsal_deg ≠ null:
    si max(elevación dorsal MCP) > max_mcp_dorsal_deg → JOINT
si motivo = ninguno y U existe:
    v = ‖c − c_U‖ / ((palma2 + palma2_U) / 2) · 1000 / (t − t_U)
    si v > max_palm_speed_per_s → JUMP

si motivo ≠ ninguno:
    si r0 no existe: r0 = t
    si t − r0 < reset_after_ms: el frame es INVÁLIDO (IMPLAUSIBLE, motivo); fin
    vaciar R y U                     # la referencia ya no describe a esta mano
r0 = ninguno;  añadir L a R;  U = (c, palma2, t)
```

La mediana de una cantidad par es la media de los dos centrales. La elevación
dorsal es `landmark_stats.mcp_dorsal_elevation`; hoy está desactivada
(`null`). Los frames sin mano no reinician nada.

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

`handedness` es la **mano declarada** (§0.2), no la que diga el detector. Nunca
se infiere ni se vota: una votación sobre la etiqueta del detector fallaría justo
en las letras con la palma de lado y volcaría el trazo entero sin ningún síntoma.

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

Se usa el **tamaño de palma** como unidad (v3): la mayor de las distancias 2D entre
la muñeca y los cuatro nudillos, y entre el nudillo del índice y el del meñique.
No involucra ninguna punta, así que es estable frente a la flexión de los dedos, y
no se colapsa con la palma de canto: la palma es aproximadamente plana y muñeca →
nudillo 9 y nudillo 5 → nudillo 17 son casi perpendiculares, así que al girarla no
pueden encogerse las dos a la vez.

```
P = ( (0,5), (0,9), (0,13), (0,17), (5,17) )      # en este orden
m = max_{(a,b) ∈ P} sqrt((p_a.x − p_b.x)² + (p_a.y − p_b.y)²)     # 2D, ignora z

si m < 1e-6:
    frame INVÁLIDO — descartar

p_i ← p_i / m    para todo i

s = sqrt(p_9.x² + p_9.y²)       # antes de dividir: la escala de hasta la v2
r = s / m                       # qué tan de frente está la palma; lo usa el paso 5
```

Hasta la v2 la escala era `s`, muñeca → nudillo 9. Sigue calculándose porque el
paso 5 necesita `r`, y porque el metadato de distancia a la cámara
(`mean_scale_px`) se calibró con ella; ya no divide nada.

### Paso 5 — Rotación en el plano XY

Alinea el eje de la palma con el eje +Y. Da tolerancia a la inclinación de la
muñeca.

Desde la v3 el ángulo **se decide sobre la secuencia**, porque con la palma de
canto el de un frame suelto es ruido (40° de desviación con la mano quieta en la
X, ADR 0020). En dos pasadas:

```
# 1. Qué frames son fiables, con histéresis sobre r (paso 4):
fiable_{-1} = verdadero
fiable_t    = falso           si r_t <  off        # features.rotation_off_ratio = 0.45
              verdadero       si r_t ≥  on         # features.rotation_on_ratio  = 0.6
              fiable_{t−1}    si no

# 2. El ángulo de cada frame:
θ^A_t = atan2(p_9.y, p_9.x)                        # tras el paso 4 del frame t
θ_t   = θ^A_t                          si fiable_t
        θ^A del último fiable anterior si no, y lo hay
        θ^A del primer fiable posterior si no hay ninguno antes

si ningún frame es fiable:
    secuencia RECHAZADA — PALM_EDGE_ON

φ_t = π/2 - θ_t

para todo i:
    x' = p_i.x * cos(φ) - p_i.y * sin(φ)
    y' = p_i.x * sin(φ) + p_i.y * cos(φ)
    p_i.x ← x'
    p_i.y ← y'
    # p_i.z no se modifica
```

Tras este paso `p_0 = (0, 0)`, y en los frames fiables `p_9` queda sobre el eje
+Y (en `(0, s/m)`; hasta la v2, con `m = s`, en `(0, 1)`).

- **Una secuencia de un solo frame** —los casos por frame de los golden— es fiable
  si `r ≥ off` (empieza fiable y solo `r < off` lo apaga).
- **No se mezclan referencias**: el ángulo de un frame de canto no se reemplaza
  por el de otra construcción (la línea de nudillos), cuyo desfase con `θ^A`
  depende de la forma de la mano —entre 16° y 154°, ADR 0020—.
- **La velocidad del §6.1 no pasa por aquí**: una mano de canto se sigue
  moviendo aunque su secuencia no se pueda clasificar.

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

**Sobre los componentes constantes.** Por construcción `p_0 = (0,0)` en todas las
muestras, y `p_9.x = 0` en los frames fiables del paso 5. Hasta la v2 también
`p_9 = (0,1)`; desde la v3 `p_9.y = s/m`, que en las estáticas va de 0.80 a 1.00 y
dice cuánto se ve de frente la palma. Se conservan deliberadamente: las
constantes aportan distancia cero en cualquier métrica, y mantener los 21 índices
alineados con la numeración de MediaPipe elimina una fuente crónica de errores
off-by-one al depurar y al reimplementar en TypeScript.

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
δ_t = ln( m_t / m_0 )        ∈ ℝ          (v4)
```

- Origen en la muñeca del primer frame → invariante a la posición en el encuadre.
- Escala en unidades de mano → invariante a la distancia a la cámara.
- **δ (v4, ADR 0034)** es la profundidad relativa: `m_t` es el tamaño de palma
  del paso 4 (la mayor distancia 2D entre los segmentos de la palma), así que
  δ > 0 cuando la mano se acerca a la cámara y δ < 0 cuando se aleja. τ vive en
  el plano de la imagen y no ve ese eje: la X va hacia la cámara y vuelve, y sin
  δ su trayectoria es casi un punto. Con el tamaño de palma y no con la escala
  muñeca → nudillo 9, un giro de la palma no se lee como acercamiento (ADR 0020,
  tabla de δ). `δ_0 = 0` exactamente; `math.log` en punto flotante doble.

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
remuestrea `f_t`, luego `τ_t`, luego δ_t (como filas de una componente), y solo
entonces se concatenan.

### 3.3 Vector por frame para el clasificador dinámico

```
g_t = concat( f_t , w_τ · τ_t , w_δ · δ_t )        ∈ ℝ⁴⁵        (v4)
```

`w_τ = config.features.trajectory_weight`, por defecto `1.0`.
`w_δ = config.features.depth_weight`, por defecto `1.0` (v4). `w_δ = 0` apaga
δ sin cambiar el ancho de `g_t`: es como se mide con y sin δ (ADR 0034). El
modelo dinámico exporta los dos pesos.

> **Calibrado en la Fase 5** (ADR 0016): el punto de partida era `4.0`; el barrido
> con leave-one-signer-out dio `1.0`. La fórmula no cambia —por eso
> `FEATURE_SPEC_VERSION` tampoco—, pero los `dynamic_rows` de los golden vectors
> se regeneraron con el valor nuevo, que el archivo declara en su bloque
> `config`. El modelo dinámico exporta el peso con que se entrenó, y quien lo
> ejecute tiene que construir `g_t` con ese.

Justificación del peso: el canal de forma aporta 42 componentes y el de trayectoria
solo 2. Sin ponderación, la distancia euclidiana queda dominada por la forma y el
trazo —que es precisamente lo que distingue estas letras— resulta invisible. El valor
por defecto es un punto de partida; debe ajustarse empíricamente y el ajuste
registrarse.

### 3.4 Distancia

DTW sobre las secuencias `(24, 45)` con distancia euclidiana local, ventana de
Sakoe-Chiba de radio `config.dtw.band_radius` (por defecto `6`), y costo normalizado
por la longitud del camino de alineación.

> **Precisión, no cambio** (Fase 5, `docs/adr/0016-clasificador-dinamico-y-barrido.md`).
> El párrafo de arriba no fijaba lo bastante para que dos implementaciones den los
> mismos bits. Lo siguiente es la definición normativa; `FEATURE_SPEC_VERSION` no
> cambia porque ningún modelo exportado usaba todavía esta sección.

Sean `a` y `b` dos secuencias de `T_ref` filas de `g_t`.

```
c(i, j) = sqrt( Σ_k (a_i[k] − b_j[k])² )        # k ascendente, raíz al final

celdas admitidas: |i − j| ≤ band_radius

D[0][0] = c(0, 0);   L[0][0] = 1
D[i][j] = c(i, j) + D[p];   L[i][j] = L[p] + 1
    donde p es el predecesor de menor D entre, EN ESTE ORDEN,
    (i−1, j−1), (i−1, j), (i, j−1); solo gana uno posterior si su D es
    ESTRICTAMENTE menor. Fuera de la banda, D = +∞.

distancia = D[T_ref−1][T_ref−1] / L[T_ref−1][T_ref−1]
```

- **El desempate es parte del contrato.** Decide qué camino se cuenta y por tanto
  `L`: con dos predecesores empatados en `D`, elegir otro cambia el divisor. Con
  este orden el DTW no es exactamente simétrico, y ninguna implementación debe
  suponer `d(a, b) = d(b, a)`.
- **La suma de `c` no usa `sum()` de Python**: desde 3.12 compensa con Neumaier y
  TypeScript no lo hará. Bucle explícito en las dos.
- Las dos secuencias miden `T_ref`: el §3.2 lo garantiza, y una banda inclinada
  para longitudes distintas no hace falta en ninguna de las dos implementaciones.

**Decisión.** Una plantilla por (letra, persona): el medoide DTW del grupo, la
muestra con menor suma de distancias al resto (empates al índice menor, con el
grupo ordenado por sesión y marca de tiempo). `d_letra` es el mínimo sobre sus
plantillas; la confianza es `d₂ / (d₁ + d₂)` entre las dos letras más cercanas
—la misma escala que `static_knn`—, y **UNKNOWN si `d₁ > config.dtw.max_distance`**.

---

## 4. Suavizado temporal: filtro One Euro (v4, ADR 0028)

> **Versión.** Sustituye a la media exponencial de α fijo de la v3 (que estaba
> desactivada, `α = 1`). Es parte de `FEATURE_SPEC_VERSION` 4 junto con el §0.4
> y δ; los golden lo cubren en `one_euro_cases`, con sus parámetros en cada caso.
> En el código está apagado por defecto; `config.yaml` lo activa con el
> **candidato (0.5, 1, 2)** (ADR 0028): ningún punto del barrido cumplió a la vez
> quitar el temblor y no retrasar los trazos, y este es el que mejora los
> intentos en vivo. El modelo exportado declara los valores con que se entrenó.

Filtro de Casiez, Roussel y Vogel (2012) sobre **cada coordenada cruda** (x, y, z
de los 21 landmarks: 63 señales con estado propio), **después** de la
plausibilidad (§0.4) y del relleno de huecos (§0.3) y **antes del paso 1**. Para
la muestra `k` de una señal, con `Δ_k = (t_k − t_{k−1}) / 1000` segundos de tiempo
real (`timestamp_ms`, o `índice · 1000 / fps`; un `Δ ≤ 0` es un error):

```
α(f, Δ) = r / (r + 1),   r = 2π · f · Δ

k = 0:  x̂_0 = x_0 ;  d̂_0 = 0
k ≥ 1:  d_k = (x_k − x̂_{k−1}) / Δ_k
        d̂_k = α(d_cutoff, Δ_k) · d_k + (1 − α(d_cutoff, Δ_k)) · d̂_{k−1}
        f_k = min_cutoff + β · |d̂_k| · e / m
        x̂_k = α(f_k, Δ_k) · x_k + (1 − α(f_k, Δ_k)) · x̂_{k−1}
```

- **`e / m`, la única diferencia con el original**: la velocidad que abre el filtro
  va en palmas por segundo. `e` es el factor del paso 1 para el eje (`a =
  width/height` para x y z, 1 para y) y `m` el tamaño de palma del paso 4 del
  frame `k`, calculado sobre el frame **crudo**; si es `< 1e-6`, el del último
  frame que no lo era, y si no hay ninguno, la velocidad vale 0. Así `β` significa
  lo mismo cerca o lejos de la cámara y en cualquier resolución.
- El orden de las operaciones es el de arriba, coordenada por coordenada en el
  orden `x_0, y_0, z_0, x_1, …`.
- **Estado**: se reinicia cuando la secuencia se corta —un hueco que no se
  rellena—. En vivo el filtro llega al trazo con la historia de la mano; en una
  muestra guardada arranca en su primer frame (`lsm.preprocessing`).
- Parámetros: `smoothing.min_cutoff`, `smoothing.beta`, `smoothing.d_cutoff`. Los
  modelos exportados los llevan en `params.preprocessing.one_euro` y la
  implementación web tiene que usar esos.

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
| `scales` | `s_t` del paso 4 por frame, en las unidades corregidas del paso 1. |
| `velocities` | `v_t` del §6, longitud `T − 1`. Es lo que `segmentation.ts` debe reproducir. |
| `resampled_trajectory` | `τ` tras el remuestreo del §3.2, sin ponderar. |
| `dynamic_rows` | `g_t` final: 24 filas de 45 componentes. |
| `dynamic_unavailable_reason` | `TOO_FEW_SOURCE_FRAMES` si la secuencia se rechazó para el canal dinámico; en ese caso los dos campos anteriores van en `null`. |

La cobertura incluye **la misma interrupción al inicio, en medio y al final**,
porque los tres casos se comportan distinto y no basta con probar uno:

- al inicio, el origen de `τ` se mueve al primer frame válido y el trazo restante
  se mide desde otro punto;
- al final, la secuencia simplemente se corta antes y `τ` conserva su origen;
- en medio quedan **dos** secuencias, no una con un salto.

---

## 6. Velocidad y estabilidad — contrato de segmentación

**`SEGMENTATION_SPEC_VERSION = 7`** (`src/lsm/segmentation.py`).

> **v7** — `docs/adr/0021-tolerancia-a-huecos-en-el-camino-dinamico.md`. Dentro de
> DYNAMIC_CANDIDATE, un hueco corto entre dos frames de la misma mano se rellena
> interpolando los landmarks crudos (§0.3) en vez de cortar el trazo. Mientras un
> hueco está abierto dentro de un trazo la máquina retiene los frames inválidos,
> y va hasta `dynamic_max_gap_ms` por detrás. Un trazo con más de
> `dynamic_max_interpolated_fraction` de frames rellenados se descarta sin
> clasificar (`DYNAMIC_TOO_MUCH_INTERPOLATED`).

> **v6** — `docs/adr/0019-cierre-del-trazo-y-cascada.md`. La velocidad del §6.1
> se divide entre el **tamaño de palma** `m` y no entre la escala del paso 4, que
> se colapsa con la palma de canto (en la X queda en 0.17 del tamaño real). El
> cierre de DYNAMIC_CANDIDATE mide el reposo con su propia ventana (§6.1.2,
> `closing_window_ms`), y tras un `DYNAMIC_TOO_LONG` puede nacer otra racha a los
> `motion_exhausted_ms` (§6.7). La escala del paso 4 y las features no cambian.

> **v5** — `docs/adr/0017-diagnostico-de-tracking.md`. La velocidad con la que
> **decide la máquina** deja de ser la del último par de cuadros y pasa a medirse
> contra el cuadro de hace `velocity_window_ms` (§6.1.1). El temblor de MediaPipe
> es por cuadro: por pares y por segundo crecía con la tasa, y a 28 fps la mano
> quieta superaba el umbral de reposo. `velocity_threshold_per_s` pasa a 0.55,
> fijado con la prueba de reposo y no por conversión. `motion_threshold_per_s`
> (0.75 → 0.60) y `motion_confirm_low_ms` (400 → 667) cambian de forma
> **provisional**, hasta medirlos en vivo. `v_t` del §6.1 y los `velocities` de
> los golden vectors no cambian.

> **v4** — `docs/adr/0017-diagnostico-de-tracking.md`. Los dos umbrales de
> velocidad pasan a unidades de mano **por segundo** (`velocity_threshold_per_s`,
> `motion_threshold_per_s`) y se convierten a por cuadro con la tasa congelada
> de la sesión (§6.5): `umbral_cuadro = umbral_por_s / fps`. `v_t` (§6.1) no
> cambia: sigue midiéndose por par de cuadros, sin tiempo. Los valores v3 se
> convirtieron con la tasa a la que se midieron, 30 fps: 0.02 → 0.6 y
> 0.025 → 0.75.

> **v2** — `docs/adr/0013-la-ventana-mezclada.md`. Se añaden §6.4 (qué ventana se
> clasifica), §6.5 (los umbrales temporales en milisegundos) y §6.6 (emisión
> progresiva). `FEATURE_SPEC_VERSION` no cambia: el promedio del §2 es el mismo y
> ningún modelo entrenado se invalida — lo que cambia es **qué frames entran**.
>
> **v3** — `docs/adr/0015-el-camino-dinamico-de-la-segmentacion.md`. Se añade la
> §6.7: el camino dinámico, que resuelve lo que la §6.3 dejó escrito sin
> resolver. `FEATURE_SPEC_VERSION` tampoco cambia.

Esta sección **no está bajo `FEATURE_SPEC_VERSION`** y se versiona aparte. Las dos
cosas cambian por motivos distintos: el vector de features cambia cuando cambia lo
que consume el clasificador, y entonces hay que reentrenar y rechazar los modelos
viejos; la segmentación cambia cuando se ajusta cómo se decide que una mano está
quieta, y eso no invalida ningún modelo. Acoplarlas obligaría a reentrenar cada vez
que se afina un umbral, lo cual es absurdo.

`segmentation.ts` deberá reproducir esta sección igual que `features.ts` reproduce
las §1 a §5. Los valores esperados viajan en los `sequence_cases` de
`golden_features.json`, en los campos `scales` y `velocities`.

### 6.1 Velocidad entre frames

Entrada: dos frames válidos y consecutivos de la misma secuencia. Sea `q_t` el
conjunto de puntos del frame `t` **tras el paso 2** —relación de aspecto corregida,
`y` invertida y lateralidad canonizada— y `m_t` su **tamaño de palma** (v6):

```
P     = ( (0,5), (0,9), (0,13), (0,17), (5,17) )     # en este orden
m_t   = max_{(a,b) ∈ P} ‖ q_{t,a} - q_{t,b} ‖₂        # en 2D
d_t   = ( Σ_i ‖ q_{t,i} - q_{t-1,i} ‖₂ ) / 21      # i ascendente, en 2D: z se ignora
m_par = ( m_{t-1} + m_t ) / 2
v_t   = d_t / m_par
```

- **Por qué la palma y no la escala del paso 4** (v6, ADR 0019). La escala del
  paso 4 es un solo segmento, muñeca → nudillo 9, y cuando apunta a la cámara su
  proyección se encoge: en la X quieta quedó en 0.17 del tamaño de la mano y la
  velocidad salió 30 veces mayor que en una estática. La palma es
  aproximadamente plana, y dentro de ella muñeca → nudillo 9 y nudillo 5 →
  nudillo 17 son casi perpendiculares: al girarla no pueden encogerse las dos a
  la vez. En las estáticas `s/m` está entre 0.80 y 1.00, así que su velocidad
  apenas cambia. Como `P` incluye muñeca → nudillo 9, `m_t ≥ s_t`.

- `‖·‖₂` es la norma euclidiana en el plano XY. `z` se ignora por la misma razón
  que en el paso 4: es ruido.
- La suma recorre los landmarks en orden de índice ascendente y divide al final,
  igual que el §5.4 exige para el promedio del §2.
- Una ventana de `T` frames produce `T - 1` velocidades. La máquina de estados usa
  la del último par **para decidir si la mano se está moviendo ahora**. Cuál es la
  ventana que se **clasifica** es otra pregunta, y la contesta la §6.4.
- `q_t` **no está trasladado ni escalado**: es el paso 2, no el 5.

### 6.1.1 La velocidad con la que decide la máquina (v5)

La máquina de estados no compara `v_t` del último par, sino la velocidad contra
un cuadro más antiguo. Sea `B` el buffer de frames válidos (§6.4), `t` su último
frame y

```
k   = frames_from_ms(velocity_window_ms, fps)      # §6.5; 3 a 30 fps con 100 ms
k'  = min(k, |B| - 1)                              # si el buffer no llega tan atrás
w_t = v(B[t - k'], B[t]) / k'                      # v = la fórmula del §6.1
```

donde `v(a, b)` es la fórmula del §6.1 aplicada a los frames `a` y `b` en vez de a
dos consecutivos. `w_t` está en unidades de mano **por cuadro** y se compara con
los umbrales ya convertidos (`umbral_por_s / fps`, §6.5). Con menos de dos frames
en el buffer no hay `w_t`.

- **Por qué.** El temblor de los landmarks es aproximadamente constante por
  cuadro y no se acumula: entre dos cuadros separados por `k` pasos pesa lo mismo
  que entre dos seguidos, mientras que el movimiento real sí se acumula. Por
  pares y por segundo el ruido crecía con la tasa; contra una ventana fija en
  milisegundos pesa lo mismo a cualquier tasa. Medido con la mano quieta a
  28 fps: p50 0.70 u/s por pares, 0.18 u/s con la ventana (ADR 0017).
- **El tiempo es el de la tasa congelada**, `k'/fps`, no el reloj de cada cuadro:
  los frames no llevan marca de tiempo, y es la misma aproximación que convierte
  a cuadros todos los umbrales en milisegundos. Con el descarte de repetidos
  (ADR 0017) los cuadros son únicos y su intervalo medio es `1/fps`.
- **Arrastre.** `w_t` mira `k'` cuadros atrás: tras una parada en seco la mano
  sigue «moviéndose» hasta `k - 1` cuadros más, y un movimiento que arranca de
  golpe tarda en superar el umbral. Los umbrales en milisegundos del §6.7 se
  cuentan sobre esa señal.
- Un hueco no rompe el buffer (§6.4): `B[t - k']` puede quedar al otro lado de
  un frame inválido, igual que el par del §6.1.
- `velocity_window_ms ≤ buffer_ms`, validado: el buffer tiene que guardar el
  cuadro contra el que se mide.

### 6.1.2 La velocidad con la que se cierra un trazo (v6)

Dentro de DYNAMIC_CANDIDATE, el reposo que cuenta para `motion_confirm_low_ms`
—el que cierra el trazo— se mide con la misma fórmula del §6.1.1 pero con su
propia ventana, `k_c = frames_from_ms(closing_window_ms, fps)`:

```
c_t = v(B[t - min(k_c, |B| - 1)], B[t]) / min(k_c, |B| - 1)
```

y un frame del candidato cuenta como movimiento si `c_t ≥ motion_threshold`. Todo
lo demás —arrancar una racha, pasar a candidato, `moving`, `still_run` y STABLE—
sigue con `w_t` del §6.1.1. Si `closing_window_ms` y `velocity_window_ms` dan los
mismos cuadros, `c_t = w_t`.

**Por qué una ventana aparte.** Con 100 ms la X quieta no pasa de 618 ms de
reposo seguido, y el trazo necesita 667: nunca cierra. Con 150 ms, 1501 ms
(ADR 0019). El precio de una ventana más larga es arrastre: tras una parada en
seco el cierre sigue viendo movimiento hasta `k_c - 1` cuadros.

### 6.2 Por qué la escala del par y no otra

(Hasta la v5 el divisor era la escala del paso 4, `s`; desde la v6 es el tamaño de
palma `m`, §6.1. Lo que sigue vale igual para `m`.)

Las tres opciones —`s_t`, `s_{t-1}` o la `s̄` de la ventana— dan números distintos
en cuanto la mano se acerca o se aleja de la cámara, así que la paridad depende de
fijar una. **Se usa la media del par, `(s_{t-1} + s_t) / 2`.**

Contra la `s̄` de la ventana: con ella, el mismo par de frames produce velocidades
distintas según qué otros frames haya en el buffer en ese instante. Una
implementación incremental —guardar el frame anterior y calcular al llegar el
siguiente, que es la forma natural de escribirlo en TypeScript sobre un stream— no
coincidiría con una que recorre la ventana entera, y la discrepancia dependería del
estado del buffer, que es la peor clase de discrepancia para depurar. Con la media
del par, el valor de un par depende solo de ese par.

Contra `s_t` o `s_{t-1}` a secas: funcionan y son locales al par, pero son
asimétricas. La media no depende de en qué dirección se recorra el tiempo, que es
una propiedad barata y conviene tener.

Dividir por una escala —cualquiera de las tres— es lo que hace la velocidad
invariante a la distancia: la misma seña ejecutada más cerca de la cámara recorre
más píxeles por frame, pero no es más rápida.

### 6.3 Consecuencia: esto no sirve para enrutar estáticas contra dinámicas

`q_t` no está trasladado, así que `v_t` mide **dos cosas a la vez**: cuánto se
desplazó la mano por el encuadre y cuánto cambió la configuración de los dedos.

Para las letras estáticas es exactamente lo que se quiere: la ventana solo es
estable si la mano ni viajó ni siguió acomodando los dedos.

**Para las dinámicas es una trampa.** En una J, una Ñ o una Z el movimiento *es* la
seña, así que `v_t` se mantiene alta mientras se ejecuta y el criterio de
estabilidad no se cumple nunca. Una máquina de estados que solo espere a
`v_t < umbral` jamás emitirá una letra dinámica: se quedará en TRACKING hasta que
la persona termine el trazo y se detenga, y para entonces la ventana ya contiene el
final del movimiento y no el trazo completo.

De ahí la consecuencia que importa registrar ahora: **el enrutamiento entre el
clasificador estático y el dinámico de la Fase 5 no puede colgar de este mismo
umbral.** Necesitará su propio criterio —energía de movimiento sostenida con
patrón, no ruido, como apunta `ARQUITECTURA.md` §4.2— y muy probablemente un camino
distinto por la máquina de estados: capturar la ventana completa mientras hay
movimiento coherente, en vez de esperar a que se detenga.

No se resuelve aquí. Se deja escrito para no descubrirlo en la Fase 5 con el
dataset ya grabado. Ver `docs/adr/0004-contrato-de-segmentacion.md`.

> **Resuelto en la v3** (§6.7). La conclusión de arriba se sostiene a medias: el
> enrutamiento no cuelga de `velocity_threshold`, pero tampoco hizo falta una
> métrica nueva. Se usa la misma `v_t` contra otro umbral, `motion_threshold`, y
> lo que distingue un trazo de un tránsito es cuánto dura, no qué se mide.

### 6.4 Qué ventana se clasifica

Sea `stable_run` el número de **pares consecutivos** cuya velocidad quedó por
debajo de `velocity_threshold`, contando desde el último par hacia atrás, y sea
`T_max = buffer_size` (§6.5). La ventana que se le pasa al clasificador son los

```
longitud = min( max(stable_run, stable_frames), T_max )
```

últimos frames del buffer, y **no el buffer entero**.

**La invariante deja de comprobarse y pasa a cumplirse por construcción.** El ADR
0004 dice que «la ventana solo es estable si la mano ni viajó ni siguió
acomodándose». Si la ventana *es* el tramo estable, todos sus frames lo son por
definición: no hay ningún bucle que los revise, y el caso en que fallaría no
existe. La v1 comprobaba quietud sobre los últimos `stable_run` frames y
clasificaba los `buffer_size` del buffer circular — hasta 18 frames sin comprobar
que, con un tránsito más corto que el buffer, eran la mano viajando de una letra
a la siguiente. Medido: un tránsito de 4 frames entre dos letras producía una
letra que nadie firmó.

Dos consecuencias que una reimplementación tiene que reproducir exactamente:

1. **Se toman `stable_run` frames, no `stable_run + 1`.** `stable_run` pares
   involucran un frame más, pero el más viejo de ellos es aquel al que la mano
   *llegó*, y su propia entrada pudo ser rápida. Dejarlo fuera es la lectura
   conservadora.
2. **Por eso el tope alcanzable es `T_max − 1` y no `T_max`.** Con un buffer lleno
   de N frames hay N−1 pares, así que el frame más viejo del buffer nunca entra en
   la ventana clasificada. No es un error de una unidad: es aritmética de la
   definición.

σ (§2) se calcula **sobre esta ventana**, no sobre el buffer.

### 6.5 Los umbrales temporales están en milisegundos

`buffer_ms`, `stable_ms`, `emit_cooldown_ms`, `reject_cooldown_ms`,
`missing_to_idle_ms`, `motion_min_ms`, `motion_confirm_low_ms`, `motion_max_ms` y
`spelling.space_after_absent_ms` son **duraciones**. Se convierten a cuadros con
la tasa de la sesión:

```
cuadros(ms, fps) = max( 1, floor( ms · fps / 1000 + 0.5 ) )
```

- **La regla de redondeo es normativa.** `Math.round` de JavaScript redondea hacia
  arriba en el empate y `round` de Python redondea al par más cercano: con 0.5
  exacto darían cuadros distintos leyendo el mismo `config.yaml`. Se adopta
  `floor(x + 0.5)`, que es la de JavaScript.
- **El piso de 1 cuadro** también es normativo: un cooldown de cero cuadros no es
  un cooldown y una ventana de cero frames no se puede clasificar.
- **La tasa se congela antes del primer frame y no se re-deriva durante la
  sesión.** Si cambiara a mitad de deletreo, la misma seña se comportaría distinto
  según lo que la máquina llevara haciendo un segundo antes, y una grabación no
  se podría reproducir. Dada `(ms, fps)`, la máquina vuelve a ser determinista bit
  a bit, que es lo que permite validarla contra golden vectors.
- Sin tasa medida —reproducir un dataset, un test— se usa la nominal,
  `capture.camera_fps`.

**Por qué**: en cuadros, el comportamiento cambiaba con la máquina sin que nadie
lo notara. Los comentarios de `config.yaml` traducían los umbrales a milisegundos
suponiendo 30 fps; la medición de `lsm-demo --medir-fps` en la máquina de
referencia dio **17.8 fps sostenidos**, de modo que cada umbral duraba 1.7 veces
lo que su comentario afirmaba: los 24 frames de buffer eran 1348 ms y no 800.

**Resuelto en la v4:** los umbrales de velocidad también se expresan por
segundo y se convierten con la tasa (`umbral / fps`), así que dejaron de
depender de ella. Lo que sigue es la excepción tal como estaba anotada hasta la
v3: `velocity_threshold` —y `motion_threshold`, que comparte métrica y
unidad— seguía en unidades de mano **por
frame** y por tanto dependía de la tasa — a menor tasa, dos frames
consecutivos están más separados en el tiempo y la misma mano física da un `v_t`
mayor. Expresarla por segundo es el arreglo pendiente; no se hizo con lo demás
porque es un eje del barrido de calibración de la Fase 2 y cambiarle la unidad
invalida esa calibración.

### 6.6 Emisión progresiva

A partir de `stable_frames`, la ventana se clasifica **en cada frame** mientras
crece, y la decisión usa dos umbrales:

| Confianza | Qué pasa |
|---|---|
| `≥ high_confidence` | emite ya |
| `[min_confidence, high_confidence)` | **acumula**: no emite, no cuesta cooldown, y se reclasifica en el frame siguiente con un frame más de evidencia. Al llegar a `T_max − 1` —ya no puede crecer— emite con lo que tenga |
| `< min_confidence` | rechaza, con `reject_cooldown_ms` de silencio |

Acumular **no es rechazar** y por eso no paga cooldown: ese es todo el mecanismo
de la latencia adaptativa. Una letra sin vecinos cercanos sale con la ventana
mínima; un par confundible acumula hasta despegarse o hasta agotar la ventana.
Medido bajo leave-one-signer-out sobre el dataset de la Fase 2, el 75.1% de las
muestras cruzan `high_confidence = 0.82` y saldrían de inmediato.

El cerrojo de letra repetida se evalúa **antes** de acumular: acumular evidencia
de una letra que el cerrojo no va a dejar salir gastaría la ventana entera para
terminar en el mismo rechazo.

### 6.7 El camino dinámico (v3)

Una letra dinámica nunca cumple la condición de STABLE: el movimiento *es* la seña
(§6.3). Tiene su propio camino por la máquina:

```
TRACKING ──(racha con ≥ motion_min frames móviles)──> DYNAMIC_CANDIDATE
DYNAMIC_CANDIDATE ──(low_run ≥ motion_confirm_low)──> DYNAMIC_EMIT
DYNAMIC_CANDIDATE ──(trazo > motion_max frames)─────> TRACKING   (descarta)
DYNAMIC_CANDIDATE ──(hueco o escala degenerada)─────> TRACKING   (descarta)
DYNAMIC_EMIT ──(clasificador dinámico acepta)───────> EMIT
DYNAMIC_EMIT ──(UNKNOWN o < min_confidence)─────────> TRACKING
```

**Movimiento** es la `v_t` del §6.1 —la del último par, la misma que decide
`moving`— contra `motion_threshold`. No hay una segunda métrica. Se exige
`motion_threshold ≥ velocity_threshold`: ningún frame cuenta a la vez como
quietud para STABLE y como movimiento para el candidato.

Contadores, evaluados en **todos los estados salvo IDLE**, también durante el
cooldown de EMIT (un trazo que empieza mientras la letra anterior se enfría sigue
siendo el mismo trazo):

| contador | qué cuenta | se reinicia |
|---|---|---|
| `low_run` | frames consecutivos con `v_t < motion_threshold` | con un frame en movimiento, o un hueco |
| `still_run` | frames consecutivos con `v_t < velocity_threshold` | con un frame que supera `velocity_threshold`, o un hueco |
| racha | frames móviles desde que empezó; no consecutivos | ver abajo |

- **Nace una racha** con un frame en movimiento si no hay una en curso. Su trazo
  empieza en el **frame anterior** a ese primer par en movimiento: es el punto de
  partida del recorrido, y el origen de τ (§3.1).
- **Muere una racha que no llegó a candidato** cuando `low_run ≥
  motion_confirm_low`, o cuando `still_run ≥ stable_frames`. El segundo corte es
  el que impide que los tránsitos cortos de un deletreo rápido se sumen hasta
  parecer un trazo largo: si la mano se detuvo lo que el camino estático llama
  quietud, lo anterior era un tránsito.
- **Los frames móviles no son consecutivos** a propósito. La cámara entrega
  cuadros duplicados y en pleno trazo `v_t` alterna alto y casi cero (medido en
  las grabaciones: ADR 0015). Exigir consecutivos no dispararía nunca.
- **Entrada**: en TRACKING —tras salir de STABLE si hacía falta—, en un frame en
  movimiento, con `moving ≥ motion_min_frames`.
- **Mientras dura DYNAMIC_CANDIDATE, STABLE está suspendido.** `stable_run` se
  sigue contando para que, si el trazo se rechaza, el camino estático retome en
  el frame siguiente sin volver a esperar `stable_frames`.
- **DYNAMIC_EMIT entrega el trazo crudo** desde su frame de partida hasta el
  **último frame en movimiento**, sin el reposo que lo cerró y sin remuestrear.
  El remuestreo (§3.2) lo hace el clasificador. Viaja con `WindowOrigin.DYNAMIC`.
- **Largo máximo**: se compara el número de frames del trazo, contados desde su
  frame de partida, con `motion_max_frames`, con `>`. Tras descartar por largo no
  nace otra racha hasta que ocurra lo primero de: `low_run ≥ motion_confirm_low`,
  un frame inválido, o `motion_exhausted_frames` cuadros desde el rechazo (v6).
  La tercera vía existe porque la primera es la condición que acaba de fallar: en
  la X y la Q, que no llegan a reposar, bloqueaba los intentos siguientes
  (ADR 0019).
- **Tras emitir una dinámica**, además del `pending_repeat` de siempre, un
  cerrojo impide que TRACKING promueva a STABLE hasta que un frame supere
  `velocity_threshold` o se pierda la mano: la pose en que termina una dinámica
  no es una letra nueva (la J acaba en la mano de la I).
- **Tras rechazar un trazo por margen** —el clasificador dinámico dio la letra D,
  con confianza bajo `min_confidence`— la estática
  `rejected_stroke_final_poses[D]`, su pose final (J → I, K → P, Ñ → N; ADR
  0022), no se emite hasta que un frame supere `velocity_threshold` o se pierda
  la mano: esa ventana se rechaza con `FINAL_POSE_OF_REJECTED_STROKE` (ADR
  0033). Es preferible no escribir nada a escribir I. Cualquier otra letra sale
  como antes, en el frame siguiente: la de llegada de un tránsito largo tiene
  que salir. Un trazo UNKNOWN o descartado sin clasificar no pone el cerrojo.
  Entra con el bloque de tolerancia en la versión siguiente de segmentación.
- La emisión dinámica **no es progresiva**: se clasifica una vez, y se emite si
  la confianza alcanza `min_confidence`.

`motion_threshold` está en unidades de mano **por frame**, con la misma deuda que
`velocity_threshold` (§6.5). Las tres duraciones van en milisegundos.

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
