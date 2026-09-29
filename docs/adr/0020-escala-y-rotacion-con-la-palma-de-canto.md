# ADR 0020 — Escala y rotación de las features con la palma de canto

- **Estado:** pasos 4 y 5 **implementados** el 2026-09-28 como
  `FEATURE_SPEC_VERSION` 3, con las decisiones de «Lo que se implementó». ρ y δ
  del Bloque 3 quedan para la versión 4.
- **Fecha:** 2026-09-28
- **Datos:** la prueba de reposo ampliada `data/diagnostico/2026-09-28-193646-habitual/`
  (estática, K, X y Q, 10 s cada una) y las 3407 muestras de `data/raw`
  (s01–s03). Fuera del repositorio.
- **Relacionadas:** ADR 0002 (formato de features), ADR 0019 (la escala de la
  velocidad), Fase 5.1 Bloque 3 (canales ρ y δ, sin implementar)

## Contexto

El ADR 0019 encontró que en la X la distancia muñeca → nudillo 9 se colapsa: el
segmento apunta a la cámara y su proyección queda en 0.17 del tamaño real de la
palma. La velocidad ya no depende de ella (SEGMENTATION_SPEC_VERSION 6), pero la
misma distancia es la base de las features:

- **paso 4**: la escala `s` que normaliza el tamaño de la mano;
- **paso 5**: el ángulo θ de ese segmento, al que se rota la mano;
- y, en el Bloque 3, los canales **δ_t = ln(s_t / s_0)** y
  **ρ_t = unwrap(θ_t) − unwrap(θ_0)**.

Con la escala colapsada el vector de forma se infla (en el golden
`velocity_foreshortened_palm` las coordenadas llegan a ±9 en vez de ±1), θ es
ruido, y δ y ρ heredan las dos cosas.

## Lo que se midió

### 1. Escala / palma por letra

`s` / tamaño de palma (`features.palm_size`: el mayor de muñeca → nudillos 5, 9,
13, 17 y nudillo 5 → 17), en todos los cuadros del dataset:

| grupo | letras | s/palma p10 / p50 | cuadros < 0.5 |
|---|---|---|---|
| estáticas, las más bajas | M, N | 0.77–0.79 / 0.80–0.81 | 0 |
| estáticas, el resto | 19 letras y NONE | 0.89–0.99 / 0.91–1.00 | 0 |
| dinámicas sin colapso | J, K, Z, LL, RR | 0.85–0.93 / 0.92–0.97 | 0–0.2% |
| **dinámicas con colapso** | **X, Ñ, Q** | 0.52–0.64 / 0.73–0.85 | **2.2–8.9%** |
| prueba de reposo: X | — | 0.07 / 0.17 | 100% |
| prueba de reposo: Q | — | 0.63 / 0.66 | 0% |

**Ninguna estática se colapsa.** La X de la prueba de reposo (0.17) no es la X del
dataset (0.73): quien firma la ejecuta hoy con la palma más de canto que cuando
se grabó. Eso se resuelve al regrabar, pero dice que el colapso no es un caso
raro: depende de cómo se haga la letra.

### 2. Ruido del ángulo del paso 5 con la mano quieta

Desviación estándar circular de θ, en grados, con cuatro referencias:

- **A** — muñeca → nudillo 9 (el paso 5 de hoy);
- **B** — la línea de nudillos 5 → 17 girada 90°;
- **C** — muñeca → centroide de los nudillos 5, 9, 13 y 17;
- **D** — suma de A y B sin normalizar (domina la proyección más larga).

| | A | B | C | D |
|---|---|---|---|---|
| reposo: estática | 0.8 | 0.7 | 0.7 | 3.0 |
| reposo: K | 0.5 | 0.8 | 0.5 | 0.8 |
| **reposo: X** | **40.1** | **2.8** | 44.0 | 6.6 |
| reposo: Q | 2.3 | 4.6 | 2.2 | 4.0 |
| estáticas del dataset, mediana por muestra (rango entre letras) | **0.40–1.05** | 0.62–3.42 | 0.46–1.10 | 0.54–2.08 |

- **En las estáticas A es la mejor** o empata: B se degrada con los dedos
  curvados (O 3.42°, C 2.01°, F 2.08°) y D suma el ruido de las dos.
- **En la X A es ruido puro** (40°); B es la única estable (2.8°).
- **C se degenera igual que A** (44° en la X): el centroide de los nudillos está
  en la misma dirección que el nudillo 9.

### 3. Por qué no se pueden mezclar A y B

El desfase entre B y A, medido en los cuadros donde las dos son fiables
(s/palma ≥ 0.65), **no es constante**: depende de la forma de la mano, entre
16° y 154° según la letra (Y, G y H de lado; A, E, I de frente). La línea de
nudillos no es perpendicular al eje de la palma en la imagen cuando la mano está
girada. Una mezcla por cuadro según s/palma haría girar θ entre 16° y 154° al
cruzar la transición, y ρ lo acumularía como un giro que no ocurrió.

### 4. δ: cuánta «profundidad» inventa el escorzo

Rango de `ln(escala_t / escala_0)` dentro de cada muestra (máximo − mínimo), p50 (p90):

| letra | con `s` | con el tamaño de palma |
|---|---|---|
| X | 0.76 (**2.82**) | 0.31 (0.49) |
| Q | 0.56 (1.30) | 0.32 (0.48) |
| Ñ | 0.52 (1.28) | 0.27 (0.41) |
| K | 0.48 (0.80) | 0.34 (0.47) |
| J, Z, LL, RR | 0.13–0.27 (0.23–0.58) | 0.12–0.26 (0.21–0.46) |
| estáticas | 0.02–0.05 (0.03–0.11) | 0.02–0.05 (0.03–0.10) |

Con `s`, la X registra en el p90 un cambio de escala de e^2.8 ≈ 16× sin que la
mano se acerque a la cámara; con la palma, 1.6×. En J, Z, LL, RR y las estáticas
las dos medidas coinciden.

## Propuesta

**`FEATURE_SPEC_VERSION` 3.** Todo se recalcula desde los landmarks crudos de las
muestras: no hace falta regrabar para cambiar las features. Los modelos v2 se
rechazan al cargar, los golden se regeneran y se reentrena.

### Paso 4 — la escala es el tamaño de palma

```
m_t = max_{(a,b) ∈ P} ‖ p_{t,a} − p_{t,b} ‖₂,   P = {(0,5), (0,9), (0,13), (0,17), (5,17)}
```

en 2D, tras el paso 3, recorriendo `P` en ese orden (la misma `palm_size` que ya
usa la velocidad desde la v6, así hay una sola definición que replicar en
TypeScript). Como incluye muñeca → nudillo 9, `m_t ≥ s_t`: el umbral
`MIN_SCALE` se aplica a `m_t` y nada que hoy pase deja de pasar.

**Efecto en las estáticas:** cada vector de forma se reescala por `s/m`, entre
0.80 (M, N) y 1.00 (O, F, W), casi constante dentro de cada letra (p10 y p50
separados por ≤ 0.04). La distancia entre letras cambia poco y de forma
sistemática; hay que reentrenar y **repetir el eval de Fase 2** para confirmarlo.
Criterio: accuracy LOSO sin caer más de un punto respecto de 0.9261, mirando M y
N por separado, que son las que más se reescalan.

### Paso 5 — el ángulo de la palma, sostenido cuando se pone de canto

Por secuencia, en dos pasadas:

1. Cada frame es **fiable** si `s_t / m_t ≥ τ_rot` (`features.rotation_min_ratio`,
   propuesto **0.5**). Su ángulo es el de hoy, `θ^A_t` (muñeca → nudillo 9).
2. Un frame no fiable toma el θ del **último frame fiable anterior** de la misma
   secuencia; si no hay ninguno antes, el del **primero fiable después**.
3. Si la secuencia **no tiene ningún frame fiable**, todos sus frames usan `θ^B_t`
   (la línea de nudillos 5 → 17 girada 90°, con signo fijo en el contrato).

- **0.5**: por encima de la X de la prueba de reposo (máximo 0.467, p90 0.31) y muy por
  debajo de cualquier estática (p10 mínimo 0.77). Solo cae por debajo el 8.9% de
  los cuadros de la X, el 3.8% de la Ñ, el 2.2% de la Q y el 0.2% de la K.
- **Las estáticas no cambian en rotación:** ninguna tiene un frame bajo 0.5, así
  que el paso 5 les da exactamente el θ de hoy.
- **No se mezclan referencias dentro de una secuencia**: o A (con los frames
  colapsados sostenidos), o B entera. Con eso ρ no acumula el desfase entre A y
  B del punto 3.
- **Ninguna secuencia del dataset cae en el caso 3** (0 de 3407). Existe para
  ejecuciones como la X de hoy, con toda la secuencia de canto; con B su ruido
  queda en 2.8° en vez de 40°.
- Las ventanas estáticas que clasifica la máquina en vivo son secuencias como
  las demás: la regla se les aplica igual, y no cambia nada porque no se
  colapsan.

### Bloque 3 — δ con el tamaño de palma, ρ con el θ sostenido

```
δ_t = ln( m_t / m_0 )
ρ_t = unwrap(θ_t) − unwrap(θ_0)        # θ_t del paso 5 de arriba
```

con el `unwrap` exacto que ya fija el Bloque 3. Mientras la palma está de canto, ρ
queda plano en vez de acumular ruido, y δ ya no confunde un giro de la palma con
un acercamiento.

**Orden propuesto:** esta versión (pasos 4 y 5) como `FEATURE_SPEC_VERSION` 3, y
los canales ρ y δ del Bloque 3 como la 4, para poder medir por separado qué
aporta cada cambio, que es lo que pide la ablación del Bloque 3.

## Lo que no resuelve

- **La X sigue temblando 4 veces más en la imagen** (ADR 0019): es MediaPipe con
  el índice en gancho. Estas features dejan de amplificarlo, no lo quitan.
- **Con τ_rot se añade un umbral y un salto:** un frame que cruza 0.5 pasa de
  «sostenido» a «propio». Como el θ sostenido es el de un frame fiable cercano,
  el salto es el giro real que hubo mientras la palma estaba de canto, no el
  desfase entre referencias.
- **`scale_to_pixels`** (el metadato `mean_scale_px` de las muestras, que mide la
  distancia a la cámara) sigue con `s`: cambiarlo movería el umbral de
  `Distance`, y no es parte de las features. Se decide aparte.

## Decisiones abiertas

1. **τ_rot = 0.5.** Separa los dos grupos medidos (la X de reposo llega a 0.467;
   la estática más baja tiene p10 0.77), pero el margen del lado de la X es
   corto: 0.033. 0.6 lo ampliaría a 0.13 a costa de sostener más cuadros en la
   X, la Ñ y la Q del dataset (sus p10 están entre 0.52 y 0.64). Se puede fijar
   después de regrabar la X tal como se ejecuta hoy.
2. **El caso sin ningún frame fiable:** B entera (propuesto) o sin rotar
   (θ = π/2, identidad). B mantiene el vector comparable entre repeticiones con
   distinta inclinación; sin rotar es más simple, pero deja la orientación de la
   mano en el vector.
3. **Versiones separadas (3 y 4) o una sola** para pasos 4–5 y el Bloque 3.

## Lo que se implementó (2026-09-28)

Aprobado con tres decisiones:

1. **ρ y δ en una versión aparte** (`FEATURE_SPEC_VERSION` 4), para que la
   ablación del Bloque 3 los mida por separado. La v3 son solo los pasos 4 y 5.
2. **Secuencias sin ningún cuadro fiable: se rechazan** (`InvalidReason.PALM_EDGE_ON`),
   en vez de caer a la línea de nudillos.
3. **Histéresis** en el umbral de palma de canto, porque hay letras cerca del
   corte.

### La histéresis, con los números

Cambios de estado fiable / no fiable en las muestras del dataset, y secuencias que
parpadean (más de 2 cambios):

| letra | s/palma mínimo | corte único 0.5 | 0.45 / 0.55 | 0.5 / 0.6 | **0.45 / 0.6** |
|---|---|---|---|---|---|
| X (101) | 0.007 | 506 · 45 | 289 · 36 | 351 · 41 | **234 · 31** |
| Q (100) | 0.027 | 209 · 27 | 111 · 14 | 171 · 25 | **109 · 14** |
| Ñ (100) | 0.108 | 145 · 16 | 65 · 7 | 87 · 11 | **51 · 6** |
| K (121) | 0.181 | 20 · 3 | 16 · 2 | 16 · 2 | **16 · 2** |
| Z (100) | 0.444 | 2 · 0 | 2 · 0 | 2 · 0 | **2 · 0** |

- **`rotation_off_ratio` = 0.45, `rotation_on_ratio` = 0.6.** La banda más ancha
  de las medidas reduce a menos de la mitad los cambios de la X, la Q y la Ñ.
- **Ninguna estática baja de 0.702** en ningún cuadro, así que con cualquiera de
  estas bandas nunca dejan de ser fiables: su rotación es exactamente la de la v2.
- **Ninguna muestra del dataset queda sin cuadro fiable** con ninguna banda. La X
  de la prueba de reposo (máximo 0.467) sí: se rechazaría, como se decidió.

### Cómo se evita que el rechazo rompa el camino dinámico

La máquina de estados extrae el buffer en cada cuadro. Si una ventana de palma de
canto reiniciara la máquina como una escala degenerada, la X no podría trazarse.

- **`PALM_EDGE_ON` no reinicia la máquina**: solo la escala degenerada lo hace.
  Una ventana estable de canto no se clasifica
  (`RejectionReason.PALM_EDGE_ON`) y la máquina sigue; un trazo de canto llega al
  DTW, que lo rechaza como UNKNOWN.
- **La velocidad no pasa por el paso 5**: `features.pair_velocity` hace el
  suavizado, los pasos 1 a 3 y el tamaño de palma, igual que antes. La mano de
  canto sigue moviéndose aunque su secuencia no se pueda clasificar.
- **`mean_scale_px`** (distancia a la cámara) sigue con muñeca → nudillo 9
  (`features.reference_scales`), con la que se calibró `Distance`.

### Golden

Se regeneraron. Casos nuevos: `palm_edge_on` (un frame de canto: `PALM_EDGE_ON`),
`rotation_hysteresis` (s/palma ≈ 1, 0.2, 0.5, 0.5, 0.9: el 0.5 conserva el estado
no fiable y sostiene el ángulo) y `velocity_foreshortened_palm`, que ahora empieza
con un cuadro de frente. `scale_just_below_minimum` y `collapsed_scale` colapsan la
palma entera: mover solo el nudillo 9 ya no es escala degenerada, es palma de canto
(`synthetic.edge_on_palm`).

### Verificación

Se reentrenaron los dos modelos (`lsm-train --sin-sintetico`); los v2 se rechazan
al cargar.

**Eval de Fase 2 (LOSO):** mejora.

| | v2 | **v3** |
|---|---|---|
| accuracy | 0.9261 | **0.9288** |
| macro | 0.9201 | **0.9240** |
| UNKNOWN | 0.0573 | 0.0549 |
| mejor punto del barrido | 0.9578 | 0.9602 |

Por letra solo cambian S (62 → 68 de 80) y W (92 → 93 de 102). **M y N**, las que
más se reescalan (s/palma 0.80), quedan exactamente igual: 103/114 y 89/92. El
criterio era no caer más de un punto.

**Eval y replay de Fase 5:** la segmentación no cambia —611 trazos enteros, 11
partidos, 0 perdidos, como en la v2— y el clasificador dinámico mejora.

| | v2 | **v3** |
|---|---|---|
| DTW fuera de línea (LOSO): accuracy | 0.7926 | **0.8087** |
| macro / UNKNOWN | 0.7896 / 0.0177 | **0.8059 / 0.0000** |
| replay: dinámicas acertadas | 248 | **270** |
| Ñ / Q / Z acertadas | 27 / 11 / 46 | **45 / 14 / 47** |
| J / K / X acertadas | 92 / 70 / 2 | 92 / 70 / 2 |
| replay: estáticas, 1.ª letra correcta | 0.927 | 0.929 |

La mejora cae donde el escorzo pesaba: la Ñ (s/palma p10 0.59) pasa de 27 a 45.
La X sigue en 2 de 101: sus plantillas son del dataset viejo y, como se midió, la
X que se ejecuta hoy no es la grabada.
