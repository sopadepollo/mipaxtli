# ADR 0017 — Por qué fallan las dinámicas en vivo: diagnóstico de tracking

- **Estado:** decisión 1 **aceptada e implementada** el 2026-09-26 (opción B, mano
  declarada); decisión 2 pendiente del Bloque 1
- **Fecha:** 2026-09-26
- **Fase:** 5.1, Bloque 0
- **Datos:** `data/diagnostico/2026-09-26-005123-habitual/` y
  `data/diagnostico/2026-09-26-005610-lampara/` (fuera del repositorio), generados
  con `lsm-demo diagnosticar`, 8 letras dinámicas × 10 repeticiones, commit `1242773`
- **Relacionadas:** ADR 0015 (camino dinámico), ADR 0016 (clasificador dinámico)

## Contexto

En vivo casi ninguna dinámica sale. La hipótesis de partida era una sola causa:
poca luz → desenfoque → MediaPipe pierde la mano → un hueco descarta el trazo.
Se instrumentó la demo (`src/lsm/tracking_diagnostics.py`) y se grabaron dos
sesiones guiadas, con la luz de siempre y con una lámpara de frente.

## Lo que se midió

**El candidato dinámico casi nunca termina en un trazo.**

| sesión | episodios de candidato | interrumpidos por hueco | «demasiado largos» | trazo entregado |
|---|---|---|---|---|
| habitual | 94 | 49 | 32 | **13** |
| lámpara | 86 | 47 | 22 | **17** |

Tres causas distintas, en orden de peso:

### 1. MediaPipe cambia la lateralidad a mitad del trazo

Un tercio de los cuadros llega como `LEFT` (2565 de 7738 y 2263 de 6076), y hay
**217 y 190 cambios** de lateralidad entre cuadros consecutivos. Quien firmó
confirmó después que alterna de mano a propósito —para entrenar con todo tipo de
usuario—, así que el tercio `LEFT` puede ser en parte real; **los cambios entre
cuadros consecutivos no**: nadie cambia de mano en 33 ms. **Todos** los
picos de velocidad absurdos (`v_t > 2`, hasta 57 manos por cuadro) son esos
cambios: el paso 2 del contrato espeja la mano cuando dice `LEFT`, así que un
cambio de etiqueta espeja los 21 puntos de un cuadro al siguiente. Para la
máquina de estados es un salto enorme; el candidato no encuentra reposo
(«demasiado largos»: 67 de 68 cuadros «en movimiento», reposo máximo 1 cuadro)
y, cuando sí cierra, la trayectoria está rota.

Se concentran en las letras con la palma de lado o hacia abajo:

| letra | cambios (habitual) | cambios (lámpara) |
|---|---|---|
| J | 118 | 124 |
| Q | 41 | 26 |
| X | 15 | 25 |
| Ñ | 20 | 4 |
| K, LL, RR, Z | 0–19 | 0–7 |

Y **el dataset no lo muestra**: 0 de las 3407 muestras grabadas mezclan
lateralidades, porque la captura rechaza `MIXED_HANDEDNESS`. Las J y las Q del
entrenamiento son solo los intentos en que MediaPipe mantuvo la etiqueta.

### 2. Huecos de detección, repentinos y ligados a la velocidad

Todos los cuadros inválidos son `NO_HAND` (4305 y 2075): MediaPipe no encuentra
mano, no es el filtro de score. El score del cuadro previo a una pérdida es alto
(p50 0.97): la mano desaparece de golpe, sin aviso. La tasa de pérdida sube con
la velocidad (por cuartil de `v_t`: 0.1% → 1.2% → 1.4% → 2.7% en la sesión
habitual) y **no se correlaciona con la luminancia** (Pearson 0.01 y 0.03).

Duración de los huecos dentro de un trazo: en la sesión habitual 40 de 62 duran
1–3 cuadros (≤ ~100 ms); con lámpara 23 de 53, mediana 5 cuadros (178 ms).

### 3. La tasa se congela mal y la cámara duplica cuadros

- La cámara entrega 29.5 fps, pero **el 44.5% son cuadros repetidos**, en patrón
  alterno (`.d.d.d..d`): **16.4 fps reales**, igual en las dos sesiones y en todos
  los cuartiles de luminancia.
- La tasa que se congela al arrancar midió **18.4 y 18.8 fps**, pero la sesión
  corrió a 29.5. Todos los umbrales en milisegundos se convirtieron a cuadros con
  la tasa equivocada y duran ~1.6 veces menos de lo configurado: los
  «demasiado largos» se cortan a los ~2.2 s y no a los 4 s de `motion_max_ms`.
- El reloj nominal que recibe MediaPipe (33.0 ms) coincide con el real entregado
  (33.9 ms). **Esa hipótesis queda descartada.**

### Sobre la luz

La lámpara **no cambió la luminancia medida del cuadro** (0.47–0.69 contra
0.49–0.71): la exposición automática la compensa. Así que la prueba no puede
decir si más luz ayudaría. Lo que sí dice es que, dentro de cada sesión, las
pérdidas no dependen de la luminancia y los duplicados tampoco. **Más luz no es
la solución principal**; el Bloque 2 (tolerancia a huecos) deja de ser una red de
seguridad y pasa a ser necesario, pero **no basta**: sin resolver la causa 1, el
Bloque 2 abortaría (su condición es que la lateralidad coincida a los dos lados
del hueco) y los trazos seguirían rotos.

## Decisión 1 — la lateralidad de una secuencia es una (planteamiento)

El paso 2 del `feature-spec` canoniza **cada cuadro** con la etiqueta de ese
cuadro. Las mediciones dicen que la etiqueta por cuadro no es fiable con la palma
de lado. La arquitectura documentada no funciona aquí y hay que cambiar el
contrato, no parchear la segmentación:

- **A. Lateralidad por pista (recomendada).** Mientras la mano no se pierda
  (`HandAcquired` → `HandLost`), la lateralidad es la de la mayoría de los
  primeros N cuadros de la pista, y se aplica a todos. Automática y sin interfaz.
  Cambia el §0 y el paso 2: `FEATURE_SPEC_VERSION = 2`, junto con el Bloque 3,
  que ya la sube.
- **B. Lateralidad de la sesión.** Quien firma declara su mano al empezar (la
  calibración ya registra la convención) y la etiqueta de MediaPipe se ignora.
  La más robusta; obliga a declararla y no admite cambiar de mano.
- **C. Tratar un cambio de etiqueta como un hueco.** No arregla nada: la J
  cambia 12 veces por repetición y se quedaría sin trazo. Descartada.

Con A o B hay que re-derivar el dataset (los landmarks crudos no cambian; solo
cómo se canonizan) y **volver a grabar J y Q** sin el filtro `MIXED_HANDEDNESS`
de la captura, que sesgó la muestra hacia los intentos «fáciles».

### Resolución: B, mano declarada, con un aviso

**Se descartó A** por dos motivos que dio quien decidió: la votación ocurre al
**principio** del trazo, que en la J es la fase con la palma de lado —donde
MediaPipe más se equivoca—, y si falla el trazo sale volteado de forma coherente
y se clasifica mal sin ningún síntoma; y cada hueco abriría una votación nueva
que puede contradecir a la anterior y abortar la interpolación del Bloque 2, que
exige la misma mano a los dos lados del hueco.

Lo que se implementó:

- **La mano se declara por sesión**: `--mano derecha|izquierda`, obligatorio en
  `lsm-capture grabar` y en la demo en vivo (`lsm-demo`, `lsm-demo diagnosticar`).
  El adaptador del detector (`io/hands.py`) la pone en `RawFrame.handedness`, que
  es lo que lee el paso 2; la etiqueta de MediaPipe queda en
  `RawFrame.detected_handedness`, solo para diagnóstico. Para cambiar de mano se
  abre otra sesión.
- **Contrato**: `feature-spec.md` §0.2, §0.3 y paso 2; **`FEATURE_SPEC_VERSION` =
  2**. Los modelos v1 se rechazan al cargar. Los golden vectors se regeneraron:
  solo cambia su número de versión, porque sus entradas ya declaraban la mano.
- **Modelo exportado**: `handedness_convention` desaparece y entra
  `detector_input = "UNMIRRORED"` —lo que de verdad tienen que compartir dos
  implementaciones desde que la etiqueta no decide—.
- **`lsm-capture calibrar`** comprueba ahora que el cuadro llega al detector sin
  espejar, geométricamente y sin mirar la etiqueta: con la mano derecha junto al
  hombro derecho, la muñeca tiene que caer en la mitad izquierda de la imagen
  (`lsm.hand_check.input_looks_unmirrored`), `capture.calibration_frames` cuadros
  seguidos. El registro pasa a v2 (`entrada_sin_espejar`); una entrada v1 se lee
  pero no habilita a grabar, así que **hay que recalibrar una vez**.
- **Aviso «¿cambiaste de mano?»** (`lsm.hand_check.HandMismatchWatcher`): la
  etiqueta de MediaPipe contradice a la mano declarada con score ≥
  `hands.mismatch_min_score` durante `hands.mismatch_ms` seguidos **con la mano
  quieta**. Solo avisa en el preview de la captura y de la demo; nunca cambia la
  mano. Los dos valores son puntos de partida, no medidos.
- **Captura**: desaparece `Rejection.MIXED_HANDEDNESS`; con mano declarada el
  caso ya no existe.
- **Dataset**: esquema de muestra v3, con `handedness_source`. La v2 se sigue
  leyendo como `DETECTED`. Al cargar, la mano de la muestra se aplica a todos sus
  frames y la etiqueta original queda en `detected_handedness`.

**Por qué v2 y no «v2 junto con el Bloque 3».** Se pidió que la mano declarada
fuera en la misma versión que los canales ρ y δ del Bloque 3, pero el orden de
trabajo pone la mano declarada primero y el Bloque 3 después. Si «v2» significara
hoy «mano declarada» y mañana «mano declarada + ρ + δ», un modelo entrenado en
medio cargaría sin quejarse bajo el formato nuevo, que es exactamente lo que la
versión existe para impedir. El Bloque 3 será `FEATURE_SPEC_VERSION = 3`.

### El dataset existente

**No tiene mano declarada.** Quien grabó alternó de mano **dentro** de cada
sesión, así que tampoco hay una mano por firmante que declarar a posteriori. Lo
único que hay por muestra es la etiqueta de MediaPipe, que la captura obligaba a
ser la misma en todos los frames. Esa se usa como mano de la muestra
(`handedness_source = DETECTED`):

| firmante | estáticas | `RIGHT` | `LEFT` | etiqueta distinta a la mano de la muestra |
|---|---|---|---|---|
| s01 | 755 | 385 | 370 | 0 |
| s02 | 1260 | 628 | 632 | 0 |
| s03 | 570 | 285 | 285 | 0 |

El cero de la última columna es **por construcción**, no una medida: la mano de
la muestra *es* la etiqueta de MediaPipe. Lo que confirma la tabla es la
alternancia (mitad y mitad en las tres personas), y lo que no puede decir es si
en alguna muestra MediaPipe se equivocó de mano de principio a fin —ese error
existiría igual con la v1 y con la v2—.

Consecuencia: re-derivar las estáticas con la mano de cada muestra da **las
mismas features que antes**, porque antes cada frame ya se canonizaba con esa
misma etiqueta. La evaluación de la Fase 2 repetida con la v2 da **el mismo reporte, cifra a cifra** —accuracy 0.9261, macro 0.9201, UNKNOWN 0.0573, mejor punto del barrido 0.9578— salvo el commit y las versiones registradas.

Las **J y Q se regraban** con mano declarada; el resto de dinámicas, en el
Bloque 4.

## Decisión 2 — la tasa se mide sin duplicados y después del arranque

**Implementado (Bloque 1), pendiente de medir:**

- `io/camera.py` descarta los cuadros repetidos antes de MediaPipe
  (`capture.drop_duplicate_frames`, activado), reconociéndolos por su miniatura
  en grises; `CameraFrame.skipped_duplicates` dice cuántos saltó.
- El calentamiento de la demo descarta `telemetry.warmup_discard_frames` cuadros
  antes de medir la tasa, y la mide sobre cuadros nuevos.
- `capture.fourcc` y `capture.backend` son configurables; el formato se pide
  **antes** que la resolución.
- `lsm-demo medir-camara` mide backend × formato × resolución sin descartar y
  reporta fps entregados y fps nuevos. Con esos números se fija la configuración
  y se repite el diagnóstico de huecos; los parámetros del Bloque 2 se calibran
  con los números nuevos, no con los de 16 fps.
- La cámara de referencia llega por **MSMF nativo en Windows** (cabecera de los
  dos reportes), no por WSL/usbipd, así que esa comparación no aplica.

### Lo que midió `medir-camara` (2026-09-26)

Doce configuraciones, con y sin MediaPipe en el bucle
(`data/diagnostico/camara-2026-09-26-*.md`):

| configuración | fps entregados | fps nuevos |
|---|---|---|
| MSMF / auto, formato del driver o MJPG, 1280x720 o 640x480 | 28.5–30.1 | **16.0–16.7** |
| DSHOW 640x480 (pedido MJPG, aceptado YUY2) | 16.6 | **16.6** (0% repetidos) |
| DSHOW 1280x720 (pedido MJPG, aceptado YUY2) | 10.0 | 10.0 |

**Ninguna combinación pasa de ~16.6 fps reales**, y MediaPipe en el bucle no
cambia nada: no es la tubería, y tampoco el formato ni el transporte —DSHOW no
acepta MJPG y cae a YUY2; MSMF no informa el formato, y con MJPG pedido da lo
mismo—. El intervalo entre cuadros nuevos es ~60 ms constante (≈ 1/16.6 s), que
es la firma de una exposición automática larga o de un límite del sensor, no de
un formato. Lo que no se probó es fijar la exposición a mano
(`CAP_PROP_AUTO_EXPOSURE` / `CAP_PROP_EXPOSURE`): es la palanca que queda.

**Configuración fijada**: la de siempre —`backend: auto`, `fourcc: null`,
1280x720— con los repetidos descartados. Da los mismos fps nuevos que cualquier
otra y más resolución que DSHOW 640x480, que es la única sin repetidos de origen.

### El diagnóstico repetido (2026-09-27), con mano declarada y sin repetidos

| | 26 habitual | 26 lámpara | 27 habitual | 27 lámpara |
|---|---|---|---|---|
| picos `v_t > 2` (cambios de etiqueta) | 217 | 192 | **0** | **0** |
| trazos entregados | 13 | 17 | **36** | 8 |
| interrumpidos por hueco | 49 | 47 | **70** | **70** |
| mediana de hueco dentro de trazo | 2 cuadros | 5 cuadros | **15** | **16** |
| pérdida en el cuartil más rápido de `v_t` | 2.7% | 2.2% | **8.7%** | **8.3%** |

**La mano declarada funcionó**: MediaPipe sigue cambiando de etiqueta (34 y 78
veces) pero ya no espeja nada, y los picos desaparecieron. **Los huecos
empeoraron, y la causa probable la introdujo el Bloque 1**: con los repetidos
descartados el intervalo real entre cuadros pasó a ~65-68 ms, pero MediaPipe
seguía recibiendo marcas nominales de 33 ms («intervalo que recibe MediaPipe
33.0 ms» en los dos reportes). Su rastreador en modo VIDEO esperaba la mitad del
movimiento que ocurría y perdía la mano en los trazos rápidos. La hipótesis del
reloj que este ADR daba por descartada lo estaba solo mientras la cámara
entregaba 30 cuadros por segundo, repetidos incluidos.

**Corregido**: `hands.real_timestamps` (activado) pasa a MediaPipe el reloj real
(`time.monotonic`), estrictamente creciente. El contador nominal sigue
disponible con `false`, porque es reproducible. Hay que **repetir el diagnóstico**
con la corrección antes de calibrar el Bloque 2: sus números son los que
cuentan.

**Una segunda consecuencia del Bloque 1, que no se corrige aquí.** `v_t` es por
cuadro (§6.5, deuda anotada desde el ADR 0013). Sin repetidos, cada par de
cuadros está el doble de separado en el tiempo y la misma mano da una `v_t` del
doble: los cuadros de «mano quieta» cayeron de 1074 a 195. `velocity_threshold`
y `motion_threshold` quedaron efectivamente el doble de estrictos. Pasarlos a
unidades por segundo es el arreglo, y es un recalibrado de umbrales de las Fases
2, 3 y 5: se propone, con esta evidencia, pero no se hace sin decisión.

### Velocidad por segundo (SEGMENTATION_SPEC_VERSION 4)

Decidido el 2026-09-27. Los dos umbrales de velocidad pasan a unidades de mano
por segundo y la máquina los convierte a por cuadro con la tasa congelada de la
sesión, igual que los umbrales en milisegundos: `umbral_cuadro = umbral / fps`.
La métrica `v_t` del §6.1 no cambia.

**No se recalibró nada: se convirtió.** Cada umbral se multiplicó por la tasa a
la que se midió. Los dos salieron del dataset grabado (la distribución de `v_t`
del diagnóstico de la Fase 2 y el barrido del detector de movimiento del ADR
0015), grabado a ~30 cuadros por segundo entregados:

| campo v3 (por cuadro) | × 30 fps | campo v4 (por segundo) |
|---|---|---|
| `velocity_threshold` 0.02 | = | `velocity_threshold_per_s` 0.6 |
| `motion_threshold` 0.025 | = | `motion_threshold_per_s` 0.75 |

Los campos cambian de nombre a propósito: un `config.yaml` viejo con `0.02` falla
al cargar en vez de leerse como 0.02 manos por segundo. Sobre el dataset, que se
reproduce a la tasa nominal de 30 fps, los umbrales por cuadro son exactamente
los de antes. El eval de Fase 2 (`lsm-eval --sin-sintetico`) dio lo mismo
—accuracy 0.9261, macro 0.9201, UNKNOWN 0.0573— y el replay de Fase 5
(`lsm-eval-dinamico --sin-sintetico --rejilla rapido`) dio resultados idénticos
salvo la versión y el commit: 621 de 622 trazos enteros, J 100/100, accuracy
0.7926. En vivo, a 16.6 fps nuevos, el umbral por
cuadro sube a 0.036 y 0.045, que es lo que compensa el doble de separación entre
cuadros que dejó el descarte de repetidos.

`capture.exposure` (manual, `None` = automática) y
`lsm-demo medir-camara --exposicion=...` quedan para medir si la exposición es lo
que limita la cámara a 16.6 fps.

## Alternativa anotada: ventana deslizante con spotting por DTW

**No implementada.** Si tras velocidad por segundo, exposición fija y la
tolerancia a huecos del Bloque 2 el camino dinámico sigue sin entregar los
trazos, se evaluará reemplazar la detección de fin de trazo —DYNAMIC_CANDIDATE
esperando reposo— por **spotting**: una ventana deslizante de duración fija que
se compara con DTW contra las plantillas en cada paso, y emite cuando el costo
cae bajo un umbral durante un mínimo de pasos, sin necesitar que la mano se
detenga.

**Criterio de activación.** En los dos diagnósticos guiados (luz habitual y
lámpara) repetidos con esas tres correcciones, la fracción de repeticiones que
entregan **exactamente un trazo completo** al clasificador (columna «trazos» = 1
en la sección 5 del reporte, sobre las repeticiones no descartadas) queda por
debajo de **~80%** en el conjunto de las letras dinámicas evaluables. Se mira
además por letra: si la falta se concentra en una o dos letras, primero se
revisa por qué antes de cambiar el mecanismo entero.

Lo que costaría: una comparación DTW por paso de ventana contra todas las
plantillas (hoy es una por trazo), una segunda puerta para no emitir dos veces
el mismo trazo, y reescribir el §6.7 del contrato.

Queda para después de medir: si con los cuadros ya deduplicados vuelve a
funcionar el criterio de frames **consecutivos** en movimiento (ADR 0015, punto
3), y si conviene bajar `hands.min_tracking_confidence`.

Con el Bloque 1 (descartar duplicados antes de MediaPipe) la tasa relevante pasa
a ser la de cuadros únicos (~16.4 fps), y el calentamiento tiene que medirla
después de que la cámara se estabilice, no en los primeros 30 cuadros. Sin esto,
cualquier umbral en milisegundos que se calibre en vivo está mal medido.

## Lo que el diagnóstico no pudo separar

- Si los huecos son desenfoque (luz) o pérdida del rastreador por velocidad: la
  lámpara no movió la exposición. Una tercera sesión con la exposición de la
  cámara fijada a mano (o más luz de verdad, medida en la luminancia) lo diría.
- `min_tracking_confidence`: las pérdidas son `NO_HAND` repentinas; bajarlo
  podría reducirlas, y el barrido del Bloque 1.7 sigue en pie, pero conviene
  hacerlo después de la decisión 1, o medirá cambios de etiqueta como pérdidas.
