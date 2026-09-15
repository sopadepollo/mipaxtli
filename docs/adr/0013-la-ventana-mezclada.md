# ADR 0013 — La ventana que se clasifica no es la ventana que se comprueba estable

- **Estado:** aceptada e implementada el 2026-09-09 (`SEGMENTATION_SPEC_VERSION = 2`)
- **Fecha:** 2026-09-09
- **Fase:** 3 (cierre)
- **Implementa:** `src/lsm/segmentation.py`
- **Continúa:** `docs/adr/0004-contrato-de-segmentacion.md`

## Por qué esto es un ADR y no una nota en un informe

`CLAUDE.md` es explícito: «Si una tarea revela que la arquitectura documentada
no funciona, detente y propón el cambio en vez de improvisar una excepción
local». Esto es esa propuesta. El hallazgo apareció mientras se construía
`tests/test_cli_demo.py` para la tarea 7 y hoy solo vive en el comentario de
`VIAJE` de ese archivo — el sitio donde nadie que toque `segmentation.ts` en la
Fase 7 va a buscarlo dentro de seis meses.

## El problema

`run_segmentation` decide que una ventana está lista para clasificarse con dos
señales:

- `moving = velocities[-1] >= velocity_threshold` — un solo par de frames, **el
  último** del buffer.
- `stable_run >= stable_frames` (5 por defecto) — cuántos frames consecutivos,
  contando desde el final, no dispararon `moving`.

Pero lo que se le pasa a `classify` no son esos 5 frames: es `window =
Sequence(frames=tuple(buffer))`, el buffer circular entero —24 frames por
defecto (`config.yaml`, `segmentation.buffer_size`)—. La máquina comprobó
quietud sobre los últimos 5 a 6 frames y clasificó los 24. Hay hasta 18 frames
sobre los que no se comprobó nada.

**Consecuencia medida, no teórica.** Si el tránsito entre dos letras dura menos
que `buffer_size` frames, el buffer que se declara estable todavía contiene
cola del tránsito anterior: la ventana clasificada mezcla dos manos. Se
descubrió construyendo el test de la tarea 7: con un tránsito de 4 frames entre
`S` y `A`, la tubería emitió una **`A` fantasma** — una letra que la persona
firmando nunca hizo. El test que hoy pasa (`VIAJE = buffer_size`, ver el
comentario en `tests/test_cli_demo.py`) pasa **porque** se eligió deliberadamente
un tránsito tan largo como el buffer para no tocar el síntoma. No lo repara: lo
evita en ese archivo.

## Por qué `quality.max_dispersion` no lo atrapa, aunque parezca que debería

`max_dispersion` existe precisamente para rechazar ventanas donde "la mano
todavía se estaba acomodando" (`docs/adr/0011-calibracion-de-la-fase-2.md`), así
que la pregunta obligada es por qué no atrapa esto. Verificado en
`src/lsm/features.py`, `aggregate_static` (§2):

**1. σ se mide solo sobre el canal de forma, ya normalizado.** `aggregate_static`
promedia `std_t` sobre los 42 componentes de `FeatureVector` — el canal de forma,
que para cada frame ya pasó por traslación (origen en la muñeca), rotación
(canonizada por el eje muñeca→nudillo medio) y escala (dividida por esa misma
distancia). El canal de trayectoria (`_trajectory_channel`, §3.1) — el que sí
registra que la mano viajó por el encuadre — **no entra en σ**: `aggregate_static`
ni lo recibe. Una ventana con la mano desplazándose 18 de 24 frames tiene
exactamente la misma σ que una mano quieta, siempre que la configuración de los
dedos no cambie mientras viaja. σ mide "la mano cambió de forma", no "la mano se
movió".

**2. Aunque la forma sí cambie entre las dos letras, la aritmética no alcanza.**
Con 18 frames de una letra y 6 de otra, superar `max_dispersion = 0.08` exige una
diferencia media de unas 0.19 unidades de mano repartida entre los 42
componentes del vector (el cálculo: `σ = mean_j(std_t(f_t[j]))`, y una mezcla de
dos valores constantes por bloques tiene una desviación estándar proporcional a
la diferencia entre ellos, atenuada por la fracción minoritaria — aquí 6/24).
Entre `A` y `S`, dos puños que solo difieren en la posición del pulgar, la
mayoría de los 42 componentes son casi idénticos entre las dos letras y la media
se hunde muy por debajo de ese umbral. No hay un valor de `max_dispersion` que
separe "ventana mezclada entre dos letras parecidas" de "temblor normal de una
mano quieta": las dos señales miden magnitudes distintas del mismo fenómeno y
una tapa a la otra según qué tan parecidas sean las letras en juego, no según si
hubo mezcla.

## La invariante que `docs/adr/0004-contrato-de-segmentacion.md` da por garantizada y no lo está

Esa ADR dice, sobre la definición de velocidad: «la ventana solo es estable si
la mano ni viajó ni siguió acomodándose». La implementación actual garantiza esa
propiedad de los últimos `stable_run` frames — normalmente 5 o 6 —, no de la
ventana de `buffer_size` frames que efectivamente se clasifica. La cita describe
la ventana; el código comprueba una cola de ella.

## Por qué urge y no se aplaza a cuando se decida la corrección

El contrato de segmentación está versionado con `SEGMENTATION_SPEC_VERSION`
justamente para que `segmentation.ts` lo reproduzca bit a bit en la Fase 7
(`docs/adr/0004-contrato-de-segmentacion.md`). Si esto se corrige después de que
exista la versión web, se corrige **dos veces** — una vez en Python y otra en
TypeScript — con paridad que hay que verificar entre las dos, en vez de una sola
vez con un test que las dos implementaciones comparten desde el principio.
Dejarlo sin registrar es la forma más barata de que la Fase 5, que reintroduce
las dinámicas y por tanto vuelve a tocar `segmentation.py`, lo redescubra desde
cero con el dataset ya grabado.

## Salidas posibles — sin elegir ninguna, porque esta fase no lo decide

- **Evaluar la estabilidad sobre la ventana que se va a clasificar, no sobre su
  último par.** Por ejemplo, `max(velocities)` en vez de `velocities[-1]`, o
  exigir `stable_run >= len(buffer)` antes de declarar `STABLE`. Cambia el
  criterio de transición de la máquina de estados.
- **Clasificar los últimos `stable_run` frames en vez del buffer entero.**
  Cambia qué ventana se le pasa a `classify`, no el criterio de estabilidad.
- **Mitigación disponible hoy, solo con configuración, sin tocar código:** subir
  `segmentation.stable_frames` hasta `segmentation.buffer_size` — `config.py` ya
  lo permite (`_coherencia_entre_umbrales` solo exige `stable_frames <=
  buffer_size`, y 24 = 24 pasa). Con `stable_frames = buffer_size` la ventana es
  pura por construcción: si los 24 frames tuvieron que estar por debajo del
  umbral de velocidad para llegar a `STABLE`, ninguno pertenece a un tránsito.
  Cuesta 24 frames de quietud sostenida por letra antes de emitir —0.8 s a 30
  fps— encima de los 0.4 s del cooldown de emisión: probablemente demasiado para
  que el deletreo se sienta ágil, y es la razón de no aplicarlo ya en
  `config.yaml` sin que alguien lo mida. La palanca hermana es bajar
  `buffer_size` para las estáticas. Hoy vale 24, el mismo número que
  `RESAMPLE_LENGTH` (`src/lsm/features.py`, el remuestreo de las dinámicas) y
  que `capture.static_frames` (`config.yaml`, con su propia justificación
  escrita: "800 ms de mano sostenida"). Ninguna de las tres coincidencias está
  documentada como intencional — no hay ADR ni comentario que ate
  `segmentation.buffer_size` a ninguna de las otras dos—, así que no se puede
  afirmar por qué vale 24 más allá de que nadie lo ha cuestionado todavía. Eso
  es en sí mismo parte de lo que habría que revisar antes de tocarlo.

## Consecuencias

**Si no se corrige:** cualquier transición entre dos letras más rápida que
`buffer_size` frames (0.8 s a 30 fps) arriesga emitir una letra que nadie firmó,
con probabilidad más alta cuanto más se parezcan las dos letras del par — el
mismo tipo de par que ya es difícil para el clasificador: `C`/`O`, con 20
confusiones medidas y el par dominante de la Fase 2
(`docs/adr/0011-calibracion-de-la-fase-2.md`), o `C`/`F`, que apareció al
cambiar a la métrica coseno. `docs/glosario-lsm.md` predice además `M`/`N` y
`S`/`T` como confundibles por forma — no lo son por distancia: `M` y `S` se
rechazaban limpio, sin una sola confusión entre ellas, por un problema de
magnitud del vector que la métrica coseno ya corrigió, así que no son un
ejemplo de este riesgo. El efecto de la ventana mezclada se suma al del
clasificador en los pares donde de verdad se parecen, en vez de ser
independiente de él.

**Si se corrige subiendo `stable_frames`:** se paga en latencia percibida, sin
tocar código, y sin garantía de que 0.8 s sea aceptable — nadie lo ha medido
con una persona firmando de verdad.

**Si se corrige cambiando el criterio o la ventana clasificada:** es el cambio
correcto a largo plazo, pero toca la máquina de estados que `segmentation.ts`
tiene que reproducir, así que exige su propio ciclo de golden vectors y
verificación cruzada antes de la Fase 7 — no es un cambio de una línea aunque el
diagnóstico quepa en una.

---

# Resolución (2026-09-09)

Lo de arriba se deja **tal como se escribió**, incluida la parte que la medición
posterior corrigió. Lo que sigue es lo que se decidió y lo que se midió.

## Lo primero fue medir, porque no había con qué decidir

Todos los umbrales están en frames, y «24 frames» significa cosas distintas a 30
fps que a 12. Se instrumentó el bucle en vivo (`lsm-demo --medir-fps`,
`src/lsm/telemetry.py`) separando la tasa de entrega de la cámara de la de
procesamiento de la tubería, con percentiles y reparto por etapas. Sesión de 60 s
con una persona deletreando delante, Windows, backend MSMF, 1280×720:

```
cuadros: 1067   pared: 59.98 s   fps sostenido: 17.8

fps por cuadro      media  mediana       p5      p95      min      max
entrega              18.4     18.4     13.8     23.6      1.6     31.3
procesamiento        27.2     27.1     18.2     36.8     12.3     61.8

latencia (ms)       media  mediana       p5      p95      min      max
camara                9.6      8.6      6.6     12.0      5.7    538.3
deteccion            36.7     35.1     25.7     52.8     16.2     78.9
segmentacion          1.8      1.7      1.0      3.5      0.0      7.7
preview               8.1      8.1      4.9     11.9      4.6     79.8
```

**La máquina de referencia corre a 17.8 fps, no a 30.** Cada umbral de
`segmentation` duraba 1.7 veces lo que su comentario afirmaba: el buffer no eran
800 ms sino **1348**, el cooldown de emisión no eran 400 ms sino 674, y el
espacio entre palabras no era un segundo sino 1.7. La lentitud reportada era, en
su mayor parte, el buffer circular vaciándose de frames de tránsito durante 1348
ms mientras cada rechazo intermedio costaba otros 225.

Dos lecturas más de la tabla, las dos con consecuencias:

- **`deteccion` es el 66% del ciclo** (35.1 ms de 53.5). Ninguna de las
  correcciones de esta ADR la toca, así que la tasa seguirá rondando los 18 fps:
  lo que se recupera es latencia de confirmación, no fluidez.
- **`segmentacion` es el 3%** (1.7 ms, techo medido 7.7). Eso es lo que hace
  asequible clasificar en cada frame en vez de uno de cada cinco, que es el
  mecanismo de la emisión progresiva. No hay que optimizar la segmentación; hay
  que dejar de esperar frames.

## Lo que se decidió

**1. La ventana de clasificación es el tramo estable** (`feature-spec.md` §6.4).
De las dos salidas que esta ADR listaba sin elegir, se toma la segunda: cambia
qué ventana se clasifica, no el criterio de estabilidad. La invariante del ADR
0004 pasa a cumplirse por construcción.

Consecuencia aceptada: promediar 5 frames reduce menos ruido que promediar 24. A
cambio, esos 5 contienen la seña y los 24 no.

Consecuencia aritmética descubierta al implementarlo: como `stable_run` cuenta
pares, el tope alcanzable de la ventana es `buffer_size − 1`, no `buffer_size`.

**2. Emisión progresiva con dos umbrales** (§6.6). Desde `stable_frames` se
clasifica en cada frame; por encima de `high_confidence` se emite ya, entre los
dos umbrales se acumula sin pagar cooldown, y por debajo del piso se rechaza como
siempre.

`high_confidence = 0.82` está **medido**, no elegido por instinto: distribución de
confianzas bajo leave-one-signer-out sobre las 2437 muestras no rechazadas del
dataset de la Fase 2.

| umbral | emite ya | precisión | errores que pasan |
|---|---|---|---|
| 0.75 | 86.3% | 0.9952 | 10 |
| 0.80 | 79.4% | 0.9979 | 4 |
| **0.82** | **75.1%** | **0.9995** | **1** |
| 0.85 | 67.1% | 0.9994 | 1 |

0.82 es la rodilla. Por encima la precisión ya no mejora —el único error que
queda es un `E→C` con 0.9316 que ningún umbral por debajo de 0.94 excluye— y la
emisión inmediata se desploma. Lo que la medida **no** cubre: se hizo sobre
muestras completas de 24 frames, no sobre las ventanas cortas de la emisión
progresiva, que son más ruidosas; en vivo se acumulará más de lo que dice la
tabla. El error va hacia tardar, no hacia escribir mal.

**3. Los umbrales temporales pasan a milisegundos** (§6.5), derivados de la tasa
medida al arrancar y congelada para toda la sesión. La conversión —redondeo hacia
arriba en el empate, piso de un cuadro— es normativa, porque `Math.round` y
`round` no coinciden y la discrepancia solo aparecería en algunas combinaciones
de umbral y tasa.

**4. `velocity_threshold` NO se convirtió**, y conviene que quede escrito por qué,
porque es la pieza que falta. Sigue en unidades de mano **por frame**, así que
sigue dependiendo de la tasa, y en la dirección mala: a menor tasa, dos frames
consecutivos están más separados en el tiempo, la misma mano física da un `v_t`
mayor y cuesta **más** declararla quieta. Es decir, la máquina lenta es también la
más exigente. No se arregló aquí porque `segmentation.velocity_threshold` es un
eje del barrido de calibración de la Fase 2 (`src/lsm/cli/evaluate.py`) y
cambiarle la unidad invalida esa calibración: pide su propia medición y su propio
ADR.

## Lo que se midió y no se tocó

- **σ y `max_dispersion`** siguen igual. El análisis de esta ADR sobre por qué no
  atrapan la ventana mezclada sigue siendo correcto, y σ pasa ahora a calcularse
  sobre la ventana estable, que es donde significa algo.
- **El apéndice A del `feature-spec`** sigue inactivo. Se activa con la matriz de
  confusión de la Fase 2 en la mano, no por instinto.
- **`capture.static_frames`** (24 frames, «800 ms a 30 fps») tiene exactamente el
  mismo defecto y se queda como está: es de la Fase 1 y cambiarlo invalidaría la
  comparabilidad del dataset ya grabado.
- **El outlier de `camara`**: 538 ms en un solo cuadro, con el mínimo de entrega en
  1.6 fps. Medio segundo de stall son diez cuadros perdidos, o sea dos ventanas
  estables enteras. Es la justificación del aviso de tasa baja del preview
  (`telemetry.min_fps`), no un problema que esta ADR resuelva.

## Cómo se verificó

El defecto original está fijado como test de regresión en
`tests/test_cli_demo.py::test_un_transito_corto_ya_no_funde_dos_manos_en_una_ventana`:
con un tránsito de 4 frames entre `S` y `A`, la tubería producía `"ssa"` —una `s`
que nadie firmó— y ahora produce `"sa"`. Se comprobó revirtiendo la ventana al
buffer entero y viendo reaparecer la letra fantasma.

La latencia adaptativa se ve en el mismo archivo: con el clasificador entrenado
sobre el corpus sintético, `A` (0.832) y `S` (0.959) emiten con la ventana mínima
y la `C` (0.743) acumula hasta agotarla. Que sea la `C` la que paga no es
casualidad: `C`/`O` es el par dominante de la matriz de confusión de la Fase 2.
