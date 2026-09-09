# ADR 0013 — La ventana que se clasifica no es la ventana que se comprueba estable (propuesta)

- **Estado:** propuesta — esta fase no la decide
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
