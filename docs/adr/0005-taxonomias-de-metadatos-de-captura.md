# ADR 0005 — Taxonomías de metadatos de captura, y sus dos medidas objetivas

- **Estado:** aceptada
- **Fecha:** 2026-09-07
- **Fase:** 0 (cierre)
- **Implementa:** `src/lsm/types.py` · `docs/dataset-schema.md`

## Contexto

`ARQUITECTURA.md` §4.7 exige que cada muestra guarde `signer_id`, `session_id`,
`timestamp`, `handedness`, `lighting`, `distance` y `label`, pero nombra los campos
sin definir sus valores. La Fase 0 los implementó con enums provisionales, y esa
provisionalidad tiene fecha de caducidad: **en cuanto empiece la primera sesión de
captura, cambiar la taxonomía invalida metadatos ya grabados.** Volver a preguntarle
a alguien "¿la luz de aquella sesión de marzo venía de frente o de lado?" no es una
opción seria.

Al revisar la taxonomía inicial aparecieron dos problemas.

**El primero: `Lighting` mezclaba dos ejes.** Sus valores eran `DIM`, `INDOOR`,
`BRIGHT`, `BACKLIT` y `MIXED`. Los tres primeros dicen *cuánta* luz hay; `BACKLIT`
dice de *dónde* viene. No son alternativas entre sí: una escena a contraluz puede
ser brillante o penumbrosa, y para el detector son situaciones distintas. Con un
solo campo hay que elegir cuál de las dos cosas se anota y se pierde la otra —
justo cuando el contraluz es la condición que más degrada la detección de manos.

**El segundo: las tres taxonomías dependen del juicio de quien graba.** Dos
personas etiquetarán distinto la misma escena, y la misma persona etiquetará
distinto un martes y un jueves. `NEAR/MEDIUM/FAR` es el caso más flagrante: es una
discretización pobre de una magnitud continua que **la tubería ya calcula**, porque
la escala del paso 4 es exactamente el tamaño aparente de la mano. Y la luminancia
media del frame se obtiene con una operación sobre píxeles que la capa de captura
tiene delante de todos modos.

## Decisión

**Se separan los dos ejes de iluminación, se congelan las tres taxonomías, y cada
categoría subjetiva viaja acompañada de una medida objetiva obligatoria.**

### Taxonomías (categóricas, anotadas a mano)

| Campo | Valores | Qué significa |
|---|---|---|
| `light_level` | `DIM`, `INDOOR`, `BRIGHT` | Cuánta luz hay. |
| `light_direction` | `FRONTAL`, `LATERAL`, `BACKLIT`, `MIXED` | De dónde viene respecto de quien firma. |
| `distance` | `NEAR`, `MEDIUM`, `FAR` | Distancia aproximada de la mano a la cámara. |

### Medidas objetivas (calculadas, obligatorias)

| Campo | Unidad | Acompaña a |
|---|---|---|
| `mean_luminance` | `[0, 1]`, media del frame promediada sobre la secuencia | `light_level` |
| `mean_scale_px` | píxeles, escala del paso 4 promediada sobre la secuencia | `distance` |

Ambos son obligatorios en `Sample` y se validan al construir: la luminancia debe
caer en `[0, 1]` y la escala debe ser positiva.

`mean_scale_px` sale de `lsm.features.scale_to_pixels`: tras el paso 1 las
coordenadas quedan en unidades de "píxel dividido por el alto del frame", así que
multiplicar por el alto devuelve la medida a píxeles. No cuesta nada calcularla —la
tubería ya la necesita para normalizar.

### Por qué se conservan las dos cosas

No es redundancia. Sirven para preguntas distintas:

- La **categoría** sirve para filtrar el dataset a ojo y para planear la captura:
  "faltan sesiones a contraluz", "todas las muestras son de interior".
- El **número** sirve para responder la pregunta que de verdad se hará al mirar la
  primera matriz de confusión: *¿el modelo empeora con poca luz, o a distancia?*
  Con `NEAR/MEDIUM/FAR` la respuesta es un gráfico de tres barras construido sobre
  el criterio de quien grabó. Con `mean_scale_px` es una correlación.

## Alternativas consideradas

**1. Dejar `Lighting` como estaba.**
Descartada. El coste de separar los ejes hoy es una línea; después de la primera
sesión, es regrabar o quedarse con metadatos ambiguos.

**2. Solo las medidas objetivas, sin categorías.**
Tentador y descartado. Los números no se leen de un vistazo: nadie planifica una
sesión de captura mirando un histograma de luminancias. Las categorías son la
interfaz humana del dataset.

**3. Solo las categorías, sin números.**
Es lo que había. Descartada por lo dicho: hace imposible el análisis serio y
convierte una magnitud continua en tres cubetas elegidas a ojo.

**4. `distance` en centímetros medidos.**
Descartada. Nadie va a medir con cinta durante la captura, y si lo hiciera, la
distancia en centímetros no es lo que le importa al modelo: le importa el tamaño
aparente de la mano en el encuadre, que depende también del campo de visión de la
cámara. `mean_scale_px` es directamente esa magnitud.

**5. Guardar la luminancia por frame en `RawFrame`.**
Descartada por ahora. Sería más informativo —permitiría detectar cambios de luz a
mitad de una muestra— pero obliga a que todo constructor de frames aporte el dato,
incluidos los sintéticos de los tests y los golden vectors, cuyo formato de entrada
quedaría atado a un valor que no interviene en las features. El promedio por
muestra cubre la necesidad actual. Si la Fase 2 muestra que hace falta el detalle
por frame, se añade entonces con su propio incremento de esquema.

## Consecuencias

**A favor**

- El contraluz se puede anotar sin perder el nivel de luz.
- Se puede responder con datos si el modelo falla por luz o por distancia, en vez
  de por el criterio de quien etiquetó.
- Las dos medidas salen gratis: una es una media de píxeles, la otra ya la calcula
  el paso 4.

**En contra**

- `Sample` pasa de siete a nueve campos de metadatos. La captura tiene que
  rellenarlos todos, y dos de ellos exigen que el CLI calcule algo en vez de
  limitarse a preguntar.
- Las categorías siguen siendo subjetivas. Esta ADR no lo arregla, solo hace que
  no sean lo único disponible.

**Cómo se cambia**

A partir de la primera sesión de captura, **no se cambian**. Añadir un valor nuevo
a una taxonomía es aceptable si las muestras viejas siguen siendo válidas; quitar o
resignificar uno obliga a migrar el dataset o a descartarlo, y esa decisión ya no
es técnica.
