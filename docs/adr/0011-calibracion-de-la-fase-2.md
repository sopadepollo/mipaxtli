# ADR 0011 — La métrica del clasificador estático es coseno, y el margen no se optimiza contra el protocolo offline

- **Estado:** aceptada
- **Fecha:** 2026-09-09
- **Fase:** 2 (cierre)
- **Implementa:** `config.yaml`
- **Continúa:** `docs/adr/0010-la-clase-negativa-y-el-modo-de-grabacion.md`

## Contexto

Con el dataset completo —3 firmantes, 3407 muestras, 2585 dentro del alcance
estático— y `config.yaml` sin calibrar, la Fase 2 daba **0.8681** contra un
criterio de ≥0.90. La tentación era darla por buena y seguir.

Dos letras concentraban el fallo, y las dos de la misma forma:

| letra | accuracy | UNKNOWN | confusiones | σ mediana |
|---|---|---|---|---|
| `M` | 0.4474 | 63 / 114 | **0** | **0.0125** |
| `S` | 0.4625 | 43 / 80 | **0** | **0.0071** |

Cero confusiones y la **σ más baja de todo el dataset**: no se parecían a otra
letra ni estaban mal grabadas. Se rechazaban por distancia. Varias rondas de
regrabación no las movieron, porque no había nada que corregir en la grabación.

`M` y `S` son las dos configuraciones de puño más cerrado del alfabeto. La
normalización del `feature-spec.md` §4 divide por la distancia muñeca→nudillo del
medio, que en un puño cerrado se comprime; el vector resultante apunta en la
dirección correcta con la **magnitud** equivocada. La distancia euclidiana mira
magnitud y dirección; la coseno ignora la primera.

## Decisión

**1. `static_knn.metric: cosine`.**

Medido con el barrido completo (2250 puntos, leave-one-signer-out). No es un
ajuste fino: son +9 puntos de accuracy global, y `M` sale del fondo de la tabla.
El comentario de `config.yaml` ya decía que esta elección *"se decide con `make
eval`, no leyendo"*; esto es esa decisión, tomada con datos.

La elección es robusta frente a la objeción de sobreajuste que sí afecta a los
umbrales: `metric` es un eje **binario**, así que el barrido lo explora entero y
no hay ningún valor intermedio sin visitar donde esconder un óptimo de suerte.

**2. `static_knn.max_distance: 0.25`.** Consecuencia de lo anterior. La coseno
vive en `[0, 2]` y la euclidiana no, así que el 1.0 anterior no es comparable con
el 0.25 nuevo: es otra unidad, no un endurecimiento.

**3. `static_knn.min_margin: 0.6`, en contra de lo que recomienda el barrido.**

Esta es la decisión que hay que justificar. El barrido recomienda **0.5**, que da
el mejor accuracy de la rejilla (0.9578) — y lo da porque **apaga la puerta**: la
confianza vive en `[0.5, 1]`, así que con 0.5 nada se rechaza nunca.

Que salga ganadora es un artefacto del protocolo. En la evaluación offline **toda
ventana es una letra de verdad**, así que rechazar solo puede restar y el óptimo
es siempre no rechazar nada. En el video en vivo de la Fase 3 la mayoría de los
frames no son ninguna letra, y esta puerta es lo único que impide escribirlas.
Optimizarla contra el protocolo offline es optimizarla contra el caso equivocado.

Medido:

| `min_margin` | accuracy | macro | `UNKNOWN` |
|---|---|---|---|
| 0.5 (puerta apagada) | 0.9578 | 0.9556 | 0.0% |
| 0.55 | 0.9427 | 0.9393 | 2.9% |
| **0.6 (elegido)** | **0.9261** | 0.9201 | 5.7% |
| 0.7 | 0.8662 | 0.8530 | 12.8% |

0.6 conserva una puerta que de verdad rechaza y aún deja 2.6 puntos de margen
sobre el criterio. Se paga 3 puntos de accuracy por tener un clasificador que
sabe callarse, y en la Fase 3 esos 3 puntos se recuperan con creces.

**4. `quality.max_dispersion` se queda en `0.08`.** El barrido completo lo elige
solo; no hay decisión que tomar. (En una corrida anterior, con la clase negativa
todavía contaminada, el barrido mínimo recomendaba 0.32 y eso habría roto la
Fase 3. Lo arregló el ADR 0010, no un juicio.)

## Consecuencias

La Fase 2 **cumple**: 0.9261 de accuracy, 0.9201 macro, 5.7% de rechazo, con
validación leave-one-signer-out sobre 3 firmantes.

### Lo que queda pendiente de refinar

El criterio se cumple, no se agota. Por orden de tamaño:

- **`C` · `O`, 20 confusiones**, el par dominante desde la primera corrida y
  predicho por el glosario antes de grabar nada. `O` acierta 0.7500. Es una
  similitud geométrica real, no un problema de datos: la ruta es `ARQUITECTURA.md`
  §4.8 —distancias pulgar-yemas, ángulos entre falanges— y eso cambia el
  `feature-spec.md`, o sea versión nueva, golden vectors regenerados y su ADR.
- **`C` · `F`, 8 confusiones**, que **apareció al cambiar a coseno** y antes no
  estaba. Es el precio de ignorar la magnitud: dos manos con la misma forma y
  distinta apertura se acercan. Vale la pena mirarlo si `F` empeora al añadir
  datos.
- **`S` 0.7750 y `V` 0.7875**, las dos más bajas después de `O`. `V` tiene 80
  muestras, de las pocas que no se completaron a 120.
- **Los umbrales están ajustados sobre el mismo conjunto con el que se miden.**
  0.9261 es optimista en una cantidad que no se ha cuantificado. Para una cifra
  independiente haría falta un cuarto firmante que no participe en la
  calibración, y es lo que debería hacerse antes de publicar el número en ningún
  sitio.
- **El contraste de hipótesis sigue sin ser legible**: 12 «refutadas» incluyen
  pares donde una de las dos letras se rechazaba en vez de confundirse. Un par no
  se puede refutar si nunca se llegó a evaluar.
