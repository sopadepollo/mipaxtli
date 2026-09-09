# ADR 0010 — La clase negativa sale del alcance estático por su `kind`, no por su etiqueta

- **Estado:** aceptada
- **Fecha:** 2026-09-09
- **Fase:** 2
- **Implementa:** `src/lsm/types.py` · `src/lsm/evaluation.py` · `src/lsm/io/dataset.py`
- **Continúa:** `docs/adr/0008-clasificador-estatico-y-protocolo-de-evaluacion.md`

## Contexto

La primera evaluación con dataset real dio 0.5253 de accuracy contra un criterio
de fase de ≥0.90. `NONE` fue la peor clase de las 22, con **0.0482** (4 aciertos
de 83) y 67 rechazos.

Medido sobre esas 83 muestras:

| modo de grabación | n | σ mediana | σ máxima | por encima de `max_dispersion` |
|---|---|---|---|---|
| `STATIC` | 51 | 0.0212 | 0.0710 | 0 / 51 |
| `DYNAMIC` | 32 | **0.1832** | **2.6758** | **28 / 32** |

Las 32 dinámicas son lo que `docs/glosario-lsm.md` §4 pide grabar: transiciones
entre letras y gestos cotidianos. Están bien grabadas y son el material del
clasificador dinámico de la Fase 5. El problema es que hasta ahora entraban a la
Fase 2, y ahí hacían daño dos veces:

1. **Al entrenar.** `static_knn` da **un solo centroide por clase** y lo construye
   promediando las 42 componentes de forma de cada muestra. El promedio de un
   saludo sobre 24 frames no es ninguna configuración de mano; es una mancha que
   arrastra el centroide de `NONE` hacia un punto que no representa nada. Que la
   mancha caiga cerca de letras reales explica las confusiones `E`·`NONE` (7) y
   `B`·`NONE` (5), que el glosario no había predicho.
2. **Al evaluar.** 28 de las 32 superan `quality.max_dispersion` y entran
   rechazadas de oficio por la primera puerta del clasificador. Contaban como
   fallo de `NONE` sin llegar a mirar un centroide.

El filtro de la fase era por **etiqueta** (`PHASE2_LABELS`), y `NONE` es una
etiqueta del alcance. No había forma de expresar «esta muestra concreta se grabó
en movimiento», porque `to_sample()` tiraba el `kind` al convertir la muestra
almacenada en `types.Sample`.

## Decisión

**1. `SampleKind` se muda a `types.py` y `Sample` recupera el campo `kind`.**

El dato ya se guardaba en disco desde la Fase 1 —`docs/dataset-schema.md` lo lista
y explica exactamente para qué: *«qué se **hizo** frente a la cámara, que no
siempre es lo que el glosario dice que la letra **es**»*—. Lo que faltaba era que
sobreviviera a `to_sample()`. Vivía en `capture.py`, que ni `evaluation.py` ni
`types.py` pueden importar sin ciclo; en `types.py` es donde le corresponde estar
según la primera regla de `CLAUDE.md` sobre por dónde se empieza.

**2. `observe()` descarta `NONE` con `kind == DYNAMIC`, y lo cuenta aparte.**

Aparte de `excluded` a propósito: aquellas salen porque su **etiqueta** está fuera
del alcance, estas porque su **grabación** lo está. Mezclarlas en el mismo
contador haría ilegible el reporte —`NONE: 32` bajo el epígrafe «descartadas por
dinámicas» sugeriría que `NONE` es una letra dinámica— así que `Dataset` lleva
`dynamic_negatives` y el reporte le da su propia fila.

La regla sigue viviendo en **un solo sitio**, que es lo que pedía el docstring de
`static_knn`: el clasificador sigue siendo agnóstico a las etiquetas y al modo.

**3. No se descarta nada del disco.** Las 32 muestras siguen en `data/raw` con su
`kind`, y son exactamente lo que la Fase 5 necesita para entrenar el rechazo del
clasificador dinámico. Esto es un cambio de **alcance de la evaluación**, no una
purga del dataset.

## Alternativas descartadas

**Filtrar por σ en vez de por `kind`.** Habría sido más corto —`observe()` ya
calcula la dispersión— pero el criterio depende de `quality.max_dispersion`, que
es justo uno de los ejes del barrido de calibración. El conjunto de entrenamiento
cambiaría en cada punto de la rejilla y las comparaciones dejarían de ser entre
umbrales para pasar a ser entre datasets distintos. `kind` no depende de ningún
umbral: es lo que se hizo frente a la cámara.

**Filtrar en `io/corpus.py`, al leer.** Habría evitado tocar `types.Sample`, pero
esparce la definición del alcance de la fase entre el lector y el evaluador. El
docstring de `static_knn.StaticKnnClassifier` advierte contra exactamente eso, y
la Fase 5 tendría que deshacerlo en dos sitios en vez de uno.

**Dar varios centroides a `NONE`.** Es la solución de fondo al problema real —una
clase multimodal comprimida a un punto— y probablemente haga falta en la Fase 3,
cuando la máquina de estados vea gestos que no son ninguna de las 21 letras. Pero
es un cambio del clasificador, no del alcance, y mezclarlo con esta corrección
impediría saber cuál de los dos movió el número. Queda anotado, no hecho.

## Consecuencias

El accuracy de la Fase 2 se mide ahora sobre `NONE` estática, que es la única que
`static_knn` puede modelar con un centroide. Es una métrica más honesta y también
más estrecha: **el sistema sigue sin tener respuesta para una mano en tránsito**,
solo que ahora eso se llama Fase 5 en vez de contarse como un fallo de `NONE`.

La consecuencia que hay que vigilar es la de la Fase 3: la demo en vivo verá manos
en movimiento constantemente, y el rechazo de esas ventanas no puede depender del
centroide de `NONE`. Depende de `quality.max_dispersion`, que es la puerta que las
para antes de clasificar. Esa puerta **no debe relajarse** con el valor que
recomiende el barrido de la Fase 2, porque el barrido optimiza sobre ventanas ya
recortadas donde esa puerta no tiene el trabajo que tendrá en video continuo.
