# ADR 0001 — La secuencia temporal es el tipo base de entrada

- **Estado:** aceptada
- **Fecha:** 2026-09-07
- **Fase:** 0 (andamiaje)

## Contexto

El alfabeto dactilológico de la LSM tiene dos clases de letras que se comportan de
manera distinta frente a un reconocedor:

- **Estáticas** (la mayoría): la letra es una configuración de la mano sostenida.
- **Dinámicas** (J, K, Ñ, Q, X, Z y las que confirme el glosario): la
  configuración de dedos no basta para identificarlas. Una J y una I tienen la
  misma forma de mano; lo que las separa es el recorrido que traza la muñeca.

La forma obvia de empezar un proyecto así es `predict(frame) -> letra`. Es más
simple, se llega antes a una demo que funciona con las letras estáticas, y es lo
que hacen casi todos los tutoriales de reconocimiento de señas que circulan.

El problema aparece después. Cuando toca soportar las letras con movimiento, un
sistema construido sobre frames sueltos no se extiende: hay que rehacer el formato
del dataset (una muestra deja de ser un vector y pasa a ser una matriz), el CLI de
captura (grabar N frames en vez de uno), el entrenamiento, la evaluación y la
interfaz del clasificador. En la práctica se reescribe todo menos la normalización
de features. Y ocurre a mitad del proyecto, que es cuando menos margen hay.

## Decisión

**Todo dato de entrada del sistema es una secuencia temporal de landmarks con
forma `(T, 21, 3)`. Ninguna API pública opera sobre un frame aislado.**

- Una seña estática es una secuencia donde los `T` frames son aproximadamente
  iguales. La agregación del §2 de `feature-spec.md` la reduce a un vector
  promediado, y de paso produce σ, un indicador de estabilidad que un frame suelto
  no puede dar.
- Una seña dinámica es una secuencia donde los frames varían. El canal de
  trayectoria del §3.1 recupera el recorrido.

Ambas comparten tubería, formato de dataset, código de captura, evaluación e
interfaz de clasificador. Lo único que cambia entre ellas es la implementación del
clasificador, detrás del `Protocol` de `classifiers/base.py`.

En código esto se materializa así:

- `lsm.types.Sequence` valida que tenga al menos un frame y expone `shape`
  `(T, 21, 3)`; `Sequence.window()` devuelve otra `Sequence`, nunca un frame.
- `features.extract_sequence_features` recibe una `Sequence`. La composición por
  frame existe (`_frame_geometry`) pero es privada.
- `Classifier.predict` recibe una `Sequence`.
- Un frame inválido no es `None` sino `InvalidFrame` con motivo, porque en un
  flujo temporal un hueco tiene consecuencias: interrumpe la secuencia.

## Alternativas consideradas

**1. `predict(frame) -> letra`, y añadir movimiento después.**
Descartada. Es más rápida al principio y más cara en total: la migración toca
captura, dataset, entrenamiento, evaluación y demo. Además obliga a mantener
durante un tiempo dos formatos de dataset incompatibles, o a regrabar todo lo
capturado hasta ese momento.

**2. Dos tuberías separadas: una para estáticas y otra para dinámicas.**
Descartada. Duplica el dataset, el código de captura y la evaluación, y exige
decidir a qué tubería mandar cada muestra *antes* de clasificarla, que es
justamente un problema difícil. También impide comparar las dos clases de letra en
una sola matriz de confusión.

**3. Secuencia de longitud fija en todas partes (por ejemplo, siempre 24 frames).**
Descartada como tipo base, aceptada como paso interno. Fijar la longitud desde la
captura tira información y obliga a decidir la duración de una seña antes de tener
datos. El remuestreo a `T_ref = 24` del §3.2 se aplica solo al entrar al
clasificador dinámico, donde tiene una razón concreta: acotar el costo del DTW.

**4. Guardar features en vez de landmarks crudos en el dataset.**
Descartada. Si cambia la normalización, un dataset de features hay que regrabarlo
con personas frente a la cámara; uno de landmarks crudos se re-deriva con un
comando. La regla quedó también en `ARQUITECTURA.md` §4.7.

## Consecuencias

**A favor**

- Las letras dinámicas no requieren rediseño: ya caben en el tipo base.
- El indicador de calidad σ y la máquina de estados salen naturalmente, porque hay
  una dimensión temporal donde medir estabilidad y velocidad.
- El dataset es uno solo, y la validación leave-one-signer-out se hace sobre él.
- Los tests se escriben con secuencias sintéticas y no necesitan cámara.

**En contra**

- Más ceremonia para el caso simple: clasificar una letra estática obliga a
  construir una `Sequence`, aunque sea de un frame.
- El almacenamiento del dataset crece: `T` frames por muestra en vez de uno. Con
  landmarks en JSON y `T ≈ 24` sigue siendo despreciable frente a video.
- La captura necesita decidir cuándo empieza y termina una muestra, lo que empuja
  la máquina de estados de `segmentation.py` hacia la Fase 0 en vez de la 3. Es un
  costo real y se asume: el problema existía igual, solo estaba escondido.

**Cómo se revierte**

No se revierte. Es la decisión que `ARQUITECTURA.md` §2 pide tomar el primer día y
no cambiar. Si se necesitara una ruta rápida por frame para latencia en el
navegador, se implementaría como un clasificador más detrás del mismo `Protocol`,
recibiendo una secuencia de longitud 1, sin tocar el tipo base.
