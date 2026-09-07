# ADR 0004 — El contrato de segmentación se versiona aparte del de features

- **Estado:** aceptada
- **Fecha:** 2026-09-07
- **Fase:** 0 (cierre)
- **Implementa:** `docs/feature-spec.md` §6 · `src/lsm/segmentation.py`

## Contexto

La máquina de estados necesita saber si la mano está quieta, y para eso mide una
velocidad. Esa medida quedó en la Fase 0 documentada solo en un docstring de
`features.py`, lo cual tiene dos problemas.

El primero es de alcance: en la Fase 7 no solo se reimplementa `features.ts`,
también `segmentation.ts`. Si la definición de velocidad vive en un comentario de
Python, la versión web la reinventará —dividiendo por otra escala, midiendo sobre
otros puntos— y el resultado será que la app rechaza o acepta ventanas distintas
que el escritorio, con los mismos umbrales en el mismo `config.yaml`. Es
exactamente el fallo que `ARQUITECTURA.md` §4.5 quiere evitar para las features, y
aplica igual aquí.

El segundo es de versionado. La tentación es meter §6 bajo `FEATURE_SPEC_VERSION`,
que ya existe. Sería un error caro: `feature_spec_version` es el campo que hace que
un modelo entrenado se **rechace al cargarse**. Si la definición de velocidad
viviera bajo esa versión, afinar el criterio de estabilidad —algo que se va a hacer
varias veces durante la calibración de la Fase 2— invalidaría todos los modelos
entrenados y obligaría a reentrenar. Y no hay ninguna razón para ello: el
clasificador nunca ve la velocidad.

## Decisión

**Se crea `SEGMENTATION_SPEC_VERSION`, independiente de `FEATURE_SPEC_VERSION`, y
la definición de velocidad sube a `docs/feature-spec.md` §6.**

- `SEGMENTATION_SPEC_VERSION` vive en `src/lsm/segmentation.py` y cubre la §6 y la
  máquina de estados.
- `FEATURE_SPEC_VERSION` sigue cubriendo las §1 a §5, que es lo que entra al
  clasificador.
- El export de modelos **no** lleva `segmentation_spec_version`: un modelo no
  depende de cómo se segmenta. Ese es precisamente el argumento de la separación.
- `golden_features.json` sí declara ambas, y los `sequence_cases` incluyen `scales`
  y `velocities` para que `segmentation.ts` pueda validarse contra el mismo
  archivo.

### La escala divisora: la media del par

La §6 tenía un hueco que había que cerrar: la velocidad se mide sobre los puntos
del paso 2, que es **anterior** al escalado del paso 4, así que hay que decir por
cuál escala se divide. Las tres candidatas dan números distintos en cuanto la mano
se acerca a la cámara.

**Se elige `s_par = (s_{t-1} + s_t) / 2`.**

Descartada la `s̄` de la ventana —que era lo que hacía la implementación inicial—
porque con ella el mismo par de frames da velocidades distintas según qué otros
frames haya en el buffer en ese instante. Una implementación incremental sobre un
stream, que es la forma natural de escribir esto en TypeScript, no coincidiría con
una que recorre la ventana, y la discrepancia dependería del estado del buffer: la
peor clase de discrepancia para depurar.

Descartadas `s_t` y `s_{t-1}` a secas por asimétricas: la velocidad de un par
dependería de en qué dirección se recorre el tiempo. La media no cuesta más y no
tiene esa propiedad rara.

## Riesgo registrado: esto no sirve para enrutar dinámicas

**Consecuencia directa de medir sobre puntos sin trasladar**, y la razón principal
por la que esta ADR existe.

`v_t` mide dos cosas a la vez: cuánto se desplazó la mano por el encuadre y cuánto
cambió la configuración de los dedos. Para las letras estáticas eso es exactamente
lo deseable —la ventana solo es estable si la mano ni viajó ni siguió
acomodándose—, y es la razón de haberlo definido así.

Para las dinámicas es una trampa. En una J, una Ñ, una Q, una X o una Z el
movimiento **es** la seña: `v_t` se mantiene alta durante toda la ejecución y el
criterio de estabilidad no se cumple nunca. Una máquina que solo espere
`v_t < umbral` no emitirá jamás una letra dinámica; se quedará en TRACKING hasta
que la persona termine y se detenga, y para entonces la ventana contiene el final
del movimiento, no el trazo.

**Por lo tanto, el enrutamiento estático/dinámico de la Fase 5 no puede colgar de
`velocity_threshold`.** Va a necesitar un criterio propio —energía de movimiento
sostenida y con patrón, no ruido, como ya apunta `ARQUITECTURA.md` §4.2— y
probablemente un camino distinto por la máquina de estados: capturar la ventana
completa mientras hay movimiento coherente, en vez de esperar a que se detenga.

No se resuelve ahora, y se deja escrito precisamente para no descubrirlo en la
Fase 5 con el dataset ya grabado y las letras dinámicas sin reconocer.

## Alternativas consideradas

**1. Dejar la definición en el docstring.**
Descartada por lo de arriba: `segmentation.ts` la reinventaría.

**2. Meter §6 bajo `FEATURE_SPEC_VERSION`.**
Descartada: acoplaría el ajuste de umbrales al reentrenamiento de modelos. Afinar
cómo se decide que una mano está quieta no cambia ni un número del vector que
consume el clasificador.

**3. Un documento aparte, `segmentation-spec.md`.**
Descartada por ahora. La §6 depende de los intermedios del §1 —los puntos del paso
2 y la escala del paso 4—, así que separarla obligaría a que un documento citara
constantemente los pasos del otro. Se mantiene en el mismo archivo con su propia
versión, que es lo que resuelve el problema real. Si la §6 crece mucho al llegar la
Fase 5, se separa entonces.

**4. Medir la velocidad sobre los puntos ya normalizados (tras el paso 5).**
Descartada, y conviene entender por qué: tras el paso 3 la posición de la mano en
el encuadre se ha destruido, de modo que la velocidad solo mediría el cambio de
configuración de los dedos. La máquina se daría por estable con una mano que viaja
por el encuadre con los dedos quietos, que es justo el caso —el tránsito entre dos
letras— que la segmentación existe para descartar.

## Consecuencias

**A favor**

- Se pueden calibrar los umbrales de segmentación sin invalidar modelos.
- `segmentation.ts` tiene un contrato que reproducir y golden vectors contra los
  que validarse, igual que `features.ts`.
- La elección de la escala hace que una implementación incremental y una por
  ventana den los mismos bits.

**En contra**

- Dos versiones que mantener y que alguien puede confundir. Se mitiga con un test
  que verifica que cada constante vive en su módulo.
- La velocidad se calcula en `features.py` aunque pertenezca a este contrato,
  porque necesita los intermedios geométricos del §1. Es una costura fea: el
  docstring de `_velocities` la señala explícitamente para que nadie asuma que está
  bajo `FEATURE_SPEC_VERSION` por vivir en ese archivo.
