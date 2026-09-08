# ADR 0002 — Formato del vector de features: 42 componentes, sin `z`

- **Estado:** aceptada
- **Fecha:** 2026-09-07
- **Fase:** 0 (andamiaje)
- **Implementa:** `docs/feature-spec.md` v1 · `src/lsm/features.py`

## Contexto

MediaPipe Hands entrega 21 landmarks por mano con coordenadas `(x, y, z)`
normalizadas al frame. Alimentar eso directamente a un clasificador no funciona:

- `x` se normaliza por el ancho e `y` por el alto, así que en un encuadre 16:9 la
  mano llega deformada horizontalmente y el mismo gesto da vectores distintos
  según la cámara.
- Si el dataset se graba a 50 cm y la demo ocurre a metro y medio, cambia la
  escala y falla.
- Si la mano se mueve a otra esquina del encuadre, cambia la posición y falla.
- Si quien firma es zurdo y el dataset lo hicieron diestros, falla.
- Si inclina la muñeca, falla.

Hay una segunda restricción, menos obvia y más cara: en la Fase 7 esta misma
transformación se reimplementa en TypeScript para correr en el navegador. Si las
dos implementaciones difieren en cualquier detalle —qué landmark se usa como
referencia, si se normaliza antes o después de espejar, el orden de las
operaciones— el modelo entrenado en Python da resultados distintos en el navegador
y la depuración es infernal, porque nada falla: solo baja la precisión.

Y una tercera: la coordenada `z` de MediaPipe. No es profundidad métrica, es una
estimación relativa a la muñeca producida por un modelo entrenado, ruidosa entre
frames consecutivos e inconsistente entre cámaras.

## Decisión

**El vector de features son 42 componentes: 21 landmarks × (x, y), tras la
normalización de `feature-spec.md` §1. La coordenada `z` se descarta en v1.**

Puntos que esta ADR fija y que el `feature-spec.md` detalla paso a paso:

1. **Orden de operaciones inmutable:** aspecto y orientación → lateralidad →
   traslación → escala → rotación → descarte de `z` → aplanado. Está prohibido
   fusionar pasos aunque sean algebraicamente equivalentes: la equivalencia
   algebraica no implica equivalencia numérica en punto flotante.
2. **Escala de referencia: muñeca → nudillo del dedo medio** (`p_0` → `p_9`),
   medida en 2D. Es estable frente a la flexión de los dedos; cualquier distancia
   que involucre una punta cambia con la propia seña.
3. **Se conservan las componentes constantes.** Por construcción `p_0 = (0,0)` y
   `p_9 = (0,1)` en todas las muestras. Aportan distancia cero en cualquier
   métrica y no afectan al clasificador; se mantienen porque conservar los 21
   índices alineados con la numeración de MediaPipe elimina una fuente crónica de
   errores off-by-one al depurar y al reimplementar en TypeScript.
4. **Aritmética en float64 y sin numpy en la ruta de features.** Las reducciones
   de numpy suman por pares: `numpy.mean` no produce los mismos bits que sumar en
   orden temporal ascendente, que es lo que exige el §5.4. Con `T ≤ 30` y 42
   componentes, hacerlo con bucles explícitos no cuesta nada y se traduce a
   TypeScript línea por línea.
5. **Las constantes del contrato no son configurables.** La dimensión 42,
   `T_ref = 24`, `MIN_SCALE = 1e-6` y `FEATURE_SPEC_VERSION` viven en
   `features.py`, no en `config.yaml`. La regla de "cero umbrales hardcodeados"
   (`CLAUDE.md` §5) aplica a lo que se calibra empíricamente —velocidades,
   ventanas, confianzas, cooldowns—, no al formato del vector. Si el formato
   fuera configurable, dos instalaciones con distinto `config.yaml` producirían
   vectores incompatibles sin que nada lo detectara.
6. **El canal `z` se sigue calculando** a lo largo de toda la tubería y se
   descarta en el paso 6. Reactivarlo como spec v2 es cambiar una línea.

### Detalles que el contrato no fijaba y aquí se cierran

Al implementar aparecieron tres huecos. Se resolvieron y se anotaron en
`feature-spec.md`, porque un hueco resuelto en silencio es una discrepancia
Python/TypeScript imposible de rastrear tres semanas después:

- **σ usa desviación poblacional (ddof = 0).** σ no es una feature y por lo tanto
  no viaja en los golden vectors por frame: una discrepancia aquí sería invisible
  para los tests y aparecería como ventanas rechazadas en el navegador que en
  escritorio pasaban. Los `sequence_cases` del archivo de golden vectors incluyen
  σ como valor esperado, precisamente para cerrar ese agujero.
- **Mapeo de índices del remuestreo:** `pos_j = (j · (T_src − 1)) / (T_ref − 1)`,
  con el producto **antes** que la división. La forma correcta hace un solo
  redondeo —el numerador es un entero exacto— y la alternativa
  `(j / (T_ref − 1)) · (T_src − 1)` hace dos, con lo que `pos_j` puede caer del
  lado equivocado de un entero. Con `T_ref = 24` y `T_src` entre 2 y 200 las dos
  formas difieren en 970 índices y en 7 de ellos cambia el `floor()`. La §3.2
  documenta el detalle.
- **Inicialización del suavizado:** `p̃_0 = p_0`. Arrancar en cero inventaría un
  movimiento desde el origen del encuadre que nadie ejecutó.

**Lo que quedó fuera de este contrato.** La velocidad que consume la máquina de
estados se calcula en `features.py`, porque necesita los intermedios geométricos
del §1, pero **no está bajo `FEATURE_SPEC_VERSION`**: la versiona
`SEGMENTATION_SPEC_VERSION` y la define la §6. La razón y el riesgo que arrastra
—que ese umbral no puede usarse para enrutar letras dinámicas— están en
`docs/adr/0004-contrato-de-segmentacion.md`.

## Alternativas consideradas

**1. Conservar `z`: 63 componentes.**
Descartada para v1. `z` es una estimación ruidosa e inconsistente entre cámaras;
con un dataset pequeño aporta varianza sin señal confiable. Se puede reactivar
como v2 sin rediseñar nada, y por eso el canal se sigue calculando.

**2. Normalizar dividiendo por el bounding box de la mano.**
Descartada. El bounding box depende de qué dedos estén extendidos, es decir, de la
propia seña: una A (puño) y una B (mano abierta) tendrían escalas de referencia
distintas, y la normalización acercaría artificialmente vectores que deberían
estar lejos.

**3. Usar la distancia muñeca → punta del dedo medio como escala.**
Descartada por lo mismo: cambia al flexionar el dedo. `p_9` es un nudillo y se
mueve poco.

**4. Espejar el dataset en vez de canonizar la lateralidad.**
Descartada. Duplica el dataset y el tiempo de captura, y no resuelve el caso de
una persona zurda usando un modelo entrenado por diestros: solo lo hace más
probable por fuerza bruta.

**5. Empezar directamente con el bloque derivado del apéndice A** (distancias
punta-muñeca, distancias entre puntas, ángulos de flexión), ℝ⁵⁶.
Descartada para v1, especificada por adelantado. Añade componentes correlacionadas
con las geométricas y complica la reimplementación en TypeScript (`acos` necesita
clamp obligatorio). Se activará si la matriz de confusión muestra que las letras
de puño (A, E, M, N, S) no se separan — es decir, con evidencia, no por si acaso.

**6. Aprender la normalización (una capa de entrada entrenada).**
Descartada. Con dataset pequeño sobreajusta, no es interpretable, y el export a
JavaScript deja de ser trivial. El proyecto prefiere transformaciones explícitas y
depurables.

## Consecuencias

**A favor**

- Las invariancias se verifican, no se suponen: los tests de `test_features.py`
  comparan la misma seña con la otra mano, a dos escalas, en las cuatro esquinas,
  rotada y en tres relaciones de aspecto, y exigen igualdad dentro de `1e-9`.
- El vector es interpretable: cada par de componentes es un landmark en un sistema
  de coordenadas fijo. Depurar es mirar puntos, no activaciones.
- El export a JSON es trivial, que es lo que permite que el modelo corra en el
  navegador sin runtime de ML.

**En contra**

- Se pierde información real: la profundidad y el tamaño absoluto de la mano.
  Si dos letras se distinguieran solo por profundidad, este formato no las separa.
- Dos de las 42 componentes son constantes y no aportan nada al clasificador. Es
  desperdicio deliberado, a cambio de que los índices coincidan con MediaPipe.
- La rotación al eje +Y descarta la orientación absoluta de la mano. Si alguna
  letra del alfabeto se distinguiera **solo** por su orientación en el encuadre,
  este paso las colapsaría. No parece ser el caso en el abecedario de la fuente
  primaria, pero es lo primero que hay que revisar si aparecen confusiones raras
  al llenar el glosario.

**Cómo se cambia**

Cualquier modificación de estos pasos obliga a: actualizar `docs/feature-spec.md`,
incrementar `FEATURE_SPEC_VERSION`, regenerar `tests/fixtures/golden_features.json`
con `make golden`, reentrenar todos los modelos y escribir un ADR nuevo. Los
modelos exportados con otra versión se rechazan al cargarse
(`classifiers/base.py::check_export_compatibility`), nunca se ejecutan: un modelo
que corre con la normalización equivocada no falla, solo acierta menos, y eso no
se detecta en una demo.
