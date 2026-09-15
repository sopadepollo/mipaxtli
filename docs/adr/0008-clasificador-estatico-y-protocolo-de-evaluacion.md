# ADR 0008 — Clasificador estático y protocolo de evaluación de la Fase 2

- **Fecha:** 2026-09-08
- **Estado:** aceptado
- **Contexto de fase:** Fase 2 (`docs/ARQUITECTURA.md` §5)

## Contexto

La Fase 2 entrega `static_knn`, `lsm-train` y `lsm-eval` con matriz de confusión.
Al implementarla hubo que cerrar siete decisiones que `ARQUITECTURA.md` dejaba
abiertas, y una de ellas contradice una expectativa razonable sobre el barrido de
calibración. Todas son reversibles con coste, que es el criterio de `CLAUDE.md`
para escribir un ADR.

## Decisiones

### 1. El alcance de la Fase 2 son 21 estáticas + `NONE`, y vive en un solo sitio

`lsm.evaluation.PHASE2_LABELS`. Las ocho dinámicas (`J`, `K`, `LL`, `Ñ`, `Q`,
`RR`, `X`, `Z`) se descartan al cargar, se cuentan y se reportan.

`static_knn` promedia los frames de la secuencia con la agregación del
`feature-spec.md` §2. El promedio de una Z no es una configuración de mano: es una
mancha que aterrizará sobre alguna estática y ensuciará su fila de la matriz. Una
matriz de confusión ilegible no se mira, y una matriz que no se mira no calibra
nada.

**El filtro no vive dentro del clasificador.** `StaticKnnClassifier` es agnóstico a
las etiquetas: clasifica lo que se le entrene. Meter la regla de alcance dentro
duplicaría la lista en el clasificador y en el evaluador, y la Fase 5 tendría que
deshacerla en los dos sitios.

### 2. Centroides por clase, no k-NN sobre el dataset

El modelo tiene que viajar a un navegador móvil como JSON y ejecutarse ahí sin
dependencias. Un k-NN de verdad obliga a enviar el dataset completo y a recorrerlo
en cada frame; 22 centroides de 42 componentes son unos 10 KB y la predicción son
22 restas. Con un dataset pequeño y clases compactas, la diferencia de precisión
pesa menos que depurar por qué el celular va a 4 fps.

Si la matriz de confusión demuestra que no basta, la tercera implementación entra
por el mismo `Protocol` sin tocar nada más — que es para lo que existe.

### 3. Tres puertas de rechazo, no una

`ARQUITECTURA.md` §4.4 pide "umbral de confianza". Al implementarlo se ve que
resuelve dos fallos distintos y hace falta separarlos:

1. **σ > `quality.max_dispersion`** — la ventana no era estable. Se decide antes de
   mirar centroide alguno: el vector promedio de una ventana temblorosa no
   representa ninguna configuración, y la distancia a los centroides no significa
   nada.
2. **d₁ > `static_knn.max_distance`** — no se parece a ninguna letra. Una mano
   saludando cae lejos de las 22 clases, pero *alguna* será la menos lejana.
3. **confianza < `static_knn.min_margin`** — dos centroides igual de cerca. Es el
   caso de las confundibles del §4.8: una duda no es una letra.

Un solo umbral no puede cubrir (2) y (3): son magnitudes distintas, una absoluta y
una relativa, y un dataset puede fallar por cualquiera de las dos por separado.

### 4. La confianza es `d₂ / (d₁ + d₂)`

Con d₁ ≤ d₂ las dos distancias menores. Vale 0.5 con dos clases empatadas y tiende
a 1 cuando la más cercana gana con holgura.

- **Frente a un softmax**: no introduce una temperatura que calibrar y se
  reimplementa en JavaScript en una línea.
- **Frente a `1 − d₁/d_max`**: esa forma depende de la escala absoluta de las
  distancias, que cambia al cambiar de métrica, y obligaría a recalibrar
  `segmentation.min_confidence` cada vez.

Vive en `[0.5, 1]`, la misma escala que `segmentation.min_confidence`, para que las
dos se puedan leer juntas: `min_margin` es el piso del clasificador y
`min_confidence` el piso, más alto, que la máquina de estados exige para escribir.

### 5. `lsm.evaluation` es un módulo puro aparte de `cli/evaluate.py`

Por el mismo motivo que `capture.py` se separó de `cli/capture.py` en la Fase 1
(ADR 0006): la parte que se equivoca en silencio es la que hay que poder testear.
Los splits, la matriz, el contraste de hipótesis y el barrido se ejercitan enteros
en CI con secuencias sintéticas; `cli/evaluate.py` solo lee el glosario y escribe
archivos.

`ARQUITECTURA.md` §3 no listaba ni este módulo ni `io/corpus.py`. Se añaden y se
anotan ahí, igual que se hizo con `capture.py`.

### 6. Con un solo firmante, `lsm-eval` se niega

Leave-one-signer-out necesita al menos dos personas. Con una sola, el comando sale
con error y ofrece `--protocolo leave-one-session-out` como degradación explícita,
que queda escrita en la cabecera del reporte.

**No degrada solo.** Una métrica por sesión y una por persona miden cosas
distintas —generalizar a otro día contra generalizar a otra mano— y la diferencia
se pierde en cuanto el número sale del reporte. Nunca hay split aleatorio de
frames: eso mezclaría frames de la misma grabación entre train y test
(`CLAUDE.md` §6).

### 7. Corpus sintético cuando `data/raw` está vacío

`data/raw` no tiene todavía ninguna grabación. Sin esta decisión, el barrido, la
matriz y el contraste de hipótesis se escribirían a ciegas y se estrenarían el día
que existan muestras, que es el peor momento para descubrir que el reporte no
compila.

`lsm.synthetic.synthetic_samples` genera un corpus determinista de 22 clases con
tres firmantes simulados. **No se parece a LSM**, y todo lo que se deriva de él va
marcado: un aviso en la cabecera del reporte, otro dentro del propio modelo
exportado —los dos archivos circulan por separado—, y un tercero en la sección de
contraste de hipótesis, porque una "confirmación" ahí es una coincidencia entre la
rejilla de flexiones del generador y la anatomía real, no evidencia sobre el
glosario.

`--sin-sintetico` exige dataset real. El día que `data/raw` tenga grabaciones, el
sintético deja de usarse sin tocar una línea de código.

## La contradicción con lo que se esperaba del barrido

El barrido de calibración recorre `trajectory_weight`, `max_dispersion`,
`velocity_threshold` y los umbrales nuevos. **Dos de esos cuatro ejes no pueden
mover el accuracy del camino estático**, y conviene dejarlo escrito porque la
expectativa contraria es razonable:

- `features.trajectory_weight` solo pondera el canal de trayectoria
  (`feature-spec.md` §3.3). `static_knn` consume la agregación del §2, que no lo
  toca.
- `segmentation.velocity_threshold` lo consume la máquina de estados para decidir
  cuándo una ventana está quieta. En la evaluación las ventanas llegan ya
  recortadas y etiquetadas: la máquina de estados no interviene.

**Se barren igualmente**, por dos motivos. Primero, para que la inercia sea una
medición y no una afirmación: `trajectory_weight` está en
`evaluation._EXTRACTION_PATHS`, así que el barrido vuelve a extraer las features de
verdad al cambiarlo, y que el resultado salga idéntico lo demuestra. Segundo, para
que nadie tenga que volver a preguntárselo dentro de seis meses.

De ahí `evaluation.STRUCTURALLY_INERT` y la distinción del reporte entre **inerte
por construcción** y **plano en este corpus**. Un eje plano puede serlo por dos
motivos muy distintos —porque no toca este camino, o porque el corpus es demasiado
fácil y ningún umbral llega a discriminar— y explicar el segundo con el argumento
del primero sería decir algo falso. El reporte además **grita** si un eje de
`STRUCTURALLY_INERT` llegara a mover una métrica: significaría que el camino
estático cambió o que hay un error, y en cualquiera de los dos casos el barrido no
sirve para calibrar hasta aclararlo.

Lo que sí calibra esos dos umbrales es la **sección de diagnóstico empírico** del
reporte: las distribuciones reales de σ, velocidad y longitud de arco por clase.
Hasta ahora sus valores eran, literalmente, "un punto de partida razonado, no
medido" (`config.yaml`), y esa sección es la que los sustituye por un número.

## Trazabilidad

Todo lo que se escribe lleva la huella del corpus: un SHA-256 del **contenido**,
más el commit del repositorio. Es lo que hace verificable la frase "estos umbrales
se calibraron contra este dataset". Un `calibracion-fase2.json` cuya huella no
coincida con el dataset actual está describiendo otra cosa.

**Nada lleva marca de tiempo**, y es deliberado: el criterio de aceptación de la
fase es que `make eval` sea reproducible byte a byte, y un `now()` lo rompería en
el primer segundo. Lo que identifica una ejecución son propiedades de la entrada,
no el instante en que se apretó *enter*.

## Consecuencias

- `config.yaml` gana la sección `static_knn` con tres campos. Se validan con
  Pydantic como todo lo demás (`CLAUDE.md` §5).
- `FEATURE_SPEC_VERSION` **no cambia**: nada de esto toca la extracción de
  features, así que los golden vectors y los modelos anteriores siguen valiendo.
- `StaticKnnClassifier.from_export` existe para que el export se pueda recargar y
  se compruebe en un test. Es la prueba ejecutable de que el archivo está
  completo: si al recargar hiciera falta algo que no viaja dentro, la app web
  tampoco lo tendría y el fallo aparecería en la Fase 7.
- El export lleva `smoothing_alpha` en `params` aunque no decida nada. Sin él la
  reimplementación de JavaScript no puede reconstruir el vector de entrada
  (`ARQUITECTURA.md` §4.5).
