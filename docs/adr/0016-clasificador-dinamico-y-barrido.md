# ADR 0016 — El clasificador dinámico y su barrido

- **Estado:** aceptada e implementada el 2026-09-25
- **Fecha:** 2026-09-25
- **Fase:** 5
- **Implementa:** `src/lsm/classifiers/dynamic_dtw.py`,
  `src/lsm/classifiers/registry.py`, `src/lsm/evaluation.py` (sección Fase 5),
  `src/lsm/cli/evaluate_dynamic.py`, `src/lsm/cli/train.py`
- **Contrato:** `docs/feature-spec.md` §3.4 (precisado, sin cambio de versión);
  `tests/fixtures/golden_features.json` regenerado por el nuevo `w_τ` por defecto
- **Relacionadas:** ADR 0015 (el camino dinámico de la segmentación),
  `docs/glosario-lsm.md` PENDIENTE-HUMANO F (la Z y la banda) e I (LL y RR)

## Contexto

`ARQUITECTURA.md` §4.3 fijó DTW contra plantillas desde el principio, y el
`feature-spec` §3 la entrada: 24 filas de `g_t = concat(f_t, w_τ · τ_t)`. Faltaba
todo lo que convierte eso en un clasificador: qué es una plantilla, cómo se pasa
de una distancia a una confianza y a un UNKNOWN, y con qué `w_τ` y qué banda.
Los dos valores de `config.yaml` —`trajectory_weight = 4.0`, `band_radius = 6`—
eran puntos de partida razonados, y el glosario llevaba desde su redacción el
aviso de que la banda podía quedarle corta a la Z (PENDIENTE-HUMANO F), riesgo
que la Fase 2 no pudo medir porque solo evaluó estáticas.

## Decisión

**Plantillas: el medoide DTW de cada (letra, persona).** Una muestra real, no un
promedio: promediar trazos con ritmos distintos da un trazo que nadie ejecutó.
Una por persona porque cada quien traza a su manera. Con el dataset de la Fase 1
son 18 plantillas (6 letras × 3 personas).

**Distancia: la del §3.4, ahora especificada al bit.** Recurrencia, orden de
desempate (diagonal, vertical, horizontal; gana solo lo estrictamente menor),
normalización por la longitud del camino elegido y suma sin `sum()`. El párrafo
original no bastaba para que TypeScript diera los mismos bits.
`FEATURE_SPEC_VERSION` no cambia: ningún modelo exportado usaba la sección.

**Decisión: la de `static_knn`, reutilizada.** Distancia a la letra = mínimo
sobre sus plantillas; confianza `d₂ / (d₁ + d₂)` entre las dos letras más cercanas
(`static_knn.rank`); **UNKNOWN si `d₁ > dtw.max_distance`**. Sin esa puerta DTW
siempre tiene una plantilla menos lejana, y cualquier movimiento que se cuele al
camino dinámico saldría como letra. El piso que decide si se escribe sigue
siendo `segmentation.min_confidence`, igual que en el camino estático.

**Export** (`ARQUITECTURA.md` §4.6): las plantillas con su persona, y en `params`
todo lo necesario para reproducir entrada y decisión en el navegador:
`band_radius`, `max_distance`, `trajectory_weight`, `min_source_frames`,
`resample_length`, `smoothing_alpha`.

**LL y RR, bloqueadas.** Su dirección canónica está pendiente de decisión humana
(glosario, PENDIENTE-HUMANO I, con la medición de que las tres personas grabaron
los dos sentidos). `vocabulary.DIRECTION_PENDING_LABELS` impide construir sus
plantillas, `lsm-train` las cuenta, y un modelo que las traiga se rechaza al
cargarse. El código no elige un sentido.

## El barrido

`lsm-eval-dinamico --sin-sintetico` (rejilla completa), dataset `fe0ada8710b6`,
622 muestras de 6 letras, leave-one-signer-out, commit `23ff9f1` más los cambios
de esta fase. Accuracy del vecino más cercano **con la puerta abierta**: la
puerta depende de `w_τ` y mezclarla aquí compararía otra cosa.

**Todas las letras:**

| | banda 2 | 4 | 6 | 8 | 12 | 23 |
|---|---|---|---|---|---|---|
| w_τ = 0 | 0.783 | 0.772 | 0.783 | 0.802 | 0.797 | 0.793 |
| **w_τ = 1** | 0.786 | 0.781 | **0.801** | 0.802 | 0.804 | 0.804 |
| w_τ = 2 | 0.760 | 0.762 | 0.770 | 0.773 | 0.775 | 0.775 |
| w_τ = 4 (antes) | 0.715 | 0.723 | 0.738 | 0.735 | 0.735 | 0.735 |
| w_τ = 8 | 0.686 | 0.627 | 0.638 | 0.632 | 0.632 | 0.638 |
| w_τ = 16 | 0.616 | 0.613 | 0.601 | 0.596 | 0.585 | 0.584 |

**La Z, aparte** (PENDIENTE-HUMANO F):

| | banda 2 | 4 | 6 | 8 | 12 | 23 |
|---|---|---|---|---|---|---|
| w_τ = 0 | 0.970 | 0.960 | 0.960 | 0.960 | 0.960 | 0.960 |
| **w_τ = 1** | 0.980 | 0.980 | **0.980** | 0.980 | 0.980 | 0.980 |
| w_τ = 2 | 0.890 | 0.890 | 0.890 | 0.890 | 0.890 | 0.890 |
| w_τ = 4 | 0.830 | 0.830 | 0.850 | 0.830 | 0.830 | 0.830 |
| w_τ = 8 | 0.710 | 0.520 | 0.510 | 0.490 | 0.490 | 0.490 |
| w_τ = 16 | 0.620 | 0.470 | 0.440 | 0.420 | 0.430 | 0.430 |

**Por letra con `w_τ = 1`, banda 6:** Ñ 0.86 · J 1.00 · K 0.89 · Q 0.51 · X 0.55 ·
Z 0.98. Las tablas por letra de toda la rejilla salen en la sección 2 de
`lsm-eval-dinamico` con la rejilla completa (copia de esta ejecución en
`data/models/eval/reporte-fase5-rejilla-completa.md`, fuera del repositorio
como todo `data/`).

### Lectura

1. **`w_τ` pasa de 4.0 a 1.0**, en `config.yaml` y como valor por defecto del
   código (el proyecto exige que coincidan). Los golden vectors se generan con
   el valor por defecto, así que se regeneraron: cambian solo sus `dynamic_rows`
   y el peso que declaran en su bloque `config`, comprobado campo a campo. La
   fórmula del §3.3 no cambia y `FEATURE_SPEC_VERSION` tampoco: el peso es un
   parámetro que el modelo exporta, no parte del formato.

   Es el eje que mueve el resultado: +6.3 puntos en total y +13 en la Z. Por encima de 2 todo empeora, y fuerte: el trazo de
   personas distintas varía más que la forma de su mano, y cuanto más pesa τ más
   pesa esa variación. La intuición de partida —«sin ponderar, el trazo queda
   invisible»— era correcta **para los pares que el glosario anota** (`I`/`J`,
   `L`/`LL`…), pero esos pares cruzan de camino: la I nunca compite con la J
   dentro del DTW, porque una llega por STABLE y la otra por DYNAMIC_EMIT. Dentro
   del conjunto dinámico las formas ya difieren, y la forma es la señal más
   estable entre personas.
2. **`band_radius` se queda en 6.** Entre las bandas 6 y 23 la diferencia es de
   0.3 puntos, un par de muestras. No hay evidencia para moverlo.
3. **El riesgo de la Z no se materializó.** Con `w_τ = 1` la Z da 0.98 con
   **cualquier** banda, incluida la de 2. El aviso del glosario suponía que el
   trazo largo exigiría deformaciones temporales grandes; con el remuestreo a 24
   filas el trazo entero se normaliza en duración y la Z resulta la segunda letra
   más fácil. PENDIENTE-HUMANO F puede cerrarse.
4. **Q y X son el problema**, y no se arregla con estos dos ejes: ninguna casilla
   de la rejilla las lleva por encima de 0.65. Las dos son «garra» de índice y
   pulgar y se distinguen por la orientación de la palma y el movimiento; el
   paso 5 del §1 rota la mano para alinear la palma y descarta z, así que parte
   de esa diferencia se pierde en la normalización. Es el mismo diagnóstico que
   el glosario anotó para X/Q en su antiguo PENDIENTE-HUMANO C, ahora medido.

### `dtw.max_distance`

Con `w_τ = 1`, banda 6:

| | p5 | p50 | p90 | p95 | p99 |
|---|---|---|---|---|---|
| d₁ de los aciertos (LOSO) | 0.75 | 1.34 | 2.87 | 3.66 | 6.17 |
| d₁ de las 2585 estáticas grabadas | 1.28 | 2.41 | 3.12 | 3.20 | 3.35 |

**La distancia no separa estáticas de dinámicas.** Cualquier umbral que conserve
el 90% de los aciertos deja pasar el 72% de las estáticas; con el 95%, todas. Una
estática remuestreada a 24 filas es una secuencia casi constante con la forma de
la mano, y contra plantillas de otras personas queda tan cerca como un trazo
auténtico.

Por eso **`max_distance = 6.0`** —el p99 de los aciertos— y no más apretado: es
una puerta gruesa contra lo que no se parece a nada, y apretarla solo perdería
letras buenas. La separación de verdad la hacen otras dos cosas, y las dos se
midieron:

- **La segmentación**: solo el 5.9% de las estáticas grabadas llega al camino
  dinámico (ADR 0015).
- **El margen**: con `segmentation.min_confidence = 0.6` —el de la Fase 3, sin
  tocar— pasan el 52.8% de los aciertos y el 0.8% de los fallos.

## Resultado con los valores elegidos

`w_τ = 1.0`, `band_radius = 6`, `max_distance = 6.0`, leave-one-signer-out:

| letra | DTW (sección 1) | emitida en la reproducción por la máquina (sección 4) |
|---|---|---|
| J | 1.00 | 100 / 100 |
| Z | 0.98 | 46 / 100 |
| K | 0.89 | 68 / 121 |
| Ñ | 0.85 | 33 / 100 |
| Q | 0.51 | 11 / 100 |
| X | 0.51 | 2 / 101 |
| **total** | **0.793** (UNKNOWN 1.8%) | 260 / 622 |

La diferencia entre las dos columnas es `segmentation.min_confidence = 0.6`
aplicado al margen: el DTW acierta la Z casi siempre, pero con un margen sobre
la segunda letra que no llega a 0.6 en la mitad de los casos. Ninguna letra se
partió ni se perdió en la segmentación (ADR 0015); lo que no sale, no sale por
confianza.

## Consecuencias

- **J y Z son reconocibles; Q y X no.** La J sale siempre; la Z, cuando sale, es
  la Z. Q y X quedan como están documentadas arriba: no es un problema de umbral
  sino de lo que el vector de 42+2 componentes conserva de la orientación de la
  palma. Candidatos, por orden de costo: activar el apéndice A del
  `feature-spec` (ángulos de flexión), recuperar la orientación que el paso 5
  descarta —cambio de spec v2, con su ADR—, o grabar más personas (Fase 6).
- **No se tocó `segmentation.min_confidence`.** Bajarlo para las dinámicas
  subiría la Z y la K, pero es un umbral calibrado en la Fase 3 para las dos
  rutas, y separar un piso dinámico es una decisión con su propia medición —
  cuántos fallos pasan a cambio—, que con tres personas y sin negativos
  dinámicos grabados no se puede hacer bien. Queda propuesto: el margen de los
  fallos no supera 0.60 en el p99, y el de los aciertos tiene su mediana en 0.61.
- **Precisión de las estáticas.** La evaluación de la Fase 2 da exactamente lo
  mismo (0.9261, macro 0.9201, UNKNOWN 0.0573): es offline y no pasa por la
  máquina. Reproducidas por la máquina, 7 de 2585 estáticas emitieron algo por
  el camino dinámico y la primera letra correcta baja de 0.8147 a 0.8139 (2
  muestras).
- **Costo en vivo**: 18 plantillas × ~0.6 ms por DTW ≈ 11 ms por trazo, una vez
  por letra dinámica. La evaluación completa tarda ~70 minutos en Python puro,
  que se acepta a cambio de que la implementación de referencia sea la misma que
  corre en producción.
- **LL y RR** siguen fuera hasta PENDIENTE-HUMANO I. Al resolverse: vaciar
  `DIRECTION_PENDING_LABELS`, reentrenar, volver a correr este reporte.
- **Validación en vivo pendiente.** Todo lo anterior es sobre grabaciones con
  leave-one-signer-out. El criterio de la Fase 5 lo cierra una sesión con
  cámara.
