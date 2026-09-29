# ADR 0021 — Tolerancia a huecos en el camino dinámico (Bloque 2)

- **Estado:** **implementada** el 2026-09-28 (SEGMENTATION_SPEC_VERSION 7).
  `dynamic_max_gap_ms` y `dynamic_max_interpolated_fraction` son **provisionales**
  hasta el barrido del Bloque 6.
- **Fecha:** 2026-09-28
- **Fase:** 5.1, Bloque 2
- **Datos:** los diagnósticos en vivo `2026-09-28-154523-habitual` y
  `2026-09-28-185100-lampara` (ADR 0017). Fuera del repositorio.
- **Relacionadas:** ADR 0004 (contrato de segmentación), ADR 0015 (camino
  dinámico), ADR 0017 (diagnóstico de tracking), ADR 0019 (cierre y cascada)

## Contexto

El contrato decía que un frame inválido interrumpe la secuencia (`feature-spec.md`
§0.3): coser dos tramos inventaría un movimiento que nadie observó. En el camino
dinámico eso tiraba trazos enteros por perder la mano uno o dos cuadros, y en
vivo es de lo más frecuente: la K se partía en 9 de 12 intentos por huecos a
mitad del trazo, y la X tenía 13 huecos en 10 intentos (ADR 0017).

## Lo que se midió

Los huecos que empiezan dentro de DYNAMIC_CANDIDATE y terminan con la mano de
vuelta, en las dos sesiones en vivo con la v5 (82 huecos):

| duración | huecos | acumulado |
|---|---|---|
| ≤ 100 ms (1–3 cuadros) | 39 | 48% |
| ≤ 150 ms | 40 | 49% |
| ≤ 200 ms | 42 | 51% |
| ≤ 300 ms | 46 | 56% |
| ≤ 500 ms | 51 | 62% |

p25 / p50 / p75: 48 / 190 / 770 ms. **La distribución es bimodal**: la mitad son
pérdidas de uno a tres cuadros y el resto son pérdidas largas, de cientos de ms,
en las que interpolar inventaría media letra. Pasar de 100 a 500 ms solo añade 14
puntos. Por letra, los huecos ≤ 200 ms son 15 de 20 en la X, 6 de 11 en la Q,
9 de 24 en la J y **2 de 12 en la K**.

## Decisión

Se implementa el Bloque 2 del plan, con estas precisiones.

### El contrato

Excepción a «los inválidos interrumpen», **solo en el camino dinámico**
(`feature-spec.md` §0.3 y §6): dentro de DYNAMIC_CANDIDATE y en las muestras
dinámicas guardadas, una racha de inválidos de como mucho `dynamic_max_gap_ms`
entre dos frames válidos se rellena interpolando linealmente los **landmarks
crudos**, `a + (b − a) · t` con `t = k / (n + 1)`. Se interpolan landmarks y no
features para que la forma, τ y los canales del Bloque 3 salgan consistentes.

- **La mano de los dos extremos tiene que coincidir.** Se compara la mano
  **declarada**, que es la que canoniza el paso 2 desde `FEATURE_SPEC_VERSION` 2.
  El plan decía «la lateralidad reportada», pero la etiqueta de MediaPipe cambia
  de opinión a mitad de un trazo y desde el ADR 0017 no decide nada: exigir que
  coincidiera abortaría justo los huecos de la J. En una sesión en vivo la
  declarada es la misma en todos los frames; la condición protege los flujos que
  mezclan manos y los cambios de resolución, que tampoco se interpolan.
- **Los huecos del borde se recortan** (en las muestras); **uno más largo que el
  máximo aborta**, como antes; **si la fracción interpolada supera el máximo**, la
  secuencia se rechaza, y en vivo el trazo se descarta sin clasificar
  (`RejectionReason.DYNAMIC_TOO_MUCH_INTERPOLATED`).
- **Los frames rellenados** heredan la resolución y la mano del frame anterior,
  llevan como scores el menor de los dos extremos —un frame inventado no es más
  fiable que los reales— y no tienen `detected_handedness`.
- **El camino estático no cambia.**

### Los valores, provisionales

- **`dynamic_max_gap_ms` = 150.** Cubre el grupo de pérdidas cortas —tres
  cuadros aun a 22 fps— y deja fuera las largas. La K, cuyos huecos son largos,
  **no se beneficia**: se seguirá partiendo cuando pierda la mano ~300 ms.
- **`dynamic_max_interpolated_fraction` = 0.25.** Un trazo de 20 cuadros admite
  cinco rellenados.
- Viven en `segmentation` y no en una sección `dynamic` como decía el plan: la
  máquina de estados es la que tiene el camino dinámico (`motion_*` están ahí), y
  las muestras las leen de la misma sección.

### Cómo lo hace la máquina en vivo

La máquina no sabe al perder la mano si el hueco va a ser corto. Dentro de
DYNAMIC_CANDIDATE **retiene** los frames inválidos sin procesarlos; si la mano
vuelve antes de `max_gap` cuadros y se puede interpolar, procesa los rellenados
—con el índice de los frames que reemplazan— y después el real. Si no, procesa
los inválidos tal cual y el trazo se corta como siempre. Mientras hay un hueco
abierto la máquina va hasta 150 ms por detrás; fuera de un trazo, nunca.

Un hueco rellenado no llega a la máquina como hueco, así que no limpia
`exhausted` (ADR 0019): solo lo hace uno que no se rellena.

### Muestras y eventos

- `WindowDynamic` y `LetterEmitted` llevan `interpolated_frames`; el diagnóstico
  lo registra y la sección 5.1 del reporte dice por letra cuántos trazos llegaron
  rellenados.
- `types.Sample.interpolated_frames` se **deriva al cargar**, no se guarda: la
  muestra en disco conserva los huecos tal como ocurrieron, y el número sale de
  ahí sin cambiar el esquema. La Fase 6 puede filtrar por él.
- `StoredSample.to_sample(gaps)` reconstruye una dinámica si se le pasa la
  política; sin política, o en una estática, cualquier hueco sigue siendo un
  error. `lsm-train`, `lsm-eval`, `lsm-eval-dinamico` y `lsm-capture verificar`
  pasan la política a la tasa nominal, la de las grabaciones.
- `lsm-capture grabar` en modo dinámico ya no rechaza una ventana con huecos que
  `to_sample` podrá reconstruir, y la mide sobre la reconstruida.

### Golden

`golden_features.json` gana el bloque `gap_cases` con los cinco casos del plan:
hueco corto en medio (se rellena), hueco largo (se rechaza), huecos en el borde
(se recortan), cambio de mano a través del hueco (se rechaza) y exceso de
fracción interpolada (se rechaza). Cada caso lleva la entrada, la política y la
salida frame a frame.

## Verificación

- **Eval de Fase 2 idéntico**: 0.9261 / 0.9201 / 0.0573, barrido 0.9578.
- **Replay de Fase 5 idéntico al de la v6** salvo la versión y el commit: 611
  trazos enteros, 248 aciertos, estáticas 0.927. Es lo esperado, porque ninguna
  muestra del dataset tiene huecos (ver «Lo que no resuelve»).
- Tests: la máquina rellena un hueco corto y entrega el trazo entero; el hueco más
  largo que se rellena es exactamente `max_gap`; un cambio de mano no se
  rellena; un trazo demasiado reconstruido no llega al clasificador dinámico;
  fuera del candidato un hueco sigue interrumpiendo.

## Lo que no resuelve

- **Los huecos largos**, la otra mitad. Son pérdidas reales de tracking —la mano
  rápida, o de canto— y rellenarlas inventaría la letra. Se miden de nuevo en vivo.
- **El replay no lo ejercita**: las muestras del dataset se grabaron cuando la
  captura rechazaba cualquier hueco, así que ninguna tiene. Lo que dice si el
  Bloque 2 sirve es el diagnóstico en vivo.
