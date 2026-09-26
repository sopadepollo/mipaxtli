# ADR 0015 — El camino dinámico de la segmentación

- **Estado:** aceptada e implementada el 2026-09-25
- **Fecha:** 2026-09-25
- **Fase:** 5
- **Implementa:** `src/lsm/segmentation.py` (v3), `src/lsm/classifiers/registry.py`,
  `src/lsm/config.py`, `config.yaml`
- **Contrato:** `docs/feature-spec.md` §6.7 — `SEGMENTATION_SPEC_VERSION` 2 → 3
- **Relacionadas:** ADR 0004 (contrato de segmentación), ADR 0013 (la ventana
  mezclada), ADR 0016 (el clasificador dinámico y su barrido)

## Contexto

Hasta la v2 la máquina solo entregaba una ventana cuando la mano llevaba
`stable_ms` quieta. Una letra dinámica no cumple eso nunca —el movimiento *es* la
seña—, así que `dynamic_dtw.py` podía quedar perfecto y no recibir jamás una
secuencia real. `ARQUITECTURA.md` §4.2 lo anunciaba como idea («si se detecta
movimiento sostenido con un patrón, la ventana completa se manda al clasificador
dinámico») sin decir qué separa un trazo de un tránsito, cuánto dura, ni cómo se
emite. El `feature-spec` §6.3 lo había dejado escrito como pendiente.

La corrección del ADR 0013 añadía un riesgo concreto: en cada cambio de
dirección de una J o una Z la velocidad cae cerca de cero unos frames, y si eso
alcanza `stable_frames` la máquina clasificaría el freno como letra estática,
partiendo la dinámica por la mitad.

## Lo que se midió antes de decidir

Sobre `v_t` (§6.1) de las 3586 muestras de s01-s03:

- **Las grabaciones corren a unos 30 fps**, no a los 17.8 de la demo: la
  separación entre muestras consecutivas acota la tasa por debajo en 26.4 fps.
  Reproducirlas a la tasa nominal es, por tanto, casi correcto.
- **La Z frena de verdad**: en la traza de `s01/.../Z/020.json` los dos cambios de
  dirección son 6 y 9 frames (~200 y ~300 ms) con `v_t` mayormente bajo 0.02. Con
  `stable_ms = 167` (5 frames) la v2 declaraba STABLE en los dos. El diagnóstico
  era correcto.
- **`v_t` alterna alto/casi cero en pleno trazo.** En la misma Z, frames 7 a 22:
  0.170, 0.020, 0.189, 0.017, 0.163, 0.012… La cámara entrega cuadros
  duplicados. Cualquier criterio de «N frames **consecutivos** en movimiento» no
  dispararía nunca.
- **El temblor de una estática cruza 0.02 a menudo**: `v_t` de las estáticas
  grabadas tiene p50 0.0125, p90 0.0269, p95 0.0366.
- **Las muestras dinámicas duran todas 90 frames**, el tope de captura
  (`dynamic_max_frames`), con movimiento de principio a fin: son «los primeros
  3 s tras pulsar la tecla», y algunas terminan con el trazo todavía en marcha.

Con eso se simuló el detector sobre las grabaciones (umbral de movimiento ×
frames móviles para entrar × reposo para cerrar). Extracto; la tabla completa se
reproduce con el script del apéndice:

| `motion_threshold` | entrar | cerrar | estáticas → candidato | dinámicas partidas | Z partidas | J partidas |
|---|---|---|---|---|---|---|
| 0.020 | 8 | 12 | 19.7% | 0.1% | 0.0% | 0.0% |
| 0.025 | 8 | 12 | 10.4% | 0.6% | 1.0% | 0.0% |
| **0.025** | **10** | **12** | **5.9%** | **0.6%** | **1.0%** | **0.0%** |
| 0.030 | 10 | 12 | 3.5% | 1.8% | 2.0% | 0.0% |
| 0.035 | 10 | 12 | 2.1% | 3.6% | 2.0% | 0.0% |
| 0.040 | 8 | 10 | 2.9% | 14.7% | 5.0% | 0.0% |
| 0.050 | 8 | 12 | 1.7% | 18.4% | 9.0% | 1.0% |

(«entrar» y «cerrar» en frames a 30 fps. Las filas hasta 0.035 llevan el corte
por quietud estática descrito abajo; las de 0.040 y 0.050 salen de una primera
pasada sin ese corte, y se dejan porque el corte solo puede bajar la columna de
estáticas, no la de partidas.) La tensión es real: un umbral bajo no parte ninguna Z
pero manda al camino dinámico una de cada cinco estáticas por temblor; uno alto
deja las estáticas en paz y parte las Z.

## Decisión

**1. Un camino propio en la máquina, especificado en el §6.7.**

```
TRACKING ──(racha ≥ motion_min)────────> DYNAMIC_CANDIDATE
DYNAMIC_CANDIDATE ──(reposo ≥ confirm)─> DYNAMIC_EMIT ──> EMIT | TRACKING
DYNAMIC_CANDIDATE ──(trazo > max)──────> TRACKING (descarta)
```

**2. Movimiento es la misma `v_t`, contra otro umbral.** No hay segunda métrica.
`motion_threshold ≥ velocity_threshold` se valida al cargar: ningún frame puede
contar a la vez como quietud para STABLE y como movimiento para el candidato.

**3. Frames móviles acumulados, no consecutivos.** Por los cuadros duplicados.
La racha muere por reposo (`motion_confirm_low_ms`) o por quietud estática.

**4. La quietud estática corta la racha.** Si `still_run ≥ stable_frames` fuera
de un candidato, lo anterior fue un tránsito. Sin este corte, los tránsitos
cortos de un deletreo rápido se sumaban hasta parecer un trazo largo; lo fija
`test_los_transitos_cortos_de_un_deletreo_rapido_no_se_suman`, que se comprobó
revirtiendo el corte y viéndolo fallar.

**5. Histéresis: STABLE suspendido mientras dura el candidato**, y un reposo más
largo que los frenos de la Z (`motion_confirm_low_ms ≥ stable_ms`, validado) para
cerrarlo. Es la exclusión mutua que pedía el diagnóstico.

**6. DYNAMIC_EMIT entrega el trazo crudo completo**, desde el frame anterior al
primer par en movimiento hasta el último frame en movimiento. Sin el reposo final
—no es parte de la letra— y sin remuestrear: eso es del clasificador (§3.2).

**7. El origen viaja con la ventana.** `Classify` recibe `(Sequence,
WindowOrigin)`; el registry obedece la ruta y no vuelve a medir el movimiento.
Correr ambos clasificadores y quedarse con la mayor confianza queda solo como red
para un caso que la máquina no produce: para `L`/`LL` y compañía el estático
ganaría con la mano equivocada (`glosario-lsm.md`, pares abiertos).

**8. Tres aristas que el diagnóstico no pedía y que la implementación necesitó.**

- *La pose final de una dinámica no es una letra.* La J termina en la mano de la
  I. Tras una emisión dinámica, TRACKING no promueve a STABLE hasta que la mano
  se mueva (`dynamic_lock`, liberado igual que `pending_repeat`). Sin esto, cada
  J sostenida escribía «ji».
- *Un hueco descarta el trazo entero* (`DYNAMIC_INTERRUPTED`): coserlo
  inventaría un movimiento que nadie observó (§0.3).
- *Tras descartar un candidato por largo* no nace otro hasta que la mano repose;
  si no, alguien gesticulando entraría y saldría de DYNAMIC_CANDIDATE sin fin.

**Valores** (`config.yaml`, medidos arriba):

| campo | valor | a 30 fps | a 17.8 fps |
|---|---|---|---|
| `motion_threshold` | 0.025 | — | — |
| `motion_min_ms` | 333 | 10 frames | 6 frames |
| `motion_confirm_low_ms` | 400 | 12 frames | 7 frames |
| `motion_max_ms` | 4000 | 120 frames | 71 frames |

`motion_max_ms` va por encima de los 3 s de trazo continuo de las grabaciones.

**No se tocó** ningún umbral calibrado en las Fases 2 y 3: ni `velocity_threshold`,
ni `stable_ms`, ni σ ni `max_dispersion`, ni los cooldowns. La interacción con el
camino nuevo no dio evidencia para hacerlo (ver «Reproducción», abajo).

## Reproducción por la máquina de estados

`lsm-eval-dinamico`, sección 4: cada grabación pasa por `run_segmentation` con
modelos entrenados sin su firmante, seguida de reposo.

| letra | grabaciones | un solo trazo | partidas | perdidas |
|---|---|---|---|---|
| J | 100 | **100** | 0 | 0 |
| Z | 100 | **100** | 0 | 0 |
| K | 121 | 120 | 1 | 0 |
| Ñ | 100 | 100 | 0 | 0 |
| Q | 100 | 100 | 0 | 0 |
| X | 101 | 101 | 0 | 0 |

621 de 622 dinámicas llegan al clasificador en un solo trazo; **ninguna J ni
ninguna Z se parte**. Qué letra sale después es cosa del clasificador y está en
el ADR 0016.

Las 2585 estáticas, con el camino dinámico y sin él (la v2):

| | |
|---|---|
| entraron a candidato dinámico | 153 (5.92% — la simulación predijo 5.9%) |
| emitieron algo por el camino dinámico | 7 (0.27%) |
| salida distinta a la v2 | 9 (0.35%) |
| primera letra correcta | 0.8139 con v3, 0.8147 con v2 |

Esa diferencia —dos muestras— es todo el costo medido en las estáticas.

Para que `lsm-demo --desde-dataset` pueda enseñar dinámicas, la reproducción
repite el último frame de cada muestra `motion_confirm_low + 1` frames antes del
hueco entre muestras: las grabaciones terminan con el trazo en marcha, y sin ese
reposo el hueco llegaba en pleno movimiento y descartaba el trazo como
interrumpido. Es lo que hace quien termina una seña: detenerse antes de bajar la
mano.

## Consecuencias

- **Latencia de una letra dinámica:** sale `motion_confirm_low_ms` (400 ms)
  después de que la mano se detiene. Es el precio de no partir la Z.
- **Latencia tras un tránsito largo.** Un viaje de más de ~333 ms de movimiento
  entre dos estáticas se vuelve candidato; el DTW lo rechaza y la letra de
  llegada sale en el frame siguiente, pero tras 400 ms de quietud en vez de 167.
  El texto no cambia; la espera sí. Lo fija
  `test_un_transito_largo_pasa_por_el_camino_dinamico_y_la_letra_sale_igual`.
- **Tests que cambiaron y por qué.** `tests/test_cli_demo.py` usa la
  configuración de fábrica con viajes de 800 ms, que ahora son candidatos: el
  criterio de la Fase 3 (cinco letras, cinco símbolos) se sigue cumpliendo
  igual, pero `QUIETO` se recalculó con la espera real de llegada
  (`motion_confirm_low_frames` en vez de `stable_frames`) para que la «segunda
  oportunidad» de cada letra siga cayendo dentro de su bloque, y los rechazos de
  la ruta dinámica se separan de los estáticos. El test de latencia estática usa
  un viaje corto. En `tests/test_segmentation.py` la configuración comprimida
  aparta el camino dinámico de los tests del estático; los del dinámico llevan
  la suya.
- **El eje `segmentation.velocity_threshold` del barrido de la Fase 2** tenía un
  valor de 0.05, que la validación nueva rechaza (supera `motion_threshold`). El
  eje es inerte por construcción para el camino estático, así que su tope pasó a
  0.025 sin cambiar nada medible.
- `SEGMENTATION_SPEC_VERSION` pasa a 3; `FEATURE_SPEC_VERSION` no cambia y ningún
  modelo se invalida. Por este ADR los golden vectors solo cambian en ese
  número; sus `dynamic_rows` cambian además por el peso calibrado del ADR 0016.

## Lo que queda abierto

- **La I antes de la J.** Si quien firma sostiene la mano de la I antes de
  trazar la J, el camino estático escribe una I antes de que el trazo empiece.
  Nada en la máquina puede saber que una forma sostenida era el preludio de un
  trazo sin retrasar todas las estáticas. Lo mismo vale para L→LL, N→Ñ y R→RR.
  Si en vivo resulta frecuente, la salida razonable es en `spelling.py`: que una
  dinámica emitida justo después de su estática confundible la reemplace.
- **No hay grabaciones de tránsitos ni de negativos dinámicos.** Cuánto de un
  deletreo real pasa por el candidato solo se mide en vivo o grabando tránsitos
  como `NONE` dinámica.
- **`motion_threshold` depende de la tasa**, con la misma deuda que
  `velocity_threshold` (§6.5).
- **Las grabaciones dinámicas están truncadas a 3 s.** El tope de captura cortó
  trazos lentos. Subir `capture.dynamic_max_frames` antes de la Fase 6.
- **Validación en vivo pendiente.** Todo lo de arriba es sobre grabaciones; el
  criterio de la Fase 5 («reconocidas en vivo») lo cierra una sesión con cámara.

## Apéndice — cómo se midió

La simulación del detector y el perfil de velocidades son dos scripts de
análisis que no quedaron en el repositorio; su lógica es la del §6.7 aplicada a
`features.velocities` de cada muestra, y la sección 4 de `lsm-eval-dinamico` la
reproduce con la máquina de verdad.
