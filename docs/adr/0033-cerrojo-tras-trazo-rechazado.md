# ADR 0033 — Cerrojo de la pose final tras un trazo rechazado por margen

- **Estado:** aceptada, 2026-09-29. Entra con el bloque de tolerancia en la
  versión siguiente de `SEGMENTATION_SPEC_VERSION` (no se sube por paso).
- **Relacionadas:** ADR 0022 (la estática de la pose final), ADR 0032
  (diagnóstico de la J), feature-spec §6.7

## Contexto

El diagnóstico de la J (ADR 0032) muestra que un trazo de J rechazado por el DTW
—confianza bajo `min_confidence`— devuelve la máquina a TRACKING con `stable_run`
intacto, y el camino estático toma la pose final en el frame siguiente. La J
acaba en la mano de la I: la persona ve **I** donde hizo una J. Lo mismo pasa con
la Ñ rechazada, que acaba en N.

No existe en el repositorio una «guía de regrabación» con un paso 5 que lo
cubriera; se implementa aquí. Tras una dinámica *emitida* ya había cerrojo
(§6.7): la pose final no promueve a STABLE hasta que la mano se mueva.

## Decisión

`segmentation.rejected_stroke_final_poses` (por defecto J → I, K → P, Ñ → N):
tras un trazo rechazado **por margen** —el clasificador dio la letra D, con
confianza bajo `min_confidence`—, la estática `final_poses[D]` no se emite hasta
que un frame supere `velocity_threshold` o se pierda la mano. La ventana se
rechaza con `FINAL_POSE_OF_REJECTED_STROKE`. **Cualquier otra letra sale como
siempre.** Es preferible no emitir nada a emitir I.

- El mapa es el del ADR 0022, medido en los diagnósticos. **Z → L queda fuera**
  por lo mismo que allí: LL no tiene plantilla y sus trazos caen en Z.
- Un trazo **UNKNOWN** (no se parece a ninguna dinámica: un tránsito) o
  descartado sin clasificar (`DYNAMIC_TOO_MUCH_INTERPOLATED`) no pone el cerrojo:
  no hay letra D.
- No es el rescate del ADR 0022: no escribe D, solo deja de escribir su pose
  final. No se toca `min_confidence` ni ningún umbral de confianza.
- Un mapa vacío (`{}`) apaga el cerrojo: es el interruptor para medirlo.

### Por qué no bloquear toda estática

La primera versión ponía el cerrojo de la dinámica emitida —ninguna estática
hasta que la mano se mueva— tras cualquier rechazo por margen. En los
diagnósticos daba lo mismo que la versión final, pero en el replay del dataset
el acierto de la primera letra estática bajaba de 0.9238 a 0.8894: 124 de las
2585 muestras estáticas entran a candidato dinámico con el movimiento de colocar
la mano, el DTW las rechaza por margen con alguna dinámica delante, y su letra
—que no es la pose final de nada— se perdía. Restringirlo a la pose final de la
letra que el DTW tenía delante conserva el beneficio y quita ese coste.

## Medición

Reproducción de las sesiones de diagnóstico de hoy con su propia configuración
(One Euro de sus metadatos) y los modelos que se usaron en vivo; emisiones dentro
de los intentos de J, X, Ñ y Q:

| sesión | sin cerrojo | con cerrojo |
|---|---|---|
| `2026-09-29-191843` (J, One Euro) | J 3, **I 8** | J 3, **I 1** |
| `2026-09-29-184727` (X, Ñ, Q, One Euro) | Ñ 10, Q 10 | Ñ 10, Q 10 |
| `2026-09-29-183332` (X, Ñ, Q, sin One Euro) | Ñ 9, **N 5** | Ñ 9, **N 2** |

En el diagnóstico de la J el cerrojo habría evitado **7 de las 8 I espurias**, sin
perder ninguna J. La I que queda (frame 8099) sigue a un trazo descartado por
`DYNAMIC_TOO_MUCH_INTERPOLATED`: no se clasificó, así que no hay letra D.
Extender el cerrojo a ese rechazo la evitaría, pero un trazo demasiado
reconstruido no dice qué dinámica era; se deja fuera hasta tener más de un caso.
En la Ñ sin One Euro evita 3 de 5 N. Ninguna emisión correcta se pierde.

**Coste en el dataset** (`lsm-eval-dinamico`, replay de las 2585 muestras
estáticas con el movimiento de colocar la mano, modelos leave-one-signer-out):

| | sin cerrojo | con cerrojo | cerrojo de toda estática (descartado) |
|---|---|---|---|
| primera letra correcta | 0.9238 | 0.9234 | 0.8894 |
| salida distinta a la de referencia | 18 | 20 | 114 |

Una muestra de 2585 pierde su letra. Las dinámicas del replay no cambian (J 124,
K 53, Ñ 49, Q 31, Z 103 aciertos en los dos casos).

## Consecuencias

- Una J, K o Ñ rechazada no deja su pose final escrita: la persona repite la
  seña. Es el mismo comportamiento que tras una dinámica emitida.
- Riesgo: «JI» (*jirafa, jinete*) con la J rechazada sale sin nada donde antes
  salía «I»; la I propia necesita un movimiento después del trazo. Con la J
  emitida ya pasaba lo mismo (cerrojo de la dinámica emitida).

## Solo trazos plausibles (2026-09-30)

En el diagnóstico `2026-09-30-090353` (A, N, I, J, Y) la N se escribió en **1 de
10** intentos: el cerrojo la bloqueó 30 veces. Colocar la mano en N es un
movimiento corto con la forma de la N, y el DTW lo lee como una Ñ dudosa; se
rechaza por margen y bloquea su pose final, que es justo la N pedida. En 183332,
en cambio, la regla Ñ → N evita 3 N espurias tras una Ñ. Las dos sesiones se
contradicen, así que el cerrojo se restringe a los trazos que **pudieron ser**
la dinámica, en vez de quitar la regla.

**Qué separa un acomodo de una dinámica real.** Todos los trazos rechazados por
margen de los 22 diagnósticos (912 trazos), con la plantilla más cercana en una
dinámica con regla:

| | reales rechazadas | acomodos (letra pedida estática) |
|---|---|---|
| Ñ: duración | p50 1795 ms; 35 de 38 ≥ 850 ms | 7 en N: **552–843 ms** |
| Ñ: longitud de arco | p50 2.35, desde 0.76 | 1.6–2.4 (se solapa) |
| Ñ: giro de la palma (ρ) | p50 1.17 rad, desde 0.20 | 0.09–0.73 (se solapa) |
| Ñ: distancia DTW d1 | p50 2.77, desde 1.11 | 1.28–2.25 (se solapa) |
| J: duración | p50 2352 ms; 52 de 62 ≥ 700 ms | 2 en I: 457 y 1181 ms |

Solo la **duración** separa: arco, giro y distancia se solapan. Las J y K reales
más cortas son de las sesiones del 26 y 27, a baja tasa (trazos de 4 a 8
cuadros). No hay acomodos a P en los datos (ninguna sesión pidió la P).

**Decisión:** `segmentation.final_pose_lock_min_stroke_ms` = **1000**, para las
tres reglas. El cerrojo solo se pone si el trazo rechazado duró al menos eso; 0
lo pone siempre. Entre 700 y 1000 ms los resultados son idénticos en las
sesiones medidas; 1000 deja más margen sobre el acomodo más largo (843 ms).

Replay con el modelo actual (intentos con la letra escrita; entre paréntesis, las
estáticas espurias):

| sesión | sin cerrojo | cerrojo sin mínimo | cerrojo ≥ 1000 ms |
|---|---|---|---|
| 090353: N | 5 | 1 | **5** |
| 090353: I | 10 | 9 | 10 |
| 090353: J (I espurias) | 2 (6) | 2 (1) | 2 (1) |
| 183332: Ñ (N espurias) | 9 (5) | 9 (2) | 9 (**2**) |
| 184727: Ñ / Q / X | 10 / 9 / 6 | igual | igual |
| 215749: J (I espurias) | 10 (5) | 10 (3) | 10 (4) |
| 191843: J (I espurias) | 3 (8) | 3 (1) | 3 (1) |

Se recuperan 4 N en 090353 sin que reaparezcan las N espurias de 183332. El
coste es una I en 215749: una J de 544 ms, más corta que el mínimo.

