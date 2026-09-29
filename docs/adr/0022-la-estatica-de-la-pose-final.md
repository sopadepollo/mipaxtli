# ADR 0022 — La estática que se cuela es la pose final, no la inicial

- **Estado:** **propuesta**, sin implementar. Cambia el plan de la «retracción de
  precursores» y toca `min_confidence` del camino dinámico, que se pidió no
  bajar antes del barrido del Bloque 6: requiere decisión.
- **Fecha:** 2026-09-28
- **Datos:** `data/diagnostico/2026-09-28-210935-habitual/` y
  `2026-09-28-212049-lampara/` (esta con una linterna; quien firmó pide no
  tomarla como referencia), commit `1b9120c`. Fuera del repositorio.
- **Relacionadas:** ADR 0015 (camino dinámico), ADR 0016 (clasificador dinámico),
  ADR 0021 (Bloque 2)

## Contexto

Con luz habitual K y Z ya salen (10 de 10). Las confusiones restantes son la
dinámica leída como su forma estática: J→I, K→P, Ñ→N, Z→L. El plan era una
**retracción de precursores**: si llega una dinámica dentro de N ms de su
estática precursora, en el mismo tramo de mano, la estática se reemplaza.

## Lo que se midió

### La estática va **después** del trazo, no antes

En cada intento de J, K, Ñ y Z con la estática colada, qué había pasado antes de
la estática:

| | habitual | linterna |
|---|---|---|
| un trazo que llegó al clasificador y se **rechazó** | 11 de 13 | 17 de 18 |
| nada: la estática fue de verdad la primera (precursora) | 1 | 1 |
| un candidato sin cerrar | 1 | — |

Y la dinámica casi nunca llega detrás de la estática: J 1 de 5 y 1 de 9, K 1 de 1 y
1 de 7, Ñ 1 de 11, Z 0 de 4. **I, P y N no son el principio de J, K y Ñ: son su
pose final.** El trazo llega al DTW, el DTW lo rechaza, y la mano reposa en la
pose final, que el camino estático escribe. Esa caída al estático es a propósito
(ADR 0015): tras un tránsito largo, la letra de llegada tiene que salir.

**La retracción de precursores arreglaría unos 2 casos de ~37.**

### El DTW acierta la letra y la rechaza por margen

Reproduciendo las dos sesiones con el modelo dinámico actual, la plantilla más
cercana de cada trazo rechazado seguido de su estática final:

| | habitual | linterna |
|---|---|---|
| más cercana = la dinámica pedida | Ñ 9, J 4, K 1 | J 9, K 11, Z 3 |
| más cercana = otra | Ñ 1 | J 2, K 1, Z 1 |

Distancias 1.0–3.3 (`max_distance` = 6.0); confianza 0.51–0.64 contra
`min_confidence` = 0.6. El DTW confunde J con K y Z, que están cerca entre sí, y
la confianza del §3.4 —el margen entre la mejor letra y la segunda— queda baja.
Las plantillas son del dataset viejo, a ~16 fps (ADR 0017): el Bloque 4 debería
subir ese margen.

## Propuesta — rescate por pose final

En la capa de deletreo: si la segmentación **rechaza un trazo cuya letra más
cercana es D**, y dentro de `spelling.rescue_window_ms` el camino estático emite
**la pose final de D**, en el mismo tramo de mano (sin `HandLost` entre medias),
se escribe D en lugar de la estática.

- **Mapa de poses finales**, construido con los diagnósticos: J → I, K → P,
  Ñ → N. **Z → L queda fuera** hasta resolver LL (Bloque 5): LL no está en el
  modelo, sus trazos caen más cerca de Z y terminan en L, y el rescate convertiría
  cada LL en Z (9 y 7 veces en las dos sesiones). Entra cuando LL tenga plantilla.
- **`rescue_window_ms` = 500.** Retraso del trazo rechazado a la estática: p50
  46–48 ms, p95 105 / 292 ms, máximo 142 / 402 ms.
- Hace falta que el rechazo del trazo lleve su letra más cercana. Hoy
  `WindowRejected(LOW_CONFIDENCE)` no la lleva: el evento gana la predicción, y la
  demo la pasa a la capa de deletreo.

**Efecto estimado** en las dos sesiones: se convertirían a la letra pedida 14 de
17 estáticas de pose final con luz habitual (J 4 de 5, K 1 de 1, Ñ 9 de 11), y
13 de 16 con linterna (J 7 de 9, K 6 de 7); con Z → L serían 3 más y 16 LL
convertidas en Z.

**Riesgo sobre estáticas legítimas.** Las 410 muestras estáticas del dataset de I,
P, N y L, reproducidas con el movimiento de colocar la mano: **0** convertidas (con
el modelo entrenado sobre todo el dataset, así que es una cota optimista). El caso
de riesgo real son palabras con la dinámica **seguida** de su pose final como
letra propia, cuando el trazo de la dinámica se rechazó: «JI» (*jirafa, jinete,
jitomate*) saldría «J» en vez de «JI»; hoy sale «I». Las palabras con **IJ** (*hijo,
dijo, hija*) no se ven afectadas: el rescate mira lo que viene *después* de un trazo
rechazado, no antes.

**Por qué requiere decisión:** acepta dinámicas con confianza entre 0.5 y 0.6 cuando
la pose final las corrobora. Es relajar `min_confidence` en un contexto concreto,
y se pidió no tocarlo antes del Bloque 6.

## Alternativas

1. **La retracción de precursores, como estaba planteada.** Arregla ~2 casos y
   borraría la I legítima de *hijo, dijo, fijo, hija, lija* cuando la J llega
   detrás dentro de N ms. No se recomienda.
2. **No hacer nada hasta el Bloque 4.** Si las plantillas regrabadas suben el margen
   de J, K y Ñ por encima de 0.6, el trazo se emite y la estática final no llega
   (el cerrojo de la dinámica ya la bloquea tras una emisión). Se puede medir
   antes de decidir: basta repetir el replay de estas dos sesiones con el modelo
   nuevo.
