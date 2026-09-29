# ADR 0019 — El cierre del trazo en X y Q, y la cascada tras un «demasiado largo»

- **Estado:** **propuesta**, sin implementar.
- **Fecha:** 2026-09-28
- **Datos:** `data/diagnostico/2026-09-28-193646-habitual/` (prueba de reposo
  ampliada: estática, K, X y Q, 10 s cada una, 28–29 fps, commit `017ad4c`) y
  los diagnósticos en vivo `2026-09-28-154523-habitual` y
  `2026-09-28-185100-lampara` (ADR 0017). Fuera del repositorio.
- **Relacionadas:** ADR 0015 (camino dinámico), ADR 0017 (velocidad contra una
  ventana, v5), ADR 0018 (marca de tiempo por cuadro)

## Contexto

En vivo, la X y la Q casi no salen (ADR 0017): el trazo no se cierra porque la
mano nunca acumula `motion_confirm_low_ms` (667 ms) de reposo, termina en
`DYNAMIC_TOO_LONG`, y tras eso la máquina no admite otro trazo hasta ver ese
mismo reposo, así que los intentos siguientes ni llegan a candidato.

La hipótesis de partida era temblor de MediaPipe en posturas que se tapan a sí
mismas, y la propuesta era medir el reposo solo con los puntos estables —muñeca
y nudillos— y con ventanas más largas. La prueba de reposo ampliada midió las
dos cosas sobre la misma grabación.

## Lo que se midió

### La X no tiembla 30 veces más: su escala se colapsa

Con la mano quieta, la velocidad actual (21 puntos, ventana de 100 ms, §6.1.1):

| postura | p50 / p95 / máx (u/s) | cuadros ≥ 0.60 |
|---|---|---|
| estática (A) | 0.11 / 0.18 / 0.25 | 0% |
| K | 0.08 / 0.12 / 0.15 | 0% |
| **X** | **4.05 / 10.43 / 17.26** | **100%** |
| Q | 0.57 / 1.15 / 1.93 | 45% |

MediaPipe no cambió de lateralidad (280 de 280 cuadros `RIGHT`) y los puntos se
mueven en la imagen lo normal para una mano quieta. Lo anómalo es el divisor. La
velocidad del §6.1 se divide entre `s`, la escala del paso 4: la distancia 2D
muñeca → nudillo 9. En la X ese segmento apunta casi a la cámara y su proyección
se encoge:

| postura | `s` (muñeca → nudillo 9) | tamaño de palma¹ | `s` / palma | desplazamiento crudo por cuadro, p50 |
|---|---|---|---|---|
| estática | 0.209 | 0.226 | 0.93 | 0.0027 |
| K | 0.272 | 0.288 | 0.95 | 0.0024 |
| **X** | **0.034** | 0.204 | **0.17** | 0.0118 |
| Q | 0.151 | 0.228 | 0.66 | 0.0091 |

¹ El mayor de las distancias 2D muñeca → nudillos 5, 9, 13 y 17, y nudillo 5 →
nudillo 17, tras el paso 2.

La X divide entre una escala ~6 veces menor, y además sus puntos sí tiemblan ~4
veces más en la imagen: juntos dan el ~30× medido. La Q tiene lo mismo en menor
grado. **Los puntos estables no lo arreglan**: se dividen entre la misma `s`, y
dan lo mismo (X: 4.13 u/s de p50 con estables contra 4.05 con los 21).

### Con el tamaño de palma como divisor

El tamaño de palma no se colapsa: la palma es aproximadamente plana, y dentro de
ella muñeca → nudillo 9 y nudillo 5 → nudillo 17 son casi perpendiculares, así
que al girarla uno puede encogerse pero no los dos. En la estática y la K es
prácticamente `s` (cociente 0.93–0.95), así que ahí nada cambia.

Reposo, p50 / p95 y fracción de cuadros ≥ `motion_threshold_per_s` (0.60), y la
**racha más larga de reposo continuo** —lo que decide si un trazo puede
cerrarse: hacen falta 667 ms—:

| postura | variante | p50 / p95 (u/s) | ≥ 0.60 | racha más larga |
|---|---|---|---|---|
| X | `s`, 21, 100 ms (hoy) | 4.05 / 10.43 | 100% | **0 ms** |
| X | `s`, 21, 200 ms | 2.03 / 5.77 | 97% | 0 ms |
| X | `s`, estables, 200 ms | 2.23 / 4.94 | 95% | 50 ms |
| X | palma, 21, 100 ms | 0.62 / 1.65 | 52% | 618 ms |
| X | **palma, 21, 150 ms** | 0.40 / 0.98 | 30% | **1501 ms** |
| X | palma, 21, 200 ms | 0.33 / 0.83 | 20% | 1277 ms |
| X | palma, estables, 150 ms | 0.41 / 0.97 | 28% | 1701 ms |
| X | palma, estables, 200 ms | 0.36 / 0.73 | 14% | 1501 ms |
| Q | `s`, 21, 100 ms (hoy) | 0.57 / 1.15 | 45% | 304 ms |
| Q | **palma, 21, 150 ms** | 0.28 / 0.50 | 2% | **3583 ms** |
| Q | palma, 21, 200 ms | 0.22 / 0.41 | 0% | 8907 ms |
| estática, K | cualquier variante | ≤ 0.13 / ≤ 0.26 | 0% | toda la postura |

### Los trazos reales siguen siendo movimiento

Cuadros en DYNAMIC_CANDIDATE de los diagnósticos en vivo (incluyen el reposo final
que cierra el trazo, por eso ninguna fila llega al 100%): p50 de la velocidad y
fracción ≥ 0.60.

| letra | sesión | `s`, 21, 100 ms (hoy) | palma, 21, 150 ms | palma, estables, 150 ms |
|---|---|---|---|---|
| J | lámpara | 1.63 · 69% | 1.54 · 68% | 1.38 · 65% |
| K | lámpara | 1.02 · 59% | 0.97 · 58% | **0.66 · 52%** |
| Ñ | lámpara | 0.70 · 53% | 0.60 · 50% | 0.53 · 48% |
| Z | lámpara | 0.72 · 54% | 0.72 · 55% | 0.62 · 51% |
| X | lámpara | 2.23 · 83% | 1.09 · 65% | 0.94 · 62% |
| Q | lámpara | 0.74 · 55% | 0.64 · 51% | 0.53 · 48% |
| K | habitual | 1.85 · 68% | 1.69 · 68% | **1.06 · 63%** |
| Ñ | habitual | 1.53 · 61% | 1.30 · 58% | 1.04 · 59% |

Con la escala de palma y los 21 puntos, J, K, Z y Ñ pierden a lo sumo ~5% de
cuadros en movimiento. Con los puntos estables la K pierde un tercio de su p50
(0.97 → 0.66 con lámpara): parte de su trazo lo hacen los dedos con la muñeca
casi quieta, y los estables no lo ven.

## Propuesta A — una velocidad de cierre

**Solo para la condición de cierre de DYNAMIC_CANDIDATE.** Dentro del candidato,
`low_run` —los cuadros seguidos de reposo que al llegar a
`motion_confirm_low_ms` cierran el trazo— cuenta con una velocidad de cierre
`c_t` en vez de con `w_t`. Todo lo demás sigue con `w_t` (§6.1.1): arrancar una
racha, pasar a candidato, `moving`, `still_run` y STABLE. El criterio de las
estáticas no cambia; su estabilidad de forma la cubre σ.

```
P       = {(0,5), (0,9), (0,13), (0,17), (5,17)}
m_t     = max_{(a,b) ∈ P} ‖ q_{t,a} − q_{t,b} ‖₂        # tamaño de palma, 2D, tras el paso 2
k_c     = frames_from_ms(closing_window_ms, fps)        # 150 ms → 4 cuadros a 28 fps
j       = t − min(k_c, |B| − 1)
c_t     = [ mean_i ‖ q_{t,i} − q_{j,i} ‖₂ / ((m_j + m_t) / 2) ] / (t − j)   # 21 puntos
```

y `c_t` se compara con el mismo `motion_threshold_per_s / fps`.

- **Los 21 puntos, no los estables.** Los datos no respaldan los estables como
  palanca: con la escala de palma ganan 200 ms de racha en la X (1501 → 1701 ms
  a 150 ms) y cuestan un tercio de la señal de la K. En el cierre eso es justo
  el riesgo: que un tramo lento hecho con los dedos cuente como reposo y parta la
  letra, como pasó con la K en el replay de la v5.
- **150 ms.** Es la ventana más corta con la que la X llega a cerrar (618 ms de
  racha con 100 ms, 1501 con 150). 200 ms no mejora la X (1277 ms) y añade
  arrastre: la ventana sigue viendo movimiento hasta ~200 ms después de parar,
  y eso se suma a la latencia de cada dinámica.
- **El mismo umbral (0.60).** Con `c_t` la estática y la K quietas quedan en
  ≤ 0.13 de p95, lejos de 0.60; un umbral aparte sería otro parámetro sin datos
  que lo distingan.
- **Contrato:** `feature-spec.md` §6.1.2 nuevo y §6.7; `SEGMENTATION_SPEC_VERSION`
  6. `FEATURE_SPEC_VERSION` no cambia: la escala del paso 4 y las features siguen
  igual. Parámetro nuevo en `config.yaml`: `segmentation.closing_window_ms`
  (150). El conjunto `P` es parte de la definición, en el contrato.

**Lo que no arregla.** La X sigue siendo marginal: en 9 s de reposo intencional
hubo **una** racha de más de 667 ms. Sus puntos tiemblan 4 veces más en la imagen
y eso es MediaPipe con el índice en gancho, no la escala. Si en vivo la X sigue
sin cerrar tras esto y el Bloque 2, queda la alternativa de la ventana deslizante
con spotting por DTW del ADR 0017, cuyo criterio ya está escrito.

## Propuesta B — salir de la cascada sin la condición que falló

Hoy, tras `DYNAMIC_TOO_LONG` la máquina marca `exhausted` y no deja nacer otra
racha hasta que:

1. se confirme el reposo (`low_run ≥ motion_confirm_low_frames`), o
2. llegue un cuadro sin mano: cualquier hueco lo limpia (`segmentation.py`,
   donde el hueco vacía el buffer).

La vía 2 ya cubre «la mano sale y vuelve a entrar». La cascada ocurre cuando la
mano sigue detectada todo el tiempo y nunca reposa: justo la X y la Q, donde la
vía 1 es la condición que acaba de fallar.

**Se propone una tercera vía, una espera fija:** `exhausted` se limpia también al
pasar `segmentation.motion_exhausted_ms` desde el rechazo, lo que ocurra antes.

- **1000 ms, provisional.** En la sesión con lámpara los intentos de X y Q
  estaban separados 2.5–3.5 s: cualquier espera menor de ~1.5 s deja pasar el
  siguiente. Se convierte a cuadros con la tasa congelada, como los demás.
- **Lo que protegía `exhausted` se conserva.** Existe para que alguien
  gesticulando sin parar no entre y salga de DYNAMIC_CANDIDATE una y otra vez.
  Con la espera, eso da como mucho un `DYNAMIC_TOO_LONG` cada `motion_max_ms` +
  1 s (5 s), cada uno rechazado y sin emitir nada.
- **Con el Bloque 2** un hueco corto se va a interpolar en vez de romper el
  trazo. Ese hueco no debe limpiar `exhausted`: la vía 2 quedaría solo para los
  huecos que no se cosen. Se decide con el Bloque 2.

## Verificación al implementar

1. Tests sintéticos: una mano quieta con `s` colapsada cierra el trazo con `c_t` y
   no con `w_t`; un tramo lento hecho solo con los dedos sigue contando como
   movimiento; tras un `DYNAMIC_TOO_LONG` sin reposo ni hueco, a los
   `motion_exhausted_ms` puede nacer otra racha.
2. Eval de Fase 2 idéntico: no toca el camino estático.
3. Replay de Fase 5: se reporta el cambio, como en la v5. `c_t` cambia el cierre
   de todas las dinámicas, y el replay sigue midiendo contra grabaciones a ~16 fps.
4. Después, el Bloque 2 y volver a medir en vivo.

## Decisiones abiertas

1. **La escala de palma para toda la velocidad y no solo para el cierre.**
   También impediría que la X y la Q quietas arranquen rachas por su velocidad
   inflada. Pero cambia `still_run` y STABLE, que se pidió no tocar. Se deja
   para después de medir A.
2. **El valor de `motion_exhausted_ms`.** 1000 ms sale de la separación entre
   intentos de una sesión; en uso real puede pedir otro.
