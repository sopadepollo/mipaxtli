# ADR 0026 — Mediciones de tolerancia a MediaPipe (Paso 1)

- **Estado:** **medición**, 2026-09-29. Fija los umbrales de los ADR 0027
  (plausibilidad), 0028 (One Euro) y 0029 (relleno ampliado).
- **Datos:** las 16 sesiones de `data/diagnostico/` (26 al 28 de septiembre,
  guiadas y pruebas de reposo) y las muestras del dataset: dinámicas del
  2026-09-29 y estáticas A de todas las fechas. Fuera del repositorio.
- **Reportes completos:** `docs/mediciones/2026-09-29-paso1-tolerancia.md`
  (`lsm-medir tolerancia`) y `docs/mediciones/2026-09-29-paso2-plausibilidad.md`
  (`lsm-medir plausibilidad`). Solo números, sin landmarks.
- **Relacionadas:** ADR 0017 (diagnóstico de tracking), ADR 0021 (huecos)

## Cómo se leen

- **·diag** son las sesiones en vivo. **La X de los diagnósticos es la de la
  definición anterior** (de perfil): la final es de las 11:35 del 2026-09-29, y
  no hay ninguna X grabada con ella todavía (ADR 0024, `corpus.exclude_before`).
  **·hoy** son las muestras dinámicas del 2026-09-29 (sin marcas de tiempo: el
  reloj es el índice a 30 fps).
- **Fases del intento**: *antes* (la mano llegando), *durante* (desde
  `pre_candidate_ms` antes de entrar a DYNAMIC_CANDIDATE hasta salir), *final*
  (la pose tras el trazo).
- **MediaPipe Tasks no expone el score de presencia**: solo el de la
  lateralidad, que es el que se reporta.

## Lo que se midió y qué decide

### Huecos → límites del relleno (ADR 0029)

Huecos que empiezan dentro del trazo, fracción acumulada:

| | ≤100 | ≤150 | ≤200 | ≤250 | ≤300 | ≤500 |
|---|---|---|---|---|---|---|
| X (diag) | 0.51 | 0.59 | 0.64 | 0.66 | 0.69 | 0.76 |
| Ñ (diag) | 0.49 | 0.54 | 0.61 | 0.61 | 0.63 | 0.73 |
| Q (diag) | 0.41 | 0.45 | 0.51 | 0.53 | 0.53 | 0.60 |
| K (diag) | 0.17 | 0.20 | 0.22 | 0.24 | 0.28 | 0.37 |

Cada 50 ms más recupera 4–8 puntos hasta 200 ms, y 0–3 después: el codo está en
**200 ms**, para el trazo y para TRACKING (los huecos que empiezan en TRACKING en
X, Ñ y Q: 0.52 / 0.58 / 0.61 a 150 / 200 / 250 ms, el mismo codo). La K apenas
gana: sus pérdidas son largas.

En la **pose sostenida** no hubo huecos en STABLE (las sesiones eran de letras
dinámicas). La pose final da 0.31–0.36 hasta 100 ms y luego queda plana hasta
~450 ms (Ñ 0.31 → 0.38, K 0.36 constante): **100 ms** para STABLE.

La tasa de detección *durante* el trazo en vivo fue 0.66 (X), 0.78 (Ñ), 0.58
(Q), 0.62 (K); en las muestras aceptadas de hoy, ≥ 0.996.

### Score antes de cada pérdida → Paso 5

p10 / p50 del score en el cuadro previo a un hueco contra todos los válidos:
X 0.68 / 0.97 contra 0.92 / 0.99; Ñ 0.81 / 0.96 contra 0.96 / 0.99; K 0.86 /
0.99 contra 0.99 / 1.00; Q 0.83 / 0.96 contra 0.83 / 0.95. **El score cae antes
de una parte de las pérdidas (la cola), pero la mediana apenas se mueve**: la
mitad de las pérdidas llega sin aviso. Bajar `min_hand_presence_confidence` puede
recuperar las primeras; las otras son pérdidas del rastreador.

### Huesos → filtro de plausibilidad (ADR 0027)

Tres formas de medir la variación del largo de los 20 huesos, peor hueso por
cuadro, p99:

| | relativa, mediana de sesión | relativa, mediana móvil 15 | **absoluta 3D, móvil 15** |
|---|---|---|---|
| A estática | 1.19 (3D) | 0.43 | **0.042** |
| reposo K | 0.32 | 0.085 | **0.016** |
| X de hoy | 0.93 | 2.74 | **0.34** |
| Q de hoy | 0.73 | 0.80 | **0.26** |
| X (diag) | 1.02 | 2.44 | **0.44** |

- La **mediana de sesión** mezcla poses: MediaPipe no conserva el largo 3D de los
  huesos entre poses (una mano quieta en la K se aleja 0.3 de la mediana de su
  sesión). Descartada.
- La **relativa** la domina el hueso más corto (falanges distales). Descartada.
- **Absoluta en 3D contra la mediana móvil**: ningún cuadro limpio (hoy, A,
  reposo) pasa de 0.45; p99.9 de la X de hoy 0.41. En diagnósticos lo pasan el
  0.8% de X y el 0.9% de K. **Umbral 0.45 palmas.** 3D es mejor que 2D en todas
  las letras (la proyección de un hueso cambia al girar; su largo 3D, menos).
- **Rachas de rechazo** (sin reinicio): 58% de 1 cuadro, 84% ≤ 3, y cola larga
  a partir de 4 — cambios de forma persistentes, sobre todo en la K, que la
  mediana móvil tarda en alcanzar. **Reinicio de la referencia tras 100 ms.**

### Ángulos → sin comprobación articular

Elevación dorsal de la falange proximal sobre el plano de la palma: en poses
limpias llega a +47° (Ñ de hoy), +45° (K de hoy), +83° (Q de hoy) y +84° (reposo
de la J), todas imposibles en una mano real. Con la palma de canto la normal de
la palma es casi toda `z`, la coordenada menos fiable de MediaPipe. Con un ángulo
con signo alrededor del eje de los nudillos pasa lo mismo, y además las falanges
dobladas saltan de ±180°. **No hay umbral que atrape algo sin tirar poses
buenas**: la comprobación queda implementada y desactivada.

### Saltos de la palma → plausibilidad

Centro de la palma entre cuadros válidos seguidos, palmas por segundo: en limpio
p99.9 ≤ 17 y máximo 21 (más un cuadro de 200 en la Q de hoy); en diagnósticos,
la fracción por encima de 25 y de 100 casi no cambia (X 0.73% → 0.42%, Q 1.07% →
0.60%): una meseta de saltos rotos. **Umbral 30.** Medir contra una base de
≥ 100 ms, como la velocidad de la máquina, no separa mejor.

### Temblor en reposo → One Euro (ADR 0028)

Velocidad de cada dedo relativa a la muñeca, ventana de 100 ms, p50 / p95 en
palmas por segundo: estática 0.14 / 0.32, K 0.10 / 0.20, Q 0.45–0.53 /
0.9–1.4, X (definición vieja) 0.83–0.88 / 2.8–3.2. Es la línea base contra la
que se calibra el filtro. Con `motion_threshold_per_s` = 0.60, el p95 de la X
quieta ya cuenta como movimiento.

## Consecuencia para la lectura de los síntomas

El filtro de plausibilidad con estos umbrales invalida ≤ 1.7% de los cuadros en
cualquier letra (1.63% X diag, 1.32% J diag, 1.24% Q diag, 1.05% K del dataset;
las estáticas, ≈ 0). Lo que en pantalla parece movimiento
inhumano es sobre todo **temblor**, que es cosa del One Euro, no cuadros
imposibles.
