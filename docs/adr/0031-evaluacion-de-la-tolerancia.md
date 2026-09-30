# ADR 0031 — Evaluación de la serie de tolerancia (Paso 6)

- **Estado:** **medición**, 2026-09-29. Sin cámara: el barrido de umbrales de
  MediaPipe (ADR 0030) y la X nueva quedan fuera.
- **Datos:** el dataset (huella `7f401543b3f6`, 3221 muestras en la Fase 2; sin
  ninguna X, ADR 0024 y `corpus.exclude_before`) y las 16 sesiones de
  `data/diagnostico/`. Reportes: `docs/mediciones/2026-09-29-paso6-ablacion.md`.
- **Relacionadas:** ADR 0026 a 0030

## Qué se comparó

Las variantes de `lsm-medir ablacion` (`cli/measure.py::VARIANTES`):

- **antes**: sin plausibilidad, sin One Euro, relleno solo dentro del trazo con
  150 ms (la tubería de antes de la serie);
- **+plausibilidad**, **+one_euro** (`0.5,1,2`, el candidato 1 del ADR 0028),
  **+relleno** (200 / 200 / 100 ms): cada componente solo;
- **plaus+relleno**: lo que queda activo por defecto;
- **todo**: además el One Euro.

## 1. Estáticas (eval de la Fase 2, leave-one-signer-out)

| variante | accuracy | macro | UNKNOWN | mejor punto del barrido |
|---|---|---|---|---|
| antes | 0.9288 | 0.9240 | 0.0549 | 0.9602 |
| plaus+relleno | 0.9288 | 0.9240 | 0.0549 | 0.9598 |
| todo | 0.9280 | 0.9230 | 0.0545 | 0.9594 |

**No empeoran.** La plausibilidad y el relleno no cambian ninguna estática; el One
Euro les quita una décima.

## 2. Segmentación en vivo, reproducida (ablación)

Cada sesión de diagnóstico reproducida por la máquina con cada variante, a la
tasa medida al arrancarla y con su reloj real. El replay de la variante «antes»
reproduce exactamente lo que la máquina hizo en vivo (comprobado en la sesión del
2026-09-28 21:09: los mismos 90 trazos y los mismos intentos enteros por letra).
**Intento entero**: un solo trazo entregado y ningún corte.

| letra (intentos) | antes | +plaus. | +One Euro | +relleno | plaus+relleno | todo |
|---|---|---|---|---|---|---|
| X (93) | 12 | 12 | **19** | 12 | 12 | **19** |
| Ñ (101) | 39 | 39 | 37 | **41** | **41** | 39 |
| Q (111) | 5 | 5 | **9** | 5 | 5 | **8** |
| K (119) | 26 | 26 | 26 | 26 | 26 | 26 |
| J (147) | 29 | 29 | 32 | 31 | 31 | **33** |
| Z (96) | 29 | 30 | 31 | 29 | 29 | 31 |

- **Lo que explica la mejora es el One Euro**: X +7, Q +4, J +3, y menos intentos
  sin ningún trazo (X 71 → 65, Q 94 → 90). En la Ñ resta 2.
- **El relleno ampliado** apenas: Ñ +2, J +2. Rellena de verdad (en una sesión,
  109 huecos del trazo contra 57, y 90 en TRACKING), pero un trazo cortado que
  vuelve a nacer ya se contaba casi siempre como entregado; lo que cambia es que
  llega completo en vez de a pedazos (columna «tras corte» del reporte).
- **La plausibilidad**, nada: invalida ≤ 1.7% de los cuadros (ADR 0027).
- **Los umbrales de MediaPipe** no se pueden medir sin cámara (ADR 0030).

### Por qué se pierden los intentos (plaus+relleno)

| causa | X | Ñ | Q |
|---|---|---|---|
| el trazo no se cerró: demasiado largo, o abierto al acabar el intento | 31 | 22 | 37 |
| cortado por un hueco más largo que el que se rellena | 25 | 16 | 26 |
| sin candidato, con mano: no se detectó movimiento | 15 | 5 | 9 |
| sin candidato, sin mano la mayor parte del intento | 0 | 0 | 22 |

**La causa principal no son los huecos: es que el trazo no cierra.** El reposo
final tiembla por encima de `motion_threshold_per_s` (0.60): con la mano quieta,
el 17% de los cuadros ya supera `velocity_threshold_per_s` (ADR 0028). Por eso el
One Euro, que baja ese temblor, es lo único que ayuda, y por eso la K y la Z,
cuyas pérdidas son otras, no se mueven.

## 2b. Dinámicas grabadas (replay de `lsm-eval-dinamico`, rejilla rápida)

Las 513 dinámicas del corpus (sin X), clasificadas y reproducidas por la máquina
con modelos entrenados sin su firmante. Aquí las muestras ya vienen enteras
(casi ninguna tiene huecos), así que lo que se ve es el efecto del filtrado:

| | antes | plaus+relleno | todo (One Euro) |
|---|---|---|---|
| accuracy / macro del DTW | 0.9045 / 0.8899 | 0.9045 / 0.8899 | 0.9025 / 0.8870 |
| aciertos en el replay: Ñ / J / K / Q / Z | 49 / 124 / 53 / 30 / 103 | 49 / 124 / 53 / 31 / 103 | 48 / **112** / 51 / 31 / 104 |
| trazos partidos o perdidos | 3 | 3 | **12** (K 3, Q 4, Z 3, Ñ 2) |
| estáticas que entran a candidato | 4.80% | 4.80% | **3.68%** |
| primera letra estática correcta | 0.9238 | 0.9238 | 0.9246 |

**El One Euro tiene un precio en el reconocimiento**: la J pierde 12 aciertos
(el gancho se suaviza) y aparecen trazos partidos. A cambio, menos estáticas
tiemblan hasta el camino dinámico. Con la ablación en vivo (sección 2) el balance
queda así: más trazos que cierran en X y Q, peor J reconocida. Ninguno de los
dos números sale del mismo protocolo, y ninguno usa la X nueva.

## 3. Intentos rechazados que pasan a ser trazos completos

**No hay ninguno que re-segmentar**: el Paso 0 empieza a guardarlos con la
próxima sesión de captura. El equivalente más cercano son los intentos de los
diagnósticos de la sección 2: de los que la máquina de antes perdía o partía, la
configuración por defecto recupera enteros 2 de la Ñ y 2 de la J; con todo activo
(One Euro incluido), 7 de la X, 3 de la Q y 4 de la J, y ninguno de la Ñ.

## Consecuencias

- El One Euro es el único componente que mueve la segmentación en vivo, pero
  cuesta aciertos en la J grabada, y la calibración no dio un punto que cumpla
  temblor y retraso a la vez (ADR 0028): **sigue apagado** y se decide a ojo con
  `lsm-demo --comparar-one-euro` y con estas dos tablas.
- El siguiente cuello de botella es el **cierre del trazo**: el umbral de reposo
  frente al temblor de la pose final. Es segmentación, no MediaPipe.
- Los huecos largos siguen siendo un tercio de las pérdidas: Paso 5.
