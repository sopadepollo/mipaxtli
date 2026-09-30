# ADR 0032 — Por qué la J y la X no se emiten (diagnósticos del 2026-09-29)

- **Estado:** **medición y decisión**, 2026-09-29.
- **Datos:** `data/diagnostico/2026-09-29-183332-habitual` (X, Ñ, Q, sin One Euro),
  `…-184727-habitual` (X, Ñ, Q, One Euro 0.5/1/2) y `…-191843-habitual` (J, One
  Euro 0.5/1/2). No hay una sesión de J sin One Euro. Modelos de `data/models/`
  entrenados a las 18:12 con el One Euro **apagado**.
- **Relacionadas:** ADR 0016 (DTW), ADR 0022 (la estática de la pose final),
  ADR 0028 (One Euro), ADR 0031 (evaluación)

## Síntomas

Ñ y Q salen 10/10 con el One Euro. La X no se emite nunca (siempre
LOW_CONFIDENCE). La J llega como trazo entero en 9 de 10 intentos, pero se
escribe J 2 veces e I 8: el DTW rechaza el trazo y la pose final se escribe como
estática (ADR 0022).

## Lo que se midió

Las sesiones se reproducen con su propia configuración (One Euro de sus
metadatos) y el modelo que se usó en vivo; en las dos sesiones con One Euro las
emisiones del replay coinciden exactamente con las de vivo.

### La X: no hay plantillas

El modelo dinámico tiene 15 plantillas —Ñ, J, K, Q y Z, una por firmante— y
**ninguna de X**: `corpus.exclude_before` deja fuera las 166 X grabadas, todas
anteriores a las 11:35. Cada trazo de X cae en la letra más parecida con
confianza 0.50–0.53, entre Ñ, Q y K, que están casi a la misma distancia (p. ej.
Q 1.99 contra Ñ 2.00). **No es un umbral: la X no puede ganar hasta que se grabe
con la definición nueva.**

### La J: la letra correcta, sin margen

| intento | 1.ª | d | 2.ª | d | confianza | emitida |
|---|---|---|---|---|---|---|
| 1 | Q, luego J | 2.75, 1.16 | Z | 3.34, 2.89 | 0.549, 0.713 | I, J |
| 2 | J | 2.37 | Z | 2.65 | 0.527 | I |
| 3 | J | 2.00 | Z | 2.57 | 0.562 | I |
| 4 | J | 2.25 | Z | 2.63 | 0.539 | I |
| 5 | J | 2.13 | Z | 2.69 | 0.558 | I |
| 6 | J | 1.88 | Z | 2.58 | 0.579 | I |
| 7 | J | 1.51 | Z | 2.58 | 0.631 | J |
| 8 | J | 2.13 | Z | 2.62 | 0.551 | I |
| 9 | J | 1.74 | Z | 2.62 | 0.601 | J |
| 10 | J | 2.29 | Z | 2.76 | 0.546 | I |

En 9 de 10 trazos la plantilla más cercana es la J y la segunda la Z; la
confianza (`d₂ / (d₁ + d₂)`) queda por debajo de `min_confidence` = 0.6 en 7.

**No es la desalineación de preprocesado**, aunque existía: el modelo se entrenó
sin One Euro y la sesión lo usó. Reentrenado con el mismo One Euro, las
confianzas cambian en la tercera cifra (J ≥ 0.6: 3 de 10 con los dos). **Tampoco
son las plantillas nuevas**: con solo las J de hoy, solo las anteriores o ninguna
de hoy, la confianza mediana queda entre 0.55 y 0.59.

**Es la ejecución.** Plantillas de J: una por firmante, medoides de s01 (10 del
09-09 y 42 del 29), s02 (12 del 09-09 y 32 del 29) y s03 (14 del 09-09 y 25 del
29). Todas acaban lejos de donde empiezan: τ final medio s01 (+1.60, −1.65), s02
(+1.09, −1.89), s03 (+0.62, −1.31) palmas. Los trazos del diagnóstico acaban en
(+0.18, −0.12). Ningún firmante se separa de los otros como para explicarlo; los
tres se separan del diagnóstico (1.5–2.4 palmas de separación media de
trayectoria, la más cercana s03).

| sesión | trazos de J | arco de la muñeca | desplazamiento neto |
|---|---|---|---|
| 2026-09-28 15:45 | 15 | 6.41 | 1.69 |
| 2026-09-28 21:09 (J 10/10 enteras) | 10 | 6.61 | 1.98 |
| **2026-09-29 19:18** | 11 | **8.01** | **0.37** |

(Palmas, medianas.) La mano recorre lo mismo o más, pero termina casi donde
empezó. **No lo causó el código**: la sesión de hoy reproducida con la máquina
de antes (sin plausibilidad, sin One Euro, relleno de 150 ms) da neto 0.41, y la
del 28 con la máquina nueva sigue en 1.95. Coincide con el cambio del glosario
(«con el meñique se traza una j»): si la J se traza con el dedo y la muñeca
vuelve, su τ es casi plana y se parece poco a las plantillas, grabadas con la
mano entera bajando.

## Decisiones

1. **Ningún umbral de confianza se toca** (como se pidió): ni para la X, que
   comparte gancho con la Q, ni para la J, porque reintroduciría I → J.
2. **El modelo declara su preprocesado y cargarlo falla si no coincide**
   (`classifiers.base.check_preprocessing`, llamado por la demo): la
   plausibilidad, el relleno y el One Euro del modelo tienen que ser los del
   runtime, igual que `feature_spec_version`. Con esto la sesión de J de hoy se
   habría negado a arrancar con un modelo sin One Euro.
3. **La J hay que decidirla antes de regrabarla**: si la ejecución válida es la
   de hoy, las plantillas tienen que grabarse así; si es la de las plantillas, el
   diagnóstico hay que repetirlo con la mano entera. Hasta entonces el cerrojo
   tras un trazo rechazado (ADR 0033) evita que se escriba la I.
