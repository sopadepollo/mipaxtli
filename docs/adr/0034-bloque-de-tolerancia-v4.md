# ADR 0034 — Cierre del bloque de tolerancia: FEATURE_SPEC 4 con δ

- **Estado:** aceptada, 2026-09-29.
- **Versiones:** `FEATURE_SPEC_VERSION` 3 → **4** y `SEGMENTATION_SPEC_VERSION`
  7 → **8**, una sola vez para todo el bloque, con los golden regenerados una
  vez. Los commits de los pasos anteriores no llevan versión intermedia.
- **Relacionadas:** ADR 0020 (δ con el tamaño de palma, planeado como v4), ADR
  0027 (plausibilidad), ADR 0028 (One Euro), ADR 0029 (huecos), ADR 0031
  (evaluación), ADR 0032 (diagnóstico de J y X), ADR 0033 (cerrojo)

## Qué entra en la v4

| componente | dónde | interruptor |
|---|---|---|
| plausibilidad anatómica | §0.4, `lsm.plausibility` | `plausibility.enabled` |
| filtro One Euro | §4, `lsm.one_euro` | `smoothing.enabled` |
| **δ_t = ln(m_t / m_0)** | §3.1, §3.3, `lsm.features` | `features.depth_weight` (0 apaga) |

Y en la v8 de segmentación, además de los dos filtros sobre el flujo: el relleno
de huecos en TRACKING y STABLE con límite por estado (ADR 0029) y el cerrojo de
la pose final (ADR 0033).

- **δ** es la profundidad relativa que el ADR 0020 dejó planeada: `m_t` es el
  tamaño de palma del paso 4, así que un giro de la palma no se lee como
  acercamiento. Va como **componente 45** de `g_t`, remuestreada aparte y
  ponderada después de interpolar, igual que τ. `w_δ = 0` la apaga **sin
  cambiar el ancho** de `g_t`: con y sin δ se mide con el mismo contrato, el
  mismo golden y el mismo formato de modelo. ρ (el giro) no entra: no se pidió
  y no hay datos que lo pidan.
- El modelo dinámico exporta `depth_weight` junto a `trajectory_weight`; los dos
  modelos declaran su preprocesado (plausibilidad, relleno, One Euro) y el
  runtime rechaza uno que no coincida (`check_preprocessing`, commit 9b122a3).
  Un modelo v3 se rechaza al cargar por la versión: **hay que reentrenar**.
- **Golden:** además de δ en cada caso de secuencia (`depth`) y un caso nuevo
  (`depth_approach`, la mano que se acerca 1.6× sin mover la muñeca), el archivo
  cubre por primera vez la plausibilidad (`plausibility_cases`: un hueso
  estirado, un salto de palma y el reinicio a los 100 ms) y el One Euro
  (`one_euro_cases`: un trazo y un temblor, los dos con dt irregular), con los
  parámetros en cada caso. Es lo que necesita el port a TypeScript.

## Ablación de δ

Medida con el contrato v4 y `w_δ` = 0 contra `w_δ` = 1 (el mismo ancho de `g_t`),
modelos reentrenados para cada variante con el preprocesado de cada réplica.

**Corpus, leave-one-signer-out** (`lsm-eval-dinamico`, One Euro apagado como en
`config.yaml` hoy):

| | sin δ | con δ |
|---|---|---|
| accuracy LOSO (513 muestras) | 0.9045 | 0.9084 |
| replay, aciertos J / K / Ñ / Q / Z | 124 / 53 / 49 / 31 / 103 | 123 / 51 / 49 / 31 / 105 |
| replay, primera letra estática correcta | 0.9234 | 0.9234 |

**Diagnósticos de hoy** (replay con el One Euro de cada sesión en vivo; intentos
con la letra escrita):

| sesión | sin δ | con δ |
|---|---|---|
| 183332 (sin One Euro): Ñ | 9 | 9 |
| 184727 (One Euro): Ñ / Q | 10 / 10 | 10 / 10 |
| 191843 (One Euro): J | 3 | 3 |

La plantilla más cercana de cada trazo no cambia en J, Ñ ni Q; la confianza
media se mueve ±0.015. **La X no se puede medir contra plantillas**: no tiene
ninguna (ADR 0032). Como aproximación, cada trazo de X de una sesión contra las
plantillas del corpus más las *otras* X de la misma sesión (optimista: misma
persona, misma sesión): gana la X en 16 de 16 con y sin δ (184727), confianza
media 0.716 sin δ y 0.713 con δ. **La X ya se separa por la forma**; lo que le
falta son plantillas.

**Por qué δ no aporta aquí.** El rango de δ dentro de un trazo (máx − mín, p50)
no distingue a la X: 0.32–0.42 en la X, 0.30–0.54 en la Q y 0.17–0.32 en la Ñ.
Todas las dinámicas cambian de tamaño aparente al girar o desplazar la mano, y
con una sola persona y una sesión por letra no se ve la curva de ida y vuelta
que la definición nueva de la X pide.

**Barrido de `w_δ`** (One Euro 0.5/1/2; LOSO por plantilla más cercana, sin
puerta):

| w_δ | LOSO | Q LOSO | Ñ en 184727 con conf. ≥ 0.6 | X (aprox.) conf. media |
|---|---|---|---|---|
| 0 | 0.9025 | 60/92 | 11/12 | 0.716 |
| 1 | 0.9006 | 60/92 | 10/12 | 0.713 |
| 2 | 0.9025 | 60/92 | 8/12 | 0.703 |
| 4 | 0.9045 | 59/92 | 8/12 | 0.686 |
| 8 | 0.9220 | 67/92 | 7/12 | 0.662 |

Con más peso sube el acierto LOSO de la Q, pero bajan los márgenes de todas las
letras —la Ñ de hoy pasa de 11 a 7 trazos por encima de 0.6— y el de la X. No
hay un peso que mejore las dos cosas.

**Decisión: `w_δ` = 1.0**, el valor neutro de partida. Con ese peso δ es
prácticamente inerte (dentro del ruido en todo lo medido) y el contrato ya lo
lleva, así que calibrarlo no pedirá otra versión. Se calibra en el barrido del
bloque 6, cuando haya X con la definición nueva de los tres firmantes: es la
única letra para la que se añadió, y hoy no hay con qué medirla.

## Consecuencias

- Todos los modelos de `data/models/` quedan inválidos hasta reentrenar.
- La demo, la captura y la evaluación usan la v4 sin cambios de interfaz.
- `web/` no se toca: su runtime queda en la v3 hasta portar el bloque.
