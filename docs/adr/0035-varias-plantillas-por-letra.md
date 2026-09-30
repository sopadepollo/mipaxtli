# ADR 0035 — Varias plantillas por letra y firmante en el DTW

- **Estado:** aceptada, 2026-09-30.
- **Relacionadas:** ADR 0016 (clasificador dinámico), ADR 0032 (diagnóstico de la
  J), ADR 0033 (cerrojo de la pose final)

## Contexto

El clasificador dinámico guardaba **una plantilla por (letra, firmante)**: el
medoide DTW del grupo. Con una sola, grabar más J de una persona solo mueve su
medoide: si la J de hoy se ejecuta distinto que la de septiembre, las muestras
nuevas se diluyen entre las viejas y la plantilla sigue siendo la de antes. En
el diagnóstico `2026-09-30-090353` la J salió en 2 de 10 intentos: la plantilla
más cercana era la J en 9 de 10 trazos, con confianza mediana de 0.57 frente a
K y Z.

## Opciones medidas

- **k-medoides por firmante**, con k = 2, 3 y 5: el BUILD de PAM, determinista.
  El primero es el medoide de siempre (con k = 1 el modelo no cambia) y cada
  siguiente es la muestra que más reduce la distancia de su grupo a su medoide
  más cercano. Sin la fase SWAP.
- **Vecino más cercano sobre todas las muestras** (k = 0).

La decisión no cambia: `d_letra = min` sobre las plantillas de la letra y
`conf = d₂ / (d₁ + d₂)` entre las dos letras más cercanas.

### Leave-one-signer-out (635 trazos del corpus, sin puertas)

| plantillas por firmante | total | más cercana correcta | emitidas a 0.6 (correctas) | emitidas erróneas a 0.6 | J: confianza p10 / p50 |
|---|---|---|---|---|---|
| 1 (antes) | 18 | 0.907 | 491 | 0.8 % | 0.62 / 0.80 |
| 2 | 36 | 0.945 | 510 | 0.2 % | 0.67 / 0.80 |
| **3** | 54 | **0.956** | 523 | 0.2 % | 0.69 / 0.82 |
| 5 | 90 | 0.969 | 533 | 0.2 % | 0.69 / 0.83 |
| todas | 635 | 0.986 | 546 | 0.0 % | 0.72 / 0.86 |

### ¿Sigue significando lo mismo el umbral?

La fórmula de la confianza no cambia, pero con más plantillas las distancias
bajan para todas las letras. Se comprobó con los trazos que **no** son una
dinámica (177 trazos de los diagnósticos, de los que se miran los que ocurren en
intentos de estáticas y en las pruebas de reposo): a 0.6 se aceptan el 11.4 %
con una plantilla y el 10.0 % con tres. El umbral que igualaría a la plantilla
única en acomodos aceptados y en emisiones erróneas del LOSO sale en **0.578**
con k = 3. Con 0.6, k = 3 no es más permisiva que antes, sino un poco más
estricta. **`min_confidence` se queda en 0.6.**

### Replay de los diagnósticos (modelos entrenados con cada k, con segmentación y cerrojo)

Intentos con la letra escrita; entre paréntesis, emisiones erróneas.

| sesión | k = 1 | k = 3 | k = 5 |
|---|---|---|---|
| 090353: J | 2 (I 1) | **3 (ninguna)** | 4 (ninguna) |
| 090353: N | 5 (Ñ 1, X 1) | 4 (Ñ 1) | 4 (Ñ 1) |
| 215749: J | 10 (I 4) | 10 (I 4) | 10 (I 4) |
| 191843: J | 3 (I 1) | **4** (I 1) | 4 (I 1) |
| 184727: Ñ / Q / X | 10 / 9 / 6 | 10 (X 1) / 9 / 6 | **7** (X 1) / 10 / 7 |
| 183332: Ñ / Q / X | 9 (N 2) / 4 / 1 | igual | 8 (N 2) / 4 / 1 |
| 220842: X (1 intento largo) | 1 | 0 | 0 |

### Latencia del DTW por trazo

Una vez por trazo, al cerrarlo, en Python y en esta máquina: p50 **13 ms** con
k = 1, **37 ms** con k = 3, 60 ms con k = 5 y 405 ms con todas las muestras.
Con la demo a 25 fps un cuadro dura 40 ms: k = 3 cuesta como mucho un cuadro de
retraso. Todas las muestras detendrían el bucle de la cámara casi medio segundo
en cada trazo.

## Decisión

`dtw.templates_per_signer` = **3** (0 = todas las muestras; 1 = el medoide de
antes). Es la mejora de LOSO sin coste en la Ñ de hoy, que k = 5 sí tiene
(184727 cae de 10 a 7), y a una latencia que la demo aguanta. El umbral no se
toca.

El formato del modelo exportado no cambia: `templates` ya era una lista de
`(label, signer_id, rows)`, ahora con varias entradas por firmante. Un runtime
que lea el JSON no necesita saber `k`.

## Consecuencias

- Hay que reentrenar (`make train`); el modelo nuevo lleva 54 plantillas en vez
  de 18.
- **Grabar más J sirve ahora para algo**: con tres plantillas por firmante, un
  grupo de J nuevas de una persona puede quedarse con uno de sus medoides en vez
  de diluirse en el de septiembre.
- La mejora en los diagnósticos es modesta (J de hoy: 2 → 3 de 10). La J de hoy
  sigue sin parecerse a ninguna grabada: lo que falta son muestras de esa
  ejecución (paso 3).
- La latencia en un navegador móvil no está medida; con 54 plantillas son 3
  veces los DTW de antes.
