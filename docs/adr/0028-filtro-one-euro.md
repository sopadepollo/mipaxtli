# ADR 0028 — Filtro One Euro en lugar del suavizado fijo (Paso 3)

- **Estado:** **implementado, apagado por defecto** (2026-09-29). **Ningún punto
  del barrido cumple a la vez los dos criterios**; no se elige uno. Queda para
  decidir a ojo con `lsm-demo --comparar-one-euro` y con la ablación del Paso 6.
  Entra en `FEATURE_SPEC_VERSION` 4 (§4) con la plausibilidad y δ.
- **Fecha:** 2026-09-29
- **Datos:** reposo: las 9 posturas de las pruebas de reposo de los diagnósticos
  (estática, J, K, Q, X de la definición anterior), con reloj real y sin el
  segundo de asentamiento. Trazos: 78 J, 61 Ñ y 64 Q grabadas el 2026-09-29 (reloj
  nominal: esas muestras no llevan marca de tiempo). Reporte completo:
  `docs/mediciones/2026-09-29-paso3-one-euro.md` (`lsm-medir one-euro`).
- **Relacionadas:** ADR 0018 (tiempo por cuadro), ADR 0026 (mediciones), ADR
  0027 (plausibilidad)

## Contexto

El filtro de plausibilidad invalida ≤ 1.7% de los cuadros: lo que en pantalla
parece movimiento imposible es sobre todo **temblor**. La media exponencial del
§4 (v3) lo quitaría a costa de retrasar y aplanar los trazos, y por eso estaba
apagada. El One Euro corta fuerte con la mano quieta y se abre con la velocidad.

## Decisión de diseño

- **Dónde**: sobre cada coordenada cruda, tras la plausibilidad y el relleno y
  antes del paso 1. En vivo, dentro de `run_segmentation` —así también la
  velocidad con la que decide la máquina ve la mano filtrada—; en las muestras,
  en `lsm.preprocessing.reconstruct`, dinámicas y estáticas. Ya no pasa por
  `extract_sequence_features`: el §4 actúa sobre el flujo, no sobre la ventana
  (con la media exponencial, cada ventana re-filtraba desde su primer frame).
- **Tiempo real**: `Δ` sale de la marca del cuadro (ADR 0018); sin marca, del
  índice a la tasa nominal.
- **Velocidad en palmas por segundo** para abrir el filtro (única desviación del
  original, en el §4): `β` no depende de la distancia a la cámara.
- **Modelos**: `static_knn` y `dynamic_dtw` exportan `params.preprocessing`
  (plausibilidad, relleno y One Euro) en lugar de `smoothing_alpha`.
- **Muestras**: el esquema 5 anota `preprocessing`, con qué se calculó su σ;
  `lsm-capture verificar` solo compara la σ cuando coincide con el de ahora.
- **Replay**: `evaluation.replay_sample` reproduce `Sample.sequence`, que ya
  viene preprocesada; la máquina no la vuelve a filtrar.

## La calibración

Dos criterios opuestos, medidos juntos. Los umbrales de «claramente» y de
«retraso aceptable» no los dan los datos: los fijé antes de mirar la tabla y
van escritos para poder discutirlos.

- **Temblor**: la velocidad con la que decide la máquina (ventana de 100 ms, 21
  puntos, palmas por segundo de reloj real) en reposo. Sin filtro: p95 1.21, y el
  17% de los cuadros quietos supera `velocity_threshold_per_s` (0.55): la máquina
  los llamaría movimiento. **Criterio: el p95 baja al menos un 40%.**
- **Retraso**: desfase que mejor alinea la señal filtrada con la cruda, en τ
  (muñeca) y en el giro (ángulo de la línea de nudillos, que es lo que rota en
  la Ñ y la Q), mediana por letra. **Criterio: ≤ 50 ms (1.5 cuadros) en J, Ñ y
  Q, y el giro conserva ≥ 0.85 de su recorrido.** Ojo: el recorrido crudo incluye
  el propio temblor, así que el techo práctico está en ~0.95 (con β = 10, que
  apenas filtra, la Q da 0.94).

Frontera, `d_cutoff` = 2 Hz (la tabla completa, con 0.5 y 1 Hz, en el reporte):

| min_cutoff | β | reposo p95 | ≥ 0.55 | J τ / giro ms | Ñ τ / giro ms, amplitud | Q τ / giro ms, amplitud |
|---|---|---|---|---|---|---|
| 0.5 | 0.3 | −59% | 0.048 | 57 / 43 | 90 / 78, 0.76 | 70 / 51, 0.70 |
| 0.5 | 1 | −42% | 0.064 | 27 / 0 | **58** / 45, 0.90 | 43 / 23, **0.84** |
| 1 | 1 | −38% | 0.073 | 24 / 0 | 47 / 40, 0.91 | 37 / 20, 0.85 |
| 2 | 0.3 | −40% | 0.071 | 39 / 28 | 41 / 46, 0.86 | 38 / 27, 0.78 |

y con `d_cutoff` = 0.5 Hz, `(0.5, 3)`: −44%, J 18 / 0, Ñ **58** / 48 (0.92), Q 43 /
28 (0.85).

**Ningún punto cumple los tres.** Los que quitan el 40% del temblor retrasan τ
en la Ñ ~58 ms o aplanan el giro de la Q; los que no retrasan quitan menos de un
40%. La relación es monótona en toda la rejilla: es la contrapartida del filtro,
no un punto mal buscado.

Candidatos para mirar en vivo, de más suave a más fiel al trazo:

1. `0.5,1,2` — el que más temblor quita sin pasar de 60 ms en ningún trazo.
2. `0.5,3,0.5` — parecido, el giro de la Q intacto (0.85).
3. `1,1,2` — −38%, todos los retrasos ≤ 47 ms y giros ≥ 0.85: cumple el retraso
   y se queda corto de temblor por 2 puntos.

## Lo que no se midió

- **El efecto en el reconocimiento**: el replay y la evaluación con el filtro
  activo son del Paso 6 (ablación).
- **La X nueva**: no hay ninguna grabada todavía.
- El retraso en la K y la Z.
