# ADR 0027 — Filtro de plausibilidad anatómica (Paso 2)

- **Estado:** **implementada** el 2026-09-29, activa por defecto
  (`plausibility.enabled`). Entra en el contrato como §0.4 de `feature-spec.md`,
  pero **sin subir `FEATURE_SPEC_VERSION` todavía**: la plausibilidad, el One Euro
  (ADR 0028) y δ van juntos en una sola versión, la 4, al cerrar el bloque, con
  los golden regenerados una vez.
- **Fecha:** 2026-09-29
- **Datos:** ADR 0026.
- **Relacionadas:** ADR 0021 (huecos), ADR 0029 (relleno ampliado)

## Contexto

En vivo, en X, Ñ y Q las falanges «se mueven de forma físicamente imposible». La
propuesta: marcar esos cuadros como inválidos y dejar que el relleno de huecos
los resuelva, como un cuadro sin mano. **No se corrigen landmarks ni se
enderezan dedos.**

## Decisión

Un cuadro con mano se invalida (`InvalidReason.IMPLAUSIBLE`, con la comprobación
en `detail`) si, en este orden:

1. **Hueso**: algún hueso de los 20 se aleja de la mediana de ese hueso en los
   últimos 15 cuadros aceptados más de **0.45 palmas**, con largos y palma en 3D.
   Sin la referencia llena no se comprueba.
2. **Articulación**: si `max_mcp_dorsal_deg` no es `null`. **Desactivada**: no
   hay umbral que no tire poses limpias (ADR 0026).
3. **Salto**: el centro de la palma se desplazó desde el último cuadro aceptado
   más de **30 palmas por segundo** de tiempo real.

Si los rechazos siguen sin interrupción **100 ms**, la referencia se reinicia con
el cuadro actual: la forma cambió de verdad. Un cuadro rechazado no toca la
referencia.

- **Dónde**: en vivo, al principio de `run_segmentation`, antes que el relleno
  de huecos; al cargar una muestra dinámica, en `lsm.preprocessing.reconstruct`,
  que es lo mismo en el mismo orden. Las estáticas guardadas todavía no pasan por
  el filtro: sin relleno en el camino estático, un cuadro invalidado las haría
  ilegibles (le pasaría a 1 de ~2900). Lo resuelve el relleno corto de STABLE
  (ADR 0029).
- **Registro**: `WindowDynamic`, `LetterEmitted` y `Sample` llevan
  `implausible_frames`: de los rellenados, cuántos sustituyen a un cuadro
  imposible.
- **Tiempo**: el salto usa la marca de tiempo del cuadro (ADR 0018) o el índice a
  la tasa nominal.

### Por qué absoluta, 3D y contra una mediana móvil

Ver la tabla del ADR 0026: la mediana de la sesión mezcla poses entre las que
MediaPipe no conserva el largo de los huesos; la desviación relativa la domina el
hueso más corto; en 2D la proyección de un hueso cambia al girar la mano.

### El reinicio, medido

Con 200 ms (el primer valor, igual al hueco más largo que se rellena) el filtro
dejaba **19 muestras dinámicas del dataset ilegibles**: 17 K con seis rechazos
seguidos por hueso —un cambio de forma persistente que la mediana móvil tardaba
en alcanzar—, una Ñ vieja con saltos en ráfaga y una Q de hoy con seis saltos
seguidos. Sin reinicio, el 84% de las
rachas dura ≤ 3 cuadros y la cola es larga: con 100 ms, ninguna muestra queda
ilegible.

## Resultado

Fracción de cuadros invalidados (reporte completo en
`docs/mediciones/2026-09-29-paso2-plausibilidad.md`): 1.63% X, 1.32% J, 1.24% Q,
0.54% Ñ, 0.69% K en los diagnósticos; 1.05% K, 0.38% Q, 0.29% Ñ, 0.18% X, 0% J en
las dinámicas del dataset; ≈ 0% en las estáticas. **Ninguna letra se acerca al
10%.**

**Lectura**: casi nada de lo que se ve en pantalla es un cuadro imposible; es
temblor. El filtro quita los saltos rotos y los cambios bruscos de largo que sí
lo son, sin tocar lo demás.

## Tests en los que se desactiva

Los tests de la máquina de estados (`test_segmentation.py`, `test_cli_demo.py`)
construyen flujos que teletransportan la mano entre bloques y manos sintéticas
que doblan los dedos encogiéndolos (`synthetic._bend`); el filtro los invalida con
razón. Llevan `plausibility.enabled: false` y el motivo escrito: prueban la
lógica de estados. El filtro tiene los suyos en `tests/test_plausibility.py`.
