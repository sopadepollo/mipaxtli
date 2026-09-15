# ADR 0014 — Los assets de texto → señas son esqueletos del dataset propio

- **Estado:** aceptada e implementada el 2026-09-14
- **Fecha:** 2026-09-14
- **Fase:** 4
- **Implementa:** `src/lsm/signs.py`, `src/lsm/io/signs.py`, `src/lsm/cli/signs.py`,
  `assets/signs/manifest.json`
- **Spec:** `docs/superpowers/specs/2026-09-14-fase-4-texto-a-senas-design.md`

## Contexto

`ARQUITECTURA.md` §4.10 avisa que la dirección texto → señas no tiene riesgo
técnico sino de **corrección del material**: casi todo lo que circula en
internet como "abecedario en lengua de señas" es ASL, y un proyecto que
presente ASL como LSM queda descalificado ante cualquier persona usuaria
(`glosario-lsm.md` §2). A eso se suma que el material fotográfico ajeno tiene
derechos, y que las ocho letras dinámicas necesitan movimiento, no una foto
con flecha.

## Decisión

**Ningún asset viene de internet.** Cada letra se dibuja como esqueleto de 21
landmarks a partir de una muestra de `data/raw`: la **medoide** de su clase (la
grabación real más cercana al centroide en el espacio de features), con
preferencia por mano derecha, que es la que muestra el diccionario. Estáticas
en PNG del frame central; dinámicas en GIF con todos los frames a
`signs.render_fps`.

`assets/signs/manifest.json` lleva por letra: archivo, `es_dinamica`,
descripción y trayectoria copiadas de `vocabulary.py`, página de *Manos con
voz*, y una `fuente` que apunta a la muestra concreta
(`signer/sesion/LETRA/NNN.json`). `manifest_drift` exige que las copias no
diverjan del glosario; `tests/test_signs_manifest.py` exige las 29 letras con
archivo, fuente y revisión `coincide`.

Los assets **se versionan**: son PNG de pocos KB y GIF de ~100-175 KB, y CI
solo puede exigir el criterio de la fase si los archivos están en el
repositorio.

## Detalles de implementación

**La fuente.** `io/signs.py` compone el cuadro de `lsm-signs reproducir` con
Pillow, no con OpenCV: OpenCV no escribe GIF y su fuente Hershey es ASCII, y
las descripciones del glosario llevan acentos y una Ñ. La fuente por defecto
de Pillow (Aileron) tampoco los tiene y los dibuja como recuadros, así que el
proyecto empaqueta `DejaVuSans.ttf` (759 KB, con su licencia al lado) como
datos del paquete en `src/lsm/io/fonts/`. Es la única fuente que viaja con el
repositorio, precisamente para que el cuadro se vea igual en CI, en Docker y
en Windows sin depender de qué fuentes tenga instaladas el sistema.

## Qué se gana

- **LSM por construcción.** Las muestras se grabaron siguiendo las páginas 15-19
  de la fuente primaria; no hay ningún paso por el que entre una imagen ajena.
- Sin derechos de terceros; reproducible con `lsm-signs render`.
- Las dinámicas tienen movimiento real, el de la grabación, no una flecha.

## Qué se pierde y cómo se compensa

- **La orientación de la palma.** Un esqueleto en 2D no distingue bien "palma
  al frente" de "palma de lado", y en LSM eso separa letras. Afecta a `C`, `F`,
  `G`, `H`, `I`, `Q` y `Y`. El reproductor muestra **siempre** la descripción y
  la trayectoria del glosario junto al dibujo; el dibujo es el apoyo, el texto
  es la norma.
- **Otras pérdidas de la proyección 2D**, anotadas letra por letra en la
  revisión: en `K` el dedo medio se proyecta más largo de lo real al pasar por
  el pico del movimiento, en `X` el desplazamiento adelante-atrás solo se ve
  como inclinación y escala, y en `LL`/`RR` el doble trazo se ve como un barrido
  único. Ninguna cambia la letra que se reconoce; todas son legibilidad del
  dibujo.
- **Legibilidad frente a una foto.** El manifest admite otros orígenes por
  `fuente.tipo`: una letra puede sustituirse por una foto verificada cambiando
  su entrada y su archivo, sin tocar código.
- **El peso de las dinámicas.** Una letra dinámica tardaba 15 s en pasar: 90
  frames a `render_fps: 12` (7.5 s) por `dynamic_loops: 2`. `render_fps` sube a
  18 —la tasa que `lsm-demo --medir-fps` midió en la Fase 3, 17.8 fps, ADR
  0013— para que el GIF reproduzca el trazo a la velocidad real de la
  grabación en vez de más lento, y `dynamic_loops` baja a 1: con los dos
  cambios una letra dinámica dura ~5 s. El GIF sigue llevando los 90 frames
  del buffer de captura completo, muchos de ellos con la mano ya quieta antes
  y después del trazo; recortarlos para dejar solo el movimiento es una
  palanca del renderizador que queda pendiente para la Fase 7 (móvil), sin
  tocar el esquema del manifest.

## Qué NO cierra este ADR

La revisión registrada en `revision` de cada letra es **contra la descripción
del glosario**, hecha por "Claude Opus 5 — revisión contra la descripción del
glosario, no validación por persona usuaria de LSM" mirando el dibujo con la
fila del glosario al lado; un segundo pase independiente repasó 21 de los
dibujos y coincidió (registro completo en el apéndice de más abajo). **Ninguna
persona ha mirado todavía los 29 assets:** las dos revisiones que hay son de
agentes contra la descripción escrita, no de una persona usuaria o intérprete
de LSM. No es la validación que pide `ARQUITECTURA.md` §4.11: el
PENDIENTE-HUMANO G del glosario sigue abierto y aplica también a estos assets.

## Alternativas descartadas

- **Fotos propias.** Más legibles; 29 assets a mano y sin ventaja de
  trazabilidad sobre el dataset, que ya existe.
- **Recortes del PDF de CONAPRED.** Fidelidad máxima, pero es material con
  derechos y no da movimiento para las dinámicas.
- **No versionar los assets.** Habría dejado el criterio de la fase fuera de CI.

## Apéndice: registro de la segunda pasada (revisión independiente, 2026-09-14)

21 de los 29 dibujos, repasados por un segundo agente revisor contra la
columna `coincide` del glosario, mirando el mismo dibujo y la misma
descripción que la primera pasada. Sigue siendo revisión contra la
descripción, no validación por persona usuaria de LSM: ver la advertencia de
arriba.

| letra | lo que se ve | de acuerdo con `coincide` |
|---|---|---|
| L | índice largo arriba, pulgar largo a la derecha en ángulo recto, tres cortos | sí |
| LL | misma mano; muñeca barre de derecha a izquierda (~150 px) en un solo sentido | sí |
| R | dos cadenas largas que se cruzan a media altura, yemas juntas arriba; pulgar recogido | sí |
| RR | misma mano que R; barrido horizontal a la izquierda, un solo sentido | sí |
| N | mano colgando, dos dedos largos hacia abajo en el lado del pulgar, dos cortos; pulgar por debajo | sí |
| Ñ | misma mano; los dedos pasan de apuntar abajo-derecha a abajo y vuelven, dos veces | sí |
| I | un dedo largo en el borde opuesto al pulgar, tres recogidos, pulgar corto apoyado | sí |
| J | misma mano; baja y luego se desplaza a la derecha y sube: gancho de j | sí |
| C | vista lateral: cuatro dedos solapados en arco, pulgar curvo por abajo, hueco claro | sí |
| F | índice corto con la yema del pulgar sobre él; medio/anular/meñique largos y paralelos | sí |
| G | mano horizontal, índice largo a la derecha, pulgar largo arriba, tres cortos | sí |
| H | igual que G con dos dedos largos juntos | sí |
| Y | pulgar largo arriba, meñique largo abajo, tres cortos | sí |
| Z | índice largo; la yema va ←, ↘, ←: tres tramos; en los últimos frames el índice se acorta (apunta a cámara) | sí |
| X | mano de lado, pulgar arqueado e índice en gancho enfrentados; el movimiento se ve como cambio de escala/inclinación | sí, con la nota |
| K | como P con el medio corto en reposo; en el pico el medio se proyecta largo por debajo de la muñeca | sí, tras corregir la nota |
| Q | índice arqueado en gancho, pulgar debajo, giro de muñeca | sí |
| P, T, M, W | P: índice arriba, medio al lado, pulgar entre ambos. T: puño con pulgar entre índice y medio. M: tres largos sobre pulgar tapado. W: tres largos separados | sí |
