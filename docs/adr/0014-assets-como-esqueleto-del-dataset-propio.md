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
- **El peso de las dinámicas.** 90 frames a 12 fps (7.5 s) por GIF es más de lo
  que hace falta para transmitir el trazo, y es un peso pensado para
  escritorio. Recortar frames o acortar la duración es una palanca del
  renderizador para la Fase 7 (móvil), sin tocar el esquema del manifest.

## Qué NO cierra este ADR

La revisión registrada en `revision` de cada letra es **contra la descripción
del glosario**, hecha por "Claude Opus 5 — revisión contra la descripción del
glosario, no validación por persona usuaria de LSM" mirando el dibujo con la
fila del glosario al lado; un segundo pase independiente repasó 21 de los
dibujos y coincidió. No es la validación por persona usuaria de LSM o
intérprete que pide `ARQUITECTURA.md` §4.11: el PENDIENTE-HUMANO G del
glosario sigue abierto y aplica también a estos assets.

## Alternativas descartadas

- **Fotos propias.** Más legibles; 29 assets a mano y sin ventaja de
  trazabilidad sobre el dataset, que ya existe.
- **Recortes del PDF de CONAPRED.** Fidelidad máxima, pero es material con
  derechos y no da movimiento para las dinámicas.
- **No versionar los assets.** Habría dejado el criterio de la fase fuera de CI.
