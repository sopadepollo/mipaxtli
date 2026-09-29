# ADR 0029 — Relleno de huecos ampliado: sostén en TRACKING y STABLE (Paso 4)

- **Estado:** **implementado** el 2026-09-29. Cambia la máquina de estados; no
  sube `SEGMENTATION_SPEC_VERSION` todavía: entra con el bloque de tolerancia
  (plausibilidad, One Euro, δ), con los golden regenerados una sola vez.
- **Fecha:** 2026-09-29
- **Datos:** ADR 0026.
- **Relacionadas:** ADR 0021 (Bloque 2: relleno dentro del trazo), ADR 0027
  (plausibilidad)

## Contexto

Desde el Bloque 2 un hueco corto se rellena, pero solo dentro de
DYNAMIC_CANDIDATE y con un límite provisional de 150 ms. Dos casos seguían
perdiendo la seña: la mano que se pierde **al arrancar** el trazo, en TRACKING,
antes de llegar a candidato; y el parpadeo del detector con la mano **quieta**,
que vaciaba el buffer y obligaba a esperar otra vez `stable_ms`.

## Decisión

1. **Límite por estado**, según el estado en que empieza el hueco, medidos en el
   ADR 0026:

   | estado | límite | de dónde |
   |---|---|---|
   | DYNAMIC_CANDIDATE | 200 ms (era 150) | codo de los huecos dentro del trazo: +4–8 puntos por cada 50 ms hasta 200, +0–3 después |
   | TRACKING | 200 ms | el mismo codo en los huecos que empiezan en TRACKING (X, Ñ y Q: 0.52 / 0.58 / 0.61 a 150 / 200 / 250 ms) |
   | STABLE | 100 ms | los huecos de la pose sostenida llegan al 31–36% a 100 ms y la distribución queda plana hasta ~450 ms |
   | IDLE, EMIT | — | no se sostiene |

   Con 200 ms se rellenan, dentro del trazo, el 64% de los huecos de la X, el 61%
   de la Ñ y el 51% de la Q (con 150: 59%, 54%, 45%). `tracking_max_gap_ms` y
   `stable_max_gap_ms` a 0 devuelven la máquina de antes (interruptor de la
   ablación).
2. **Sostén.** Mientras el hueco está abierto la máquina se queda en el último
   frame válido: no cuenta ausencia, no decide nada. Si la mano vuelve a tiempo y
   se puede interpolar, los cuadros sostenidos se sustituyen por la
   interpolación de los landmarks crudos y se procesan en orden; si no, se
   procesan los inválidos como siempre. El único precio es latencia durante el
   hueco. **No se procesan cuadros sostenidos que luego haya que deshacer**: una
   emisión o un cambio de estado no se pueden retirar.
3. **El mismo tope de fracción** (`dynamic_max_interpolated_fraction`, 0.25):
   un trazo por encima se descarta; una ventana estable por encima no se
   clasifica todavía y espera a crecer, sin cooldown.
4. **Registro.** `GapResolved` (estado, cuadros, implausibles, rellenado o no)
   llega en el cuadro que resuelve el hueco; el diagnóstico en vivo lo guarda.
   `WindowDynamic` y `LetterEmitted` —también las estáticas— cuentan
   `interpolated_frames` e `implausible_frames`; `Sample`, lo mismo al cargar.
   **Sostenidos e interpolados son los mismos cuadros** en todo lo que llega al
   clasificador: un cuadro sostenido o se interpola o corta la secuencia.
5. **Muestras estáticas**: pasan por la plausibilidad y se rellenan con el
   límite de STABLE, como en vivo. La captura estática acepta una ventana con un
   parpadeo; los huecos del borde se recortan.

## Lo que no hace

- **No rellena los huecos largos**, la mitad de los de la Q y más de la mitad de
  los de la K: son pérdidas reales del rastreador. Lo que queda es el Paso 5
  (umbrales de MediaPipe).
- **No dibuja el cuadro sostenido** en el preview: la máquina sostiene, la
  pantalla sigue enseñando lo que entrega el detector.
