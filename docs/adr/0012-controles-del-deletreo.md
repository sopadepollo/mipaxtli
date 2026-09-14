# ADR 0012 — Los controles del deletreo son teclado, no seña

- **Estado:** aceptada
- **Fecha:** 2026-09-09
- **Fase:** 3 (cierre)
- **Implementa:** `src/lsm/cli/demo.py` · `src/lsm/spelling.py`
- **Continúa:** `docs/adr/0004-contrato-de-segmentacion.md`

## Contexto

Deletrear una frase necesita más que emitir letras: hace falta cerrar una
palabra, borrar un símbolo que se escribió por error y cerrar la frase entera
para poder copiarla. `ARQUITECTURA.md` §4.2 dejó esos tres controles para la
Fase 3 sin decir cómo se disparan, y la pregunta obvia es si se señan.

No se pueden señar con un gesto **clasificado**. `static_knn` conoce 22 clases
—las 21 letras estáticas más `NONE`— y las 21 letras ya están ocupadas por su
propia letra: no queda ninguna clase libre para "cerrar palabra" o "borrar".
Añadir una exigiría grabarla, y grabar una clase nueva en este proyecto no es
gratis: son las tres personas citadas otra vez, con cámara, consentimiento y
condiciones de luz variadas, para un gesto que ni siquiera pertenece al
deletreo manual real (`docs/glosario-lsm.md` no tiene una seña de "borrar").
Es el mismo costo que pagó cada letra del alfabeto, por un símbolo que no es
una letra.

## Decisión

**Los tres controles se dividen entre un gesto que ya existe y el teclado, y
ninguno pasa por el clasificador.**

- **Bajar la mano cierra la palabra.** No es una clase nueva: es la ausencia de
  mano que `run_segmentation` ya detecta para volver a `IDLE`
  (`missing_to_idle_ms`), y `spelling.py` la reutiliza con su propio umbral,
  más largo (`spelling.space_after_absent_ms`, un segundo — y desde el ADR 0013,
  un segundo de verdad a cualquier tasa; cuando estaba en frames eran 1.7
  segundos en la máquina donde se midió). Se justifica solo: al deletrear con las manos, bajarla entre palabras es
  lo que ya hace cualquiera de forma natural, con o sin este proyecto. No hay
  nada que aprender.
- **`BACKSPACE` borra el último símbolo** de la palabra en curso.
- **`ENTER` cierra la frase** — lo que hay escrito se imprime y el buffer se
  vacía para la siguiente.
- **`q` sale**, y lo que quedó sin cerrar con `ENTER` se imprime igual al
  salir.

## Alternativas descartadas

**Un gesto de control clasificado** (por ejemplo, un puño cerrado sostenido
para "borrar"). Descartado por lo de arriba: una clase nueva son tres personas
citadas otra vez, y el costo no compra nada que el teclado no dé ya gratis.

**Un borrado disparado por gesto**, incluso reusando una clase existente (por
ejemplo, "dos manos" o "mano abierta sostenida" fuera del vocabulario de
letras). Descartado porque un gesto de borrado es peligroso de una forma que
uno de cierre no lo es: si se dispara solo —por una mano que pasa por esa
forma de camino a otra letra, o por ruido del detector— **destruye trabajo ya
escrito** sin que quien firma lo pida. Cerrar una palabra de más es reversible
con `BACKSPACE`; borrar una letra de más porque el sistema interpretó mal un
tránsito no lo es, y no hay forma de deshacer un borrado con el vocabulario
que hay.

## Consecuencia

**La demo no es señable de extremo a extremo.** Alguien sordo delante de la
cámara puede escribir letras y cerrar palabras bajando la mano, pero necesita
un teclado — y por tanto a alguien que lo use, o las dos manos libres para
alcanzarlo — para borrar un error o cerrar la frase. Es una limitación real de
esta fase, no un detalle de implementación, y no se resuelve agregando más
teclas: se resuelve solo si en una fase futura hay clases de control que
valga la pena grabar, con su propio costo evaluado aparte.
