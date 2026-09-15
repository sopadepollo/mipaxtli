# ADR 0009 — La verificación del dataset cruza dos libm, y σ no es bit a bit portable

- **Estado:** aceptada
- **Fecha:** 2026-09-08
- **Fase:** 2 (antes de leer la primera matriz de confusión con datos reales)
- **Implementa:** `src/lsm/cli/capture.py` · `tests/test_cli_capture.py`
- **Continúa:** `docs/adr/0006-deteccion-de-manos-y-captura.md`

## Contexto

La primera sesión formal dejó 613 muestras en `data/raw`. `lsm-capture verificar`,
ejecutado en WSL, denunció **100 de ellas**: la σ re-derivada no coincidía con la
anotada al grabar.

No había ninguna pérdida de precisión. Las diferencias eran de **1 a 10 ULPs**
—1.5e-15 relativo—, repartidas por igual entre letras y entre manos, y el mismo
comando ejecutado **en Windows**, que es donde se grabaron, aprobaba las 613 sin
una sola queja.

La causa es la frontera que el proyecto eligió a propósito en `docker/README.md`
§b: **la captura corre en el host, porque ahí está la webcam, y todo lo demás en
WSL o en el contenedor.** El paso 3 de `feature-spec.md` —la rotación que alinea
la muñeca con el nudillo— pasa por `math.atan2`, `math.cos` y `math.sin`, y el §3.1
por `math.hypot`. IEEE 754 **no** obliga a que esas funciones estén correctamente
redondeadas, a diferencia de `sqrt`: cada libm las implementa a su manera y la de
MSVC y la de glibc discrepan en el último bit. σ, que es un agregado de las 42
componentes sobre todos los frames, hereda esa discrepancia.

El docstring del comando afirmaba lo contrario:

> *"es el **mismo** código sobre los **mismos** números, así que cualquier
> diferencia, por pequeña que sea, delata una pérdida de precisión en el viaje a
> disco"*

La premisa es falsa en la configuración que el propio repositorio documenta. Y el
efecto práctico era peor que el fallo que prevenía: cien líneas rojas que no son
nada entrenan a quien las lee a ignorar la salida del comando. Es exactamente el
mecanismo por el que la deriva del glosario pasó desapercibida —`make test` ya
estaba en rojo esperando a una persona—, y el README lo cuenta como la lección que
no hay que repetir.

## Decisión

**1. σ se compara con tolerancia relativa `1e-12` (`SIGMA_REL_TOL`).**

El número no es un gusto. Está entre dos magnitudes medidas sobre este dataset:

| Fenómeno | Desviación relativa de σ |
|---|---|
| Ruido de libm entre MSVC y glibc (10 ULP) | `1.5e-15` |
| **`SIGMA_REL_TOL`** | **`1e-12`** |
| Truncar los landmarks a 6 decimales | `2.3e-5` |

Mil veces por encima del ruido, diez millones por debajo de la regresión que el
comando existe para cazar. Hay tanto sitio entre las dos que la elección exacta
dentro de ese rango no cambia nada, y por eso se puede fijar sin volver a medir.

**2. La igualdad exacta se conserva donde sí es legítima: la suite.**

`tests/test_cli_capture.py` escribe y relee dentro del mismo proceso y la misma
máquina. Ahí `==` sí significa lo que dice, y ahí es donde de verdad se defiende
la promesa de guardar landmarks crudos: si un día el serializador truncara
valores, el test que compara lo que estaba en memoria contra lo que quedó en el
archivo se pone rojo en CI, antes de que ninguna muestra se grabe.

Los dos criterios conviven, y la separación es la que importa: **CI comprueba que
el formato no pierde precisión; el verificador de campo comprueba que un dataset
ya grabado no se ha corrompido.** Son preguntas distintas y solo la primera admite
`==`.

**3. Se descartó comprobar el round-trip de los landmarks byte a byte.**

Fue la primera idea y no resiste el examen: re-serializar lo que se acaba de leer
del archivo es idempotente por construcción. Si el escritor truncara valores, el
archivo truncado seguiría siendo punto fijo de sí mismo y la comprobación pasaría
en verde. No añade ninguna garantía sobre la que ya da la suite.

## Consecuencias

**Lo que sigue estando cubierto.** Un landmark editado a mano, un archivo
corrompido, una muestra regrabada con otro criterio: todos mueven σ muchísimo más
de `1e-12`. El test `test_verificar_denuncia_una_perdida_real_de_precision` fija
ese lado de la tolerancia para que nadie la afloje sin romper algo.

**Lo que hay que dejar de prometer.** Ningún float derivado de `features.py` es
reproducible bit a bit entre plataformas. Esto alcanza a `make eval`, cuyo reporte
el README vende como reproducible byte a byte sobre el mismo dataset: **lo es
dentro de una misma máquina, no entre Windows y Linux.** Queda anotado en el
README, porque si no, la primera persona que compare dos calibraciones generadas
en sitios distintos buscará una diferencia que no existe.

**Lo que no cambia.** La tolerancia de `1e-6` del contrato de features sigue
siendo la de la Fase 7 y cubre otra cosa: dos implementaciones distintas, Python y
TypeScript. Esta cubre la misma implementación sobre dos libm. Que la primera sea
siete órdenes de magnitud más ancha es la medida de esa diferencia.

**Lo que esto sugiere para la Fase 7.** Si el ruido de libm entre MSVC y glibc ya
es de 10 ULPs, el de la máquina de JavaScript de un navegador móvil no va a ser
menor. Los golden vectors se comparan con `1e-6` y el margen sobra, pero conviene
saber que el margen se está usando para esto y no solo para errores de
transcripción.
