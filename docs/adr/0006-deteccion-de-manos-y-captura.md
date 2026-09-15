# ADR 0006 — Detección de manos con MediaPipe Tasks, y la forma del CLI de captura

- **Estado:** aceptada
- **Fecha:** 2026-09-08
- **Fase:** 1
- **Implementa:** `src/lsm/io/hands.py` · `src/lsm/io/camera.py` ·
  `src/lsm/io/preview.py` · `src/lsm/io/dataset.py` · `src/lsm/capture.py` ·
  `src/lsm/cli/capture.py` · `docs/dataset-schema.md`

## Contexto

`ARQUITECTURA.md` §3 dejaba escrito que `io/hands.py` sería «un wrapper de
MediaPipe, aislado tras interfaz», y la Fase 0 dejó el `Protocol` y un doble. La
Fase 1 tiene que poner el detector real detrás de esa interfaz y escribir el CLI
que recolecta el dataset.

Al implementarlo aparecieron cuatro decisiones que no estaban tomadas, y las
cuatro son caras de revertir una vez que haya gente grabada.

---

## Decisión 1 — La API de Tasks, porque `solutions` ya no existe

`mediapipe.solutions.hands` es la API con la que está escrito casi todo el
material que circula, y era la opción evidente: no necesita descargar ningún
modelo. **Desde MediaPipe 1.0 el paquete `solutions` no se distribuye.** No hubo
elección: la API de Tasks es la única.

Las consecuencias que arrastra, que sí son decisiones:

**El bundle del modelo no se versiona.** `hand_landmarker.task` pesa ~8 MB. Va a
`data/models/`, que ya está en el `.gitignore`, y se descarga con `make model`. La
ruta es configurable (`hands.model_path`) y `MediaPipeHandDetector.open()` falla
con un mensaje que dice qué comando ejecutar, no solo que faltaba un archivo.

**Modo VIDEO, no IMAGE.** En IMAGE cada cuadro se detecta desde cero y los
landmarks bailan aunque la mano esté quieta. Ese jitter va directo a la σ del
`feature-spec.md` §2, que es el criterio con el que se acepta una muestra: con
IMAGE, la captura rechazaría manos perfectamente sostenidas. VIDEO mantiene el
rastreo entre cuadros a cambio de exigir timestamps monótonos.

Los timestamps salen de un **contador interno**, no del reloj. El reloj no es
reproducible: pasar la misma grabación dos veces por el detector daría marcas
distintas y, con ellas, resultados distintos. El contador hace que dos ejecuciones
sobre los mismos cuadros sean la misma ejecución.

**`num_hands = 2` y no 1.** `feature-spec.md` §0.3 manda quedarse con la mano de
mayor score cuando hay varias. Con `num_hands = 1`, MediaPipe entrega la que él
prefiera y la regla del contrato queda sin efecto: para poder elegir hay que ver
más de una.

**Los dos scores del §0.2 son el mismo número.** La API de Tasks expone una sola
confianza por mano detectada, la de la clasificación de lateralidad; la confianza
de detección de la palma se consume dentro del grafo (vía
`min_hand_detection_confidence`) y no vuelve a salir. `handedness_score` y
`detection_score` se rellenan con el mismo valor. Se mantienen separados en
`RawFrame` porque son conceptos distintos y otro detector —o MediaPipe JS en la
Fase 7— puede darlos por separado.

### Y la trampa cara: qué mano dice que ve

Hay dos preguntas que suenan a la misma y no lo son:

- **Qué imagen se le da al detector.** Sin espejar, siempre: lo manda
  `feature-spec.md` §0.3. Espejarla invertiría la geometría de los landmarks y
  corrompería el vector.
- **Qué etiqueta devuelve ante esa imagen.** Eso depende de la librería, y hay que
  medirlo. La documentación de MediaPipe afirmaba que la lateralidad se decide
  *"assuming the input image is mirrored"*, lo que implicaría tener que invertirla.

Equivocarse en la segunda no falla. El paso 2 espeja en X según ese valor, así que
con la etiqueta al revés **todas** las muestras se canonizan hacia la mano
contraria: el vector sale espejado pero coherente consigo mismo, y el modelo
entrena tan campante hasta que alguien firma con la otra mano — o hasta la Fase 7,
con MediaPipe JS usando la otra convención.

**Se resuelve con `hands.mediapipe_reports_mirrored_handedness`.** Es un
interruptor en `config.yaml` y no una constante en el código porque es exactamente
el tipo de convención que cambia entre versiones de una librería, y porque
equivocarse no produce ningún síntoma que se pueda depurar.

> **Actualización del 2026-09-08 (Fase 1.5).** El valor por defecto era `true`,
> siguiendo la documentación de MediaPipe. Al calibrar contra una cámara real
> resultó ser **`false`**: MediaPipe 1.0.1 alimentado sin espejar devuelve la mano
> anatómica. Ver las consecuencias, abajo.

La defensa real no es el interruptor sino **el preview**: muestra la lateralidad ya
resuelta en pantalla. Levantar la mano derecha y comprobar que dice `RIGHT` cuesta
un segundo, y es la única verificación que zanja esto en una cámara concreta.

Que esa comprobación existiera es lo que salvó el asunto: **la documentación estaba
equivocada, y no había ninguna otra forma de saberlo.** Si el valor se hubiera
grabado como constante "porque lo dice el manual", el dataset entero habría salido
espejado sin un solo síntoma.

---

## Decisión 2 — El criterio de aceptación vive en código puro

`ARQUITECTURA.md` §3 no listaba `src/lsm/capture.py`. Se añade.

La razón es el criterio de aceptación de esta fase: «grabar 20 muestras de una
letra y verificar que se re-derivan features idénticas». Si la lógica que decide
qué se guarda estuviera enredada con la cámara, el teclado y el dibujo, ese
criterio solo se podría comprobar sentándose delante de una webcam — y no en CI,
que es donde se comprueba todos los días.

Así que `capture.py` es puro (buffer circular, evaluación de la ventana, geometría
del espejado) y `cli/capture.py` es el envoltorio. El único punto de contacto es
`guardar_muestra`, que tampoco toca OpenCV: es **la** puerta por la que una muestra
llega al dataset, y `tests/test_cli_capture.py` la llama veinte veces.

`io/camera.py` y `io/preview.py` también son nuevos y estaban previstos en §3
(`camera.py`) o son la capa de dibujo que §3 no nombraba (`preview.py`).

**Todos importan sus dependencias nativas de forma diferida.** Importar
`lsm.cli.capture` no debe arrastrar MediaPipe, OpenCV ni numpy, porque la suite, el
entrenamiento y la evaluación tienen que correr sin ellas. Lo verifica
`test_ningun_modulo_del_paquete_importa_mediapipe`, que ahora recorre también los
módulos nuevos.

### La quietud solo se le exige a las estáticas

`evaluate_window` aplica el umbral de σ únicamente en modo `STATIC`. En una J, una
Ñ o una Z el movimiento **es** la seña (`feature-spec.md` §6.3): aplicarles el
umbral rechazaría exactamente las muestras correctas y el dataset se quedaría sin
las seis letras con recorrido. La σ se calcula igual y se guarda en el archivo,
porque sirve para depurar, pero ahí no decide nada.

### Lo que σ no mide

Vale la pena dejarlo escrito porque es contraintuitivo y ya costó un test mal
planteado: **σ es ciega al viaje de la mano.** Se calcula sobre el vector del §2,
que ya pasó por la traslación del paso 3 y la escala del paso 4, así que una mano
que cruza el encuadre entero sin cambiar de configuración tiene σ ≈ 0 — y la
ventana se acepta, con razón, porque la seña es la misma esté donde esté la mano.

Lo que mide el desplazamiento es `v_t` del §6, que se calcula sobre puntos sin
trasladar y es insumo de la segmentación, no de la captura. Quien quiera rechazar
el viaje no debe tocar `quality.max_dispersion`.

### Una muestra guardada no tiene huecos

El formato admite huecos y los conserva, porque `dataset-schema.md` exige que el
archivo no mienta sobre lo que ocurrió frente a la cámara. Pero una ventana **con**
huecos no se acepta como muestra, y `StoredSample.to_sample()` la rechaza al
cargar.

Las dos alternativas son peores. Coser los trozos inventaría un movimiento entre
dos posiciones que nunca se observó; quedarse con el trozo más largo entregaría una
seña recortada con la etiqueta de la completa. Las dos envenenan el entrenamiento
sin ningún síntoma, así que se prefiere fallar temprano, cuando todavía se puede
regrabar.

---

## Decisión 3 — El buffer mira hacia atrás

En modo estático se guarda la ventana **anterior** a la pulsación de la tecla, no
la siguiente.

Grabar hacia adelante obligaría a sostener la seña otro segundo después de haber
decidido que ya estaba bien, que es justo el momento en que la mano se relaja. Mirar
hacia atrás guarda lo que quien firma acababa de ver aceptado en el indicador de σ.

En modo dinámico no vale lo mismo: el principio y el final del trazo son parte de
la seña y solo quien firma sabe dónde están, así que la grabación se delimita
explícitamente con dos pulsaciones.

Tras guardar, el buffer se vacía. Sin eso, pulsar la tecla dos veces seguidas
guardaría dos muestras que comparten casi todos sus frames, y el dataset tendría
veinte repeticiones donde en realidad hubo cinco — que es exactamente el fallo que
la validación leave-one-signer-out existe para no cometer.

---

## Decisión 4 — Consentimiento: dos llaves, ninguna se abre sola

`ARQUITECTURA.md` §4.11 pide que no se almacene video sin consentimiento explícito
y por escrito. Traducido a mecanismo:

1. Un registro en `data/raw/consentimiento.json` con `video: true` para esa persona.
2. La bandera `--guardar-video`, apagada por defecto.

Hacen falta las dos. Un registro que falta es una autorización que no se pidió,
nunca una que se dio, así que la ausencia del archivo significa «no». Y aun con
permiso, el video solo se guarda si se pide explícitamente en esa sesión.

El archivo **no es** el consentimiento —eso es papel firmado—; es el registro de
que existe, y `referencia` dice dónde encontrarlo. El permiso se puede revocar
volviendo a ejecutar el comando sin `--video`, porque quien firma puede cambiar de
opinión.

---

## Alternativas consideradas

**1. Detección en modo IMAGE, sin estado.**
Descartada. Más simple y sin timestamps, pero el jitter entre cuadros infla la σ y
la captura rechazaría manos quietas. El criterio de calidad dejaría de medir a
quien firma para medir al detector.

**2. Un solo umbral de aceptación para estáticas y dinámicas.**
Descartada: rechazaría las seis letras con movimiento. Ver §6.3 del contrato.

**3. Guardar features junto a los landmarks, "por si acaso".**
Descartada. Tentador porque ahorraría el paso de re-derivar, y equivocado por lo
mismo: dos representaciones de lo mismo se desincronizan, y la que se lea por
error será la vieja. Lo único que se guarda de la extracción es σ, que es un
metadato de procedencia y no una feature — está para poder preguntarle al dataset
si las muestras peores eran las más temblorosas.

**4. Aceptar muestras con huecos y coserlas al cargar.**
Descartada. Ver arriba: inventa movimiento o recorta la seña, y en los dos casos en
silencio.

**5. Poner `save_video` en `config.yaml`.**
Descartada. Un permiso que se puede dejar encendido en un archivo de configuración
acaba encendido. La bandera vive solo en la línea de comandos, donde hay que
escribirla cada vez.

**6. Pedir las condiciones de luz y distancia por muestra.**
Descartada. Se piden una vez por sesión, que es como se graban de verdad: la luz y
la distancia se eligen al montar la sesión y no cambian a mitad. El ADR 0005 pide
exactamente eso — variar las condiciones **entre** sesiones, no dentro de una.

---

## Consecuencias

**A favor**

- Todo lo que decide qué entra al dataset se puede probar sin cámara, y se prueba.
- El criterio de aceptación de la Fase 1 es un test, no una sesión de grabación.
- La trampa de la lateralidad está documentada en tres sitios y es verificable de
  un vistazo en el preview.
- El dataset se re-deriva con un comando (`lsm-capture verificar`), que es lo que
  hace verdadera la promesa de guardar landmarks crudos.

**En contra**

- Hace falta descargar un modelo de 8 MB antes de la primera captura. Es un paso
  más en el arranque, mitigado con `make model` y con un mensaje de error que dice
  qué ejecutar.
- `hands.mediapipe_reports_mirrored_handedness` era un interruptor sin verificar
  contra hardware. **Ya está verificado** (ver abajo), y resultó que el valor por
  defecto que se había elegido leyendo la documentación era el equivocado.
- `src/lsm/capture.py` es un módulo que `ARQUITECTURA.md` §3 no preveía. El árbol
  de §3 se actualiza en el mismo cambio.

**RESUELTO el 2026-09-08 — la lateralidad, verificada con una cámara real.**

Resultado: **`mediapipe_reports_mirrored_handedness: false`**.

| | |
|---|---|
| MediaPipe | 1.0.1, Tasks API, `HandLandmarker`, modo VIDEO |
| Cámara | webcam integrada, backend `MSMF`, 1280x720 |
| Observado | mano derecha real, con `swap_handedness=True` el preview decía `LEFT` |
| Conclusión | MediaPipe devuelve la mano **anatómica** ante un cuadro sin espejar |

El valor por defecto original (`true`) salía de la nota histórica de la
documentación de MediaPipe —*"handedness is determined assuming the input image is
mirrored"*—, que no describe el comportamiento de la API de Tasks en 1.0.1. Es
justo el motivo por el que esto se decidió con un interruptor y una comprobación
humana en vez de con una constante: **la documentación estaba desactualizada y no
había forma de saberlo sin mirar una pantalla.**

Ninguna muestra se grabó con el valor equivocado: la comprobación es previa a la
primera sesión, que es exactamente para lo que se puso el cerrojo.

**Cómo se cambia**

`SAMPLE_SCHEMA_VERSION` y `CAPTURE_SPEC_VERSION` se incrementan por separado.
Cambiar el criterio de aceptación (la σ que se exige, los frames mínimos) es un
incremento de `CAPTURE_SPEC_VERSION` y **no invalida nada ya grabado**: los
landmarks crudos siguen ahí. Cambiar el formato del archivo es un incremento de
`SAMPLE_SCHEMA_VERSION` y obliga a migrar el dataset, porque los archivos de otra
versión se rechazan al cargar.
