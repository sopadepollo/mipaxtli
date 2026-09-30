# Cómo probar el proyecto

Todo lo que se puede ejecutar, en el orden en que tiene sentido hacerlo, con lo
que hace falta para cada cosa. Si solo vas a leer un archivo antes de tocar el
repositorio, que sea este.

**La mayor parte no necesita cámara.** Solo tres comandos la necesitan —calibrar,
grabar y la demo en vivo— y están marcados con 📷.

| Quiero… | Necesito cámara | Sección |
|---|---|---|
| Correr los tests | no | [2](#2-la-suite) |
| Entrenar y evaluar el modelo | no | [5](#5-entrenar-y-evaluar) |
| Comprobar que el dataset no se corrompió | no | [4.5](#45-verificar-el-dataset) |
| Probar la demo con grabaciones | no | [6](#6-la-demo) |
| Calibrar una cámara | 📷 | [4.2](#42-calibrar-la-cámara-una-vez-por-cámara) |
| Grabar dataset | 📷 | [4.4](#44-grabar) |
| La demo en vivo | 📷 | [6](#6-la-demo) |
| Medir la tasa de cuadros | 📷 | [6.1](#61-medir-la-tasa-de-cuadros-) |

---

## 1. Preparar el entorno

Hace falta **Python 3.11+** y [uv](https://docs.astral.sh/uv/). `make` es
opcional: cada receta del `Makefile` es un comando de una línea.

```bash
uv sync          # o: make setup
```

Con esto ya corren los tests, el entrenamiento y la evaluación. MediaPipe y
OpenCV **no** se instalan aquí: son opcionales y solo hacen falta para grabar y
para la demo en vivo (sección 4.1).

### Si trabajas en WSL con el repositorio en Linux

El `.venv` es un entorno **Linux**. Desde Windows no se puede ejecutar
directamente: hay que entrar por WSL.

```bash
wsl -d Ubuntu -- bash -lc 'cd ~/mipaxtli && uv run pytest -q'
```

---

## 2. La suite

```bash
uv run ruff check .          # lint
uv run ruff format --check . # formato
uv run mypy                  # tipos, en modo strict
uv run pytest                # tests
```

O de una vez: `make test`. Sin cámara, sin MediaPipe, sin modelo y sin dataset.

`make test-nucleo` es lo mismo saltando los tests marcados `glosario`, que fallan
mientras `docs/glosario-lsm.md` tenga huecos que solo una persona puede cerrar.
**Hoy pasan todos**, así que los dos objetivos deberían estar verdes.

Dentro de Docker, si prefieres no instalar nada:

```bash
docker compose -f docker/docker-compose.yml run --rm test
```

### Los golden vectors

```bash
uv run lsm-golden            # o: make golden
```

Regenera `tests/fixtures/golden_features.json`, el contrato entre la
implementación de Python y la de TypeScript de la Fase 7. **Regenerarlos cambia
ese contrato**: solo se hace junto con un incremento de `FEATURE_SPEC_VERSION` y
un ADR (`CLAUDE.md`, regla 4).

---

## 3. Ver si hay dataset

```bash
ls data/raw/
```

`data/` está en el `.gitignore`, así que un clon recién hecho **no trae
dataset**. Sin él, `lsm-train` y `lsm-eval` caen a un corpus sintético
determinista y lo dicen en la cabecera del reporte y dentro del modelo exportado.
Sirve para comprobar que la tubería corre; no dice nada sobre LSM.

Para exigir dataset real, `--sin-sintetico`.

---

## 4. Grabar dataset 📷

### 4.1 Instalar lo que necesita la cámara

```bash
uv sync --extra capture      # MediaPipe y OpenCV
make model                   # descarga hand_landmarker.task (~8 MB, no se versiona)
```

Sin `make`, el modelo se baja a mano:

```bash
mkdir -p data/models
curl -sSL -o data/models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

> **WSL no ve la webcam.** No hay `/dev/video*` sin `usbipd-win`, y aun con él la
> cámara deja de funcionar en Windows mientras esté conectada. La captura se
> ejecuta **desde Windows** contra el repositorio de WSL, que se lee por UNC sin
> copiar nada. La receta comprobada está en `docker/README.md`, sección (b). Lo
> mismo vale para macOS.

### 4.2 Calibrar la cámara (una vez por cámara)

```bash
uv run lsm-capture calibrar --confirmado-por "tu nombre"
```

Levanta la mano **derecha** junto a tu hombro derecho, con la palma hacia la
cámara, y déjala ahí. Arriba tiene que decir `entrada: SIN ESPEJAR`; entonces
pulsa `S`. Si dice `ESPEJADA`, la cámara o su driver espejan la imagen por su
cuenta: desactívalo en la configuración de la webcam y repite.

Desde el ADR 0017 no se comprueba qué **nombre** le pone MediaPipe a la mano —ese
nombre ya no decide el espejo del contrato, lo decide la mano que declaras con
`--mano`— sino que la imagen llega al detector **sin espejar**. Es un error que
no se ve: con la entrada espejada tu mano derecha se trataría como izquierda y
todo el dataset saldría reflejado. Las calibraciones hechas antes del ADR 0017 no
valen: `grabar` pide repetirla una vez.

**Este minuto no se salta**, y `grabar` se niega a abrir la cámara sin él.
La calibración se invalida sola si cambias de cámara o de resolución.

### 4.3 Consentimiento, si vas a guardar video

```bash
uv run lsm-capture consentimiento --firmante s01 --referencia "expediente 3"
```

Sin `--video`, la persona queda registrada pero **no** autoriza almacenar
cuadros. Los landmarks no identifican a nadie; el video sí. Para guardar video
hacen falta las dos llaves: `--video` aquí **y** `--guardar-video` en la sesión.

Volver a ejecutarlo sin `--video` revoca el permiso: quien firma puede cambiar de
opinión.

### 4.4 Grabar

Ensaya primero, sin ensuciar el dataset:

```bash
uv run lsm-capture grabar --sesion-prueba --firmante s01 --sesion ensayo \
  --mano derecha --luz-nivel INDOOR --luz-direccion FRONTAL --distancia MEDIUM
```

Escribe en `data/raw/pruebas/`, que no entra al dataset. Es para encuadrar la
cámara antes de citar a nadie.

La sesión formal:

```bash
uv run lsm-capture grabar --firmante s01 --sesion 2026-09-09-manana \
  --mano derecha --luz-nivel INDOOR --luz-direccion FRONTAL --distancia MEDIUM
```

**`--mano` es la mano con la que vas a firmar en toda la sesión** (ADR 0017). Es
la que canoniza el contrato; lo que diga MediaPipe se guarda solo para
diagnóstico, porque con la palma de lado cambia de opinión a mitad de una J. Para
grabar con la otra mano, abre otra sesión. Si firmas con la mano contraria a la
declarada, el preview pregunta **¿cambiaste de mano?** —solo pregunta, no cambia
nada—.

Las condiciones son obligatorias a propósito: anotarlas después, de memoria, no
funciona (`docs/adr/0005-taxonomias-de-metadatos-de-captura.md`). **Varía luz y
distancia entre sesiones, nunca dentro de una**, o los metadatos mienten sobre la
mitad de las muestras.

En el preview: **ESPACIO** guarda, `n`/`p` cambian de letra, `m` alterna entre
estática y dinámica, `r` descarta la grabación en curso, `q` sale.

**Una dinámica no se cierra con la tecla** (Bloque 4, ADR 0023). ESPACIO la
**arma**; desde ahí haz la letra y **detén la mano al terminar**. La delimita la
misma máquina de estados que usa la demo: la muestra empieza donde empieza el
trazo que la máquina entregaría y termina cuando lo entrega, con el reposo que lo
cerró incluido. Si en `capture.dynamic_max_ms` (6 s) la máquina no entrega un
trazo cerrado, no se guarda nada y el preview dice por qué («no se cerro el
trazo: deten la mano al terminar», o «trazo demasiado largo»). `r` cancela.

**Un intento rechazado no se pierde** (Paso 0, ADR 0024): se guarda entero, en
crudo y con el motivo, en `<LETRA>/rechazados/`, que el entrenamiento no lee. El
preview lo dice («intento en rechazados/003.json»). Para saber cuántos saldrían
enteros con la configuración de hoy, sin cámara:

```bash
uv run lsm-capture rechazados
```

No escribe nada: re-segmenta cada intento con la misma máquina y la misma
evaluación que la captura, y cuenta por letra y motivo.

La barra de calidad dice si la ventana serviría *ahora mismo*, y mide cosas
opuestas según el modo:

| Modo | Barra | Verde cuando |
|---|---|---|
| estática | `sigma` | la mano está **quieta** |
| dinámica | `arco` | el trazo ya **recorrió** lo suficiente |

Guardar con la barra en rojo no se puede.

Si se pierde la mano: en modo estático cualquier hueco impide guardar. En modo
dinámico, desde el Bloque 2 (ADR 0021), un hueco corto —como mucho
`segmentation.dynamic_max_gap_ms`, 150 ms— no lo impide: la muestra se guarda
con el hueco tal como ocurrió, y al entrenar se rellena interpolando. Uno más
largo, o demasiados frames rellenados, sí.

La clase `NONE` **no es opcional** y es la que todo el mundo olvida: mano
relajada, transiciones entre letras y gestos cotidianos. Sin ella el clasificador
asigna una letra aunque la persona se esté rascando la nariz. Grábala sobre todo
en modo estático: las muestras dinámicas de `NONE` quedan fuera del alcance de la
Fase 2 (`docs/adr/0010-la-clase-negativa-y-el-modo-de-grabacion.md`).

**Objetivo de cobertura** (`ARQUITECTURA.md` §4.7): 3 personas, 2 sesiones cada
una, ~20 repeticiones por letra.

### 4.5 Verificar el dataset

```bash
uv run lsm-capture verificar     # o: make verify
```

No necesita cámara. Relee `data/raw`, re-deriva las features de cada muestra y
comprueba que coinciden con las que se anotaron al grabar. Es la garantía de que
guardar landmarks crudos sirve para algo: si cambia la normalización, el dataset
se re-deriva con un comando en vez de volver a citar a tres personas.

La σ solo se compara en las muestras que la anotaron con la versión actual del
contrato de features (esquema v4, `feature_spec_version`): las anteriores se
revisan en todo lo demás —huecos, extracción— y se cuentan aparte.

**Dinámicas truncadas** (Bloque 4). Las dinámicas grabadas antes del Bloque 4
se cortaban en un tope de cuadros y muchas acaban con el trazo en marcha. Para
dejarlas fuera del entrenamiento y la evaluación sin borrarlas:

```bash
uv run lsm-capture marcar-truncadas
```

Escribe `data/raw/truncadas.json` con las que acaban en movimiento y por qué;
`lsm-train`, `lsm-eval` y `lsm-eval-dinamico` las excluyen al cargar. Borrar el
archivo las devuelve. Conviene marcarlas **después** de grabar las dinámicas
nuevas: con las 822 de hoy quedarían entre 20 y 38 por letra.

---

## 5. Entrenar y evaluar

Ninguno de los dos necesita cámara ni MediaPipe.

```bash
uv run lsm-train --sin-sintetico           # o: make train
uv run lsm-eval  --sin-sintetico           # o: make eval
uv run lsm-eval-dinamico --sin-sintetico   # o: make eval-dinamico
```

`lsm-train` exporta **dos** modelos y **no mide nada** a propósito: la precisión
honesta la dan `lsm-eval` (estáticas) y `lsm-eval-dinamico` (dinámicas).

| Modelo | Qué letras |
|---|---|
| `data/models/static_knn.json` | las 21 estáticas más `NONE` |
| `data/models/dynamic_dtw.json` | J, K, Ñ, Q, X, Z. **LL y RR no**: su dirección canónica está pendiente de decisión humana (`docs/glosario-lsm.md`, PENDIENTE-HUMANO I) y el entrenamiento las cuenta como bloqueadas |

`lsm-eval` escribe tres archivos en `data/models/eval/`:

| Archivo | Para qué |
|---|---|
| `reporte-fase2.md` | para leer: accuracy, matriz de confusión, contraste de hipótesis |
| `resultados-fase2.json` | para que lo lea otro programa |
| `calibracion-fase2.json` | los umbrales que recomienda el barrido, con la huella del dataset |

Ninguno lleva marca de tiempo: dos ejecuciones sobre el mismo dataset dan los
mismos bytes, que es lo que permite comparar dos calibraciones. **Dentro de una
misma máquina**: entre Windows y Linux los últimos decimales pueden diferir
(`docs/adr/0009-verificacion-entre-plataformas.md`).

Opciones que se usan de verdad:

```bash
uv run lsm-eval --barrido minimo                    # pasada rápida, 64 puntos
uv run lsm-eval --protocolo leave-one-session-out   # con un solo firmante grabado
uv run lsm-eval --sin-sintetico                     # exige dataset real
```

Con un solo firmante, `lsm-eval` **se niega** en vez de degradar solo: una
métrica por sesión y una por persona no son comparables, y la degradación tiene
que quedar escrita en el reporte.

La validación es **leave-one-signer-out**, nunca un split aleatorio de frames:
frames consecutivos de la misma grabación son casi idénticos y repartirlos entre
train y test mide memorización, no generalización.

### 5.1 Las letras dinámicas

`lsm-eval-dinamico` escribe `reporte-fase5.md` y `resultados-fase5.json` en
`data/models/eval/`. Cuatro secciones: el resultado por letra con la
configuración actual; el barrido `trajectory_weight × band_radius` con **la Z
aparte**; las distancias con las que se fija `dtw.max_distance`; y la
reproducción de cada grabación por la máquina de estados —cuántas dinámicas salen
enteras, partidas o perdidas, y qué cambió en las estáticas—. Es el sustento del
ADR 0016.

Con el dataset de la Fase 1 tarda unos 70 minutos (la rejilla completa son
36 puntos, y la reproducción pasa 3200 grabaciones por la máquina): el DTW es Python puro a
propósito, porque es la implementación de referencia que TypeScript tiene que
reproducir. Para una pasada rápida:

```bash
uv run lsm-eval-dinamico --rejilla rapido --sin-reproduccion
```

### 5.2 Mediciones de tolerancia a MediaPipe

Sin cámara. Leen `data/diagnostico/` (las sesiones de `lsm-demo diagnosticar`) y
`data/raw/`, y escriben un reporte en Markdown; son las que fijan los umbrales de
la plausibilidad, el One Euro y el relleno de huecos (ADR 0026 a 0029):

```bash
# Paso 1: huecos por fase y por estado, score antes de cada pérdida, largo de
# los huesos, elevación de las falanges, saltos de la palma, temblor por dedo.
uv run lsm-medir --salida docs/mediciones/tolerancia.md tolerancia

# Paso 2: qué fracción de cuadros invalida el filtro de plausibilidad, por letra.
uv run lsm-medir --salida docs/mediciones/plausibilidad.md plausibilidad
```

`tolerancia` toma las dinámicas desde `--desde` (por defecto el 2026-09-29) y la
A estática como control; ~30 s. `plausibilidad` recorre todo el dataset y los
diagnósticos con la configuración de `config.yaml`: para medir otro umbral,
`--config` con otra copia.

```bash
# Paso 6: ablación. Reproduce la máquina sobre el flujo crudo de cada sesión de
# diagnóstico con cada componente (plausibilidad, One Euro, relleno ampliado)
# solo y juntos, y cuenta por letra los intentos con el trazo entero, partido o
# perdido. Una variante tarda varios minutos; se pueden pedir sueltas.
uv run lsm-medir --salida docs/mediciones/ablacion.md ablacion
uv run lsm-medir ablacion --variantes antes todo
```

```bash
# Paso 3: barrido del One Euro. Temblor en reposo contra retraso y amplitud del
# trazo en J, Ñ y Q. ~40 s por valor de d_cutoff.
uv run lsm-medir --salida docs/mediciones/one-euro.md one-euro --d-cutoff 2
```

### 5.3 El One Euro a ojo 📷

El barrido no encontró un punto que quite el temblor sin retrasar los trazos
(ADR 0028); el filtro viene **encendido con el candidato `0.5,1,2`**, elegido
por la ablación en vivo. Los modelos se entrenan con el mismo filtro y la demo
rechaza un modelo entrenado con otro: si lo apagas o cambias sus valores en
`config.yaml`, reentrena (`make train`). Para verlo en vivo, con el esqueleto
crudo en gris y el filtrado en color encima:

```bash
uv run lsm-demo --mano derecha --comparar-one-euro --one-euro-params 0.5,1,2
```

`--comparar-one-euro` solo cambia el dibujo; la máquina sigue con lo que diga
`smoothing.enabled`. `--one-euro off` lo apaga para la máquina y `--one-euro on`
con otros `--one-euro-params` lo cambia; en los dos casos hace falta un modelo
entrenado con ese mismo preprocesado. Los tres candidatos del ADR 0028 son
`0.5,1,2` (el activo), `0.5,3,0.5` y `1,1,2`.

---

## 6. La demo

Necesita cámara, MediaPipe y los modelos entrenados (`make train`):

```bash
uv run lsm-demo --mano derecha           # 📷 sesión en vivo
```

Carga `data/models/static_knn.json` y, si existe, `data/models/dynamic_dtw.json`
(`--modelo` y `--modelo-dinamico` para otras rutas). Sin el dinámico avisa y
sigue: las estáticas se deletrean igual y los trazos salen como rechazo en vez
de caer al clasificador estático. Una letra que sale por el camino dinámico
lleva `(trazo)` en el mensaje del HUD.

Sin `--extra capture` instalado, imprime el mismo aviso que `make setup-capture`
y sale con código 1: el resto del proyecto corre sin esas dependencias y la
demo no es la excepción a la hora de fallar con un mensaje útil.

Para probarla **sin cámara**, reproduce una sesión ya grabada en `data/raw/`:

```bash
uv run lsm-demo --desde-dataset data/raw
# o: make demo ARGS="--desde-dataset data/raw"
```

La ruta es la **raíz del dataset**, la misma que consume `lsm-eval`: dentro se
buscan `<firmante>/<sesion>/<letra>/NNN.json`. Apuntar a una sesión concreta no
encuentra nada — antes eso salía como una línea en blanco y código 0, que es
indistinguible de «no reconoció nada»; ahora lo dice y sale con 1.

Esto es lo que ejercita `tests/test_cli_demo.py` en CI, sin cámara y sin
MediaPipe: el criterio de la fase —deletrear una palabra de cinco letras sin
errores de segmentación— pasa ahí, con secuencias sintéticas.
`test_una_palabra_de_cinco_letras_produce_cinco_simbolos` produce `"casas"` a
partir de cinco señas.

### Controles

| Tecla / gesto | Efecto |
|---|---|
| bajar la mano | cierra la palabra en curso y abre una nueva |
| `BACKSPACE` | borra el último símbolo escrito |
| `ENTER` | cierra la frase entera (se imprime y el buffer se vacía) |
| `q` | sale; lo que quedó sin cerrar con `ENTER` se imprime igual |

### Qué esperar en pantalla

La demo **mide la tasa antes de arrancar** (los primeros
`telemetry.fps_window_frames` cuadros) y la imprime junto a en cuántos cuadros se
tradujo cada umbral. Esa tasa se congela para toda la sesión: los umbrales de
`config.yaml` están en milisegundos y esto es lo que los convierte.

Una vez dentro, arriba a la derecha van las dos tasas en vivo (`fps` de entrega,
`proc` de procesamiento) y se ponen en rojo por debajo de `telemetry.min_fps`,
con un aviso que sustituye al de las letras dinámicas: a esa tasa el problema es
de rendimiento y no de la seña.

En la línea de mensajes, una letra que no acaba de salir dice `acumulando C: 9
frames`: está por encima del piso de confianza y por debajo del umbral que emite
sin esperar, así que sigue juntando evidencia. Es lo que distingue «la máquina
está dudando entre C y O» de «el detector no encuentra la mano», que antes se
veían igual.

Bajar la mano es el único gesto de control del proyecto, y no está clasificado:
es la ausencia de mano que la segmentación ya detecta, con un umbral propio y
más largo (`spelling.space_after_absent_ms`, un segundo) para que
no baste un parpadeo del detector. Borrar y cerrar la frase van por teclado
porque el clasificador ya usa sus 22 clases en las 21 letras más `NONE`, y no
hay ninguna libre para un gesto de control sin grabar una clase nueva. El
razonamiento completo está en `docs/adr/0012-controles-del-deletreo.md`, y su
consecuencia se dice ahí sin adornos: la demo **no es señable de extremo a
extremo** — cerrar la frase o corregir un error necesita un teclado.

Las **ocho letras dinámicas** (`J`, `K`, `LL`, `Ñ`, `Q`, `RR`, `X`, `Z`) no se
reconocen todavía: llegan en la Fase 5 con `dynamic_dtw`. El HUD lo avisa en
pantalla, en la franja inferior, mientras dura la sesión.

**La sesión en vivo —con cámara de verdad— no se ha ejecutado todavía.** Lo
único verificado hasta ahora es `--desde-dataset`, que no abre cámara ni toca
MediaPipe. Cómo se ve el HUD con una persona firmando delante no lo dice ningún
test. La **latencia** sí se puede medir ya, y es lo que hace la sección
siguiente.

### 6.1 Medir la tasa de cuadros 📷

```bash
uv run lsm-demo --mano derecha --medir-fps                    # 60 s
uv run lsm-demo --mano derecha --medir-fps --medir-segundos 20
# o: make medir-fps ARGS="--medir-segundos 20"
```

Corre una sesión de demo normal —se puede deletrear mientras mide— durante
`telemetry.benchmark_seconds` y al terminar imprime la distribución. Necesita lo
mismo que la demo: cámara, MediaPipe y modelo entrenado. Sobre
`--desde-dataset` se niega a correr: sin cámara no hay tasa de entrega que medir
y el número no diría nada sobre la máquina.

> **Con el repositorio en WSL, esto se ejecuta desde Windows**, como la captura
> y como la demo en vivo: WSL no ve la webcam (sección 4.1). Con el entorno de
> Windows de `docker/README.md` (b) ya creado:
>
> ```powershell
> $repo = "\wsl.localhost\Ubuntu\home\<usuario>\mipaxtli"
> Set-Location $repo; $env:PYTHONPATH = "$repo\src"
> & $HOME\lsm-win\Scripts\python.exe -m lsm.cli.demo --medir-fps
> ```
>
> Y **la medida es de esa máquina**: los fps de Windows con el backend `MSMF` no
> son los de WSL ni los de otro equipo. Anotar con qué se midió, igual que se
> anota con qué dataset se calibró un umbral.

**Por qué hace falta antes de tocar cualquier umbral.** Todos los umbrales de
`segmentation` están ahora expresados en **milisegundos** y se convierten a
frames con la tasa que mide este comando. Antes estaban en frames y los
comentarios de `config.yaml` los traducían suponiendo 30 fps: la primera medición
dio 17.8, así que cada umbral duraba 1.7 veces lo que su comentario afirmaba — el
buffer eran 1348 ms y no 800. Ver `docs/adr/0013-la-ventana-mezclada.md`.

Por eso esta medición sigue haciendo falta después del cambio: **la tasa de tu
máquina es lo que decide en cuántos cuadros se traduce cada umbral**, y la demo
la mide sola al arrancar (los primeros `telemetry.fps_window_frames` cuadros) y
la imprime antes de empezar.

El volcado tiene esta forma:

```
== medicion de fps ==
cuadros: 300   pared: 10.28 s   fps sostenido: 29.2

fps por cuadro      media  mediana       p5      p95      min      max
entrega              29.8     29.7     26.1     34.9     12.1     38.1
procesamiento        35.5     35.5     30.7     42.6     13.0     45.1

latencia (ms)       media  mediana       p5      p95      min      max
camara                1.9      1.8      1.1      2.9      1.0      3.0
deteccion            22.9     22.0     18.4     25.8     18.0     70.5
segmentacion          6.0      6.1      4.2      7.8      4.0      8.0
preview               3.4      3.3      2.1      4.9      2.0      5.0

manda la tubería: el reconocimiento tarda más que la espera de la cámara, ...
```

Cómo leerlo:

| Fila | Qué es |
|---|---|
| `entrega` | cuadros por segundo que completa el bucle. Es la tasa en la que están expresados los umbrales, y su techo es `capture.camera_fps`. |
| `procesamiento` | los que sostendría la tubería si la cámara entregara infinitamente rápido: sin la espera de `camera.read()` ni el dibujo del preview. |
| `camara` | espera, no trabajo. Si la tubería es más lenta que la cámara, el cuadro ya está en el buffer del driver y esto sale casi cero. |
| `deteccion` | MediaPipe. |
| `segmentacion` | features + máquina de estados + clasificador, cuando toca. |
| `preview` | HUD, `imshow` y `waitKey`. |

**Los percentiles no son adorno: son el dato.** Una tubería que promedia 28 fps
pero cae a 9 durante medio segundo produce exactamente la lentitud que se
percibe, y en la media no se ve — el `p5` es el que la denuncia. La última línea
dice quién es el cuello de botella, que es la pregunta que cambia qué hacer con
el número: si manda la cámara, subir `capture.camera_fps` o bajar la resolución;
si manda la tubería, el tiempo está en `deteccion` o en `segmentacion` y ahí es
donde hay que mirar.

Los mismos dos números —`fps` de entrega y `proc` de procesamiento— van **en
vivo** arriba a la derecha del HUD, promediados sobre los últimos
`telemetry.fps_window_frames` cuadros (30, un segundo a 30 fps). Sin verlos, «la
demo tarda en confirmar» y «la tubería va a 9 fps» se ven exactamente igual en
pantalla.

### 6.2 Diagnosticar pérdidas de tracking 📷

Para saber **por qué** se pierde la mano a mitad de una letra dinámica (Fase 5.1,
Bloque 0). Una sesión guiada pide cada una de las ocho dinámicas diez veces:

```bash
uv run lsm-demo diagnosticar --mano derecha --iluminacion habitual
uv run lsm-demo diagnosticar --mano derecha --iluminacion lampara
# o: make diagnostico ARGS="--iluminacion habitual"
```

**Prueba de reposo.** Antes de las letras la sesión pide cuatro posturas con la
mano quieta: una estática (la A), y la K, la X y la Q en su posición inicial.
Cada una se arranca con **ESPACIO**, dura `diagnostics.rest_ms` (10 s) y avanza
sola; **BACKSPACE** la reinicia. El reporte añade la sección 6: velocidad de la
mano quieta en unidades por segundo (p50 / p95 / máx), por pares de cuadros y
contra el cuadro de hace `segmentation.velocity_window_ms` (100 ms) —la que usa la
máquina desde la v5—, y qué fracción supera `velocity_threshold_per_s` y
`motion_threshold_per_s`. Cuenta el último intento de cada postura, sin sus
primeros `diagnostics.rest_settle_ms` (1 s), en que la mano se acomoda. Si te
mueves a mitad de una postura, reiníciala: el reporte no puede distinguir un
ajuste de la mano de su temblor. La sección 6.1 mide la misma grabación de
varias formas —con los 21 landmarks o solo con muñeca y nudillos
(`diagnostics.stable_landmarks`), y con cada ventana de
`diagnostics.rest_windows_ms` (100, 150 y 200 ms)— para comparar variantes sin
volver a grabar. Para hacer solo esa prueba, o solo las letras:

```bash
uv run lsm-demo diagnosticar --mano derecha --iluminacion habitual --solo-reposo
uv run lsm-demo diagnosticar --mano derecha --iluminacion habitual --sin-reposo
```

Para diagnosticar solo algunas letras, `--letras X,ENIE,Q`. Admite también
estáticas (`--letras A,N,I`): cuando una letra no sale en vivo pero sí en la
evaluación, la sesión guarda el flujo para ver sin cámara qué decidió la
máquina y con qué confianza.

En pantalla, abajo, dice qué letra hacer y cuántas van. **ESPACIO** da la
repetición por hecha, **BACKSPACE** la descarta y la vuelve a pedir, **q**
termina (lo registrado hasta ese momento se escribe igual). Hacer cada letra
como en un deletreo normal, con la mano quieta un momento antes y después.

La sección 5.1 del reporte da, por letra, los intentos con el trazo entero,
partido o perdido, la latencia entre el último cuadro en movimiento y la
entrega del trazo (p50 / p90 / máx, en ms), y cuántos trazos llegaron con algún
hueco rellenado por interpolación (Bloque 2, ADR 0021).

Al salir escribe en `data/diagnostico/<fecha>-<iluminacion>/`:

| Archivo | Qué es |
|---|---|
| `diagnostico.md` | el reporte: detección dentro de los trazos contra mano quieta, huecos en frames y ms, correlación de las pérdidas con velocidad y luminancia, fps reales sin cuadros duplicados, y una fila por repetición |
| `diagnostico.json` | lo mismo más cada cuadro registrado, para rehacer el análisis |
| `flujo.json` | los landmarks de la sesión, para reproducir sus huecos sin cámara |

`data/` no se versiona: esos archivos llevan landmarks de quien firma.

Cualquier sesión en vivo se puede instrumentar igual, deletreando libremente:
`uv run lsm-demo --mano derecha --diagnostico habitual`.

Los parámetros del detector que se diagnostican están en la sección `hands` de
`config.yaml` y quedan anotados en la cabecera de cada reporte. MediaPipe Tasks
no expone un score de presencia: el reporte registra si hubo mano, el motivo del
frame inválido y el score de lateralidad, que es el único número que devuelve.

Desde Windows, como la demo en vivo (ver 6.1):

```powershell
& $HOME\lsm-win\Scripts\python.exe -m lsm.cli.demo diagnosticar --iluminacion habitual
```

**Barrido de umbrales de MediaPipe** (Paso 5, ADR 0030). `--presencia` y
`--tracking` cambian `hands.min_hand_presence_confidence` y
`hands.min_tracking_confidence` para esa sesión, y quedan en los metadatos del
reporte junto con la plausibilidad, el relleno y el One Euro. Una sesión por par,
misma luz, solo X, Ñ y Q:

```powershell
foreach ($p in @("0.5 0.5", "0.4 0.35", "0.3 0.2")) {
  $pres, $trk = $p.Split(" ")
  & $HOME\lsm-win\Scripts\python.exe -m lsm.cli.demo diagnosticar --mano derecha `
    --iluminacion habitual --letras X,ENIE,Q --sin-reposo `
    --presencia $pres --tracking $trk
}
```

Después, sin cámara, la tabla por par y letra (detección durante el trazo,
implausibles, trazos enteros):

```bash
uv run lsm-medir --salida docs/mediciones/mediapipe.md mediapipe
```

### 6.3 Medir la cámara: formato y resolución 📷

Para encontrar la configuración que entrega más cuadros **nuevos** por segundo
(Fase 5.1, Bloque 1). La webcam de referencia entregaba 29.5 fps de los que solo
16.4 eran nuevos: el resto repetía el cuadro anterior (ADR 0017).

```bash
uv run lsm-demo medir-camara                    # solo la cámara, 8 s por configuración
uv run lsm-demo medir-camara --con-deteccion    # con MediaPipe en el bucle, como la demo
```

Prueba cada combinación de backend (`auto`, `MSMF`, `DSHOW`), formato (el del
driver o `MJPG`) y resolución (1280x720 y 640x480), descartando antes los
primeros cuadros, que llegan lentos. No hace falta hacer nada delante de la
cámara. Escribe `data/diagnostico/camara-<fecha>.md` con una tabla ordenada de
mejor a peor por **fps nuevos**; la de arriba es la candidata para `capture.backend`,
`capture.fourcc`, `capture.frame_width` y `capture.frame_height` en `config.yaml`.

**Exposición manual.** Con la exposición automática ninguna configuración pasó
de ~16.6 fps nuevos (ADR 0017). Para barrer exposiciones fijas sobre la
configuración de `config.yaml`:

```bash
uv run lsm-demo medir-camara --exposicion=-5,-6,-7,-8
```

Con `=`: sin él, `-5` se lee como otra opción. En Windows los valores suelen ser
log2 de segundos (-6 ≈ 1/64 s). La tabla añade la luminancia media: si un valor
sube los fps pero deja la imagen demasiado oscura, no sirve sin más luz. El
elegido va a `capture.exposure` (hoy -5, ADR 0017).

**La columna «aceptado por el driver» no es confiable para la exposición.** Con
la webcam de referencia en MSMF dijo `exp -5 (auto 0)` en todas las filas
mientras la luminancia bajaba de 0.44 a 0.06: el driver sí aplicaba cada valor,
pero lee de vuelta otro. No la uses para decidir; la luminancia y los fps son
lo que se mide de verdad.

Barre de día **y** con la luz con la que vas a usar la demo: con luz de día todas
las exposiciones, la automática incluida, dieron ~28 fps nuevos, así que de día
el barrido no discrimina. Los 16.6 fps de la primera medición eran la exposición
automática alargándose con poca luz.

Las filas de `MSMF` y `DSHOW` solo funcionan en Windows. Desde Windows, como la demo:

```powershell
& $HOME\lsm-win\Scripts\python.exe -m lsm.cli.demo medir-camara
```

**Descarte de repetidos.** Con `capture.drop_duplicate_frames` (activado por
defecto) la cámara ya no entrega a MediaPipe los cuadros que repiten al anterior;
la tasa que ven la demo y la captura es la de cuadros nuevos. El sondeo los mide
sin descartar, porque lo que quiere es contarlos.

---

## 7. Texto a señas 🖼️

La dirección inversa: escribes y ves las señas. No necesita cámara ni MediaPipe,
pero sí Pillow y OpenCV (`make setup-capture`) para la ventana.

```bash
uv run lsm-signs reproducir "hola mundo"
# o: make signs TEXTO="hola mundo"
```

Se abre una ventana: la seña a la izquierda; la letra, su descripción del
glosario, el estado (`REPRODUCIENDO` / `PAUSA` / `FIN`) y la barra de progreso
a la derecha; el texto completo abajo con el símbolo actual resaltado. `"ll"`
y `"rr"` son un solo símbolo; los acentos se quitan; `ñ` es `Ñ`. Dígitos y
puntuación se rechazan con la lista exacta.

**Controles:** `ESPACIO` pausa, `n`/`p` siguiente/anterior, `r` reinicia,
`+`/`-` velocidad, `q` sale. Las duraciones viven en `config.yaml`, sección
`signs`: una letra estática se sostiene `static_hold_ms`; una dinámica,
`duracion_ms` (del manifest) `× dynamic_loops` — con los valores por defecto,
~5 s.

### 7.1 Regenerar los assets

Los 29 assets de `assets/signs/` están versionados y se dibujan desde el
dataset propio, nunca de internet (`docs/adr/0014-assets-como-esqueleto-del-dataset-propio.md`).
Para regenerarlos tras regrabar:

```bash
uv run lsm-signs render --revisor "tu nombre"
# o: make signs-render REVISOR="tu nombre"
```

Elige por letra la muestra más típica de `data/raw` (mano derecha si la hay) y
escribe `manifest.json`. Las revisiones ya hechas se conservan si la muestra
elegida no cambió; si cambió, la letra vuelve a `pendiente` y hay que mirarla
otra vez contra `docs/glosario-lsm.md` §3 y ponerla en `coincide` a mano.

### 7.2 Verificar

```bash
uv run lsm-signs verificar
# o: make signs-verificar
```

Exige 29 letras, archivos presentes, dinámicas en GIF con los frames que dice
`duracion_ms`, cero deriva contra el glosario y todas las revisiones en
`coincide`. Es lo mismo que exige `tests/test_signs_manifest.py` en CI.

La revisión registrada es contra la descripción del glosario, no una
validación por persona usuaria de LSM o intérprete: ese pendiente sigue
abierto (`docs/adr/0014-assets-como-esqueleto-del-dataset-propio.md`).

---

## 8. Cuando algo falla

### `ModuleNotFoundError: No module named 'lsm'`

El entorno perdió el archivo que hace importable el paquete. **`uv sync` no lo
repara**: ve el `dist-info` del proyecto, da el entorno por sincronizado y no
repone nada. Hay que reconstruirlo:

```bash
rm -rf .venv && uv venv && uv sync
```

Para confirmar que es eso, compara los `.pth` que hay con los que debería haber:

```bash
ls .venv/lib/python*/site-packages/*.pth
cat .venv/lib/python*/site-packages/lsm_translator-*.dist-info/RECORD | head
```

### MediaPipe escribe cosas que parecen errores

```
Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI.
```

Sale del grafo que MediaPipe distribuye ya compilado dentro del `.task` y no se
puede silenciar desde la API de Python. Aparece **solo cuando hay una mano en el
encuadre**: si lo ves, es que la detección funciona. Para comprobarlo sin fiarte
de esto, mira que el esqueleto verde se dibuje sobre la mano y la siga.

### `verificar` denuncia muestras que se grabaron bien

Si las diferencias son de unos pocos ULPs, es ruido de punto flotante entre
sistemas operativos: `atan2`, `sin` y `cos` no están correctamente redondeadas
por IEEE 754 y la libm de Windows y la de Linux discrepan en el último bit. El
comando ya lo tolera con una tolerancia relativa de `1e-12`. Si la desviación es
mucho mayor —del orden de `1e-5`— entonces sí hay una pérdida real de precisión.
Ver `docs/adr/0009-verificacion-entre-plataformas.md`.

### `grabar` se niega a arrancar

Es a propósito, y el mensaje dice cuál de los tres cerrojos falta:

1. **calibración vigente** para esa cámara → sección 4.2
2. **glosario validado** por una persona usuaria de LSM (sección 5 de
   `docs/glosario-lsm.md`) → o usa `--sesion-prueba`, que no lo exige
3. **consentimiento registrado**, si pediste `--guardar-video` → sección 4.3

Un aviso en una terminal antes de cuarenta minutos de grabación con otra persona
esperando no lo lee nadie, así que son rechazos y no avisos.

---

## 9. Dónde seguir leyendo

En este orden:

1. `CLAUDE.md` — reglas no negociables del repositorio
2. `docs/ARQUITECTURA.md` — la fuente de verdad sobre el diseño
3. `docs/feature-spec.md` — el contrato de la extracción de features
4. `docs/dataset-schema.md` — qué se guarda por muestra y por qué
5. `docs/glosario-lsm.md` — qué letras se reconocen y cuáles llevan movimiento
6. `docs/adr/` — las decisiones y por qué se tomaron
7. `docker/README.md` — cámara, contenedor y el caso WSL + Windows
