# Traductor de deletreo manual — LSM

Traductor bidireccional del **alfabeto dactilológico** de la Lengua de Señas
Mexicana. Reconoce por cámara las señas manuales del abecedario y las convierte en
texto; a la inversa, muestra las señas correspondientes a un texto escrito.

## Qué NO es esto

**No es un traductor de lengua de señas.** La LSM es una lengua completa, con
gramática propia, señas léxicas y componentes no manuales (expresión facial,
movimiento corporal, uso del espacio). Este proyecto reconoce únicamente el
**deletreo manual**: las 27 letras del abecedario, una por una.

El deletreo manual es un recurso puntual dentro de la LSM —se usa sobre todo para
nombres propios y préstamos—, no la lengua. Esta herramienta es un apoyo acotado y
**no sustituye a un intérprete**.

Tampoco hay reconocimiento facial ni identificación de personas, y ningún frame
sale del dispositivo: todo el procesamiento es local por diseño.

## Estado

**Fase 1 — captura, cerrada.** Al núcleo puro de la Fase 0 (tipos, features,
segmentación, contrato de clasificadores, configuración, golden vectors) se le suma
la recolección de dataset: el detector real de MediaPipe, la cámara, el preview con
landmarks y el CLI `lsm-capture`, con criterio de calidad para los dos modos y los
tres cerrojos que impiden grabar mal (calibración, glosario validado,
consentimiento). No hay todavía entrenamiento, demo ni app web. Ver el plan de
fases en `docs/ARQUITECTURA.md` §5.

**Fase 2 — clasificador y evaluación, cerrada.** `static_knn` (vecino más cercano
por centroides, con tres puertas de rechazo y export a JSON portable), `make train`
y `make eval`. La evaluación usa leave-one-signer-out y produce accuracy global y
por letra, matriz de confusión, pares más confundidos, barrido de calibración,
diagnóstico empírico de umbrales y el contraste de la columna `confundible_con` del
glosario contra la matriz real. La fase se restringe a las **21 letras estáticas más
`NONE`**; las 8 dinámicas esperan a la Fase 5. Ver
`docs/adr/0008-clasificador-estatico-y-protocolo-de-evaluacion.md`.

**Criterio cumplido: 0.9261 de accuracy** (0.9201 macro, 5.7% de rechazo) con
validación leave-one-signer-out sobre 3 firmantes y 2585 muestras estáticas. Los
umbrales de `config.yaml` ya no son «razonados, no medidos»: salen del barrido
contra el dataset `fe0ada8710b6`. La decisión que más pesó fue la métrica —coseno
en vez de euclidiana, +9 puntos— y está explicada en
`docs/adr/0011-calibracion-de-la-fase-2.md`, junto con la lista de lo que queda por
refinar: el par `C`·`O`, las letras `O`, `S` y `V`, y el hecho de que los umbrales
se ajustaron sobre el mismo conjunto con el que se miden.

**El dataset ya existe**: 3 firmantes, 2 sesiones cada uno, 3407 muestras. Sin él,
`make train` y `make eval` caen a un corpus **sintético** determinista y lo dicen en
la cabecera del reporte, dentro del modelo exportado y en la sección de contraste de
hipótesis. `--sin-sintetico` exige dataset real.

**Fase 3 — deletreo en vivo, cerrada.** `cli/demo.py` conecta cámara → MediaPipe
→ features → segmentación → clasificador → `spelling.py`, el buffer que
acumula letras en palabras y frases. Controles: bajar la mano cierra la
palabra (es el mismo gesto de ausencia que ya detectaba la segmentación, no
una clase nueva); `BACKSPACE` borra un símbolo y `ENTER` cierra la frase, por
teclado — el clasificador ya usa sus 22 clases en las 21 letras más `NONE` y no
hay ninguna libre para un gesto de control (`docs/adr/0012-controles-del-deletreo.md`).

**Criterio cumplido:** deletrear una palabra de cinco letras sin errores de
segmentación. `tests/test_cli_demo.py::test_una_palabra_de_cinco_letras_produce_cinco_simbolos`
lo comprueba en CI, sin cámara y sin MediaPipe, con un `StaticKnnClassifier`
entrenado de verdad sobre secuencias sintéticas: cinco señas producen cinco
símbolos, `"casas"`.

Las **ocho letras dinámicas** (`J`, `K`, `LL`, `Ñ`, `Q`, `RR`, `X`, `Z`) siguen
sin reconocerse —llegan en la Fase 5 con `dynamic_dtw`— y el HUD de la demo lo
avisa en pantalla mientras dura la sesión, para que no se confunda con un
fallo. **La sesión en vivo con cámara real no se ha ejecutado todavía**: lo
único probado es `lsm-demo --desde-dataset`, que reproduce una grabación sin
abrir cámara ni tocar MediaPipe. Ver la sección 6 de `docs/COMO-PROBAR.md`.

Esa propuesta de cambio arquitectónico ya se resolvió:
`docs/adr/0013-la-ventana-mezclada.md` documentaba que la máquina de estados
comprobaba quietud sobre los últimos frames pero clasificaba el buffer entero, y
que con un tránsito corto entre dos letras eso emitía una letra que nadie firmó.
Está corregido en `SEGMENTATION_SPEC_VERSION = 2`: la ventana que se clasifica es
el tramo verificado estable, la emisión es progresiva —una letra segura sale con
la ventana mínima y un par confundible acumula evidencia— y los umbrales
temporales pasan a milisegundos derivados de la tasa **medida**, que resultó ser
17.8 fps y no los 30 que suponían los comentarios. `lsm-demo --medir-fps` mide esa
tasa; ver la sección 6.1 de `docs/COMO-PROBAR.md`.

**Fase 4 — texto a señas, cerrada.** Texto → señas: `lsm-signs reproducir
"casa"`. Los 29 assets son esqueletos renderizados desde el dataset propio, con
fuente y revisión por letra en `assets/signs/manifest.json`; ver ADR 0014. La
revisión registrada la hizo un agente contra la descripción del glosario, no
una persona: **ninguna persona ha mirado todavía los 29 assets**, y la
validación por persona usuaria de LSM (el PENDIENTE-HUMANO G del glosario)
sigue pendiente.

**Nada de eso hace falta para trabajar en el núcleo.** MediaPipe y OpenCV son
dependencias opcionales: `make test` pasa sin cámara, sin modelo y sin ninguna de
las dos instaladas.

## Por dónde empezar

**[`docs/COMO-PROBAR.md`](docs/COMO-PROBAR.md)** — todo lo que se puede ejecutar,
en el orden en que tiene sentido hacerlo: la suite, la calibración de la cámara,
la captura de dataset, el entrenamiento, la evaluación, la demo y qué hacer
cuando algo falla. Si solo vas a leer un archivo antes de tocar el repositorio,
que sea ese.

## Documentos normativos

Antes de tocar código, en este orden:

1. `CLAUDE.md` — reglas no negociables del repositorio.
2. `docs/ARQUITECTURA.md` — fuente de verdad sobre el diseño.
3. `docs/feature-spec.md` — **contrato** de la extracción de features (§1-§5) y de
   la segmentación (§6, versionada aparte). Cualquier implementación, Python o
   TypeScript, debe reproducirlo dentro de `1e-6`.
4. `docs/adr/` — decisiones de arquitectura y por qué se tomaron.
5. `docs/glosario-lsm.md` — qué letras se reconocen y cuáles llevan movimiento.
6. `docs/COMO-PROBAR.md` — cómo se ejecuta cada cosa, con qué hace falta para cada una.

## Requisitos

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) para gestionar el entorno
- `make` (opcional; el `Makefile` es un atajo, ver equivalentes abajo)

Nada de esto necesita cámara ni MediaPipe: el núcleo se testea con secuencias
sintéticas.

## Uso

```bash
make setup     # uv sync — instala dependencias
make test      # pytest + mypy strict + ruff. Sin cámara, sin MediaPipe, sin dataset
make lint      # solo ruff (check + format --check)
make golden    # regenera tests/fixtures/golden_features.json
```

### Entrenar y evaluar

Tampoco necesitan cámara ni MediaPipe: leen `data/raw` y, mientras esté vacío, un
corpus sintético determinista.

```bash
make train     # entrena static_knn -> data/models/static_knn.json
make eval      # reporte completo   -> data/models/eval/
```

`make eval` escribe tres archivos: `reporte-fase2.md` para leer,
`resultados-fase2.json` para que lo lea otro programa, y `calibracion-fase2.json`
con los umbrales que el barrido recomienda **y la huella del dataset contra el que
se ajustaron**. Ninguno lleva marca de tiempo: dos ejecuciones sobre el mismo
dataset dan los mismos bytes, que es lo que permite comparar dos calibraciones.

> **Dentro de una misma máquina.** `atan2`, `sin` y `cos` no están correctamente
> redondeadas por IEEE 754, así que la libm de Windows y la de Linux discrepan en
> el último bit y todo lo que se deriva de `features.py` lo hereda. Dos reportes
> generados en sistemas distintos pueden diferir en los últimos decimales sin que
> nada esté mal. Comparar calibraciones exige generarlas en el mismo sitio; ver
> `docs/adr/0009-verificacion-entre-plataformas.md`.

Opciones que se usan de verdad:

```bash
make eval ARGS="--barrido minimo"                    # pasada rápida
make eval ARGS="--protocolo leave-one-session-out"   # con un solo firmante grabado
make eval ARGS="--sin-sintetico"                     # exige dataset real
```

Con un solo firmante en el dataset, `make eval` **se niega** en vez de degradar
solo: una métrica por sesión y una por persona no son comparables, y la degradación
tiene que quedar escrita en el reporte.

### Grabar dataset

Solo esto necesita cámara. Dos pasos de una vez:

```bash
make setup-capture   # instala MediaPipe y OpenCV (dependencias opcionales)
make model           # descarga hand_landmarker.task (~8 MB, no se versiona)
```

> **¿Trabajas en WSL?** WSL2 no ve la webcam: no hay `/dev/video*` sin
> `usbipd-win`, y aun con él la cámara deja de funcionar en Windows mientras esté
> conectada. La captura se ejecuta desde Windows contra el repositorio de WSL, que
> se lee por UNC sin copiar nada. La receta comprobada está en `docker/README.md`,
> sección (b). Lo mismo vale para macOS: la captura va fuera del contenedor.

Y una vez por cámara, la calibración:

```bash
uv run lsm-capture calibrar --confirmado-por "tu nombre"
```

Levanta la mano **derecha** y comprueba que el preview dice `RIGHT`. Si dice
`LEFT`, cancela con `q`, **invierte** `hands.mediapipe_reports_mirrored_handedness`
en `config.yaml` y repite.

El valor de fábrica (`false`, o sea "no invertir") está medido contra MediaPipe
1.0.1 y una webcam real, no deducido de la documentación — que dice lo contrario.
Si tu combinación de cámara y versión no coincide, esta pantalla es donde se ve.

> **Por qué este minuto no se salta.** MediaPipe decide la lateralidad asumiendo
> una imagen espejada, y aquí se le alimenta sin espejar (`feature-spec.md` §0.3).
> Si el ajuste está al revés, el paso 2 canoniza **todas** las muestras hacia la
> mano contraria — y no pasa nada: el modelo entrena igual de bien, infiere igual
> de bien y la precisión es idéntica. El error solo aparece en la Fase 7, cuando
> MediaPipe JS use la convención contraria y la app web confunda cada seña con su
> espejo. Ninguna prueba automática lo detecta; solo un ojo humano.
>
> Por eso `grabar` **se niega a abrir la cámara** sin calibración vigente, y por
> eso la calibración se invalida sola si alguien toca ese ajuste.

Y luego, por sesión:

```bash
make capture ARGS="--firmante s01 --sesion 2026-09-08-manana \
                   --luz-nivel INDOOR --luz-direccion FRONTAL --distancia MEDIUM"
```

Las condiciones de luz y distancia son obligatorias a propósito: anotarlas
después, de memoria, no funciona (`docs/adr/0005-taxonomias-de-metadatos-de-captura.md`).

En el preview: **ESPACIO** guarda, `n`/`p` cambian de letra, `m` alterna entre
estática y dinámica, `r` descarta la grabación en curso, `q` sale.

La barra de calidad dice si la ventana serviría *ahora mismo*, y mide cosas
opuestas según el modo:

| Modo | Barra | Verde cuando |
|---|---|---|
| estática | `sigma` | la mano está **quieta** |
| dinámica | `arco` | el trazo ya **recorrió** lo suficiente |

Guardar con la barra en rojo no se puede. La segunda existe porque una "J" en la
que la mano apenas se movió es, en el dataset, indistinguible de una "I", y
mirando el archivo después no hay forma de saber cuál era cuál.

### Ensayar sin ensuciar el dataset

```bash
uv run lsm-capture grabar --sesion-prueba --firmante s01 --sesion ensayo \
  --luz-nivel INDOOR --luz-direccion FRONTAL --distancia MEDIUM
```

Escribe en `data/raw/pruebas/`, que no entra al dataset, y no exige el glosario
validado. Es para encuadrar la cámara antes de citar a nadie.

**El modo formal sí lo exige**: se niega a arrancar si la sección 5 del glosario
está vacía. Tres personas × 29 letras × dos sesiones contra un glosario que no ha
revisado un intérprete es el error caro del proyecto, y no se arregla salvo
volviendo a citar a todo el mundo.

Para comprobar que el dataset se re-deriva bien (no necesita cámara):

```bash
make verify    # relee data/raw y re-extrae las features de cada muestra
```

**El video está apagado por defecto y necesita dos llaves**: consentimiento
registrado para esa persona y `--guardar-video` en la línea de comandos. Ver
`docs/dataset-schema.md`.

```bash
uv run lsm-capture consentimiento --firmante s01 --video --referencia "expediente 3"
```

Sin `make` instalado, los equivalentes directos:

```bash
uv sync
uv run pytest
uv run mypy
uv run ruff check . && uv run ruff format --check .
uv run lsm-golden
```

### Los tests marcados `glosario`

Fallan mientras `docs/glosario-lsm.md` tenga huecos que solo una persona puede
cerrar. No son fallos de código: son la señal de que grabar dataset todavía
costaría veinte repeticiones por letra de una seña que quizá esté mal.

**Hoy pasan todos.** La tabla está completa y los pares de confusión son
simétricos.

Un test que **no** lleva esa marca, y no debe llevarla nunca, es el que comprueba
que `src/lsm/vocabulary.py` sigue diciendo lo mismo que la tabla del glosario. Esa
deriva ya ocurrió una vez, y lo que la hizo pasar desapercibida no fue la falta de
un test —lo había— sino que `make test` ya estaba en rojo esperando a una persona.
Tiene que romper `make test-nucleo`, que es el que siempre debe estar limpio.

En Docker (sin cámara, para tests y futuro entrenamiento):

```bash
docker compose -f docker/docker-compose.yml run --rm test
```

Ver `docker/README.md` para el detalle de por qué la captura por cámara se ejecuta
**fuera** del contenedor en Windows y macOS.

## Estructura

El núcleo es código puro y aislado del hardware, que es lo que permite correr la
suite completa en CI:

- `src/lsm/types.py` — tipos base. El de entrada es una secuencia `(T, 21, 3)`.
- `src/lsm/features.py` — implementación normativa de `docs/feature-spec.md`.
- `src/lsm/segmentation.py` — máquina de estados que decide cuándo empieza y
  termina una seña.
- `src/lsm/capture.py` — criterio con el que se acepta o se rechaza una muestra:
  σ para las estáticas, longitud de arco para las dinámicas.
- `src/lsm/evaluation.py` — splits leave-one-signer-out, matriz de confusión,
  barrido de calibración y contraste de hipótesis.
- `src/lsm/classifiers/` — `Protocol` común y `static_knn`; `dynamic_dtw` llega en
  la Fase 5.
- `src/lsm/io/` — única frontera con cámara, MediaPipe y disco.

`src/lsm/features.py`, `segmentation.py`, `capture.py`, `evaluation.py` y
`classifiers/` no
importan OpenCV ni MediaPipe, no leen disco y no abren la cámara. Y los módulos que
sí dependen de ellas las importan de forma **diferida**, así que ni siquiera
importar `lsm.cli.capture` las arrastra. Esa regla no es estética: es lo que hace
que `make test` pase en una máquina sin webcam, y hay un test que lo comprueba.

## Ética y consentimiento

- El procesamiento ocurre en el dispositivo. Ningún frame se envía a un servidor.
- No se almacena video sin consentimiento explícito y por escrito de quien firma.
- El glosario y los assets deben verificarse contra la fuente primaria de LSM
  (**no ASL**) y revisarse con una persona usuaria de LSM o un intérprete antes de
  presentar el proyecto.
