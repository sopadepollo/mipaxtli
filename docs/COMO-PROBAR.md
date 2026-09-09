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

Levanta la mano **derecha** y comprueba que el preview dice `RIGHT`. Si dice
`LEFT`, cancela con `q`, invierte `hands.mediapipe_reports_mirrored_handedness`
en `config.yaml` y repite.

**Este minuto no se salta**, y `grabar` se niega a abrir la cámara sin él. Si el
ajuste está al revés no pasa nada visible: el modelo entrena igual de bien y la
precisión es idéntica. El error solo aparece en la Fase 7, cuando la app web
confunda cada seña con su espejo. Ninguna prueba automática lo detecta; solo un
ojo humano. Ver `docs/adr/0007-cierre-de-captura.md`.

La calibración se invalida sola si cambias de cámara, de resolución o ese ajuste.

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
  --luz-nivel INDOOR --luz-direccion FRONTAL --distancia MEDIUM
```

Escribe en `data/raw/pruebas/`, que no entra al dataset. Es para encuadrar la
cámara antes de citar a nadie.

La sesión formal:

```bash
uv run lsm-capture grabar --firmante s01 --sesion 2026-09-09-manana \
  --luz-nivel INDOOR --luz-direccion FRONTAL --distancia MEDIUM
```

Las condiciones son obligatorias a propósito: anotarlas después, de memoria, no
funciona (`docs/adr/0005-taxonomias-de-metadatos-de-captura.md`). **Varía luz y
distancia entre sesiones, nunca dentro de una**, o los metadatos mienten sobre la
mitad de las muestras.

En el preview: **ESPACIO** guarda, `n`/`p` cambian de letra, `m` alterna entre
estática y dinámica, `r` descarta la grabación en curso, `q` sale.

La barra de calidad dice si la ventana serviría *ahora mismo*, y mide cosas
opuestas según el modo:

| Modo | Barra | Verde cuando |
|---|---|---|
| estática | `sigma` | la mano está **quieta** |
| dinámica | `arco` | el trazo ya **recorrió** lo suficiente |

Guardar con la barra en rojo no se puede.

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

---

## 5. Entrenar y evaluar

Ninguno de los dos necesita cámara ni MediaPipe.

```bash
uv run lsm-train --sin-sintetico     # o: make train
uv run lsm-eval  --sin-sintetico     # o: make eval
```

`lsm-train` exporta a `data/models/static_knn.json` y **no mide nada** a
propósito: la precisión honesta la da `lsm-eval`.

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

---

## 6. La demo

> **Fase 3, en construcción.** El diseño está en
> `docs/superpowers/specs/2026-09-09-fase-3-deletreo-en-vivo-design.md`. Esta
> sección se completa cuando `lsm-demo` exista; hasta entonces, `make demo`
> avisa de que todavía no está.

Lo que habrá:

```bash
uv run lsm-demo                          # 📷 sesión en vivo
uv run lsm-demo --desde-dataset RUTA     # reproduce grabaciones, sin cámara
```

Y el criterio de la fase —deletrear una palabra de cinco letras sin errores de
segmentación— se comprueba en la suite, sin cámara, con
`tests/test_cli_demo.py`.

---

## 7. Cuando algo falla

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

## 8. Dónde seguir leyendo

En este orden:

1. `CLAUDE.md` — reglas no negociables del repositorio
2. `docs/ARQUITECTURA.md` — la fuente de verdad sobre el diseño
3. `docs/feature-spec.md` — el contrato de la extracción de features
4. `docs/dataset-schema.md` — qué se guarda por muestra y por qué
5. `docs/glosario-lsm.md` — qué letras se reconocen y cuáles llevan movimiento
6. `docs/adr/` — las decisiones y por qué se tomaron
7. `docker/README.md` — cámara, contenedor y el caso WSL + Windows
