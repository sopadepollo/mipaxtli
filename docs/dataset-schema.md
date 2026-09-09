# Esquema del dataset

Define qué se guarda por muestra y por qué. La estructura vive en
`src/lsm/types.py` (`Sample`) y el formato en disco en `src/lsm/io/hands.py`.

El dataset es el cuello de botella real del proyecto (`ARQUITECTURA.md` §4.7): el
modelo no será mejor que los datos, y los metadatos de aquí abajo no son
burocracia, son lo que permite medir si el modelo generaliza.

## Qué es una muestra

Una `Sample` es **una secuencia etiquetada con sus metadatos**:

| Campo | Tipo | Para qué sirve |
|---|---|---|
| `sequence` | `(T, 21, 3)` landmarks crudos | La seña. Estática o dinámica, mismo tipo. |
| `label` | `str` | La letra. Mayúsculas y sin acentos: `Ñ` → `ENIE`, `LL` → `DOBLE_L`, `RR` → `DOBLE_R`. Más la clase negativa `NONE`. |
| `signer_id` | `str` | Quién firma. **Sin esto no hay leave-one-signer-out.** |
| `session_id` | `str` | Qué grabación. Frames de la misma sesión no pueden repartirse entre train y test. |
| `timestamp` | `datetime` con zona | Cuándo. ISO-8601 con offset; se exige zona horaria para que sea comparable entre máquinas. |
| `handedness` | `LEFT` / `RIGHT` | Con qué mano firma. La normalización canoniza a derecha, pero el dato se conserva para poder medir si el modelo falla más con zurdos. |
| `light_level` | `DIM`, `INDOOR`, `BRIGHT` | Cuánta luz hay. Anotado a mano. |
| `light_direction` | `FRONTAL`, `LATERAL`, `BACKLIT`, `MIXED` | De dónde viene la luz. Anotado a mano. |
| `distance` | `NEAR`, `MEDIUM`, `FAR` | Distancia aproximada a la cámara. Etiqueta gruesa para filtrar a ojo. |
| `mean_luminance` | `float` en `[0, 1]` | Luminancia media del frame, promediada sobre la secuencia. **Calculada.** |
| `mean_scale_px` | `float` > 0 | Escala del paso 4 en píxeles, promediada sobre la secuencia. **Calculada.** |

### Dos ejes de luz, no uno

`BACKLIT` no es un nivel de iluminación, es una dirección: una escena a contraluz
puede ser brillante o penumbrosa, y para el detector son problemas distintos. Con
un solo campo habría que elegir cuál de las dos cosas se anota y se perdería la
otra, justo cuando el contraluz es la condición que más degrada la detección.

### Cada categoría lleva su número

Las tres taxonomías dependen del juicio de quien graba, y dos personas etiquetarán
distinto la misma escena. Por eso cada una viaja acompañada de una medida objetiva
que sale gratis:

- `light_level` ↔ `mean_luminance`, que es una media de píxeles.
- `distance` ↔ `mean_scale_px`, que es el tamaño aparente de la mano — la magnitud
  que el paso 4 **ya calcula** para normalizar, devuelta a píxeles con
  `lsm.features.scale_to_pixels`.

`NEAR/MEDIUM/FAR` no es más que una discretización pobre de ese número. Se conservan
las dos porque sirven para cosas distintas: la categoría para filtrar el dataset a
ojo y planear la captura ("faltan sesiones a contraluz"), el número para responder
con datos si el modelo empeora con poca luz o a distancia, que es la pregunta que
se hará al leer la primera matriz de confusión.

Las taxonomías están **congeladas** en `docs/adr/0005-taxonomias-de-metadatos-de-captura.md`:
cambiarlas después de la primera sesión invalida metadatos ya grabados.

## Se guardan landmarks crudos, no features

Regla no negociable. Si cambia la normalización de `feature-spec.md`, un dataset
de features hay que **regrabarlo con personas frente a la cámara**; uno de
landmarks crudos se re-deriva con un comando.

El costo es despreciable: 21 landmarks × 3 floats × `T` frames por muestra son
unos pocos kilobytes en JSON. El video, en cambio, no se guarda salvo con
consentimiento explícito y por escrito de quien firma.

## Los huecos se guardan como huecos

Un frame donde el detector no encontró la mano se escribe con su centinela
(`{"valid": false, "reason": "NO_HAND"}`), no se omite ni se interpola. Si se
perdiera al guardar, una secuencia interrumpida se convertiría en una continua y
el dataset mentiría sobre lo que ocurrió frente a la cámara.

## Formato en disco

### El flujo de frames

Es el bloque que comparten los fixtures de los tests y las muestras del dataset:

```json
{
  "schema_version": 1,
  "frames": [
    {
      "valid": true,
      "width": 1280,
      "height": 720,
      "handedness": "RIGHT",
      "handedness_score": 0.98,
      "detection_score": 0.95,
      "landmarks": [[0.51, 0.62, 0.0], "... 21 tripletas ..."]
    },
    { "valid": false, "reason": "NO_HAND", "detail": "la mano salió del encuadre" }
  ]
}
```

Un archivo de otra `schema_version` se rechaza al cargar. Hay un ejemplo generado
por `make golden` en `tests/fixtures/sequences/ejemplo_trazo_j.json`.

Que las dos cosas compartan la representación de un frame no es comodidad:
significa que una muestra grabada de verdad se puede pegar tal cual como fixture
de un test, y que un fixture se lee con el mismo código que lee el dataset. Lo
implementa `lsm.io.hands.frames_to_json` y su inverso.

### Una muestra

`data/raw/<firmante>/<sesión>/<LETRA>/007.json`. Los metadatos de la tabla de
arriba, envolviendo el mismo arreglo `frames`:

```json
{
  "schema_version": 2,
  "capture_spec_version": 2,
  "label": "A",
  "signer_id": "s01",
  "session_id": "2026-09-08-manana",
  "timestamp": "2026-09-08T11:30:12.482913-06:00",
  "handedness": "RIGHT",
  "light_level": "INDOOR",
  "light_direction": "FRONTAL",
  "distance": "MEDIUM",
  "mean_luminance": 0.4412,
  "mean_scale_px": 98.53,
  "kind": "STATIC",
  "dispersion": 0.0123,
  "arc_length": 0.0412,
  "handedness_swapped": true,
  "handedness_convention": "SIGNER",
  "video": null,
  "frames": ["... como arriba ..."]
}
```

Los cinco últimos campos antes de `frames` no describen la seña sino **cómo se
grabó**, y sirven para auditar el dataset sin volver a procesarlo:

| Campo | Para qué |
|---|---|
| `capture_spec_version` | Con qué criterio de aceptación se admitió (`lsm.capture.CAPTURE_SPEC_VERSION`). Se versiona aparte de `FEATURE_SPEC_VERSION` porque cambia por otros motivos: ajustar cuánta quietud se le exige a quien graba no invalida ninguna muestra ya grabada. |
| `kind` | `STATIC` o `DYNAMIC`: qué se **hizo** frente a la cámara, que no siempre es lo que el glosario dice que la letra **es**. `NONE` se graba de las dos formas, y una estática grabada por error en modo dinámico tiene que poder reconocerse al depurar. |
| `dispersion` | La σ del `feature-spec.md` §2 en el momento de aceptar. Decide en las estáticas. Se guarda también en las dinámicas, donde no decide: al leer la primera matriz de confusión permite preguntar si las muestras peores eran las más temblorosas, y esa pregunta no se puede hacer si el número no se guardó. |
| `arc_length` | Longitud de arco de la trayectoria τ (§3.1), en unidades de mano. El espejo del anterior: decide en las dinámicas y se guarda sin decidir en las estáticas. |
| `handedness_swapped` | Valor **efectivo** de `hands.mediapipe_reports_mirrored_handedness` al grabar. Ver abajo. |
| `handedness_convention` | Qué mano nombra `handedness`: `SIGNER` (la anatómica de quien firma) o `IMAGE` (la de la imagen espejada). Es la promesa semántica; `handedness_swapped` es cómo se llegó a ella. |
| `video` | Nombre del archivo de video hermano, o `null`. **Nunca se rellena sin consentimiento registrado.** |

`video: null` se escribe explícitamente y no se omite: una clave que falta se
confunde con un archivo truncado.

#### Por qué la lateralidad se anota dos veces

Porque el error que tapa **no produce ningún síntoma**. Si
`mediapipe_reports_mirrored_handedness` está al revés, el paso 2 canoniza *todas*
las muestras hacia la mano contraria: las dos poblaciones de vectores difieren por
un espejo global y nada más, el modelo entrena igual de bien y la precisión es
idéntica.

Sin `handedness_swapped`, cambiar el interruptor a mitad del proyecto dejaría el
dataset mezclado —media parte con una convención, media con la otra, todo con la
misma pinta— y no habría forma de saber qué muestra es cuál. Con él, la reparación
es un filtro y un espejo en vez de una nueva ronda de grabaciones.

`handedness_convention` viaja además en el JSON del modelo exportado
(`ARQUITECTURA.md` §4.6) y el runtime rechaza la carga si no coincide con la suya.
Es lo que protegerá a la Fase 7 de MediaPipe JS, y hace falta porque **los golden
vectors no cubren esto**: reciben la lateralidad ya resuelta como entrada, así que
el test de paridad pasaría en verde con la app web reconociendo cada seña al revés.
Ver `docs/adr/0007-cierre-de-captura.md`.

### Disposición en `data/raw/`

```
data/raw/
├── consentimiento.json           # registro de permisos, por firmante
├── calibracion.json              # registro de lateralidad, por cámara
├── pruebas/                      # sesiones de ensayo; NO entran al dataset
│   └── <signer_id>/…
└── <signer_id>/
    └── <session_id>/
        └── <LABEL>/
            ├── 001.json          # una muestra
            ├── 001.mp4           # video hermano, SOLO con consentimiento
            └── 002.json
```

`pruebas/` guarda lo que graba `lsm-capture grabar --sesion-prueba`: sesiones para
encuadrar la cámara y ensayar antes de citar a nadie. Vive **dentro** de la raíz
para compartir los dos registros —son del mismo equipo y las mismas personas— pero
un nivel más abajo, de modo que el recorrido del dataset no lo alcanza. Para
recorrerlas a propósito, `--raiz data/raw/pruebas`.

**La jerarquía empieza por la persona** porque las dos operaciones que más se
hacen sobre este árbol son por firmante: el split es leave-one-signer-out, y el
borrado a petición de quien firma es de todas sus muestras. Con las letras arriba,
las dos serían un recorrido del árbol entero.

`signer_id`, `session_id` y `label` viajan a nombres de carpeta, así que se
restringen a `[A-Za-z0-9][A-Za-z0-9_-]{0,63}`. No es cosmética: sin la
restricción, un `../` en un identificador escribiría fuera de `data/raw/`.

**La ruta duplica lo que ya dice el archivo, y el archivo manda.** La ruta sirve
para navegar y para contar muestras sin abrir nada —es lo que alimenta el contador
del preview, que se refresca treinta veces por segundo—, pero si alguien mueve una
carpeta, la verdad sigue estando dentro del JSON.

Los números de muestra no se reutilizan: se toma el mayor existente más uno. Si la
003 se borró por mala, la siguiente es la 006, para que la bitácora de la sesión no
acabe hablando de una muestra distinta a la que hay en disco.

### El modelo del detector no vive aquí

`data/models/hand_landmarker.task` es el bundle de MediaPipe Tasks, pesa unos 8 MB
y **no se versiona**. Se descarga con `make model`. No es parte del dataset: es una
dependencia del detector, y está bajo `data/` solo porque `data/` ya está en el
`.gitignore`.

## Calibración de la cámara

`data/raw/calibracion.json`:

```json
{
  "schema_version": 1,
  "camaras": {
    "V4L2:0@1280x720": {
      "swap_handedness": true,
      "convention": "SIGNER",
      "fecha": "2026-09-08T09:15:00-06:00",
      "width": 1280,
      "height": 720,
      "confirmado_por": "quien grabó"
    }
  }
}
```

Registra que **una persona miró la pantalla** y confirmó que la lateralidad que
reporta el detector es la mano real. Es el único dato del proyecto que ninguna
prueba puede producir: el sistema no tiene forma de saber qué mano hay delante de
la cámara.

Se escribe con `lsm-capture calibrar` y `lsm-capture grabar` lo exige: sin
calibración vigente no abre la cámara. Un rechazo, no un aviso — el fallo que tapa
no deja ningún rastro en el dataset.

La clave es `backend:índice@anchoxalto`. No hay forma portable de leer el número de
serie de una webcam, así que se compone con lo que sí cambia el resultado: el
índice distingue la cámara integrada de la externa, la resolución **realmente
entregada** cambia el recorte y la relación de aspecto, y el backend de OpenCV
cambia el comportamiento del mismo dispositivo.

Una calibración deja de ser vigente cuando cambia la cámara o cuando cambia
`hands.mediapipe_reports_mirrored_handedness`, porque el registro dice que alguien
vio `RIGHT` **con aquel ajuste**. No caduca por tiempo: no se estropea sola.

## Validación: leave-one-signer-out

**Nunca un split aleatorio de frames.** Frames consecutivos de la misma grabación
son casi idénticos: repartirlos entre train y test mide memorización, no
generalización, y da métricas infladas que se derrumban en la demo.

El split se hace **por persona**: se entrena con todas las personas menos una y se
evalúa con la que quedó fuera, rotando. Por eso `signer_id` no puede ir vacío, y
por eso `types.py` lo valida al construir la muestra en vez de confiar en que
alguien se acuerde.

## Objetivo de cobertura

Mínimo, según `ARQUITECTURA.md` §4.7:

- **3 personas distintas**, 2 sesiones cada una, ~20 repeticiones por letra.
- Variar iluminación y distancia entre sesiones, no dentro de una.
- Incluir la clase negativa `NONE`: mano relajada, transiciones entre letras y
  gestos cotidianos que no son señas. Sin ella, el clasificador asigna una de las
  29 letras aunque la persona no esté firmando. El vocabulario exacto —29 letras
  más `NONE`— está en `src/lsm/vocabulary.py`, transcrito del glosario.

Son unos 40 minutos por persona. El error clásico —un dataset de una persona, una
sesión, una iluminación— da 98% en validación y 40% en la demo.

## Consentimiento

No se almacena video sin consentimiento explícito y por escrito de quien firma. Los
landmarks no son identificables, el video sí. El consentimiento se registra por
`signer_id`, y quien firma puede pedir que se borren sus muestras.

### Cómo se registra

`data/raw/consentimiento.json`:

```json
{
  "schema_version": 1,
  "firmantes": {
    "s01": {
      "video": true,
      "fecha": "2026-09-08T10:55:03-06:00",
      "referencia": "carta firmada, expediente 2026-03"
    }
  }
}
```

**Este archivo no es el consentimiento.** El consentimiento es explícito y por
escrito, en papel o su equivalente; esto es el registro de que existe, y
`referencia` es dónde encontrarlo. Se escribe con:

```bash
lsm-capture consentimiento --firmante s01 --video --referencia "expediente 2026-03"
```

Sin `--video`, la persona queda registrada pero **no** autoriza almacenar cuadros:
son dos preguntas distintas y la segunda tiene su propia bandera. Volver a
ejecutar el comando sin `--video` revoca el permiso, porque quien firma puede
cambiar de opinión.

### Dos llaves para guardar video

Grabar video exige las dos a la vez, y ninguna se abre sola:

1. Un consentimiento registrado con `video: true` **para esa persona concreta**.
2. La bandera `--guardar-video` en la línea de comandos de la sesión.

Un registro que falta es una autorización que no se pidió, nunca una que se dio:
sin archivo, la respuesta es «no». Y aun con permiso, el video solo se guarda si
se pide explícitamente: apagado por defecto significa apagado por defecto.

Si se piden cuadros sin permiso, la sesión **no arranca** —falla antes de abrir la
cámara, no a mitad de grabar— y el mensaje dice cómo registrar el consentimiento.

El video se guarda **sin espejar y sin landmarks dibujados encima**: es el registro
de lo que ocurrió frente a la cámara, no una captura de pantalla del programa. Si
algún día hay que re-derivar landmarks de él —que es la única razón seria para
guardarlo— tienen que salir los mismos.
