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
