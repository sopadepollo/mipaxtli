# ADR 0018 — Marca de tiempo real por cuadro

- **Estado:** **implementada en parte** el 2026-09-29: §1 (el tipo), §2 (la
  función única, `lsm.timing.frame_times_ms`) y §4 (formatos: muestra esquema 5,
  fixture esquema 2). **§3 no**: la velocidad sigue con la tasa congelada. El
  primer consumidor del tiempo real es el filtro One Euro (ADR 0028), que lo
  pedía explícitamente. Las dinámicas del 2026-09-29 se grabaron sin marca.
- **Fecha:** 2026-09-28
- **Relacionadas:** ADR 0001 (secuencias como tipo base), ADR 0004 (contrato de
  segmentación), ADR 0013 (umbrales en milisegundos), ADR 0017 (velocidad contra
  una ventana, v5)

## Contexto

Los frames no llevan tiempo. Todo lo temporal se deriva del **índice** del cuadro
y de una tasa congelada al arrancar la sesión: los umbrales en milisegundos se
convierten a cuadros (`FrameThresholds`, ADR 0013) y, desde la v5, la velocidad
contra la ventana divide entre `k/fps` en vez de entre el tiempo que pasó de
verdad (ADR 0017).

Funciona mientras los cuadros lleguen a intervalos regulares, y hoy no llegan:

- El descarte de repetidos (ADR 0017, Bloque 1) quita cuadros del flujo. Tras un
  descarte, dos cuadros consecutivos están separados por 66 ms o 100 ms, y la
  máquina los trata como si fueran 33 ms.
- La tasa congelada es una media: con poca luz la cámara cae de 28 a 16 fps a
  mitad de sesión y ningún umbral se entera.
- El dataset viejo se grabó a ~16 cuadros nuevos por segundo con cuadros casi
  repetidos intercalados, y el replay lo pasa a 30 fps nominales. Parte de la
  regresión del replay de la v5 viene de ahí (ADR 0017).

El reloj real ya existe en la tubería: `io/hands.py` le pasa a MediaPipe
`time.monotonic` en milisegundos (`hands.real_timestamps`), y el diagnóstico
registra `wall_ms` por cuadro. Lo que falta es que ese tiempo viaje **con el
cuadro** hasta la segmentación y quede guardado en las muestras.

## Propuesta

### 1. El tipo base

```python
@dataclass(frozen=True, slots=True)
class RawFrame:
    ...
    #: Milisegundos de un reloj monótono, con origen arbitrario por flujo: solo
    #: importan las diferencias. `None` en frames sin tiempo (dataset viejo,
    #: sintéticos).
    timestamp_ms: float | None = None

@dataclass(frozen=True, slots=True)
class InvalidFrame:
    ...
    timestamp_ms: float | None = None
```

- **En el cuadro y no en un arreglo paralelo.** Cada cuadro se describe solo, y
  lo que copia frames con `replace` —la mano declarada de `to_sample`, el
  descarte, el replay— conserva el tiempo sin tener que saber de él. Un arreglo
  paralelo en `Sequence` y en `FrameStream` obligaría a que cada corte, cada
  `split_valid_runs` y cada buffer mantuvieran dos tuplas alineadas.
- **También en `InvalidFrame`**: la duración de un hueco es tiempo, no cuadros.
- **La regla 1 no cambia**: la entrada sigue siendo una secuencia `(T, 21, 3)`;
  el tiempo es metadato de cada fila.
- **Los módulos puros no leen el reloj**: el tiempo llega en el cuadro. Lo pone
  `io/hands.py`, con el **mismo** valor que recibe MediaPipe, para que haya un
  solo reloj en toda la tubería.

**Invariantes, validados al construir un flujo** (`Sequence`, `split_valid_runs`,
la lectura de muestras):

1. Todo o nada: o todos los cuadros de un flujo llevan `timestamp_ms` o ninguno.
   Un flujo mixto se rechaza; mezclar tiempo real con tiempo derivado daría saltos
   que no ocurrieron.
2. Estrictamente creciente. Un cuadro que retrocede es un error de captura y se
   rechaza, no se ordena.

### 2. El tiempo de un cuadro sin marca

Una sola función, en `segmentation.py`, es la única que decide qué hora es:

```python
def frame_time_ms(slot: FrameSlot, index: int, fps: float) -> float:
    """El tiempo del cuadro; sin marca, `index × 1000 / fps`."""
```

- **Muestras guardadas sin marca** (esquemas ≤ 3, todo el dataset actual):
  `fps` = `capture.camera_fps` nominal (30), que es lo que el replay usa hoy. No
  es arbitrario: el dataset se grabó con la cámara **entregando** ~29.5 fps
  (33.9 ms entre cuadros, ADR 0017), aunque la mitad fueran casi repetidos, así
  que `índice × 33.3 ms` es fiel al reloj de entrega.
- **Flujo en vivo sin marca**: no debería pasar desde que `io/hands.py` la pone;
  si pasa, la tasa congelada de la sesión, como hoy.
- **Sintéticos** (`lsm.synthetic`, tests): sin marca, la nominal. Los tests de
  la máquina no cambian.

La propiedad que hace segura la transición: **un flujo sin marcas da exactamente
los mismos números que la v5**, porque `(t_i − t_j) = (i − j) × 1000/fps` es la
división que la v5 ya hace. El eval de Fase 2 y el replay de Fase 5 sobre el
dataset viejo tienen que salir idénticos, y eso es un test.

### 3. Qué usa el tiempo

**En esta propuesta, solo la velocidad** (`feature-spec.md` §6.1.1,
`SEGMENTATION_SPEC_VERSION` 6):

```
j   = el cuadro más reciente del buffer con  t_t − t_j ≥ velocity_window_ms
      (el más antiguo del buffer si ninguno llega)
w_t = v(B[j], B[t]) · 1000 / (t_t − t_j)          # unidades de mano por segundo
```

y `w_t` se compara directamente con `velocity_threshold_per_s` y
`motion_threshold_per_s`, sin dividir entre `fps`. Es lo que la v5 aproxima con
`k/fps` y lo que la prueba de reposo ya mide con el reloj real, así que los
umbrales fijados con ella siguen valiendo. `hand_check.py` y el diagnóstico usan
la misma función.

**Fuera de esta propuesta, anotado como siguiente paso:** medir también las
duraciones (`stable_ms`, `motion_min_ms`, `motion_confirm_low_ms`, cooldowns,
`missing_to_idle_ms`, `motion_max_ms`) en tiempo en vez de en cuadros. Eliminaría
la tasa congelada, pero cambia cada contador de la máquina y el razonamiento del
ADR 0013 sobre reproducibilidad; conviene hacerlo con los datos regrabados
delante y como versión aparte (7), no mezclado con la velocidad.

**No cambia:** `FEATURE_SPEC_VERSION`. El remuestreo del §3.2 a `T_ref = 24` sigue
siendo por índice; remuestrear por tiempo es un cambio de features (reentrenar,
rechazar modelos) que esta propuesta no hace. Los `velocities` de los golden
vectors tampoco cambian.

### 4. Formatos

| archivo | cambio | lectura de lo viejo |
|---|---|---|
| muestra (`io/dataset.py`, `dataset-schema.md`) | `SAMPLE_SCHEMA_VERSION` 4: cada entrada de `frames`, válida o hueco, lleva `"t_ms"` | v2 y v3 se siguen leyendo; sin `"t_ms"` → `None` |
| flujo (`flujo.json`, `tests/fixtures/sequences`) | `FIXTURE_SCHEMA_VERSION` +1, mismo `"t_ms"` | la versión anterior se lee sin marcas |
| `golden_features.json` | sin cambio en las features; los `sequence_cases` de segmentación ganan casos con marcas irregulares para `segmentation.ts` | — |
| modelos exportados | sin cambio: no llevan frames | — |

- **La captura escribe siempre v4** (`lsm-capture grabar`), con el tiempo que
  puso `io/hands.py`. Una sesión de grabación en la que el detector no dé marca
  se niega a guardar, en vez de escribir una v4 sin tiempo.
- **Sin migración del dataset viejo**: no se inventan marcas que no se midieron.
  Las muestras v3 quedan como están y se leen con el tiempo derivado; al
  regrabar (Bloque 4) se reemplazan.
- **`web/` no se toca ahora.** `segmentation.ts` tendrá que implementar
  `frame_time_ms` y el mismo invariante; los casos nuevos del golden son los que
  lo verifican (ADR 0009).

### 5. Lo que hay que adaptar

- `evaluation.replay_sample` rellena el reposo final repitiendo el último frame.
  Con marcas, los cuadros de relleno tienen que avanzar el reloj
  (`t_último + i × 1000/fps`) o romperían el invariante 2.
- El descarte de repetidos ya no necesita compensar nada: el cuadro descartado no
  entra y el salto de tiempo queda visible.
- `TrackingRecorder` puede leer el tiempo del cuadro en vez de recibir `wall_ms`
  aparte; el reporte no cambia.
- Igualdad de frames: dos cuadros con los mismos landmarks y distinta marca dejan
  de ser iguales. Hay que revisar los tests que comparan frames o secuencias
  enteras (los de `test_segmentation.py` comparan ventanas) — se construyen sin
  marca, así que en principio no cambian, pero es lo primero que puede romperse.

## Criterios de aceptación de la implementación

1. Dataset viejo (sin marcas): eval de Fase 2 y replay de Fase 5 **idénticos**
   a la v5.
2. Un mismo movimiento sintético muestreado a 15 y a 30 fps, con marcas, da la
   misma `w_t` (dentro de la tolerancia del muestreo) y las mismas decisiones.
3. El mismo movimiento con cuadros descartados (saltos de 66 y 100 ms) da la
   `w_t` del movimiento real, no el doble.
4. Round-trip de una muestra v4; una v3 se lee con `timestamp_ms = None`.
5. Un flujo mixto o que retrocede se rechaza con un mensaje que dice qué cuadro.

## Decisiones abiertas

1. **¿La marca entra en la igualdad de `RawFrame`?** Por defecto sí. La
   alternativa, `field(compare=False)`, deja iguales dos cuadros con la misma
   mano en distinto momento, que es cómodo para los tests pero esconde un campo.
   Se recomienda dejarla en la igualdad.
2. **¿Duraciones en tiempo en la misma versión o en la siguiente?** Se recomienda
   la siguiente (ver §3).
3. **¿Qué reloj?** `time.monotonic` en el momento en que el cuadro sale de la
   cámara, que es el que ya recibe MediaPipe. `CAP_PROP_POS_MSEC` del driver
   sería más cercano a la exposición, pero su lectura de vuelta ya demostró no ser
   fiable (la exposición, ADR 0017) y no todos los backends lo dan.
