# ADR 0003 — Formato de export del modelo: un JSON versionado

- **Estado:** aceptada
- **Fecha:** 2026-09-07
- **Fase:** 0 (andamiaje)
- **Implementa:** `src/lsm/classifiers/base.py`

## Contexto

El modelo entrenado en Python tiene que ejecutarse en dos sitios: el escritorio
(Fases 2 a 6) y un navegador móvil (Fase 7). Son dos runtimes distintos, escritos
en dos lenguajes, y el segundo no puede cargar un pickle de Python ni un
checkpoint de PyTorch sin arrastrar un runtime de ML entero al teléfono.

Además hay un problema de versionado que en este proyecto es especialmente
peligroso. La entrada del clasificador no son los landmarks crudos sino el vector
de features de `docs/feature-spec.md`. Si esa normalización cambia —y va a cambiar
al menos una vez, porque el apéndice A ya prevé una v2— un modelo entrenado con la
normalización vieja **sigue funcionando**: carga sin errores, devuelve etiquetas
con confianzas de aspecto razonable, y simplemente acierta menos. No hay
excepción, no hay traza, no hay síntoma. Se descubre en una demo, delante de
alguien, sin saber por qué.

## Decisión

**El modelo se exporta como un único archivo JSON con seis campos obligatorios, y
se rechaza al cargarse si alguna de sus dos versiones no coincide con el
runtime.**

```json
{
  "schema_version": 1,
  "feature_spec_version": 1,
  "classifier": "static_knn",
  "labels": ["A", "B", "..."],
  "params": { "...": "..." },
  "data":   { "...": "..." }
}
```

- **`schema_version`** versiona la *estructura del archivo*.
- **`feature_spec_version`** versiona el *significado de los números que lleva
  dentro*. Son independientes a propósito: se puede reorganizar el archivo sin
  tocar la normalización, y al revés.
- **`classifier`** identifica la implementación (`static_knn`, `dynamic_dtw`), que
  es lo que permite que el registry elija cómo interpretar `params` y `data`.
- **`labels`** es el vocabulario, en orden estable.
- **`params`** son los hiperparámetros necesarios para reproducir la inferencia
  (por ejemplo `band_radius` en DTW, `k` en KNN).
- **`data`** es el modelo propiamente dicho: centroides, plantillas, vectores.

`build_export()` arma el dict con las dos versiones ya puestas, para que ninguna
implementación se invente el formato ni olvide `feature_spec_version`, que es el
campo del que depende todo el mecanismo.

`check_export_compatibility()` verifica al **cargar** —nunca al predecir— que
estén los seis campos y que las dos versiones coincidan. Si no, lanza
`IncompatibleModelError` y el modelo no se ejecuta.

**No hay migración automática.** Reentrenar cuesta minutos con este tamaño de
dataset; depurar un modelo que corre con la normalización equivocada cuesta días.

## Alternativas consideradas

**1. Pickle de Python.**
Descartada. No se lee desde JavaScript, no es estable entre versiones de Python,
y ejecutar un pickle de origen ajeno es ejecutar código arbitrario.

**2. ONNX.**
Descartada. Resuelve un problema que este proyecto no tiene: los clasificadores
previstos son KNN sobre centroides y DTW sobre plantillas, es decir, vectores y
una función de distancia de sesenta líneas. ONNX metería una dependencia pesada en
el navegador y en el entrenamiento a cambio de nada, y no cubre el DTW.

**3. TensorFlow.js con un modelo entrenado en Keras.**
Descartada junto con la decisión de no usar redes (`ARQUITECTURA.md` §4.3): con un
dataset de 3-5 personas una red sobreajusta, y el runtime pesa megabytes en un
teléfono.

**4. Archivos binarios (`.npy`, protobuf) para los vectores.**
Descartada. El modelo son unos pocos kilobytes de floats: no hay problema de
tamaño que resolver. JSON se inspecciona con un editor de texto, se versiona con
diff legible en git y se carga con `JSON.parse`.

**5. Una sola versión en vez de dos.**
Descartada. Mezclar "cómo está organizado el archivo" con "qué significan los
números" obligaría a invalidar todos los modelos por un cambio cosmético de
estructura, o a no poder expresar que la estructura es la misma pero las features
cambiaron.

**6. Migrar automáticamente modelos de versiones anteriores.**
Descartada. Migrar features es imposible en general: de un centroide calculado con
la normalización v1 no se recupera la muestra original. Cualquier "migración"
sería una aproximación silenciosa, que es exactamente el fallo que esta ADR busca
evitar.

## Consecuencias

**A favor**

- El mismo archivo lo consumen Python y TypeScript, sin conversión.
- Un modelo entrenado con otra normalización falla ruidosamente al cargar, con un
  mensaje que dice qué hacer (reentrenar).
- El modelo es auditable: se abre y se lee. En un proyecto académico esto también
  significa que se puede mostrar y explicar en la defensa.
- Añadir un clasificador nuevo no cambia el formato: cambian `classifier`,
  `params` y `data`.

**En contra**

- JSON es verboso para floats. Con centroides de 42 componentes por clase y 27
  clases son decenas de kilobytes; irrelevante aquí, pero pondría un techo si en
  el futuro alguien quisiera exportar un modelo grande. Ese techo es intencional:
  empuja a mantener los modelos pequeños, que es lo que este proyecto quiere.
- La precisión depende de la serialización de floats. `json.dumps` de Python emite
  el `repr` más corto que hace round-trip, y `JSON.parse` de JavaScript lo lee como
  float64 exacto, así que no hay pérdida. Conviene no cambiar eso por un formateo
  "más bonito" con decimales fijos.
- Cada implementación nueva tiene que acordarse de llamar a `build_export()`. Es
  una convención sostenida por el `Protocol` y por los tests, no por el sistema de
  tipos.

**Cómo se cambia**

Añadir un campo opcional no requiere subir `schema_version` si los lectores
antiguos lo ignoran sin romperse. Quitar o resignificar un campo sí. Cambiar la
normalización de features no toca `schema_version` en absoluto: sube
`feature_spec_version`, y de eso se encarga la ADR 0002.
