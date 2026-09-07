# CLAUDE.md

Instrucciones permanentes para trabajar en este repositorio.

## Qué es este proyecto

Traductor bidireccional del alfabeto dactilológico de la Lengua de Señas Mexicana
(LSM). Reconoce señas manuales por cámara y las convierte en texto, y a la inversa
muestra las señas correspondientes a un texto escrito.

Lee `docs/ARQUITECTURA.md` antes de hacer cambios estructurales. Ese documento es la
fuente de verdad sobre el diseño.

## Reglas no negociables

1. **El tipo base de entrada es una secuencia temporal `(T, 21, 3)`, nunca un frame
   suelto.** Una seña estática es una secuencia corta y estable. No introduzcas APIs
   que acepten un solo frame.

2. **`src/lsm/features.py`, `segmentation.py` y `classifiers/` son código puro.** Sin
   OpenCV, sin MediaPipe, sin acceso a disco, sin cámara. Toda la I/O vive en
   `src/lsm/io/` y en `src/lsm/cli/`. Esto permite testear el núcleo en CI sin
   hardware.

3. **MediaPipe solo se importa en `src/lsm/io/hands.py`.** El resto del código
   consume la interfaz definida ahí.

4. **Cualquier cambio en la extracción de features requiere:** actualizar
   `docs/feature-spec.md`, incrementar `feature_spec_version`, regenerar
   `tests/fixtures/golden_features.json` y anotarlo en un ADR. Los modelos exportados
   con una versión anterior deben rechazarse al cargarse.

5. **Cero umbrales hardcodeados.** Velocidades, ventanas, confianzas mínimas y
   cooldowns viven en `config.yaml`, validado con Pydantic.

6. **La evaluación usa validación leave-one-signer-out.** Nunca split aleatorio de
   frames: mezclaría frames de la misma grabación entre train y test y daría métricas
   infladas.

7. **Todo clasificador implementa el `Protocol` de `classifiers/base.py`**, incluido
   `export()` a un dict JSON-serializable que pueda consumirse desde JavaScript.

## Comandos

```bash
make setup      # instalar dependencias
make test       # pytest + mypy + ruff — debe pasar sin cámara
make capture    # CLI de recolección de dataset
make train      # entrenar y exportar modelo a data/models/
make eval       # métricas + matriz de confusión
make demo       # demo en vivo
```

## Al implementar

- Empieza por los tipos en `src/lsm/types.py` y por los tests. El núcleo se testea
  con secuencias sintéticas y con fixtures grabados, no con webcam.
- Cuando una decisión sea reversible pero costosa, escribe un ADR en `docs/adr/`
  antes de implementar.
- Prefiere soluciones simples y depurables (KNN, DTW) sobre modelos grandes. El
  dataset será pequeño y el modelo tiene que correr en un navegador móvil.
- Si una tarea revela que la arquitectura documentada no funciona, detente y propón
  el cambio en vez de improvisar una excepción local.

## Terminología

- El proyecto traduce **deletreo manual**, no lengua de señas. La LSM es una lengua
  con gramática propia y componentes no manuales. No uses "traductor de lengua de
  señas" en código, docs ni UI.
- `signer` = persona que ejecuta la seña. `label` = letra. `sample` = una secuencia
  etiquetada con sus metadatos.

## Fuera de alcance

No agregues reconocimiento facial, identificación de personas, ni envío de video a
servidores. El procesamiento es local por diseño y así se le comunica al usuario.
