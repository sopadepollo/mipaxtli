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

**Fase 0 — andamiaje.** Están el núcleo puro (tipos, features, segmentación,
contrato de clasificadores), la configuración, los tests y los golden vectors.
No hay todavía captura, entrenamiento, demo ni app web. Ver el plan de fases en
`docs/ARQUITECTURA.md` §5.

## Documentos normativos

Antes de tocar código, en este orden:

1. `CLAUDE.md` — reglas no negociables del repositorio.
2. `docs/ARQUITECTURA.md` — fuente de verdad sobre el diseño.
3. `docs/feature-spec.md` — **contrato** de la extracción de features (§1-§5) y de
   la segmentación (§6, versionada aparte). Cualquier implementación, Python o
   TypeScript, debe reproducirlo dentro de `1e-6`.
4. `docs/adr/` — decisiones de arquitectura y por qué se tomaron.
5. `docs/glosario-lsm.md` — qué letras se reconocen y cuáles llevan movimiento.

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

Sin `make` instalado, los equivalentes directos:

```bash
uv sync
uv run pytest
uv run mypy
uv run ruff check . && uv run ruff format --check .
uv run lsm-golden
```

### La suite está en rojo a propósito

Los tests marcados `glosario` fallan hasta que `docs/glosario-lsm.md` esté completo
y verificado contra la fuente primaria. **Es la señal de que la Fase 1 no puede
empezar**: capturar dataset con un glosario incompleto significa grabar veinte
repeticiones por letra de una seña que quizá esté mal.

Para trabajar en el núcleo mientras tanto:

```bash
uv run pytest -m "not glosario"
```

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
- `src/lsm/classifiers/` — `Protocol` común; las implementaciones llegan en fases
  posteriores.
- `src/lsm/io/` — única frontera con cámara, MediaPipe y disco.

`src/lsm/features.py`, `segmentation.py` y `classifiers/` no importan OpenCV ni
MediaPipe, no leen disco y no abren la cámara. Esa regla no es estética: es lo que
hace que `make test` pase en una máquina sin webcam.

## Ética y consentimiento

- El procesamiento ocurre en el dispositivo. Ningún frame se envía a un servidor.
- No se almacena video sin consentimiento explícito y por escrito de quien firma.
- El glosario y los assets deben verificarse contra la fuente primaria de LSM
  (**no ASL**) y revisarse con una persona usuaria de LSM o un intérprete antes de
  presentar el proyecto.
