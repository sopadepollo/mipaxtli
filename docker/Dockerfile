# Imagen para tests y entrenamiento. Sin cámara y sin MediaPipe.
#
# El núcleo del proyecto (`features`, `segmentation`, `classifiers`) es código
# puro sobre arreglos, así que todo lo que se hace aquí dentro —tests, y más
# adelante entrenamiento y evaluación— funciona sin acceso a hardware de video.
# Ver `docker/README.md` para el caso de la captura, que es distinto.

FROM python:3.12-slim-bookworm

# uv viene de su imagen oficial: evita instalarlo con pip y fija la versión.
COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Las dependencias se instalan antes de copiar el código: mientras no cambien
# `pyproject.toml` ni `uv.lock`, esta capa se reutiliza y el build es instantáneo.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev \
    && uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

# Correr como usuario sin privilegios. El contenedor no necesita root para nada,
# y los archivos que escriba en volúmenes montados no quedan como root del host.
RUN useradd --create-home --uid 1000 lsm && chown -R lsm:lsm /app /opt/venv
USER lsm

# Por defecto, la suite completa: ruff, mypy y pytest. Es el criterio de
# aceptación de la Fase 0 y lo que debería correr CI.
CMD ["sh", "-c", "ruff check . && ruff format --check . && mypy && pytest"]
