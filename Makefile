# Atajos del proyecto. Nada de lo que hay aquí necesita cámara ni MediaPipe.
#
# Si no tienes `make`, cada receta es un comando de una línea: mira el README.

UV ?= uv

.PHONY: help setup test test-nucleo lint format golden docker-test capture train eval demo

help:
	@echo "setup        instala dependencias con uv"
	@echo "test         ruff + mypy strict + pytest (sin camara, sin dataset)"
	@echo "test-nucleo  igual, saltando los tests del glosario en rojo"
	@echo "lint         solo ruff: check y verificacion de formato"
	@echo "format       aplica formato y correcciones automaticas"
	@echo "golden       regenera tests/fixtures/golden_features.json"
	@echo "docker-test  corre la suite dentro del contenedor"

setup:
	$(UV) sync

# El orden es deliberado: lo barato primero. Un error de formato no debería
# esperar a que corra la suite entera.
#
# ESTE OBJETIVO ESTA EN ROJO A PROPOSITO. Los tests marcados `glosario` fallan
# hasta que docs/glosario-lsm.md este completo y verificado contra la fuente
# primaria; es la senal de que la Fase 1 no puede empezar. Ver el README.
test: lint
	$(UV) run mypy
	$(UV) run pytest

# La suite sin los tests que bloquean la Fase 1, para poder trabajar en el nucleo
# mientras el glosario sigue pendiente. Este si debe pasar limpio siempre.
test-nucleo: lint
	$(UV) run mypy
	$(UV) run pytest -m "not glosario"

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:
	$(UV) run ruff format .
	$(UV) run ruff check . --fix

# Python es la referencia normativa de los golden vectors. Regenerarlos cambia el
# contrato con la implementacion de TypeScript: solo se hace junto con un
# incremento de FEATURE_SPEC_VERSION y un ADR (CLAUDE.md, regla 4).
golden:
	$(UV) run lsm-golden

docker-test:
	docker compose -f docker/docker-compose.yml run --rm test

# --------------------------------------------------------------------------- #
# Comandos de fases posteriores. Existen como recordatorio de que estan
# planeados y de en que fase llegan; ver docs/ARQUITECTURA.md seccion 5.
# --------------------------------------------------------------------------- #

capture:
	@echo "make capture: llega en la Fase 1 (src/lsm/cli/capture.py)."; exit 1

train:
	@echo "make train: llega en la Fase 2 (src/lsm/cli/train.py)."; exit 1

eval:
	@echo "make eval: llega en la Fase 2 (src/lsm/cli/evaluate.py)."; exit 1

demo:
	@echo "make demo: llega en la Fase 3 (src/lsm/cli/demo.py)."; exit 1
