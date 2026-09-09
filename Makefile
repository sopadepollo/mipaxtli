# Atajos del proyecto. Nada de lo que hay aquí necesita cámara ni MediaPipe.
#
# Si no tienes `make`, cada receta es un comando de una línea: mira el README.

UV ?= uv

.PHONY: help setup setup-capture model calibrar verify test test-nucleo lint format golden docker-test capture train eval demo

help:
	@echo "setup        instala dependencias con uv"
	@echo "test         ruff + mypy strict + pytest (sin camara, sin dataset)"
	@echo "test-nucleo  igual, saltando los tests del glosario en rojo"
	@echo "lint         solo ruff: check y verificacion de formato"
	@echo "format       aplica formato y correcciones automaticas"
	@echo "golden       regenera tests/fixtures/golden_features.json"
	@echo "docker-test  corre la suite dentro del contenedor"
	@echo ""
	@echo "Solo para grabar dataset (necesitan camara, fuera de Docker):"
	@echo "setup-capture  instala MediaPipe y OpenCV"
	@echo "model          descarga el modelo de MediaPipe (~8 MB)"
	@echo "calibrar       confirma la lateralidad de la camara (una vez)"
	@echo "capture        sesion de captura; pasa ARGS=..."
	@echo "verify         relee data/raw y re-deriva las features"

setup:
	$(UV) sync

# MediaPipe y OpenCV son dependencias OPCIONALES: la suite, el entrenamiento y
# la evaluacion corren sin ellas. Solo hacen falta para grabar dataset y para la
# demo en vivo, y en Windows/macOS eso se ejecuta FUERA del contenedor.
setup-capture:
	$(UV) sync --extra capture

# El bundle del detector: ~8 MB que no se versionan en el repositorio. Ver
# docs/adr/0006-deteccion-de-manos-y-captura.md.
MODEL_URL ?= https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
MODEL_PATH ?= data/models/hand_landmarker.task

model:
	@mkdir -p $(dir $(MODEL_PATH))
	curl -sSL -o $(MODEL_PATH) $(MODEL_URL)
	@ls -la $(MODEL_PATH)

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

# Confirma a ojo que la lateralidad que reporta el detector es la mano real, y
# lo deja escrito. `capture` se niega a arrancar sin esto. Es un minuto, una vez
# por camara, y es la unica defensa contra un error que no da ningun sintoma:
# ver docs/adr/0007-cierre-de-captura.md.
calibrar:
	$(UV) run lsm-capture calibrar $(ARGS)

# Sesion de captura. Los metadatos de condiciones son obligatorios a proposito:
# anotarlos despues, de memoria, no funciona (ver docs/adr/0005-...).
#
#   make capture ARGS="--firmante s01 --sesion 2026-09-08-manana #                      --luz-nivel INDOOR --luz-direccion FRONTAL --distancia MEDIUM"
#
# Necesita `make setup-capture` y `make model` una sola vez.
CAPTURE_ARGS ?= --help

capture:
	$(UV) run lsm-capture grabar $(if $(ARGS),$(ARGS),$(CAPTURE_ARGS))

# Relee data/raw, re-deriva las features y comprueba que salen identicas a las
# del momento de grabar. No necesita camara ni MediaPipe.
verify:
	$(UV) run lsm-capture verificar

# Entrena static_knn sobre las 21 letras estaticas mas NONE y exporta el modelo
# a data/models/static_knn.json. Las 8 letras dinamicas quedan fuera hasta la
# Fase 5. No mide precision a proposito: eso es `make eval`.
#
# Mientras data/raw este vacio usa un corpus SINTETICO y lo dice en pantalla y
# dentro del propio archivo. Para exigir dataset real: ARGS="--sin-sintetico".
train:
	$(UV) run lsm-train $(ARGS)

# Reporte completo de la Fase 2 en data/models/eval/: accuracy global y por
# letra, matriz de confusion, pares mas confundidos, contraste de hipotesis
# contra la columna confundible_con del glosario, barrido de calibracion y
# diagnostico empirico de umbrales.
#
# Validacion leave-one-signer-out. Con un solo firmante grabado se NIEGA a
# correr; la degradacion se pide a mano y queda escrita en el reporte:
#
#   make eval ARGS="--protocolo leave-one-session-out"
#
# El barrido completo son 2250 puntos de rejilla. Para una pasada rapida:
#
#   make eval ARGS="--barrido minimo"
#
# No lleva marca de tiempo en ningun archivo: dos ejecuciones sobre el mismo
# dataset dan los mismos bytes, que es lo que permite comparar dos calibraciones.
eval:
	$(UV) run lsm-eval $(ARGS)

# Demo en vivo: senas -> texto. Necesita camara, MediaPipe y un modelo
# entrenado (`make train`). Controles: bajar la mano cierra la palabra,
# BACKSPACE borra un simbolo, ENTER cierra la frase, q sale. Las 8 letras
# dinamicas (J, K, LL, N~, Q, RR, X, Z) no se reconocen todavia: el HUD lo avisa
# en pantalla (Fase 5). Para probarla sin camara, contra una sesion ya grabada:
#
#   make demo ARGS="--desde-dataset data/raw/s01/2026-09-09-manana"
demo:
	$(UV) run lsm-demo $(ARGS)
