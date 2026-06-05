# CorSeaCare_yolo — common tasks. Run `make help` for the list.
.DEFAULT_GOAL := help
.PHONY: help setup test demo manifest calibrate serve label-studio clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Install engine + Label Studio (macOS / Apple Silicon)
	./scripts/setup.sh

test: ## Run the engine test-suite
	uv run pytest -q

demo: ## Build a synthetic dataset and run the whole pipeline with Fake backends
	uv run python scripts/make_test_dataset.py
	CORSEACARE_FAKE=1 uv run corseacare count data/test_run/images --out counts.csv --mm-per-px 0.1

manifest: ## Regenerate samples.csv (photo -> sieve mapping)
	uv run python scripts/make_manifest.py

calibrate: ## Apply mesh-as-ruler px/mm to samples.csv (needs mesh_calibration.json)
	uv run python scripts/calibrate_mesh.py apply

serve: ## Serve data/ over HTTP+CORS for Label Studio (http://localhost:8081)
	uv run python scripts/serve_images.py

label-studio: ## Launch Label Studio (http://localhost:8080)
	./scripts/launch_label_studio.sh

clean: ## Remove caches and pipeline outputs (keeps your data/ photos & annotations)
	rm -rf .pytest_cache **/__pycache__ data/ls_export_yolo data/*_pred_tiled data/*_tiles
