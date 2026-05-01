.PHONY: help install download download-artifacts source-data model-data calibration assessment figures workflow workflow-refresh

help:
	@printf '%s\n' \
	  'Available targets:' \
	  '  make install                 Install/update the pixi environment' \
	  '  make download                Download and unpack the required source data' \
	  '  make download-artifacts      Optionally download and extract a published generated-artifact cache' \
	  '  make source-data             Build prepared source-data artifacts from raw resources' \
	  '  make model-data              Run cached model-data generation' \
	  '  make calibration             Run cached model calibration' \
	  '  make assessment              Run cached performance assessment artifact generation' \
	  '  make figures                 Generate publication figures from cached assessment artifacts' \
	  '  make workflow                Run the paper workflow (reuse existing outputs when available)' \
	  '  make workflow-refresh        Recompute the full workflow from source inputs and overwrite outputs'

install:
	pixi install

download:
	pixi run python scripts/download_data.py

download-artifacts:
	pixi run download-artifacts

source-data:
	pixi run workflow-source-data

model-data:
	pixi run workflow-model-data

calibration:
	pixi run workflow-calibration

assessment:
	pixi run workflow-assessment

figures:
	pixi run workflow-figures

workflow:
	pixi run workflow

workflow-refresh:
	pixi run workflow-refresh