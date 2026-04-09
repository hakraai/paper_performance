.PHONY: help install download workflow workflow-refresh data calibration assessment figures

help:
	@printf '%s\n' \
	  'Available targets:' \
	  '  make install                 Install/update the pixi environment' \
	  '  make download                Download and unpack the required source data' \
	  '  make workflow                Run the paper workflow (reuse existing outputs when available)' \
	  '  make workflow-refresh        Recompute the full workflow from source inputs and overwrite outputs' \
	  '  make data                    Run cached model-data generation' \
	  '  make calibration             Run cached model calibration' \
	  '  make assessment              Run cached performance assessment artifact generation' \
	  '  make figures                 Generate publication figures from cached assessment artifacts'

install:
	pixi install

download:
	pixi run python scripts/download_data.py

workflow:
	pixi run workflow

workflow-refresh:
	pixi run workflow-refresh

data:
	pixi run workflow-data

calibration:
	pixi run workflow-calibration

assessment:
	pixi run workflow-assessment

figures:
	pixi run workflow-figures