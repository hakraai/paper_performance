.PHONY: help install download-resources download-artifacts prepare-artifacts package-resources package-artifacts source-data model-data calibration assessment figures workflow workflow-refresh

help:
	@printf '%s\n' \
	  'Available targets:' \
	  '  make install                 Install/update the pixi environment' \
	  '  make download-resources      Download and unpack the required raw resources' \
	  '  make download-artifacts      Optionally download downstream caches; source-data stays local' \
	  '  make prepare-artifacts       Download resources, build source-data, then download downstream caches' \
	  '  make package-resources       Build the raw-resource bundle for Zenodo release upload' \
	  '  make package-artifacts       Build the downstream generated-artifact cache archive for release upload' \
	  '  make source-data             Build prepared source-data artifacts from raw resources' \
	  '  make model-data              Run cached model-data generation' \
	  '  make calibration             Run cached model calibration' \
	  '  make assessment              Run cached performance assessment artifact generation' \
	  '  make figures                 Generate publication figures from cached assessment artifacts' \
	  '  make workflow                Run the paper workflow (reuse existing outputs when available)' \
	  '  make workflow-refresh        Recompute the full workflow from source inputs and overwrite outputs'

install:
	pixi install

download-resources:
	pixi run download-resources

download-artifacts:
	pixi run download-artifacts

prepare-artifacts: download-resources source-data download-artifacts

package-resources:
	pixi run package-resources

package-artifacts:
	pixi run package-artifacts

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