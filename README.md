# Performance Testing, Comparison, and Selection of Induced Seismicity Rate Models for the Groningen Gas Field

This repository contains the scripted workflow, configuration, and data interfaces used to generate the results and figures for the manuscript:

> **"Performance testing, comparison and selection of induced seismicity rate models for the Groningen gas field"**
> *D.A. Kraaijpoel, F.M. Aben, S. Osinga, M.P.D. Pluymaekers*
> *Journal submission under review*

The supported public workflow is command-line and configuration driven. `make workflow` and `pixi run workflow` are the primary production entrypoints for the paper workflow, reusing existing outputs when available.


## Repository Structure

```text
paper_performance/
├── configs/                   # Workflow configuration files
├── data/
│   ├── resources/             # External inputs used by the workflow
│   ├── generated_*/           # Generated workflow artifacts
│   └── obsolete/              # Archived local artefacts no longer on the active path
├── figures/                   # Generated figures
├── scripts/
│   ├── workflow_support/      # Shared workflow modules used by the runners
│   └── run_*.py               # Supported workflow runners
├── src/
│   ├── chaintools/            # Generic tools (submodule)
│   └── model_chain_inference/ # Core model, calibration, and assessment code
├── Makefile                   # Convenience task wrapper
├── pixi.toml                  # Environment and task definitions
├── pixi.lock                  # Locked dependency set
└── README.md
```

## Environment

Use [pixi](https://pixi.sh/) to create the project environment.

```bash
git clone --recurse-submodules https://github.com/hakraai/paper_performance.git
cd paper_performance
pixi install
```

If you already cloned the repository without submodules, initialize the bundled `chaintools` dependency with:

```bash
git submodule update --init --recursive
```

The activation environment defined in `pixi.toml` sets `PYTHONPATH` for the local source tree and enables a persistent JAX compilation cache under `.cache/jax_compilation`.

If the source datasets are not present locally, download and unpack them with:

```bash
make download
```

or

```bash
pixi run python scripts/download_data.py
```

The download helper retrieves the published upstream source datasets from Zenodo record `17816284` into `data/resources/`. The current record contains:

- `Cm_grids_rotliechend.csv`
- `Faultdata_NAM_reformatted_cleaned_selected.sqlite3`
- `Groningen_field_outline.csv`
- `ReservoirModel_compressibility_20180122.csv`
- `ReservoirModel_thickness_20171013.csv`
- `XY_PRF_ROTL_GY_V2a_2023.csv`
- `rticm_stress.h5`

These downloads are not the same as the prepared runtime inputs consumed by the supported workflow runners.

If you want to reuse a published generated-artifact cache instead of rebuilding every stage locally, you can optionally download that cache bundle after filling in `artifact_archive_url` in `configs/generated_artifacts_download.yaml`:

```bash
make download-artifacts
```

or

```bash
pixi run download-artifacts
```

The generated-artifact download is optional. If you skip it, the supported workflow can still build the cache locally from the raw resources.

## Workflow

Run the production paper workflow with:

```bash
make workflow
```

or

```bash
pixi run workflow
```

This runs the full pipeline in cache-reuse mode across source-data generation, model-data generation, model calibration, performance assessment, and figure generation. Existing outputs are reused when present, so the command acts as the normal production workflow for day-to-day work on the repository.

Force a full recomputation of all supported workflow outputs from the configured source inputs with:

```bash
make workflow-refresh
```

or

```bash
pixi run workflow-refresh
```

This runs the same pipeline but refreshes any existing outputs instead of reusing them.

The individual workflow stages are also available through `make source-data`, `make model-data`, `make calibration`, `make assessment`, and `make figures`.

## Data and Artifacts

The repository now distinguishes between raw upstream inputs and prepared runtime inputs:

- raw upstream inputs downloaded from Zenodo live under `data/resources/`
- optional published generated-artifact cache bundles are downloaded and extracted into the same generated output paths used by the workflow
- prepared runtime source data generated from those inputs live under `data/generated_source_data/`
- archived legacy prepared artifacts kept for parity checking live under `data/obsolete/legacy_source_data_reference/`
- active experiment YAMLs live under `configs/experiments/groningen_1995_2025/`

The supported workflow starts from the prepared source-data set under `data/generated_source_data/`, specifically:

- `data/generated_source_data/event_data.h5`
- `data/generated_source_data/fault_data.h5`
- `data/generated_source_data/grid_data.h5`
- `data/generated_source_data/groningen_polygons.{shp,shx,dbf,prj,cpg}`
- `configs/experiments/groningen_1995_2025/model_specs.yaml`
- `configs/experiments/groningen_1995_2025/data_scenarios.yaml`

Build that prepared source-data set with:

```bash
make source-data
```

or

```bash
pixi run workflow-source-data
```

The source-data stage reads the raw Zenodo bundle directly and writes the prepared runtime inputs into `data/generated_source_data/`.

The optional generated-artifact archive is expected to unpack directly into the repository cache layout, specifically:

- `data/generated_source_data/`
- `data/generated_model_data/`
- `data/generated_calibrations/`
- `data/generated_assessment/`
- `figures/generated_paper/`

Legacy migration and parity-check helpers are kept under `forensics/` and are not part of the normal workflow surface.

Run the model-data stage with:

```bash
make model-data
```

or

```bash
pixi run workflow-model-data
```

Additional files currently present under `data/resources/` are not all on the active supported path:

- `model_specs-perf_1995_2025.yaml`, `model_specs-performance_1995_2025.yaml`, `data_scenarios-perf_1995_2025.yaml`, and `data_scenarios-performance_1995_2025.yaml` are legacy alternate config variants that are not referenced by the active configs.
- `grid_data_flat.h5` is not referenced by the active workflow.

Generated artifacts are organized as follows:

- prepared source data: `data/generated_source_data/`
- model-data artifacts: `data/generated_model_data/`
- calibration artifacts: `data/generated_calibrations/`
- performance-assessment artifacts: `data/generated_assessment/`
- figures: `figures/generated_paper/`

## Workflow Surface

The main workflow configuration files are:

- `configs/model_data.yaml`
- `configs/model_calibration.yaml`
- `configs/model_calibration_settings.yaml`
- `configs/performance_assessment.yaml`
- `configs/figure_generation.yaml`
- `configs/paper_workflow.yaml`

The core implementation is in `src/model_chain_inference/`, including:

- `generate_data.py` for source-data assembly helpers
- `data_prep.py` for inference-data preparation
- `performance.py` and `testsuite.py` for assessment logic

`pixi.toml`, `Makefile`, `configs/`, and the `run_*.py` entrypoints in `scripts/` define the supported task surface for reconstructing the paper workflow. The modules under `scripts/workflow_support/` are internal shared helpers for those runners. Auxiliary local exploration material is outside the supported workflow contract.

## License

This project is licensed under the EUPL v1.2. See [LICENSE](LICENSE).

## Citation

If you use this repository or its methodology, cite the accompanying manuscript.

```bibtex
@article{Kraaijpoel202X,
  title={Performance testing, comparison and selection of induced seismicity rate models for the Groningen gas field},
  author={Kraaijpoel, D.A. and Aben, F.M. and Osinga, S. and Pluymaekers, M.P.D.},
  journal={Journal Name},
  year={202X},
  doi={...}
}
```

## Contact

For repository questions, open an issue in this repository.