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
│   └── release/               # Downloaded publication archives
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
make download-resources
```

or

```bash
pixi run download-resources
```

The download helper retrieves and extracts `resources-bundle.zip` from Zenodo record `20025710` into `data/resources/`. The archive contains:

- `Cm_grids_rotliechend.csv`
- `Faultdata_NAM_reformatted_cleaned_selected.sqlite3`
- `Groningen_field_outline.csv`
- `ReservoirModel_compressibility_20180122.csv`
- `ReservoirModel_thickness_20171013.csv`
- `XY_PRF_ROTL_GY_V2a_2023.csv`
- `rticm_stress.h5`

These raw resources are not the same as the prepared runtime inputs consumed by the supported workflow runners. The same Zenodo record also provides the optional downstream cache archive consumed by `make download-artifacts`.

If you want to reuse published downstream caches instead of rebuilding every downstream stage locally, you can optionally download that cache bundle with:

```bash
make download-artifacts
```

or

```bash
pixi run download-artifacts
```

The generated-artifact download is optional. If you skip it, the supported workflow can still build the downstream caches locally from the raw resources.

If you want the downloaded downstream cache to be immediately usable for the later workflow stages, run:

```bash
make prepare-artifacts
```

This downloads the raw-resource bundle, builds `data/generated_source_data/`, and then downloads the downstream cache archive.

Prepared source-data is not part of the published cache archive. Always generate `data/generated_source_data/` locally from the raw Zenodo bundle before running the downstream stages. The current source-data stage writes about 72 GB under `data/generated_source_data/`, so reserve at least 80 GB of free disk space before running `make source-data`.

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

The repository now distinguishes between raw upstream inputs, locally generated source-data, and optional downstream caches:

- raw upstream inputs downloaded from Zenodo live under `data/resources/`
- optional published downstream cache bundles are downloaded and extracted into the same generated output paths used by the workflow
- downloaded Zenodo archives live under `data/release/`
- prepared runtime source data generated from those inputs live under `data/generated_source_data/`
- active experiment YAMLs live under `configs/experiments/groningen_1995_2025/`

The supported workflow always starts from the locally generated source-data set under `data/generated_source_data/`, specifically:

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

The source-data bundle is intentionally not shipped in the Zenodo release cache. Generate it locally once, then combine it with downloaded downstream caches as needed.

The optional generated-artifact archive is expected to unpack directly into the repository cache layout for downstream stages only, specifically:

- `data/generated_model_data/`
- `data/generated_calibrations/`
- `data/generated_assessment/`
- `figures/generated_paper/`

A release consumer who wants to recreate figures from the published downstream caches should use this sequence:

```bash
make prepare-artifacts
pixi run python scripts/run_figure_generation.py --config configs/figure_generation.yaml --cache refresh
```

That combination uses locally rebuilt `data/generated_source_data/` together with downloaded downstream caches under `data/generated_model_data/`, `data/generated_calibrations/`, `data/generated_assessment/`, and `figures/generated_paper/`.

Reproducibility note: the workflow reproduces the assessment boolean outcomes from the published downstream caches, but not every published PNG figure is pixel-perfect when regenerated from the current code path. This came from an earlier mistake in the assessment random-seed plumbing: the configured assessment seed was added to the workflow, but one multiscale spatial simulation call still failed to receive that RNG. As a result, the multiscale spatial summary values drifted slightly between refresh runs even though the derived boolean test results remained unchanged. The affected PNG figures are `multi_prospective.png`, `multi_retrospective.png`, `multi_bs_prospective.png`, and `multi_bs_retrospective.png`; the other generated paper PNGs match pixel-perfectly.

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
@article{Kraaijpoel2026,
  title={Performance testing, comparison and selection of induced seismicity rate models for the Groningen gas field},
  author={Kraaijpoel, D.A. and Aben, F.M. and Osinga, S. and Pluymaekers, M.P.D.},
  journal={BSSA},
  year={2026},
  doi={...}
}
```

## Contact

For repository questions, open an issue in this repository.