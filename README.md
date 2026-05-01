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

The download helper retrieves the published workflow inputs from Zenodo record `17816284` and extracts any bundled archives into `data/resources/`.

## Workflow

Run the production paper workflow with:

```bash
make workflow
```

or

```bash
pixi run workflow
```

This runs the full pipeline in cache-reuse mode across data generation, model calibration, performance assessment, and figure generation. Existing outputs are reused when present, so the command acts as the normal production workflow for day-to-day work on the repository.

Force a full recomputation of all supported workflow outputs from the configured source inputs with:

```bash
make workflow-refresh
```

or

```bash
pixi run workflow-refresh
```

The individual workflow stages are also available through `make data`, `make calibration`, `make assessment`, and `make figures`.

## Data and Artifacts

The supported workflow expects external inputs under `data/resources/`, including at least:

- `data/resources/event_data.h5`
- `data/resources/fault_data.h5`
- `data/resources/grid_data.h5`
- `data/resources/groningen_polygons.shp`
- `data/resources/model_specs-*.yaml`
- `data/resources/data_scenarios-*.yaml`

Generated artifacts are organized as follows:

- model-data artifacts: `data/generated_model_data/`
- calibration artifacts: `data/generated_calibrations/`
- performance-assessment artifacts: `data/generated_assessment/`
- figures: `figures/generated_paper/`

## Workflow Surface

The main workflow configuration files are:

- `configs/model_data.yaml`
- `configs/model_calibration.yaml`
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