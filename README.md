# Performance Testing, Comparison, and Selection of Induced Seismicity Rate Models for the Groningen Gas Field

This repository contains the source code, data processing scripts, and analysis notebooks used to generate the results and figures for the manuscript:

> **"Performance testing, comparison and selection of induced seismicity rate models for the Groningen gas field"**  
> *Authors: D.A. Kraaijpoel, F.M. Aben, S. Osinga, M.P.D. Pluymaekers*  
> *Journal: [xxxxx] (Under Review)*

> **Warning:** The code in this repository is currently in a preliminary state. The scripts as provided have been used for the creation of the results and figure described in the manuscript, but the environment specification, paths, and required data are not fully specified. Reproducing the results may be challenging.


## Repository Structure

```text
paper_performance/
├── data/                   # Input datasets (catalogs, production data, etc.)
├── scripts/                # Download data script
├── notebooks/              # Jupyter notebooks for interactive analysis and figure generation
│   ├── 01_preprocessing.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_model_calibration.ipynb
│   └── 04_performance_assessmet.ipynb
├── src/                        # Core Python source code
│   ├── chaintools/             # Generic tools (submodule)
│   └── model_chain_inference/  # Inference codes
├── pixi.toml               # Pixi environment specification
├── pixi.lock               # Pixi lock file
└── README.md               # This file
```

## Installation

To reproduce the environment used for this analysis, we recommend using [pixi](https://pixi.sh/).

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/hakraai/paper_performance.git
    cd paper_performance
    ```

2.  **Create the environment:**
    <!-- ```bash
    conda env create -f environment.yml
    conda activate gronstat
    ``` -->
    You can use [pixi](https://pixi.sh/) to manage the environment:
    ```bash
    pixi install
    pixi run start
    ```

## Usage

### Reproducing Figures
The primary results can be reproduced by running the notebooks in the `notebooks/` directory in sequential order.

1.  Start the Jupyter server:
    ```bash
    jupyter notebook
    ```
2.  Open `notebooks/04_performance_assessment.ipynb` to view the main performance comparison workflow and results.

## Data Availability

To download and unpack the necessary data files from Zenodo (Record 10245813), run the following script:

```bash
pixi run python scripts/download_data.py
```

*Note: Some pre-processed intermediate data files may be required to run specific notebooks without re-calculating the entire model chain.*

## License

This project is licensed under the EUPL v1.2 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code or the methodology in your research, please cite the accompanying manuscript:

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

For questions regarding the code or the manuscript, please open an issue in this repository.
