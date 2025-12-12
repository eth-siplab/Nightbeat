# Nightbeat: Heart Rate Estimation From a Wrist-Worn Accelerometer During Sleep

[Max Moebus<sup>1</sup>](https://maxmoebus.com/), Lars Hauptmann<sup>1</sup>, Nicolas Kopp<sup>1</sup>, [Berken Demirel](https://berken-demirel.github.io/)<sup>1</sup>, [Björn Braun](https://bjoernbraun.com/)<sup>1</sup>, [Christian Holz<sup>1</sup>](https://www.christianholz.net)

<sup>1</sup> [Sensing, Interaction & Perception Lab](https://siplab.org), Department of Computer Science, ETH Zürich, Switzerland <br/>

>Today’s fitness bands and smartwatches typically track heart rates (HR) using optical sensors. Large behavioral studies such as the UK Biobank use activity trackers without such optical sensors and thus lack HR data, which could reveal valuable health trends for the wider population. In this paper, we present the first dataset of wrist-worn accelerometer recordings and electrocardiogram references in uncontrolled at-home settings to investigate the recent promise of IMU-only HR estimation via ballistocardiograms. Our recordings are from 42 patients during the night, totaling 310 hours. We also introduce a frequency-based method to extract HR via curve tracing from IMU recordings while rejecting motion artifacts. Using our dataset, we analyze existing baselines and show that our method achieves a mean absolute error of 0.88 bpm—76% better than previous approaches. Our results validate the potential of IMUonly HR estimation as a key indicator of cardiac activity in existing longitudinal studies to discover novel health insights.

# Overview
**Nightbeat-DB:** a novel dataset to enable heart rate monitoring during sleep from wrist-worn accelerometers in *uncontrolled, at-home* settings.<br/>
**Nightbeat:** a novel algorithm that is based on robust signal aggregation and combines effective motion artifact removal, curve tracing of the heart rate in the frequency domain, and simple post processing to push the average error of heart rate estimation from wrist-worn accelerometers during sleep below 1 bpm MAE.

![overview](./figures/overview_small.png)

# Results

Comparison of Nightbeat against three baseline approaches on Nightbeat-DB and the AW dataset (Walch et al. [1]). Approaches from related work do not manage to beat Mean-Pred. baseline: predicting each participant's average heart rate across the entire night.

| Approach          | Dataset      | MAE   | RMSE  | Cor  |
|-------------------|--------------|-------|-------|------|
| BioInsights [2]  | Nightbeat-DB | 17.47 | 21.38 | 0.00 |
|                   | AW           | 21.12 | 24.53 | -0.03 |
| Jerks [3]        | Nightbeat-DB | 13.23 | 20.63 | 0.04 |
|                   | AW           | 3.68  | 7.11  | 0.50 |
| PWR [4]          | Nightbeat-DB | 8.83  | 10.91 | 0.22 |
|                   | AW           | 6.05  | 7.77  | 0.37 |
| Base: Mean-Pred.  | Nightbeat-DB | 3.16  | 4.07  | 0    |
|                   | AW           | 2.98  | 4.09  | 0    |
| **Nightbeat (ours)** | **Nightbeat-DB** | **0.88** | **2.24** | **0.81** |
|                   | **AW**       | **1.68** | **3.38** | **0.64** |

Across all participants, the correlation of Nightbeat exceeds 0.95.

<img src="./figures/all_participants_small.png" alt="all participants" width="50%">

For a single participant, the correlation approaches 1 with an MAE as low as 0.41 across the entire night.

<img src="./figures/single_participant_small.png" alt="single participant" width="50%">

# NightbeatDB

For scientific use, we make the data of 40 participants of NightbeatDB publicly available. The study protocol was approved by the ethics committee of ETH Zurich (23 ETHICS-002). All participants provided written informed consent and 40 out of 42 participants provided consent for the publication of their anonymized data for scientific, non-commercial purposes only.

The anonymized data can be downloaded [here](https://polybox.ethz.ch/index.php/s/oszYjY5aZ2xwMZp) or also [here](https://polybox.ethz.ch/index.php/s/oszYjY5aZ2xwMZp). Please read below for instructions on how to incorporate NightbeatDB into the Nightbeat repository.

# Codebase and Usage

This repository contains the official implementation of **Nightbeat**, and implementations of benchmarking algorithms.

## 📂 Repository Structure

*   **`BCGAlgorithms/`**: Core implementations of the signal processing algorithms.
    *   `nightbeat.py`: The proposed method using adaptive bandpass and curve tracing.
    *   `pwr.py`: Pulse Wave Reconstruction (axis selection + Hilbert envelope).
    *   `jerks.py`: Jerk-based estimation.
    *   `bioinsights.py`: Baseline bandpass filtering method.
*   **`helpers/`**: Utilities for data loading, cleaning, and preprocessing.
*   **`transform.py`**: Feature extraction pipeline (converts raw acceleration to spectrograms/features).
*   **`predict.py`**: Estimating Heart Rate from the transformed features.
*   **`align.py`**: Helper script to download and align public datasets.
*   **`evaluate.py`**: Metrics (MAE, RMSE, Correlation) and visualization tools.
*   **`runner.sh`**: Prepares the two datasets, and executes all algorithms on both datasets.
*   **`visualize_results.ipynb`**: Compute performance metrics and create visualizations.

## 🚀 Getting Started

### 1. Installation

This repository relies on specific versions of SciPy (for `ShortTimeFFT`) and Polars. We recommend using Conda.

**Option A: Conda (Recommended)**
Run:
```bash
conda env create -f environment.yml
conda activate nightbeat
```

If you have issues installing PyTorch, create a conda environment with Python=3.12, install whatever version of torch works for you and then install the other packages. The correct SciPy version is crucial for the repository to work. The correct Polars version is helpful to keep file sizes small using the Float16 datatype (which is not implemented in older versions). However, the repository does not break if the Polars version is not correct.

**Option A: venv**
Alternatively, you can also set up a virtual (venv) environment with Python version 3.12. Simply take the required versions out of the environment.yml file and install the packages via pip.

### 2. Data Preparation

This repository supports two datasets:
1.  **AW (Apple Watch):** A public dataset from *Walch et al. (2019)* [1].
2.  **NightbeatDB:** (Internal/Private dataset).

First, download **NightbeatDB** from [here](https://polybox.ethz.ch/index.php/s/oszYjY5aZ2xwMZp) (~4GB in zipper format) and place all files into `data/raw/nighbeatdb` (e.g., `data/raw/nighbeatdb/00_acc-a.npz`). Then to prepare **NightbeatDB**, and download (~5GB) and prepare the **AW** dataset, run:

```bash
python align.py
```
*This will download data from PhysioNet to `data/raw/aw`, and process it into `data/aligned/aw`. And process NightbeatDB into `data/aligned/nightbeatdb`*

## ⚡ Usage Pipeline

The pipeline consists of two stages: **Transformation** (signal processing) and **Prediction** (HR estimation).

### Step 1: Transformation
Prepares the data into the format needed for the respective algorithms. This step is computationally expensive but results are saved as `.parquet` files and only need to be computed once per algorithm.

**Available Algorithms:** `nightbeat`, `pwr`, `jerks`, `bioinsights`.

```bash
# Run a specific algorithm with 4 parallel workers
python transform.py --algorithm nightbeat --datasets aw --workers 4

# Run all algorithms
python transform.py --algorithm all --datasets nightbeatdb --workers 4
```

### Step 2: Prediction
Estimates the heart rate from the transformed data.

```bash
# Predict using Nightbeat with a 20-second window
python predict.py --algorithm nightbeat --datasets aw --workers 4

# Predict using all implemented algorithms
python predict.py --algorithm all --datasets aw
```

### Step 3: Evaluation & Visualization
You can analyze results using the provided notebook or scripts.

```bash
# Start Jupyter to view compute and visualize results
jupyter notebook visualize_results.ipynb
```

### Full run

If you've set up the environment, you can also simply run

```bash
bash runner.sh
```

to execute all algorithms on both the **aw** dataset and **NightbeatDB** (including download and data preparation). Note that the created files will take up quite a bit of disk space. With the correct polars version, the nightbeat algorithm will create files that take up ~10GB of disk space for NightbeatDB and ~4GB of disk space for the aw dataset. With earlier polars versions, this will double. Using this current implementation the file created when executing the other algorithms will take up similar amounts of disk space, so things can add up.

## ⚙️ Configuration

*   **GPU Acceleration:** If you have a GPU (specifically for PWR), you can edit the `possible_gpus` list in `transform.py` to enable CUDA acceleration.
*   **Threading:** The scripts automatically handle thread management (`POLARS_MAX_THREADS=1`) to prevent CPU oversubscription when using multiprocessing.

# Follow-up(s)

Stay tuned for follow-up projects that evaluate the value of Nightbeat for disease prediction and advance what we can extract from wrist-worn accelerometers.

# Citation

If you use this code in your research, please cite:

```bibtex
@INPROCEEDINGS{moebus2024nightbeat,
  title={Nightbeat: Heart Rate Estimation From a Wrist-Worn Accelerometer During Sleep},
  booktitle={2024 IEEE EMBS International Conference on Biomedical and Health Informatics (BHI)},
  author={Moebus, Max and Hauptmann, Lars and Kopp, Nicolas and Demirel, Berken and Braun, Bj{\"o}rn and Holz, Christian},
  year={2024}
}
```

# References

[1] Walch, Olivia. "Motion and heart rate from a wrist-worn wearable and labeled sleep from polysomnography." PhysioNet 101 (2019).

[2] Hernandez, Javier, Daniel J. McDuff, and Rosalind W. Picard. "BioInsights: Extracting personal data from “Still” wearable motion sensors." 2015 IEEE 12th International Conference on Wearable and Implantable Body Sensor Networks (BSN). IEEE, 2015.

[3] Zschocke, Johannes, et al. "Reconstruction of pulse wave and respiration from wrist accelerometer during sleep." IEEE Transactions on Biomedical Engineering 69.2 (2021).

[4] Weaver, R. Glenn, et al. "Jerks are Useful: Extracting pulse rate from wrist-placed accelerometry jerk during sleep in children." Sleep (2024).
