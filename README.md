# A Reanalysis‑infused Attentive Message Passing Network for Multi‑Station Meteorological Imputation


This repository contains the official implementation of **RAMP-Net**, a novel reanalysis-infused attentive message-passing framework designed for robust meteorological time-series imputation by integrating physical priors from reanalysis products with dynamic inter-station correlations.

---

## 🌟 Key Features

[//]: # "> Our model leverages spectral fusion to bridge the gap between large-scale atmospheric dynamics and local-scale convective patterns."
> **RAMP-Net:** Reanalysis-infused Attentive Message Passing Network.
> Our model leverages ERA5 reanalysis data as physical priors and dynamically learns inter-station dependencies to achieve robust meteorological time-series imputation.

* **Prior-guided Spatiotemporal Modeling:** Introduces a two-stage framework that first extracts large-scale physical priors from ERA5 reanalysis data and then mines spatiotemporal correlations between stations, enabling effective heterogeneous data alignment and accurate station-level modeling.
* **Attentive Message Passing:** Designs a cross-attention module jointly conditioned on meteorological features and geographic coordinates, moving beyond static topologies to dynamically capture inter-station dependencies and long-range teleconnection-like relationships.
* **Superior Performance Across Diverse Regions and Missing Scenarios:**  Extensive experiments over Guangdong (China) and California (USA) demonstrate that RAMP-Net consistently outperforms state-of-the-art baselines, particularly under high missing rates and extreme weather conditions.

### 🏗️ Model Architecture
[//]: # "![Model Architecture]&#40;docs/assets/model_architecture.png&#41;"
<p align="center">
  <img src="docs/Overall Framework.png" width="900" alt="Model Architecture">
</p>


[//]: # "*Figure 1: Overview of the RAMP-Net: framework.*"
<p align="center">
  <em>Figure 1: Overview of the RAMP-Net framework.</em>
</p>


---

## 📊 Implemented Models

| Category                     | Models                                               |
| :--------------------------- | :--------------------------------------------------- |
| **Proposed**                 | **RAMP-Net**                                         |
| **Geostatistical Baselines** | Kriging                                              |
| **Single-Station Baselines** | LR, Autoformer, TimesNet, Non-stationary Transformer |
| **Multi-Station Baselines**  | MPNN                                                 |

---

## 📂 Repository Structure

```text
├── code/
│   ├── data_provider/      # Data loading and preprocessing
│   ├── exp/                # Training and evaluation pipelines
│   ├── layers/             # Neural network building blocks
│   ├── mask_rate/          # Missing-value mask files
│   ├── models/             # RAMP-Net and baseline implementations
│   ├── scaler/             # Normalization statistics and result files
│   ├── utils/              # Utility functions
│   ├── run.py              # Multi-station experiment entry
│   └── run_s.py            # Single-station experiment entry
│
├── data/                   # Processed datasets for Guangdong and California regions
└── README.md
```

---

## 📥 Dataset Preparation

We evaluate RAMP-Net on two meteorologically distinct regions, **Guangdong** (China) and **California** (USA), using station observations from **WEATHER-5K** and atmospheric background fields from **ERA5 reanalysis data**.

### 1. WEATHER-5K Dataset

**Download:** [Weather-5K](https://hkustconnect-my.sharepoint.com/personal/thanad_connect_ust_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fthanad%5Fconnect%5Fust%5Fhk%2FDocuments%2Fresearch%5Fproject%2FWEATHER%5F5k%2FOpenSourced%2FWEATHER%2D5K%2Ezip&parent=%2Fpersonal%2Fthanad%5Fconnect%5Fust%5Fhk%2FDocuments%2Fresearch%5Fproject%2FWEATHER%5F5k%2FOpenSourced&ga=1)

### 2. ERA5 Dataset

**Download:** [ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download)

### 3. Study Regions

To evaluate the generalization capability of RAMP-Net under different climatic conditions, two geographically and meteorologically distinct regions are selected:

**Guangdong, China** (109°E–118°E, 20°N–26°N)`

**California, USA**  (125°W–114°W, 32°N–43°N)

### 4.Processed Data

The processed datasets are provided in the `data/` directory. To run an experiment, extract the contents of the `Guangdong` or `California` folder from the corresponding archive directly into the project root (alongside the `code/` folder).

---

## 🏃 Quick Start

### Training

Training is performed using model-specific shell scripts. You can find all training scripts in the `code/scripts/` directory, organized by model name (e.g., `code/scripts/RAMP-Net/`).

**Important:** You must navigate to the `code/` directory before executing these scripts to ensure the environment is correctly set up. To train a model, use the following commands:

## 📈 Experimental Results
### Table 1: verage quantitative evaluation results of RAMP-Net and existing methods under different missing rates
| Dataset   | Var. | Missing rate | Metric | OK            | LR            | Autoformer    | NS-Trans               | TimesNet               | MPNN                         | RAMP-Net (Ours)              |
| --------- | ---- | ------------ | ------ | ------------- | ------------- | ------------- | ---------------------- | ---------------------- | ---------------------------- | ---------------------------- |
| Guangdong | U    | 12.5%        | MAE    | 1.237 ± 0.002 | 1.074 ± 0.002 | 0.556 ± 0.002 | 0.526 ± 0.001          | 0.498 ± 0.004          | *0.491* ± 0.002               | **0.372** ± 0.005            |
| Guangdong | U    | 12.5%        | $R^2$  | 0.074 ± 0.002 | 0.259 ± 0.002 | 0.696 ± 0.002 | 0.702 ± 0.003          | 0.727 ± 0.004          | *0.743* ± 0.002               | **0.819** ± 0.003            |
| Guangdong | U    | 25%          | MAE    | 1.238 ± 0.001 | 1.077 ± 0.002 | 0.593 ± 0.002 | 0.573 ± 0.001          | *0.526* ± 0.002        | 0.534 ± 0.001                | **0.419** ± 0.002            |
| Guangdong | U    | 25%          | $R^2$  | 0.075 ± 0.000 | 0.258 ± 0.001 | 0.668 ± 0.003 | 0.673 ± 0.002          | 0.710 ± 0.001          | *0.713* ± 0.002               | **0.788** ± 0.002            |
| Guangdong | U    | 50%          | MAE    | 1.238 ± 0.001 | 1.076 ± 0.001 | 0.694 ± 0.003 | 0.670 ± 0.002          | *0.641* ± 0.002        | 0.649 ± 0.001                | **0.574** ± 0.003            |
| Guangdong | U    | 50%          | $R^2$  | 0.073 ± 0.001 | 0.258 ± 0.001 | 0.580 ± 0.003 | 0.600 ± 0.001          | *0.615* ± 0.002        | 0.607 ± 0.001                | **0.668** ± 0.005            |
| Guangdong | U    | 75%          | MAE    | 1.237 ± 0.000 | 1.076 ± 0.001 | 0.908 ± 0.001 | 0.801 ± 0.001          | *0.779* ± 0.001        | 0.826 ± 0.001                | **0.718** ± 0.003            |
| Guangdong | U    | 75%          | $R^2$  | 0.074 ± 0.001 | 0.257 ± 0.001 | 0.415 ± 0.001 | 0.500 ± 0.001          | *0.520* ± 0.001        | 0.482 ± 0.001                | **0.561** ± 0.003            |
| Guangdong | V    | 12.5%        | MAE    | 1.350 ± 0.002 | 1.161 ± 0.003 | 0.570 ± 0.003 | 0.554 ± 0.004          | 0.495 ± 0.005          | *0.478* ± 0.003               | **0.368** ± 0.004            |
| Guangdong | V    | 12.5%        | $R^2$  | 0.145 ± 0.002 | 0.365 ± 0.003 | 0.777 ± 0.001 | 0.764 ± 0.005          | 0.809 ± 0.004          | *0.833* ± 0.003               | **0.872** ± 0.003            |
| Guangdong | V    | 25%          | MAE    | 1.351 ± 0.001 | 1.161 ± 0.002 | 0.634 ± 0.005 | 0.620 ± 0.002          | *0.543* ± 0.002        | 0.546 ± 0.002                | **0.440** ± 0.004            |
| Guangdong | V    | 25%          | $R^2$  | 0.143 ± 0.002 | 0.363 ± 0.002 | 0.726 ± 0.004 | 0.719 ± 0.002          | 0.771 ± 0.001          | *0.786* ± 0.002               | **0.826** ± 0.002            |
| Guangdong | V    | 50%          | MAE    | 1.350 ± 0.001 | 1.160 ± 0.002 | 0.768 ± 0.002 | 0.741 ± 0.001          | *0.689* ± 0.002        | 0.691 ± 0.002                | **0.626** ± 0.005            |
| Guangdong | V    | 50%          | $R^2$  | 0.144 ± 0.002 | 0.363 ± 0.001 | 0.624 ± 0.002 | 0.636 ± 0.002          | 0.665 ± 0.002          | *0.680* ± 0.002               | **0.707** ± 0.002            |
| Guangdong | V    | 75%          | MAE    | 1.350 ± 0.000 | 1.160 ± 0.000 | 1.026 ± 0.003 | 0.900 ± 0.001          | *0.864* ± 0.002        | 0.927 ± 0.013                | **0.782** ± 0.002            |
| Guangdong | V    | 75%          | $R^2$  | 0.144 ± 0.001 | 0.364 ± 0.002 | 0.462 ± 0.003 | 0.534 ± 0.001          | *0.565* ± 0.002        | 0.525 ± 0.008                | **0.613** ± 0.001            |
| Guangdong | T    | 12.5%        | MAE    | 1.439 ± 0.001 | 1.056 ± 0.002 | 0.304 ± 0.003 | 0.205 ± 0.001          | *0.161* ± 0.001        | 0.216 ± 0.000                | **0.143** ± 0.001            |
| Guangdong | T    | 12.5%        | $R^2$  | 0.891 ± 0.000 | 0.922 ± 0.001 | 0.994 ± 0.000 | *0.996* ± 0.000        | **0.997** ± 0.000      | *0.996* ± 0.000               | **0.997** ± 0.000            |
| Guangdong | T    | 25%          | MAE    | 1.441 ± 0.002 | 1.056 ± 0.002 | 0.361 ± 0.004 | *0.223* ± 0.001        | **0.182** ± 0.000      | 0.256 ± 0.001                | 0.197 ± 0.008                |
| Guangdong | T    | 25%          | $R^2$  | 0.891 ± 0.000 | 0.922 ± 0.000 | 0.993 ± 0.000 | *0.996* ± 0.000        | **0.997** ± 0.000      | 0.995 ± 0.000                | **0.997** ± 0.000            |
| Guangdong | T    | 50%          | MAE    | 1.442 ± 0.001 | 1.057 ± 0.001 | 0.502 ± 0.003 | *0.303* ± 0.001        | **0.268** ± 0.001      | 0.381 ± 0.001                | 0.336 ± 0.015                |
| Guangdong | T    | 50%          | $R^2$  | 0.891 ± 0.000 | 0.922 ± 0.001 | 0.987 ± 0.000 | **0.994** ± 0.000      | **0.994** ± 0.000      | 0.992 ± 0.000                | *0.993* ± 0.001               |
| Guangdong | T    | 75%          | MAE    | 1.442 ± 0.001 | 1.057 ± 0.001 | 1.645 ± 0.048 | *0.599* ± 0.002        | 0.639 ± 0.005          | 0.938 ± 0.262                | **0.470** ± 0.004            |
| Guangdong | T    | 75%          | $R^2$  | 0.891 ± 0.000 | 0.922 ± 0.001 | 0.862 ± 0.012 | *0.983* ± 0.000        | 0.982 ± 0.000          | 0.963 ± 0.022                | **0.988** ± 0.000            |
| Guangdong | P    | 12.5%        | MAE    | 0.580 ± 0.001 | 0.394 ± 0.002 | 0.171 ± 0.003 | 0.109 ± 0.001          | 0.066 ± 0.001          | *0.058* ± 0.000               | **0.048** ± 0.003            |
| Guangdong | P    | 12.5%        | $R^2$  | 0.983 ± 0.000 | 0.988 ± 0.000 | *0.999* ± 0.000 | *0.999* ± 0.000 | **1.000** ± 0.000 | **1.000** ± 0.000 | **1.000** ± 0.000  |
| Guangdong | P    | 25%          | MAE    | 0.580 ± 0.001 | 0.393 ± 0.001 | 0.201 ± 0.002 | 0.113 ± 0.000          | **0.079** ± 0.000      | 0.084 ± 0.000                | *0.081* ± 0.007               |
| Guangdong | P    | 25%          | $R^2$  | 0.983 ± 0.000 | 0.988 ± 0.000 | 0.998 ± 0.000 | *0.999* ± 0.000        | **1.000** ± 0.000      | **1.000** ± 0.000            | **1.000** ± 0.000            |
| Guangdong | P    | 50%          | MAE    | 0.581 ± 0.000 | 0.394 ± 0.001 | 0.277 ± 0.002 | 0.147 ± 0.001          | **0.127** ± 0.000      | 0.161 ± 0.002                | *0.143* ± 0.008               |
| Guangdong | P    | 50%          | $R^2$  | 0.983 ± 0.000 | 0.988 ± 0.000 | 0.997 ± 0.000 | **0.999** ± 0.000      | **0.999** ± 0.000      | **0.999** ± 0.000            | **0.999** ± 0.000            |
| Guangdong | P    | 75%          | MAE    | 0.581 ± 0.000 | 0.394 ± 0.001 | 1.799 ± 0.119 | **0.302** ± 0.001      | *0.314* ± 0.005        | 0.683 ± 0.098                | 0.388 ± 0.019                |
| Guangdong | P    | 75%          | $R^2$  | 0.983 ± 0.000 | 0.988 ± 0.000 | 0.853 ± 0.019 | **0.996** ± 0.000      | 0.993 ± 0.001          | 0.985 ± 0.004                | *0.994* ± 0.000               |
| California| U    | 12.5%        | MAE    | 1.603 ± 0.002 | 1.327 ± 0.003 | 0.984 ± 0.002 | 0.944 ± 0.002          | 0.978 ± 0.002          | *0.927* ± 0.002               | **0.890** ± 0.002            |
| California| U    | 12.5%        | $R^2$  | 0.228 ± 0.001 | 0.422 ± 0.001 | 0.670 ± 0.001 | 0.689 ± 0.001          | 0.670 ± 0.001          | *0.702* ± 0.001               | **0.715** ± 0.001            |
| California| U    | 25%          | MAE    | 1.603 ± 0.001 | 1.329 ± 0.001 | 1.009 ± 0.001 | 0.971 ± 0.000          | 0.999 ± 0.001          | *0.954* ± 0.001               | **0.914** ± 0.001            |
| California| U    | 25%          | $R^2$  | 0.229 ± 0.002 | 0.421 ± 0.001 | 0.653 ± 0.001 | 0.673 ± 0.001          | 0.657 ± 0.001          | *0.686* ± 0.000               | **0.703** ± 0.001            |
| California| U    | 50%          | MAE    | 1.602 ± 0.001 | 1.328 ± 0.001 | 1.090 ± 0.000 | 1.029 ± 0.001          | 1.061 ± 0.001          | *1.022* ± 0.001               | **0.998** ± 0.002            |
| California| U    | 50%          | $R^2$  | 0.229 ± 0.001 | 0.422 ± 0.001 | 0.601 ± 0.001 | 0.636 ± 0.001          | 0.616 ± 0.001          | *0.643* ± 0.001               | **0.657** ± 0.001            |
| California| U    | 75%          | MAE    | 1.603 ± 0.000 | 1.328 ± 0.000 | 1.219 ± 0.003 | *1.084* ± 0.001        | 1.104 ± 0.001          | 1.092 ± 0.002                | **1.043** ± 0.002            |
| California| U    | 75%          | $R^2$  | 0.229 ± 0.001 | 0.421 ± 0.001 | 0.499 ± 0.003 | 0.583 ± 0.001          | 0.569 ± 0.002          | *0.593* ± 0.001               | **0.614** ± 0.002            |
| California| V    | 12.5%        | MAE    | 1.719 ± 0.002 | 1.370 ± 0.002 | 0.977 ± 0.001 | 0.936 ± 0.001          | 0.971 ± 0.001          | *0.918* ± 0.001               | **0.882** ± 0.001            |
| California| V    | 12.5%        | $R^2$  | 0.130 ± 0.003 | 0.410 ± 0.001 | 0.686 ± 0.001 | 0.704 ± 0.001          | 0.686 ± 0.001          | *0.719* ± 0.000               | **0.732** ± 0.001            |
| California| V    | 25%          | MAE    | 1.721 ± 0.001 | 1.371 ± 0.001 | 0.999 ± 0.001 | 0.963 ± 0.001          | 0.991 ± 0.000          | *0.944* ± 0.001               | **0.906** ± 0.001            |
| California| V    | 25%          | $R^2$  | 0.130 ± 0.001 | 0.410 ± 0.001 | 0.674 ± 0.000 | 0.689 ± 0.001          | 0.675 ± 0.000          | *0.705* ± 0.001               | **0.720** ± 0.000            |
| California| V    | 50%          | MAE    | 1.719 ± 0.001 | 1.370 ± 0.000 | 1.080 ± 0.001 | 1.026 ± 0.001          | 1.055 ± 0.001          | *1.016* ± 0.001               | **0.990** ± 0.006            |
| California| V    | 50%          | $R^2$  | 0.130 ± 0.001 | 0.410 ± 0.001 | 0.622 ± 0.001 | 0.650 ± 0.000          | 0.635 ± 0.001          | *0.660* ± 0.001               | **0.675** ± 0.003            |
| California| V    | 75%          | MAE    | 1.720 ± 0.000 | 1.370 ± 0.001 | 1.268 ± 0.003 | *1.077* ± 0.001        | 1.098 ± 0.001          | 1.088 ± 0.003                | **1.060** ± 0.001            |
| California| V    | 75%          | $R^2$  | 0.131 ± 0.000 | 0.410 ± 0.001 | 0.483 ± 0.001 | 0.599 ± 0.001          | 0.584 ± 0.001          | *0.605* ± 0.002               | **0.620** ± 0.001            |
| California| T    | 12.5%        | MAE    | 2.825 ± 0.002 | 1.437 ± 0.002 | 0.658 ± 0.002 | 0.567 ± 0.001          | 0.583 ± 0.001          | *0.546* ± 0.000               | **0.513** ± 0.002            |
| California| T    | 12.5%        | $R^2$  | 0.764 ± 0.000 | 0.921 ± 0.000 | 0.983 ± 0.000 | 0.986 ± 0.000          | 0.986 ± 0.000          | *0.987* ± 0.000               | **0.989** ± 0.000            |
| California| T    | 25%          | MAE    | 2.823 ± 0.002 | 1.438 ± 0.001 | 0.730 ± 0.003 | 0.588 ± 0.000          | 0.602 ± 0.000          | *0.578* ± 0.000               | **0.565** ± 0.019            |
| California| T    | 25%          | $R^2$  | 0.764 ± 0.000 | 0.921 ± 0.000 | 0.980 ± 0.000 | 0.985 ± 0.000          | 0.985 ± 0.000          | *0.986* ± 0.000               | **0.987** ± 0.001            |
| California| T    | 50%          | MAE    | 2.825 ± 0.000 | 1.438 ± 0.001 | 0.913 ± 0.001 | **0.663** ± 0.000      | *0.682* ± 0.001        | 0.697 ± 0.002                | 0.718 ± 0.022                |
| California| T    | 50%          | $R^2$  | 0.764 ± 0.000 | 0.921 ± 0.000 | 0.969 ± 0.000 | **0.981** ± 0.000      | *0.980* ± 0.000        | *0.980* ± 0.000               | 0.979 ± 0.001                |
| California| T    | 75%          | MAE    | 2.824 ± 0.000 | 1.438 ± 0.001 | 2.323 ± 0.082 | *0.926* ± 0.003        | 1.002 ± 0.002          | 0.994 ± 0.045                | **0.833** ± 0.006            |
| California| T    | 75%          | $R^2$  | 0.764 ± 0.000 | 0.921 ± 0.000 | 0.833 ± 0.012 | *0.965* ± 0.000        | 0.961 ± 0.000          | 0.962 ± 0.003                | **0.972** ± 0.000            |
| California| P    | 12.5%        | MAE    | 1.156 ± 0.001 | 0.521 ± 0.002 | 0.232 ± 0.001 | 0.184 ± 0.000          | 0.187 ± 0.000          | *0.155* ± 0.001               | **0.141** ± 0.001            |
| California| P    | 12.5%        | $R^2$  | 0.924 ± 0.000 | 0.974 ± 0.000 | 0.995 ± 0.000 | 0.996 ± 0.000          | 0.996 ± 0.000          | *0.997* ± 0.000               | **0.998** ± 0.000            |
| California| P    | 25%          | MAE    | 1.156 ± 0.001 | 0.522 ± 0.001 | 0.265 ± 0.002 | 0.191 ± 0.000          | 0.198 ± 0.001          | *0.169* ± 0.000               | **0.159** ± 0.001            |
| California| P    | 25%          | $R^2$  | 0.924 ± 0.000 | 0.974 ± 0.000 | 0.993 ± 0.000 | *0.996*± 0.000          | 0.995 ± 0.000          | **0.997** ± 0.000            | **0.997** ± 0.000            |
| California| P    | 50%          | MAE    | 1.156 ± 0.001 | 0.522 ± 0.001 | 0.350 ± 0.001 | *0.215* ± 0.000        | 0.232 ± 0.001          | 0.218 ± 0.002                | **0.212** ± 0.005            |
| California| P    | 50%          | $R^2$  | 0.924 ± 0.000 | 0.974 ± 0.000 | 0.988 ± 0.000 | **0.995** ± 0.000      | *0.993* ± 0.000          | **0.995** ± 0.000            | **0.995** ± 0.000            |
| California| P    | 75%          | MAE    | 1.156 ± 0.000 | 0.522 ± 0.001 | 1.215 ± 0.027 | **0.334** ± 0.001      | *0.389* ± 0.003        | 1.040 ± 0.162                | 0.417 ± 0.014                |
| California| P    | 75%          | $R^2$  | 0.924 ± 0.000 | 0.974 ± 0.000 | 0.865 ± 0.006 | **0.988** ± 0.000      | 0.984 ± 0.001          | 0.918 ± 0.023                | *0.985* ± 0.001               |
### Table 2: Bin-wise performance evaluation (in MAE) of imputation methods across wind-speed regimes under different missing rates.
| Dataset   | Var. | Missing rate | Speed (m/s)   | OK                  | LR                   | Autoformer           | NS-Trans             | TimesNet             | MPNN                 | RAMP-Net (Ours)      |
| --------- | ---- | ------------ | ------------- | ------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| Guangdong | U    | 50%          | $10 \sim 12.5$ | 3.645 ± 0.034       | 3.149 ± 0.037        | 2.100 ± 0.084        | 2.121 ± 0.083        | *2.007* ± 0.112      | 2.071 ± 0.085        | **1.853** ± 0.088    |
| Guangdong | U    | 50%          | $12.5 \sim 15$ | 6.740 ± 0.127       | 5.125 ± 0.156        | 2.975 ± 0.217        | 3.022 ± 0.141        | *2.837* ± 0.181      | 2.975 ± 0.184        | **2.666** ± 0.165    |
| Guangdong | U    | 50%          | $15 \sim 17.5$ | 6.432 ± 0.162       | 4.488 ± 0.326        | 2.267 ± 0.226        | 2.460 ± 0.353        | *1.851* ± 0.226      | 2.336 ± 0.230        | **1.796** ± 0.195    |
| Guangdong | U    | 50%          | $17.5 \sim 20$ | 7.045 ± 0.519       | 7.862 ± 0.487        | 9.145 ± 0.814        | 7.832 ± 1.323        | *7.375* ± 1.920      | 9.019 ± 0.938        | **6.006** ± 1.619    |
| Guangdong | U    | 50%          | $\ge 20.0$    | /                   | /                    | /                    | /                    | /                    | /                    | /                    |
| Guangdong | U    | 75%          | $10 \sim 12.5$ | 3.676 ± 0.019       | 3.191 ± 0.023        | 2.963 ± 0.095        | *2.282* ± 0.102      | 2.526 ± 0.110        | 2.509 ± 0.091        | **2.277** ± 0.107    |
| Guangdong | U    | 75%          | $12.5 \sim 15$ | 6.725 ± 0.019       | 5.115 ± 0.093        | 5.234 ± 0.282        | *4.122* ± 0.164      | 4.634 ± 0.272        | 4.276 ± 0.066        | **3.810** ± 0.113    |
| Guangdong | U    | 75%          | $15 \sim 17.5$ | 6.507 ± 0.116       | 4.626 ± 0.220        | 4.381 ± 0.341        | *3.554* ± 0.133      | 3.899 ± 0.456        | 3.612 ± 0.159        | **3.365** ± 0.186    |
| Guangdong | U    | 75%          | $17.5 \sim 20$ | **5.851** ± 0.262   | 6.682 ± 0.297        | 8.634 ± 0.741        | 7.159 ± 0.423        | 6.743 ± 0.643        | 6.851 ± 0.349        | *6.552* ± 0.429      |
| Guangdong | U    | 75%          | $\ge 20.0$    | /                   | /                    | /                    | /                    | /                    | /                    | /                    |
| Guangdong | V    | 50%          | $10 \sim 12.5$ | 5.500 ± 0.032       | 4.875 ± 0.026        | 2.874 ± 0.152        | 2.881 ± 0.116        | 2.560 ± 0.163        | *2.559* ± 0.103      | **2.160** ± 0.125    |
| Guangdong | V    | 50%          | $12.5 \sim 15$ | 5.232 ± 0.120       | 4.351 ± 0.127        | 2.552 ± 0.164        | 2.431 ± 0.117        | 2.466 ± 0.175        | *2.268* ± 0.138      | **1.997** ± 0.095    |
| Guangdong | V    | 50%          | $15 \sim 17.5$ | 6.496 ± 0.096       | 3.753 ± 0.119        | 2.395 ± 0.154        | 2.046 ± 0.205        | *1.994* ± 0.210      | 2.183 ± 0.188        | **1.810** ± 0.145    |
| Guangdong | V    | 50%          | $17.5 \sim 20$ | 11.383 ± 0.466      | 10.904 ± 0.718       | 9.467 ± 1.553        | 8.644 ± 1.069        | 6.850 ± 0.435        | *6.144* ± 0.503      | **4.869** ± 0.610    |
| Guangdong | V    | 50%          | $\ge 20.0$    | /                   | /                    | /                    | /                    | /                    | /                    | /                    |
| Guangdong | V    | 75%          | $10 \sim 12.5$ | 5.524 ± 0.012       | 4.882 ± 0.009        | 4.469 ± 0.103        | *3.336* ± 0.102      | 3.382 ± 0.133        | 3.676 ± 0.116        | **3.081** ± 0.082    |
| Guangdong | V    | 75%          | $12.5 \sim 15$ | 5.215 ± 0.084       | 4.331 ± 0.072        | 3.901 ± 0.047        | *3.019* ± 0.054      | 3.174 ± 0.077        | 3.084 ± 0.084        | **2.696** ± 0.119    |
| Guangdong | V    | 75%          | $15 \sim 17.5$ | 6.553 ± 0.027       | 3.831 ± 0.056        | 4.113 ± 0.245        | **2.852** ± 0.080    | *3.103* ± 0.162      | 3.364 ± 0.167        | 3.106 ± 0.124        |
| Guangdong | V    | 75%          | $17.5 \sim 20$ | 11.413 ± 0.218      | 10.915 ± 0.329       | 10.591 ± 0.680       | 9.936 ± 0.589        | *9.637* ± 0.323      | 9.939 ± 0.499        | **8.318** ± 0.484    |
| Guangdong | V    | 75%          | $\ge 20.0$    | /                   | /                    | /                    | /                    | /                    | /                    | /                    |
| California| U    | 50%          | $10 \sim 12.5$ | 3.922 ± 0.021       | 4.014 ± 0.019        | 2.382 ± 0.027        | 2.367 ± 0.032        | 2.291 ± 0.041        | *2.205* ± 0.026      | **2.152** ± 0.038    |
| California| U    | 50%          | $12.5 \sim 15$ | 5.050 ± 0.018       | 5.231 ± 0.015        | 2.808 ± 0.035        | 2.888 ± 0.050        | 2.697 ± 0.051        | *2.612* ± 0.020      | **2.556** ± 0.021    |
| California| U    | 50%          | $15 \sim 17.5$ | 7.408 ± 0.032       | 7.411 ± 0.027        | 3.921 ± 0.123        | 4.130 ± 0.133        | 3.829 ± 0.122        | **3.600** ± 0.107    | *3.699* ± 0.140      |
| California| U    | 50%          | $17.5 \sim 20$ | 8.534 ± 0.049       | 8.894 ± 0.041        | 4.234 ± 0.141        | 4.658 ± 0.109        | 4.120 ± 0.118        | *3.761* ± 0.130      | **3.734** ± 0.165    |
| California| U    | 50%          | $\ge 20.0$    | 11.687 ± 0.033      | 11.154 ± 0.020       | 4.136 ± 0.191        | 4.851 ± 0.141        | 3.981 ± 0.132        | *3.334* ± 0.099      | **3.233** ± 0.123    |
| California| U    | 75%          | $10 \sim 12.5$ | 3.931 ± 0.008       | 4.028 ± 0.011        | 3.425 ± 0.032        | *2.693* ± 0.009      | 2.947 ± 0.037        | 2.834 ± 0.028        | **2.420** ± 0.018    |
| California| U    | 75%          | $12.5 \sim 15$ | 5.057 ± 0.017       | 5.238 ± 0.022        | 4.222 ± 0.026        | *3.433* ± 0.049      | 3.785 ± 0.037        | 3.531 ± 0.052        | **2.909** ± 0.057    |
| California| U    | 75%          | $15 \sim 17.5$ | 7.403 ± 0.018       | 7.410 ± 0.017        | 5.814 ± 0.118        | 5.231 ± 0.080        | 5.533 ± 0.049        | *5.081* ± 0.098      | **4.193** ± 0.064    |
| California| U    | 75%          | $17.5 \sim 20$ | 8.524 ± 0.019       | 8.883 ± 0.023        | 7.121 ± 0.053        | 6.168 ± 0.064        | 6.414 ± 0.178        | *5.306* ± 0.142      | **4.212** ± 0.068    |
| California| U    | 75%          | $\ge 20.0$    | 11.702 ± 0.032      | 11.173 ± 0.034       | 8.221 ± 0.211        | 7.064 ± 0.042        | 7.827 ± 0.129        | *5.601* ± 0.100      | **4.074** ± 0.207    |
| California| V    | 50%          | $10 \sim 12.5$ | 5.075 ± 0.014       | 4.297 ± 0.017        | 2.396 ± 0.020        | 2.351 ± 0.017        | 2.348 ± 0.031        | *2.190* ± 0.008      | **2.186** ± 0.048    |
| California| V    | 50%          | $12.5 \sim 15$ | 6.642 ± 0.017       | 5.540 ± 0.030        | 2.997 ± 0.036        | 2.956 ± 0.022        | 2.958 ± 0.018        | **2.704** ± 0.036    | *2.711* ± 0.080      |
| California| V    | 50%          | $15 \sim 17.5$ | 9.087 ± 0.013       | 7.607 ± 0.029        | 4.217 ± 0.066        | 4.377 ± 0.084        | 4.103 ± 0.037        | **3.920** ± 0.062    | *4.069* ± 0.047      |
| California| V    | 50%          | $17.5 \sim 20$ | 10.976 ± 0.047      | 9.544 ± 0.063        | 4.708 ± 0.108        | 5.225 ± 0.083        | 4.648 ± 0.084        | **4.149** ± 0.092    | *4.264* ± 0.080      |
| California| V    | 50%          | $\ge 20.0$    | 14.117 ± 0.029      | 12.779 ± 0.038       | 4.371 ± 0.196        | 5.655 ± 0.192        | 4.131 ± 0.284        | **3.333** ± 0.150    | *3.662* ± 0.181      |
| California| V    | 75%          | $10 \sim 12.5$ | 5.089 ± 0.006       | 4.307 ± 0.019        | 3.884 ± 0.027        | *2.739* ± 0.015      | 3.015 ± 0.035        | 2.831 ± 0.032        | **2.383** ± 0.034    |
| California| V    | 75%          | $12.5 \sim 15$ | 6.638 ± 0.009       | 5.536 ± 0.021        | 5.171 ± 0.035        | *3.670* ± 0.033      | 4.096 ± 0.044        | 3.706 ± 0.049        | **2.982** ± 0.064    |
| California| V    | 75%          | $15 \sim 17.5$ | 9.083 ± 0.024       | 7.592 ± 0.052        | 6.981 ± 0.058        | *5.427* ± 0.031      | 6.014 ± 0.083        | 5.761 ± 0.074        | **4.714** ± 0.076    |
| California| V    | 75%          | $17.5 \sim 20$ | 10.951 ± 0.031      | 9.515 ± 0.039        | 8.187 ± 0.072        | 6.661 ± 0.110        | 7.189 ± 0.039        | *6.168* ± 0.096      | **4.995** ± 0.191    |
| California| V    | 75%          | $\ge 20.0$    | 14.127 ± 0.022      | 12.798 ± 0.030       | 9.909 ± 0.070        | 8.523 ± 0.096        | 9.385 ± 0.221        | *6.923* ± 0.186      | **5.199** ± 0.139    |


## ✍️ Citation


