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

### Table 1: Quantitative evaluation on the SEVIR dataset\

<div style="width: 100%; overflow-x: auto; border: 1px solid #ddd; padding: 10px;">
  <table style="width: 2600px; border-collapse: collapse; text-align: center; font-size: 13px; font-family: sans-serif;">
  <thead>
    <tr style="background-color: #f2f2f2;">
      <th style="border: 1px solid #ddd; padding: 4px;">Dataset</th><th style="border: 1px solid #ddd; padding: 4px;">Var.</th><th style="border: 1px solid #ddd; padding: 4px;">Missing rate</th><th style="border: 1px solid #ddd; padding: 4px;">Metric</th><th style="border: 1px solid #ddd; padding: 4px;">OK</th><th style="border: 1px solid #ddd; padding: 4px;">LR</th><th style="border: 1px solid #ddd; padding: 4px;">Autoformer</th><th style="border: 1px solid #ddd; padding: 4px;">NS-Trans</th><th style="border: 1px solid #ddd; padding: 4px;">TimesNet</th><th style="border: 1px solid #ddd; padding: 4px;">MPNN</th><th style="border: 1px solid #ddd; padding: 4px;">RAMP-Net (Ours)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td rowspan="8" style="border: 1px solid #ddd; padding: 4px;">Guangdong</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">U</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">12.5%</td><td>MAE</td><td>1.237 ± 0.002</td><td>1.074 ± 0.002</td><td>0.556 ± 0.002</td><td>0.526 ± 0.001</td><td>0.498 ± 0.004</td><td><u>0.491</u> ± 0.002</td><td>0.372 ± 0.005</td></tr>
    <tr><td>R²</td><td>0.074 ± 0.002</td><td>0.259 ± 0.002</td><td>0.696 ± 0.002</td><td>0.702 ± 0.003</td><td>0.727 ± 0.004</td><td><u>0.743</u> ± 0.002</td><td>0.819 ± 0.003</td></tr>
    <tr><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">U</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">25%</td><td>MAE</td><td>1.238 ± 0.001</td><td>1.077 ± 0.002</td><td>0.593 ± 0.002</td><td>0.573 ± 0.001</td><td><u>0.526</u> ± 0.002</td><td>0.534 ± 0.001</td><td>0.419 ± 0.002</td></tr>
    <tr><td>R²</td><td>0.075 ± 0.000</td><td>0.258 ± 0.001</td><td>0.668 ± 0.003</td><td>0.673 ± 0.002</td><td>0.710 ± 0.001</td><td><u>0.713</u> ± 0.002</td><td>0.788 ± 0.002</td></tr>
    <tr><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">U</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">50%</td><td>MAE</td><td>1.238 ± 0.001</td><td>1.076 ± 0.001</td><td>0.694 ± 0.003</td><td>0.670 ± 0.002</td><td><u>0.641</u> ± 0.002</td><td>0.649 ± 0.001</td><td>0.574 ± 0.003</td></tr>
    <tr><td>R²</td><td>0.073 ± 0.001</td><td>0.258 ± 0.001</td><td>0.580 ± 0.003</td><td>0.600 ± 0.001</td><td><u>0.615</u> ± 0.002</td><td>0.607 ± 0.001</td><td>0.668 ± 0.005</td></tr>
    <tr><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">U</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">75%</td><td>MAE</td><td>1.237 ± 0.000</td><td>1.076 ± 0.001</td><td>0.908 ± 0.001</td><td>0.801 ± 0.001</td><td><u>0.779</u> ± 0.001</td><td>0.826 ± 0.001</td><td>0.718 ± 0.003</td></tr>
    <tr><td>R²</td><td>0.074 ± 0.001</td><td>0.257 ± 0.001</td><td>0.415 ± 0.001</td><td>0.500 ± 0.001</td><td><u>0.520</u> ± 0.001</td><td>0.482 ± 0.001</td><td>0.561 ± 0.003</td></tr>
    <tr><td rowspan="8" style="border: 1px solid #ddd; padding: 4px;">Guangdong</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">V</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">12.5%</td><td>MAE</td><td>1.350 ± 0.002</td><td>1.161 ± 0.003</td><td>0.570 ± 0.003</td><td>0.554 ± 0.004</td><td>0.495 ± 0.005</td><td><u>0.478</u> ± 0.003</td><td>0.368 ± 0.004</td></tr>
    <tr><td>R²</td><td>0.145 ± 0.002</td><td>0.365 ± 0.003</td><td>0.777 ± 0.001</td><td>0.764 ± 0.005</td><td>0.809 ± 0.004</td><td><u>0.833</u> ± 0.003</td><td>0.872 ± 0.003</td></tr>
    <tr><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">V</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">25%</td><td>MAE</td><td>1.351 ± 0.001</td><td>1.161 ± 0.002</td><td>0.634 ± 0.005</td><td>0.620 ± 0.002</td><td><u>0.543</u> ± 0.002</td><td>0.546 ± 0.002</td><td>0.440 ± 0.004</td></tr>
    <tr><td>R²</td><td>0.143 ± 0.002</td><td>0.363 ± 0.002</td><td>0.726 ± 0.004</td><td>0.719 ± 0.002</td><td>0.771 ± 0.001</td><td><u>0.786</u> ± 0.002</td><td>0.826 ± 0.002</td></tr>
    <tr><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">V</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">50%</td><td>MAE</td><td>1.350 ± 0.001</td><td>1.160 ± 0.002</td><td>0.768 ± 0.002</td><td>0.741 ± 0.001</td><td><u>0.689</u> ± 0.002</td><td>0.691 ± 0.002</td><td>0.626 ± 0.005</td></tr>
    <tr><td>R²</td><td>0.144 ± 0.002</td><td>0.363 ± 0.001</td><td>0.624 ± 0.002</td><td>0.636 ± 0.002</td><td>0.665 ± 0.002</td><td><u>0.680</u> ± 0.002</td><td>0.707 ± 0.002</td></tr>
    <tr><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">V</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">75%</td><td>MAE</td><td>1.350 ± 0.000</td><td>1.160 ± 0.000</td><td>1.026 ± 0.003</td><td>0.900 ± 0.001</td><td><u>0.864</u> ± 0.002</td><td>0.927 ± 0.013</td><td>0.782 ± 0.002</td></tr>
    <tr><td>R²</td><td>0.144 ± 0.001</td><td>0.364 ± 0.002</td><td>0.462 ± 0.003</td><td>0.534 ± 0.001</td><td><u>0.565</u> ± 0.002</td><td>0.525 ± 0.008</td><td>0.613 ± 0.001</td></tr>
    <tr><td rowspan="8" style="border: 1px solid #ddd; padding: 4px;">Guangdong</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">T</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">12.5%</td><td>MAE</td><td>1.439 ± 0.001</td><td>1.056 ± 0.002</td><td>0.304 ± 0.003</td><td>0.205 ± 0.001</td><td><u>0.161</u> ± 0.001</td><td>0.216 ± 0.000</td><td>0.143 ± 0.001</td></tr>
    <tr><td>R²</td><td>0.891 ± 0.000</td><td>0.922 ± 0.001</td><td>0.994 ± 0.000</td><td><u>0.996</u> ± 0.000</td><td>0.997 ± 0.000</td><td><u>0.996</u> ± 0.000</td><td>0.997 ± 0.000</td></tr>
    <tr><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">T</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">25%</td><td>MAE</td><td>1.441 ± 0.002</td><td>1.056 ± 0.002</td><td>0.361 ± 0.004</td><td><u>0.223</u> ± 0.001</td><td>0.182 ± 0.000</td><td>0.256 ± 0.001</td><td>0.197 ± 0.008</td></tr>
    <tr><td>R²</td><td>0.891 ± 0.000</td><td>0.922 ± 0.000</td><td>0.993 ± 0.000</td><td><u>0.996</u> ± 0.000</td><td>0.997 ± 0.000</td><td>0.995 ± 0.000</td><td>0.997 ± 0.000</td></tr>
    <tr><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">T</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">50%</td><td>MAE</td><td>1.442 ± 0.001</td><td>1.057 ± 0.001</td><td>0.502 ± 0.003</td><td><u>0.303</u> ± 0.001</td><td>0.268 ± 0.001</td><td>0.381 ± 0.001</td><td>0.336 ± 0.015</td></tr>
    <tr><td>R²</td><td>0.891 ± 0.000</td><td>0.922 ± 0.001</td><td>0.987 ± 0.000</td><td>0.994 ± 0.000</td><td>0.994 ± 0.000</td><td>0.992 ± 0.000</td><td><u>0.993</u> ± 0.001</td></tr>
    <tr><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">T</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">75%</td><td>MAE</td><td>1.442 ± 0.001</td><td>1.057 ± 0.001</td><td>1.645 ± 0.048</td><td><u>0.599</u> ± 0.002</td><td>0.639 ± 0.005</td><td>0.938 ± 0.262</td><td>0.470 ± 0.004</td></tr>
    <tr><td>R²</td><td>0.891 ± 0.000</td><td>0.922 ± 0.001</td><td>0.862 ± 0.012</td><td><u>0.983</u> ± 0.000</td><td>0.982 ± 0.000</td><td>0.963 ± 0.022</td><td>0.988 ± 0.000</td></tr>
    <tr><td rowspan="8" style="border: 1px solid #ddd; padding: 4px;">Guangdong</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">P</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">12.5%</td><td>MAE</td><td>0.580 ± 0.001</td><td>0.394 ± 0.002</td><td>0.171 ± 0.003</td><td>0.109 ± 0.001</td><td>0.066 ± 0.001</td><td><u>0.058</u> ± 0.000</td><td>0.048 ± 0.003</td></tr>
    <tr><td>R²</td><td>0.983 ± 0.000</td><td>0.988 ± 0.000</td><td><u>0.999</u> ± 0.000</td><td><u>0.999</u> ± 0.000</td><td>1.000 ± 0.000</td><td>1.000 ± 0.000</td><td>1.000 ± 0.000</td></tr>
    <tr><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">P</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">25%</td><td>MAE</td><td>0.580 ± 0.001</td><td>0.393 ± 0.001</td><td>0.201 ± 0.002</td><td>0.113 ± 0.000</td><td>0.079 ± 0.000</td><td>0.084 ± 0.000</td><td><u>0.081</u> ± 0.007</td></tr>
    <tr><td>R²</td><td>0.983 ± 0.000</td><td>0.988 ± 0.000</td><td>0.998 ± 0.000</td><td><u>0.999</u> ± 0.000</td><td>1.000 ± 0.000</td><td>1.000 ± 0.000</td><td>1.000 ± 0.000</td></tr>
    <tr><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">P</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">50%</td><td>MAE</td><td>0.581 ± 0.000</td><td>0.394 ± 0.001</td><td>0.277 ± 0.002</td><td>0.147 ± 0.001</td><td>0.127 ± 0.000</td><td>0.161 ± 0.002</td><td><u>0.143</u> ± 0.008</td></tr>
    <tr><td>R²</td><td>0.983 ± 0.000</td><td>0.988 ± 0.000</td><td>0.997 ± 0.000</td><td>0.999 ± 0.000</td><td>0.999 ± 0.000</td><td>0.999 ± 0.000</td><td>0.999 ± 0.000</td></tr>
    <tr><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">P</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">75%</td><td>MAE</td><td>0.581 ± 0.000</td><td>0.394 ± 0.001</td><td>1.799 ± 0.119</td><td>0.302 ± 0.002</td><td><u>0.314</u> ± 0.005</td><td>0.683 ± 0.098</td><td>0.388 ± 0.019</td></tr>
    <tr><td>R²</td><td>0.983 ± 0.000</td><td>0.988 ± 0.000</td><td>0.853 ± 0.019</td><td>0.996 ± 0.000</td><td>0.993 ± 0.001</td><td>0.985 ± 0.004</td><td><u>0.994</u> ± 0.000</td></tr>
    <tr><td rowspan="8" style="border: 1px solid #ddd; padding: 4px;">California</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">U</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">12.5%</td><td>MAE</td><td>1.603 ± 0.002</td><td>1.327 ± 0.003</td><td>0.984 ± 0.002</td><td>0.944 ± 0.002</td><td>0.978 ± 0.002</td><td><u>0.927</u> ± 0.002</td><td>0.890 ± 0.002</td></tr>
    <tr><td>R²</td><td>0.228 ± 0.001</td><td>0.422 ± 0.001</td><td>0.670 ± 0.001</td><td>0.689 ± 0.001</td><td>0.670 ± 0.001</td><td><u>0.702</u> ± 0.001</td><td>0.715 ± 0.001</td></tr>
    <tr><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">U</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">25%</td><td>MAE</td><td>1.603 ± 0.001</td><td>1.329 ± 0.001</td><td>1.009 ± 0.001</td><td>0.971 ± 0.000</td><td>0.999 ± 0.001</td><td><u>0.954</u> ± 0.001</td><td>0.914 ± 0.001</td></tr>
    <tr><td>R²</td><td>0.229 ± 0.002</td><td>0.421 ± 0.001</td><td>0.653 ± 0.001</td><td>0.673 ± 0.001</td><td>0.657 ± 0.001</td><td><u>0.686</u> ± 0.000</td><td>0.703 ± 0.001</td></tr>
    <tr><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">U</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">50%</td><td>MAE</td><td>1.602 ± 0.001</td><td>1.328 ± 0.001</td><td>1.090 ± 0.000</td><td>1.029 ± 0.001</td><td>1.061 ± 0.001</td><td><u>1.022</u> ± 0.001</td><td>0.998 ± 0.002</td></tr>
    <tr><td>R²</td><td>0.229 ± 0.001</td><td>0.422 ± 0.001</td><td>0.601 ± 0.001</td><td>0.636 ± 0.001</td><td>0.616 ± 0.001</td><td><u>0.643</u> ± 0.001</td><td>0.657 ± 0.001</td></tr>
    <tr><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">U</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">75%</td><td>MAE</td><td>1.603 ± 0.000</td><td>1.328 ± 0.000</td><td>1.219 ± 0.003</td><td><u>1.084</u> ± 0.001</td><td>1.104 ± 0.001</td><td>1.092 ± 0.002</td><td>1.043 ± 0.002</td></tr>
    <tr><td>R²</td><td>0.229 ± 0.001</td><td>0.421 ± 0.001</td><td>0.499 ± 0.003</td><td>0.583 ± 0.001</td><td>0.569 ± 0.002</td><td><u>0.593</u> ± 0.001</td><td>0.614 ± 0.002</td></tr>
    <tr><td rowspan="8" style="border: 1px solid #ddd; padding: 4px;">California</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">V</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">12.5%</td><td>MAE</td><td>1.719 ± 0.002</td><td>1.370 ± 0.002</td><td>0.977 ± 0.001</td><td>0.936 ± 0.001</td><td>0.971 ± 0.001</td><td><u>0.918</u> ± 0.001</td><td>0.882 ± 0.001</td></tr>
    <tr><td>R²</td><td>0.130 ± 0.003</td><td>0.410 ± 0.001</td><td>0.686 ± 0.001</td><td>0.704 ± 0.001</td><td>0.686 ± 0.001</td><td><u>0.719</u> ± 0.000</td><td>0.732 ± 0.001</td></tr>
    <tr><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">V</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">25%</td><td>MAE</td><td>1.721 ± 0.001</td><td>1.371 ± 0.001</td><td>0.999 ± 0.001</td><td>0.963 ± 0.001</td><td>0.991 ± 0.000</td><td><u>0.944</u> ± 0.001</td><td>0.906 ± 0.001</td></tr>
    <tr><td>R²</td><td>0.130 ± 0.001</td><td>0.410 ± 0.001</td><td>0.674 ± 0.000</td><td>0.689 ± 0.001</td><td>0.675 ± 0.000</td><td><u>0.705</u> ± 0.001</td><td>0.720 ± 0.000</td></tr>
    <tr><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">V</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">50%</td><td>MAE</td><td>1.719 ± 0.001</td><td>1.370 ± 0.000</td><td>1.080 ± 0.001</td><td>1.026 ± 0.001</td><td>1.055 ± 0.001</td><td><u>1.016</u> ± 0.001</td><td>0.990 ± 0.006</td></tr>
    <tr><td>R²</td><td>0.130 ± 0.001</td><td>0.410 ± 0.001</td><td>0.622 ± 0.001</td><td>0.650 ± 0.000</td><td>0.635 ± 0.001</td><td><u>0.660</u> ± 0.001</td><td>0.675 ± 0.003</td></tr>
    <tr><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">V</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">75%</td><td>MAE</td><td>1.720 ± 0.000</td><td>1.370 ± 0.001</td><td>1.268 ± 0.003</td><td><u>1.077</u> ± 0.001</td><td>1.098 ± 0.001</td><td>1.088 ± 0.003</td><td>1.060 ± 0.001</td></tr>
    <tr><td>R²</td><td>0.131 ± 0.000</td><td>0.410 ± 0.001</td><td>0.483 ± 0.001</td><td>0.599 ± 0.001</td><td>0.584 ± 0.001</td><td><u>0.605</u> ± 0.002</td><td>0.620 ± 0.001</td></tr>
    <tr><td rowspan="8" style="border: 1px solid #ddd; padding: 4px;">California</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">T</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">12.5%</td><td>MAE</td><td>2.825 ± 0.002</td><td>1.437 ± 0.002</td><td>0.658 ± 0.002</td><td>0.567 ± 0.001</td><td>0.583 ± 0.001</td><td><u>0.546</u> ± 0.000</td><td>0.513 ± 0.002</td></tr>
    <tr><td>R²</td><td>0.764 ± 0.000</td><td>0.921 ± 0.000</td><td>0.983 ± 0.000</td><td>0.986 ± 0.000</td><td>0.986 ± 0.000</td><td><u>0.987</u> ± 0.000</td><td>0.989 ± 0.000</td></tr>
    <tr><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">T</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">25%</td><td>MAE</td><td>2.823 ± 0.002</td><td>1.438 ± 0.001</td><td>0.730 ± 0.003</td><td>0.588 ± 0.000</td><td>0.602 ± 0.000</td><td><u>0.578</u> ± 0.000</td><td>0.565 ± 0.019</td></tr>
    <tr><td>R²</td><td>0.764 ± 0.000</td><td>0.921 ± 0.000</td><td>0.980 ± 0.000</td><td>0.985 ± 0.000</td><td>0.985 ± 0.000</td><td><u>0.986</u> ± 0.000</td><td>0.987 ± 0.001</td></tr>
    <tr><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">T</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">50%</td><td>MAE</td><td>2.825 ± 0.000</td><td>1.438 ± 0.001</td><td>0.913 ± 0.001</td><td>0.663 ± 0.000</td><td><u>0.682</u> ± 0.001</td><td>0.697 ± 0.002</td><td>0.718 ± 0.022</td></tr>
    <tr><td>R²</td><td>0.764 ± 0.000</td><td>0.921 ± 0.000</td><td>0.969 ± 0.000</td><td>0.981 ± 0.000</td><td><u>0.980</u> ± 0.000</td><td><u>0.980</u> ± 0.000</td><td>0.979 ± 0.001</td></tr>
    <tr><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">T</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">75%</td><td>MAE</td><td>2.824 ± 0.000</td><td>1.438 ± 0.001</td><td>2.323 ± 0.082</td><td><u>0.926</u> ± 0.003</td><td>1.002 ± 0.002</td><td>0.994 ± 0.045</td><td>0.833 ± 0.006</td></tr>
    <tr><td>R²</td><td>0.764 ± 0.000</td><td>0.921 ± 0.000</td><td>0.833 ± 0.012</td><td><u>0.965</u> ± 0.000</td><td>0.961 ± 0.000</td><td>0.962 ± 0.003</td><td>0.972 ± 0.000</td></tr>
    <tr><td rowspan="8" style="border: 1px solid #ddd; padding: 4px;">California</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">P</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">12.5%</td><td>MAE</td><td>1.156 ± 0.001</td><td>0.521 ± 0.002</td><td>0.232 ± 0.001</td><td>0.184 ± 0.000</td><td>0.187 ± 0.000</td><td><u>0.155</u> ± 0.001</td><td>0.141 ± 0.001</td></tr>
    <tr><td>R²</td><td>0.924 ± 0.000</td><td>0.974 ± 0.000</td><td>0.995 ± 0.000</td><td>0.996 ± 0.000</td><td>0.996 ± 0.000</td><td><u>0.997</u> ± 0.000</td><td>0.998 ± 0.000</td></tr>
    <tr><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">P</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">25%</td><td>MAE</td><td>1.156 ± 0.001</td><td>0.522 ± 0.001</td><td>0.265 ± 0.002</td><td>0.191 ± 0.000</td><td>0.198 ± 0.001</td><td><u>0.169</u> ± 0.000</td><td>0.159 ± 0.001</td></tr>
    <tr><td>R²</td><td>0.924 ± 0.000</td><td>0.974 ± 0.000</td><td>0.993 ± 0.000</td><td>0.996 ± 0.000</td><td>0.995 ± 0.000</td><td>0.997 ± 0.000</td><td>0.997 ± 0.000</td></tr>
    <tr><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">P</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">50%</td><td>MAE</td><td>1.156 ± 0.001</td><td>0.522 ± 0.001</td><td>0.350 ± 0.001</td><td><u>0.215</u> ± 0.000</td><td>0.232 ± 0.001</td><td>0.218 ± 0.002</td><td>0.212 ± 0.005</td></tr>
    <tr><td>R²</td><td>0.924 ± 0.000</td><td>0.974 ± 0.000</td><td>0.988 ± 0.000</td><td>0.995 ± 0.000</td><td>0.993 ± 0.000</td><td>0.995 ± 0.000</td><td>0.995 ± 0.000</td></tr>
    <tr><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">P</td><td rowspan="2" style="border: 1px solid #ddd; padding: 4px;">75%</td><td>MAE</td><td>1.156 ± 0.000</td><td>0.522 ± 0.001</td><td>1.215 ± 0.027</td><td>0.334 ± 0.001</td><td><u>0.389</u> ± 0.003</td><td>1.040 ± 0.162</td><td>0.417 ± 0.014</td></tr>
    <tr><td>R²</td><td>0.924 ± 0.000</td><td>0.974 ± 0.000</td><td>0.865 ± 0.006</td><td>0.988 ± 0.000</td><td>0.984 ± 0.001</td><td>0.918 ± 0.023</td><td><u>0.985</u> ± 0.001</td></tr>
  </tbody>
</table>
</div>
### Table 2: Quantitative evaluation on the MeteoNet dataset
| Type | Model | CSI 12 | CSI 24 | CSI 32 | CSI Avg | HSS 12 | HSS 24 | HSS 32 | HSS Avg | MSE | MAE | PSNR | SSIM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Unimodal** | PredRNN v2 | 0.3344 | 0.1554 | 0.0350 | 0.1749 | 0.4897 | 0.2647 | 0.0671 | 0.2738 | 9.3032 | 0.7996 | 34.0562 | 0.8501 |
|  | SimVP v2 | 0.3250 | 0.1400 | 0.0195 | 0.1615 | 0.4786 | 0.2415 | 0.0379 | 0.2527 | 9.4314 | 0.8343 | 34.9568 | 0.8499 |
|  | TAU | 0.3444 | 0.1413 | 0.0248 | 0.1702 | 0.5013 | 0.2446 | 0.0482 | 0.2647 | 8.3152 | 0.7907 | 34.8399 | 0.8474 |
|  | Earthformer | 0.3628 | 0.1839 | 0.0573 | 0.2013 | 0.5218 | 0.3072 | 0.1079 | 0.3123 | 7.9094 | 0.7642 | 35.4281 | 0.8572 |
|  | PastNet | 0.3526 | 0.1644 | 0.0268 | 0.1813 | 0.5110 | 0.2789 | 0.0518 | 0.2806 | 8.1057 | 0.9666 | 34.6571 | 0.7681 |
|  | AlphaPre | 0.3722 | 0.1854 | 0.0575 | 0.2050 | 0.5326 | 0.3098 | 0.1083 | 0.3169 | 7.6863 | 1.0026 | 34.7310 | 0.7636 |
|  | NowcastNet | 0.3414 | 0.1595 | 0.0660 | 0.1890 | 0.4990 | 0.2722 | 0.1233 | 0.2982 | 7.8276 | 0.7268 | 35.5236 | 0.8651 |
|  | LMC-Memory | 0.3505 | 0.1659 | 0.0448 | 0.1871 | 0.5092 | 0.2817 | 0.0854 | 0.2921 | 7.8226 | 0.7675 | 35.2575 | 0.8496 |
|  | AFNO | 0.3739 | 0.2018 | 0.0859 | 0.2205 | 0.5343 | 0.3325 | 0.1576 | 0.3415 | 7.5631 | 0.9341 | 34.9673 | 0.7864 |
| **Multimodal** | LightNet | 0.3539 | 0.1649 | 0.0503 | 0.1897 | 0.5128 | 0.2801 | 0.0955 | 0.2961 | 7.6877 | 0.6864 | 35.5315 | 0.8716 |
|  | MM-RNN | 0.3456 | 0.2187 | 0.0519 | 0.2054 | 0.5015 | 0.3539 | 0.0981 | 0.3178 | 9.6015 | 0.7580 | 35.2290 | 0.8737 |
|  | CM-STjointNet | 0.3285 | 0.1553 | 0.0542 | 0.1793 | 0.4832 | 0.2656 | 0.1022 | 0.2837 | 8.8143 | 0.9429 | 34.3279 | 0.7974 |
|  | **Ours** | **0.3744** | **0.2206** | **0.1022** | **0.2324** | **0.5353** | **0.3579** | **0.1848** | **0.3593** | **7.3844** | **0.6603** | **35.8116** | **0.8740** |


## ✍️ Citation

If you find this work or code useful for your research, please consider citing:

```bibtex
@article{qin2026extending,
  title={Extending Precipitation Nowcasting Horizons via Spectral Fusion of Radar Observations and Foundation Model Priors},
  author={Yuze Qin, Qingyong Li, Zhiqing Guo, Wen Wang, Yan Liu, Yangli-ao Geng},
  journal={arXiv preprint arXiv:2603.21768},
  year={2026}
}
```
