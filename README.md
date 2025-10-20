# Anomaly Detection – Water Treatment Plant Project

**Author:** Alberto J. Maldonado Rodríguez  
**Email:** apimaldo@gmail.com  
**Version:** 1.0  
**Environment:** Python 3.13 + Virtual Environment (.venv)

---

##  Project Overview

This project performs **anomaly detection** on multivariate time series sensor data from a **water treatment plant**.  
It combines **dimensionality reduction (PCA)** with several unsupervised **anomaly detection algorithms** to identify unusual readings in sensor data that may indicate system malfunctions or outliers.

The workflow integrates:
- Exploratory Data Analysis (EDA)
- Preprocessing and normalization
- PCA (Principal Component Analysis)
- Multiple anomaly detection techniques:
  - Interquartile Range (IQR)
  - K-Means clustering
  - Isolation Forest
  - DBSCAN clustering
- Visualization and export of detected anomalies

---

## Methods Implemented

| Method | Description |
|--------|--------------|
| **PCA** | Reduces the dimensionality of high-dimensional sensor data to 2 components for visualization and pattern discovery. |
| **IQR (Interquartile Range)** | Detects outliers based on the spread of PCA-reduced features. |
| **K-Means** | Clusters data into groups and flags points that deviate from cluster centers. |
| **Isolation Forest** | Detects anomalies by isolating points that behave differently from the majority. |
| **DBSCAN** | Density-based clustering that identifies sparse regions as anomalies. |

---

## Project Structure

Anomaly_Detection_WaterPlant/
├── .venv/ # Virtual environment (ignored in Git)
├── data/ # Datasets
│ ├── sensor.csv
│ └── water_treatment_plant.csv
├── outputs/ # Generated anomaly plots and summary CSVs
├── anomaly_detection.py # Core class implementing all algorithms
├── main.py # CLI entry point
├── requirements.txt # Dependencies list
└── .gitignore # Files/folders excluded from Git



---

## Dependencies

Install the required libraries via:

```bash
pip install -r requirements.txt



How to Run the Project
Activate your environment
cd Anomaly_Detection_WaterPlant
source .venv/bin/activate

Run the anomaly detection pipeline
Example 1: Basic dataset
python3 main.py -f "data/sensor.csv"

Example 2: Advanced dataset
python3 main.py -f "data/water_treatment_plant.csv"

 Outputs and Visualization

After running, all plots and results are automatically saved in the outputs/ folder:

Output File	Description
pca_scatter_IQR.png	PCA + IQR anomaly visualization
pca_scatter_KMeans.png	PCA + K-Means clustering anomalies
pca_scatter_IForest.png	PCA + Isolation Forest anomalies
pca_scatter_DBSCAN.png	PCA + DBSCAN anomalies
report_summary.csv	Summary with anomaly counts per method

Example visualization (saved automatically):

plt.savefig('outputs/pca_scatter_DBSCAN.png', dpi=300, bbox_inches='tight')

 Example CLI Options

You can customize thresholds and methods:

python3 main.py -f "data/sensor.csv" --pc 3 --iqr-k 1.8 --kmeans-k 4 --if-cont 0.07 --db-eps 0.6 --db-min 10

Educational Purpose

This project is part of the Hands-On Engineering and Feature Engineering Series, developed to help students and professionals understand the integration of dimensionality reduction and unsupervised anomaly detection in real-world sensor systems.

License

This repository is open for educational and research use.
Attribution to the author is appreciated if adapted or extended.

Contact

Alberto J. Maldonado Rodríguez
Agronomist • Data Scientist in Development
apimaldo@gmail.com
