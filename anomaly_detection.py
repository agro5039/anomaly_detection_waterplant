# Purpose: Anomaly detection on multivariate time series (water treatment plant) using PCA + IQR, KMeans, IsolationForest, DBSCAN
# Author: Alberto J. Maldonado Rodríguez (apimaldo@gmail.com)
# Version: 1.0

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Tuple, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest

sns.set_context("talk")

class AnomalyPipeline:
    def __init__(self, csv_path: str, ts_column: str = "timestamp", normalize: bool = True, n_components: int = 2):
        """
        csv_path: path to CSV with a timestamp column (default 'timestamp') and sensor columns
        """
        self.csv_path = csv_path
        self.ts_column = ts_column
        self.normalize = normalize
        self.n_components = n_components

        self.df_raw = None
        self.df = None              # numeric features only (preprocessed)
        self.scaler = None
        self.pca = None
        self.X = None               # scaled features
        self.Z = None               # PCA components (2D)
        self.models = {}
        self.labels_ = {}           # method -> anomaly labels (1 normal, -1 anomaly)
        self.report = {}            # method -> summary dict

    # ---------- Loading & EDA ----------
    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"File not found: {self.csv_path}")

        # Try parse_dates
        try:
            df = pd.read_csv(self.csv_path, parse_dates=[self.ts_column])
        except Exception:
            # fallback: read without parsing, then try to parse
            df = pd.read_csv(self.csv_path)
            if self.ts_column in df.columns:
                df[self.ts_column] = pd.to_datetime(df[self.ts_column], errors="coerce")

        # Set index if timestamp exists
        if self.ts_column in df.columns:
            df = df.set_index(self.ts_column)

        self.df_raw = df.sort_index()
        return self.df_raw

    def quick_eda(self, outdir: str = "outputs", prefix: str = ""):
        os.makedirs(outdir, exist_ok=True)
        # Basic info
        desc = self.df_raw.describe(include="all")
        desc.to_csv(os.path.join(outdir, f"{prefix}describe.csv"))

        # Missing values
        mv = self.df_raw.isna().sum().sort_values(ascending=False)
        mv.to_csv(os.path.join(outdir, f"{prefix}missing_values.csv"))

        # Simple correlation heatmap (numeric only)
        num = self.df_raw.select_dtypes(include=np.number)
        if num.shape[1] >= 2:
            plt.figure(figsize=(10, 8))
            sns.heatmap(num.corr(), annot=False)
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{prefix}corr_heatmap.png"), dpi=150)
            plt.close()

    # ---------- Preprocess ----------
    def preprocess(self, resample_rule: str = None, fill_method: str = "ffill"):
        """
        - Keep numeric columns
        - Optional resampling (e.g., '5min','H','D' if index is datetime)
        - Fill missing values
        """
        df = self.df_raw.copy()

        # Optional resample (only if datetime index)
        if isinstance(df.index, pd.DatetimeIndex) and resample_rule:
            df = df.resample(resample_rule).mean()

        # keep numeric
        df_num = df.select_dtypes(include=np.number)

        # handle missing
        if fill_method == "ffill":
            df_num = df_num.ffill().bfill()
        elif fill_method == "bfill":
            df_num = df_num.bfill().ffill()
        else:
            df_num = df_num.fillna(df_num.median())

        # drop columns with zero variance
        nunique = df_num.nunique()
        df_num = df_num.loc[:, nunique > 1]

        self.df = df_num
        return self.df

    # ---------- PCA ----------
    def fit_pca(self):
        X = self.df.values
        if self.normalize:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        self.X = X

        self.pca = PCA(n_components=self.n_components, random_state=42)
        self.Z = self.pca.fit_transform(self.X)
        return self.Z

    # ---------- Anomaly Methods ----------
    def anomalies_iqr(self, k: float = 1.5) -> np.ndarray:
        """
        IQR-based outlier detection on PCA space (per component; union of outliers).
        Returns labels: 1 normal, -1 anomaly
        """
        z = self.Z
        outlier_mask = np.zeros(z.shape[0], dtype=bool)
        for j in range(z.shape[1]):
            q1, q3 = np.percentile(z[:, j], [25, 75])
            iqr = q3 - q1
            lower, upper = q1 - k * iqr, q3 + k * iqr
            outlier_mask |= (z[:, j] < lower) | (z[:, j] > upper)
        labels = np.where(outlier_mask, -1, 1)
        self.labels_["IQR"] = labels
        self.report["IQR"] = {"anomaly_rate": float((labels == -1).mean())}
        return labels

    def anomalies_kmeans(self, k: int = 3, std_thresh: float = 2.5) -> np.ndarray:
        """
        KMeans on PCA space. Mark points far from cluster centers as anomalies (z-score on distance).
        """
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(self.Z)
        d = np.linalg.norm(self.Z - km.cluster_centers_[km.labels_], axis=1)
        z = (d - d.mean()) / (d.std() + 1e-8)
        labels = np.where(z > std_thresh, -1, 1)
        self.models["KMeans"] = km
        self.labels_["KMeans"] = labels
        self.report["KMeans"] = {"anomaly_rate": float((labels == -1).mean()),
                                 "k": k, "std_thresh": std_thresh}
        return labels

    def anomalies_isoforest(self, contamination: float = 0.05) -> np.ndarray:
        iso = IsolationForest(random_state=42, contamination=contamination)
        labels = iso.fit_predict(self.Z)  # 1 normal, -1 anomaly
        self.models["IForest"] = iso
        self.labels_["IForest"] = labels
        self.report["IForest"] = {"anomaly_rate": float((labels == -1).mean()),
                                  "contamination": contamination}
        return labels

    def anomalies_dbscan(self, eps: float = 0.5, min_samples: int = 10) -> np.ndarray:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        db_labels = db.fit_predict(self.Z)
        # DBSCAN: -1 is noise (anomaly)
        labels = np.where(db_labels == -1, -1, 1)
        self.models["DBSCAN"] = db
        self.labels_["DBSCAN"] = labels
        self.report["DBSCAN"] = {"anomaly_rate": float((labels == -1).mean()),
                                 "eps": eps, "min_samples": min_samples}
        return labels

    # ---------- Plotting ----------
    def plot_pca_scatter(self, method: str, outdir: str = "outputs", title: str = None):
        os.makedirs(outdir, exist_ok=True)
        if method not in self.labels_:
            print(f"⚠️ No labels for method '{method}'. Run the method first.")
            return
        labels = self.labels_[method]
        fig, ax = plt.subplots(figsize=(9, 7))
        normal = self.Z[labels == 1]
        anom = self.Z[labels == -1]

        ax.scatter(normal[:, 0], normal[:, 1], alpha=0.6, label="Normal")
        ax.scatter(anom[:, 0], anom[:, 1], alpha=0.9, marker="x", label="Anomaly")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title or f"PCA scatter — {method}")
        ax.legend()
        fig.tight_layout()

        # Save and close (no ventana bloqueante)
        fname = f"pca_scatter_{method}.png".replace(" ", "_")
        path = os.path.join(outdir, fname)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"💾 Saved: {path}")

    def export_report(self, outdir: str = "outputs", prefix: str = "report_") -> str:
        os.makedirs(outdir, exist_ok=True)
        summary = pd.DataFrame(self.report).T
        path = os.path.join(outdir, f"{prefix}summary.csv")
        summary.to_csv(path)
        print(f"💾 Saved: {path}")
        return path
