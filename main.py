# Purpose: CLI for anomaly detection with PCA + IQR, KMeans, IsolationForest, DBSCAN
# Author: Alberto J. Maldonado Rodríguez
# Version: 1.0

import argparse
from anomaly_detection import AnomalyPipeline

def main():
    parser = argparse.ArgumentParser(description="Anomaly detection on water treatment plant sensor data.")
    parser.add_argument("-f", "--file", required=True, help="Path to CSV file (e.g., data/sensor.csv)")
    parser.add_argument("-t", "--tscol", default="timestamp", help="Timestamp column name (default: timestamp)")
    parser.add_argument("--resample", default=None, help="Optional Pandas resample rule (e.g., '5min','H','D')")
    parser.add_argument("--no-normalize", action="store_true", help="Disable normalization (default: normalize)")

    # PCA
    parser.add_argument("--pc", type=int, default=2, help="PCA components (default=2)")

    # Methods params
    parser.add_argument("--iqr-k", type=float, default=1.5, help="IQR multiplier (default=1.5)")
    parser.add_argument("--kmeans-k", type=int, default=3, help="KMeans k (default=3)")
    parser.add_argument("--kmeans-z", type=float, default=2.5, help="KMeans z-threshold on distance (default=2.5)")
    parser.add_argument("--if-cont", type=float, default=0.05, help="IsolationForest contamination (default=0.05)")
    parser.add_argument("--db-eps", type=float, default=0.5, help="DBSCAN eps (default=0.5)")
    parser.add_argument("--db-min", type=int, default=10, help="DBSCAN min_samples (default=10)")

    args = parser.parse_args()

    pipe = AnomalyPipeline(
        csv_path=args.file,
        ts_column=args.tscol,
        normalize=not args.no_normalize,
        n_components=args.pc
    )

    print(f"\n📂 Loading: {args.file}")
    df_raw = pipe.load()
    print(f"✅ Loaded shape: {df_raw.shape}")

    print("🔎 Running quick EDA...")
    pipe.quick_eda(outdir="outputs")

    print("🧽 Preprocessing...")
    pipe.preprocess(resample_rule=args.resample, fill_method="ffill")

    print(f"📉 PCA (n_components={args.pc})...")
    pipe.fit_pca()

    # --- Run all methods ---
    print("🧪 IQR...")
    pipe.anomalies_iqr(k=args.iqr_k)
    pipe.plot_pca_scatter("IQR", outdir="outputs")

    print("🧪 KMeans...")
    pipe.anomalies_kmeans(k=args.kmeans_k, std_thresh=args.kmeans_z)
    pipe.plot_pca_scatter("KMeans", outdir="outputs")

    print("🧪 Isolation Forest...")
    pipe.anomalies_isoforest(contamination=args.if_cont)
    pipe.plot_pca_scatter("IForest", outdir="outputs")

    print("🧪 DBSCAN...")
    pipe.anomalies_dbscan(eps=args.db_eps, min_samples=args.db_min)
    pipe.plot_pca_scatter("DBSCAN", outdir="outputs")

    # Export summary
    pipe.export_report(outdir="outputs")

    print("\n🎯 Done. Check the 'outputs/' folder for PNG plots and the summary CSV.\n")

if __name__ == "__main__":
    print("\n=== Anomaly Detection – Water Treatment Plant ===")
    main()
