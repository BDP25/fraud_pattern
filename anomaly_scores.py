import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from glob import glob

sns.set_theme(style="whitegrid")


def load_scores_from_json(path_pattern):
    all_scores = {}
    for filepath in glob(path_pattern):
        name = os.path.splitext(os.path.basename(filepath))[0]
        with open(filepath) as f:
            scores = json.load(f)
        all_scores[name] = scores
    return all_scores


def compute_z_scores(score_dict):
    df = pd.DataFrame(score_dict).T 
    df = df.apply(pd.to_numeric, errors='coerce')
    z_df = (df - df.mean()) / df.std(ddof=0)
    return df, z_df


def plot_histograms(z_df, title_suffix=""):
    for col in z_df.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(z_df[col], bins=40, kde=True)
        plt.axvline(2, color="red", linestyle="--", label="z > 2")
        plt.axvline(-2, color="red", linestyle="--")
        plt.title(f"Z-Score Verteilung: {col} {title_suffix}")
        plt.xlabel("Z-Score")
        plt.ylabel("Anzahl der Leistungserbringer")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./results/hist_{col}.png")
        plt.close()


def plot_scatter(df, z_df):
    if df.shape[1] >= 2:
        cols = df.columns[:2]
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df[cols[0]], y=df[cols[1]])
        high_z = (z_df[cols[0]].abs() > 2) | (z_df[cols[1]].abs() > 2)
        sns.scatterplot(x=df.loc[high_z, cols[0]], y=df.loc[high_z, cols[1]],
                        color="red", label="auffällig", marker="X", s=80)
        plt.title("Anomalie-Streuung (Top 2 Metriken)")
        plt.xlabel(cols[0])
        plt.ylabel(cols[1])
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./results/scatter_{cols[0]}_{cols[1]}.png")
        plt.close()


def export_top_doctors(z_df, output_file="./results/top_outliers.csv", z_thresh=2.0):
    outliers = z_df[(z_df.abs() > z_thresh).any(axis=1)]
    outliers = outliers.sort_values(by=z_df.columns.tolist(), ascending=False)
    outliers.to_csv(output_file)
    print(f"{len(outliers)} highscore anomalies saved as {output_file}")


def export_top_doctors_tabular(z_df, output_file="./results/top_outliers_tabular.csv", z_thresh=2.0):
    """
    Export top outliers in tabular format with columns: doctor, transformer, anomaly_score
    """
    outlier_records = []
    
    # Find all outliers (doctors with any score > threshold)
    outlier_mask = (z_df.abs() > z_thresh).any(axis=1)
    outlier_doctors = z_df[outlier_mask].index
    
    # Convert to tabular format
    for doctor in outlier_doctors:
        for transformer in z_df.columns:
            score = z_df.loc[doctor, transformer]
            # Only include scores that are actually outliers
            if abs(score) > z_thresh:
                outlier_records.append({
                    'doctor': doctor,
                    'transformer': transformer,
                    'anomaly_score': score
                })
    
    # Convert to DataFrame and sort by anomaly score (descending by absolute value)
    outlier_df = pd.DataFrame(outlier_records)
    outlier_df['abs_score'] = outlier_df['anomaly_score'].abs()
    outlier_df = outlier_df.sort_values('abs_score', ascending=False).drop('abs_score', axis=1)
    
    # Save to CSV
    outlier_df.to_csv(output_file, index=False)
    print(f"{len(outlier_df)} outlier records (doctor-transformer pairs) saved to {output_file}")
    
    return outlier_df


def export_all_scores_tabular(z_df, output_file="./results/all_scores_tabular.csv"):
    """
    Export all scores in tabular format with columns: doctor, transformer, anomaly_score
    """
    all_records = []
    
    # Convert entire matrix to tabular format
    for doctor in z_df.index:
        for transformer in z_df.columns:
            score = z_df.loc[doctor, transformer]
            # Skip NaN values
            if not pd.isna(score):
                all_records.append({
                    'doctor': doctor,
                    'transformer': transformer,
                    'anomaly_score': score
                })
    
    # Convert to DataFrame and sort by anomaly score (descending by absolute value)
    all_df = pd.DataFrame(all_records)
    all_df['abs_score'] = all_df['anomaly_score'].abs()
    all_df = all_df.sort_values('abs_score', ascending=False).drop('abs_score', axis=1)
    
    # Save to CSV
    all_df.to_csv(output_file, index=False)
    print(f"{len(all_df)} total records (doctor-transformer pairs) saved to {output_file}")
    
    return all_df


def main():
    os.makedirs("./results", exist_ok=True)

    print("loading transformer anomaly-scores...")
    transformer_scores = load_scores_from_json("./results/transformer_scores_*.json")
    merged_scores = {}
    for name, score_dict in transformer_scores.items():
        for doc_id, score in score_dict.items():
            merged_scores.setdefault(doc_id, {})[name] = score

    print("loading available stat scores...")
    if os.path.exists("./results/statistical_scores.json"):
        with open("./results/statistical_scores.json") as f:
            stats_scores = json.load(f)
        for doc_id, score_dict in stats_scores.items():
            for k, v in score_dict.items():
                merged_scores.setdefault(doc_id, {})[f"stat_{k}"] = v

    print("computing z-scores...")
    df, z_df = compute_z_scores(merged_scores)

    print("saving distributions and diagrams...")
    plot_histograms(z_df)
    plot_scatter(df, z_df)

    print("exporting outlier lists...")
    # Export top outliers in tabular format
    export_top_doctors_tabular(z_df)
    
    # Export all scores in tabular format
    export_all_scores_tabular(z_df)
    
    # Export top outliers in matrix format
    export_top_doctors(z_df)  # AEK: Original approach to be used for matrix output


if __name__ == "__main__":
    main()
