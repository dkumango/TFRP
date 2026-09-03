###############################################################################
# Execution Environment: Python 3.12 (.venv_312)
###############################################################################
"""Experimental Results Summary and Statistical Aggregation.

This script aggregates benchmark result CSVs across multiple model architectures
(ResNet50, VGG16, InceptionV3, MobileNetV2) and attribution rules (z+, epsilon, alpha-beta, composite).
It computes top-trimmed means (omitting the top 5% outliers) to summarize target metrics such as
Structure-Tolerance, Total Leakage, Sparsity, Heatmap Entropy, and Execution Time.

Outputs:
- Exported summary CSVs in `exp_result/`.
"""

import os
import pandas as pd
import numpy as np


def top_trimmed_mean(series: pd.Series, percentage: float = 0.05) -> float:
    """Compute the mean of a series excluding the top n% outlier values.

    Args:
        series (pd.Series): Numerical data series.
        percentage (float, optional): Fraction of top values to trim. Defaults to 0.05.

    Returns:
        float: Trimmed mean value.
    """
    threshold = series.quantile(1.0 - percentage)
    trimmed_series = series[series <= threshold]
    return float(trimmed_series.mean())


def get_summary_table(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Aggregate target column metrics by Global_Rule using top-trimmed mean.

    Args:
        df (pd.DataFrame): Benchmark results DataFrame containing 'Global_Rule' and target column.
        target_column (str): Name of the metric column to summarize.

    Returns:
        pd.DataFrame: Pivoted summary table ordered by rule.
    """
    summary_table = df.groupby('Global_Rule')[target_column].apply(
        lambda x: top_trimmed_mean(x, percentage=0.05)
    ).reset_index()

    summary_table.columns = ['Global_Rule', 'row']
    pivot_table = summary_table.set_index('Global_Rule').T
    desired_order = ['z_plus', 'epsilon', 'alpha_beta', 'composite']
    final_table = pivot_table[desired_order]

    return final_table


def main():
    """Aggregate benchmark results across models for a selected target metric."""
    model_list = [
        "resnet50", "vgg16", "inception_v3", "mobilenet_v2"
    ]

    target_column = 'Structure-Tolerance_Add'
    # Other candidate columns:
    # 'Structure-Tolerance_Gap', 'Structure-Tolerance_Padding',
    # 'Execution_Time', 'Heatmap_Entropy', 'Sparsity', 'Total_Leakage_(%)'

    Total_summary = None
    for model_name in model_list:
        TFRP_file = f"exp_result/TFRP_500/tfrp_{model_name}_500.csv"
        LRP_file = f"exp_result/LRP_500/lrp_{model_name}_500.csv"

        if not (os.path.exists(TFRP_file) and os.path.exists(LRP_file)):
            print(f"Skipping {model_name}: Result files not found.")
            continue

        TFRP_df = pd.read_csv(TFRP_file)
        LRP_df = pd.read_csv(LRP_file)

        TFRP_summary = get_summary_table(TFRP_df, target_column)
        LRP_summary = get_summary_table(LRP_df, target_column)
        Total_summary = pd.concat([Total_summary, LRP_summary, TFRP_summary])

    if Total_summary is not None:
        method_column = ['LRP', 'TFRP'] * len(model_list)
        model_column = []
        for m in model_list:
            model_column.extend([m] * 2)

        Total_summary.insert(0, 'Method', method_column[:len(Total_summary)])
        Total_summary.insert(0, 'Model', model_column[:len(Total_summary)])
        Total_summary.index = [target_column] * len(Total_summary)

        out_csv = f"exp_result/{target_column}_summary_table.csv"
        os.makedirs("exp_result", exist_ok=True)
        Total_summary.to_csv(out_csv, index=True)
        print(f"Saved aggregated summary table to: '{out_csv}'")


if __name__ == '__main__':
    main()
