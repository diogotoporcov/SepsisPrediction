import math
import random
from enum import Enum, auto
from types import MappingProxyType
from typing import Mapping, List, Tuple, cast

import matplotlib.patches
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


class PlotMode(Enum):
    COMBINED = auto()
    SEPARATE = auto()
    BOTH = auto()


def plot_combined_patient_data(df: pd.DataFrame) -> None:
    is_sepsis_present = 'SepsisLabel' in df.columns
    sepsis_rows = df[df['SepsisLabel'] == 1] if is_sepsis_present else pd.DataFrame()

    plt.figure(figsize=(18, 6))

    if is_sepsis_present and not sepsis_rows.empty:
        idx = sepsis_rows.index
        for i, row in sepsis_rows.iterrows():
            hour = row["ICULOS"]

            start = cast(int, hour - 0.5 if i != idx[0] else hour)
            end = cast(int, hour + 0.5 if i != idx[-1] else hour)

            plt.axvspan(start, end, color='lightcoral', alpha=0.15, zorder=0)

        first_pos_hour = sepsis_rows["ICULOS"].min()
        plt.axvline(
            first_pos_hour,
            color="red",
            linestyle="--",
            linewidth=1,
            label="First positive H",
            zorder=1,
        )

    variables_to_plot = df.columns.tolist()
    variables_to_plot.remove('ICULOS')
    variables_to_plot.remove('Patient_ID')
    if is_sepsis_present:
        variables_to_plot.remove('SepsisLabel')

    for var in variables_to_plot:
        if var not in vital_sign_colors:
            continue
        color = vital_sign_colors[var]
        plt.plot(df['ICULOS'], df[var], label=var, alpha=1, color=color, zorder=2)
        plt.scatter(df['ICULOS'], df[var], color=color, s=3, alpha=1, zorder=3)

    patient_id = df['Patient_ID'].iloc[0] if 'Patient_ID' in df.columns else 'Unknown'
    plt.title(f"Patient {patient_id}")
    plt.xlabel('ICULOS')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=1)
    plt.tight_layout(rect=(0.0, 0, 0.85, 1))
    plt.show()


def plot_patient_individual_data(df: pd.DataFrame) -> None:
    is_sepsis_present = 'SepsisLabel' in df.columns
    sepsis_hours = df[df['SepsisLabel'] == 1]['ICULOS'].tolist() if is_sepsis_present else []

    variables_to_plot = df.columns.tolist()
    variables_to_plot.remove('ICULOS')
    variables_to_plot.remove('Patient_ID')
    if is_sepsis_present:
        variables_to_plot.remove('SepsisLabel')

    num_vars = len(variables_to_plot)
    cols = 3
    rows = (num_vars + cols - 1) // cols
    plt.figure(figsize=(18, rows * 4))

    sepsis_intervals = []
    if is_sepsis_present and sepsis_hours:
        start = None
        prev = None
        for hour in sepsis_hours + [None]:
            if start is None:
                start = hour
            elif hour is None or hour != prev + 1:
                sepsis_intervals.append((start, prev + 1))
                start = hour
            prev = hour

    filtered_vars = [v for v in variables_to_plot if v in vital_sign_colors]

    for i, var in enumerate(filtered_vars, 1):
        plt.subplot(rows, cols, i)

        for start, end in sepsis_intervals:
            plt.axvspan(start, end, color='lightcoral', alpha=0.15, zorder=0)

        plt.plot(df['ICULOS'], df[var], label=var, alpha=1, zorder=2)
        plt.scatter(df['ICULOS'], df[var], s=15, alpha=1, zorder=3)

        plt.title(var)
        plt.xlabel('ICULOS')
        plt.ylabel(var)
        plt.grid(True)

    patient_id = df['Patient_ID'].iloc[0] if 'Patient_ID' in df.columns else 'Unknown'
    plt.suptitle(f"Patient {patient_id}", y=1.02, fontsize=18)
    plt.tight_layout()
    plt.show()


def plot_patient_data(df: pd.DataFrame, method: PlotMode) -> None:
    if method in (PlotMode.BOTH, PlotMode.COMBINED):
        plot_combined_patient_data(df)

    if method in (PlotMode.BOTH, PlotMode.SEPARATE):
        plot_patient_individual_data(df)


def plot_sepsis_patient_counts(n_neg: int, n_pos: int) -> None:
    labels = ['Without Sepsis', 'With Sepsis']
    counts = [n_neg, n_pos]

    bars = plt.bar(labels, counts, color=['skyblue', 'salmon'])

    plt.title('Number of Patients')
    plt.ylabel('Number of Patients')

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom'
        )

    plt.show()


def plot_stats(
    stats: pd.DataFrame,
    title: str,
    start_from_zero: bool = False,
    *,
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    ax = stats.T.plot(kind='bar', legend=False, figsize=figsize)
    plt.title(title)
    plt.ylabel('Value')
    plt.xlabel('Statistic')
    plt.xticks(rotation=45)

    for p in ax.patches:
        p: matplotlib.patches.Rectangle
        height = p.get_height()
        ax.annotate(
            f'{height:.2f}',
            (p.get_x() + p.get_width() / 2, height),
            xytext=(0, 5),
            textcoords='offset points',
            ha='center', va='bottom'
        )

    heights = [p.get_height() for p in ax.patches]
    min_val = 0 if start_from_zero else min(heights)
    max_val = max(heights)
    value_range = max_val - min_val
    pad = value_range * 0.075

    ax.set_ylim(
        bottom=min_val - pad if not start_from_zero else 0,
        top=max_val + pad
    )

    plt.tight_layout()
    plt.show()


_vital_sign_names: List[str] = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets"
]

Color = Tuple[float, float, float]


def _color_distance(c1: Color, c2: Color) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


def _generate_distinct_random_colors(
        n: int,
        existing_colors: List[Color],
        min_distance: float = 0.35,
        max_tries: int = 10000
) -> List[Color]:
    random.seed(42)
    colors: List[Color] = []
    tries = 0
    while len(colors) < n and tries < max_tries:
        candidate: Color = (random.random(), random.random(), random.random())
        if all(_color_distance(candidate, c) >= min_distance for c in existing_colors + colors):
            colors.append(candidate)
        else:
            tries += 1
    if len(colors) < n:
        raise ValueError("Could not generate enough distinct colors. Try lowering min_distance.")
    return colors


def _rgba_to_rgb(color: Tuple[float, float, float, float]) -> Color:
    r, g, b, _ = color
    return r, g, b


_cmap = plt.get_cmap('tab20')
_tab20_colors: List[Color] = [_rgba_to_rgb(_cmap(i)) for i in range(20)]

_remaining = len(_vital_sign_names) - len(_tab20_colors)
_generated_colors = _generate_distinct_random_colors(_remaining, existing_colors=_tab20_colors)

_all_colors: List[Color] = _tab20_colors + _generated_colors

vital_sign_colors: Mapping[str, Color] = MappingProxyType({
    name: color for name, color in zip(_vital_sign_names, _all_colors)
})


def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"]
    )

    plt.title("Prediction Confusion Matrix")
    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def plot_correct_prediction_counts(arr, N):
    counts = np.array([
        (arr == 0).sum(),
        (arr == 1).sum(),
    ])

    data = counts.reshape(1, 2)

    plt.figure(figsize=(6, 3))
    sns.heatmap(
        data,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Incorrect", "Correct"],
        yticklabels=[]
    )

    plt.title(f"Correct Predictions {6-N} Hours Before Sepsis")
    plt.xlabel("Prediction")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_sepsis_prediction_evolution(patient_df, *, patient_id, threshold):
    plt.figure(figsize=(14, 5))
    ax = sns.lineplot(
        data=patient_df,
        x="ICULOS",
        y="PredProb",
        linewidth=1.5,
        label="Raw prob",
    )

    ax.axhline(
        threshold,
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({threshold:.2f})",
    )

    pos_rows = patient_df[patient_df["SepsisLabel"] == 1]

    for i, row in pos_rows.iterrows():
        start = row["ICULOS"] - 0.5 if i != pos_rows.index[0] else row["ICULOS"]
        end = row["ICULOS"] + 0.5 if i != pos_rows.index[-1] else row["ICULOS"]

        ax.axvspan(start, end, color="red", alpha=0.15)

    pos_mask = patient_df["SepsisLabel"] == 1
    if pos_mask.any():
        first_pos_hour = patient_df.loc[pos_mask, "ICULOS"].min()
        ax.axvline(first_pos_hour, color="red", linestyle="--", linewidth=1, label="First positive H")

        prob_at_first_pos = patient_df.loc[patient_df["ICULOS"] == first_pos_hour, "PredProb"].iloc[0]
        ax.scatter(first_pos_hour, prob_at_first_pos, s=20, zorder=6)

    ax.set_title(f"Prediction Evolution – Patient {patient_id}")
    ax.set_ylabel("Predicted Probability")
    ax.set_xlabel("Hour (ICULOS)")
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_nan_percent(df: pd.DataFrame) -> None:
    total = len(df)
    percentages = (df.isna().sum() / total * 100).sort_values(ascending=False)

    plt.figure(figsize=(9, 4))
    ax = percentages.plot(kind='bar', width=0.7)
    plt.ylabel('NaN percent')
    plt.title('Percentage of NaNs per column')
    plt.tight_layout()

    for i, percentage in enumerate(percentages):
        percentage: float
        x = math.floor(percentage)

        if math.isclose(percentage % 1, 0, abs_tol=1e-2):
            label = str(x)
        else:
            label = f".{x:02d}"

        ax.text(i, percentage, label, ha='center', va='bottom', fontsize=8)

    plt.show()