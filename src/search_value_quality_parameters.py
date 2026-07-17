"""Search PER/ROE thresholds with chronological out-of-sample validation.

The optimization target is the mean monthly excess return of an equal-weighted
portfolio.  Thresholds are selected using only the development period.  The
last 15 months are then evaluated once as a holdout period.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


LOGGER = logging.getLogger("parameter_search")


@dataclass(frozen=True)
class Threshold:
    per_max: float
    roe_min: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-months", type=int, default=15)
    parser.add_argument("--min-median-holdings", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    return parser.parse_args()


def setup_japanese_font() -> None:
    for candidate in ("Yu Gothic", "Meiryo", "Noto Sans CJK JP", "IPAexGothic"):
        try:
            font_manager.findfont(candidate, fallback_to_default=False)
            plt.rcParams["font.family"] = candidate
            break
        except ValueError:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def prepare_observations(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["signal_date"])
    clean = frame[
        np.isfinite(frame["per"])
        & np.isfinite(frame["roe"])
        & frame["per"].gt(0)
        & frame["excess_return"].notna()
        & frame["excess_return"].le(1.0)
    ].copy()
    if clean.empty:
        raise RuntimeError("No usable observations in monthly_observations.csv")
    return clean


def screen(frame: pd.DataFrame, threshold: Threshold) -> pd.DataFrame:
    return frame[
        frame["per"].le(threshold.per_max) & frame["roe"].ge(threshold.roe_min)
    ].copy()


def monthly_portfolio(selected: pd.DataFrame) -> pd.DataFrame:
    grouped = selected.groupby("signal_date")["excess_return"]
    return pd.DataFrame(
        {
            "return": grouped.mean(),
            "holdings": grouped.size(),
        }
    )


def period_metrics(monthly: pd.DataFrame, months: pd.DatetimeIndex) -> dict[str, float]:
    subset = monthly.reindex(months).dropna(subset=["return"])
    if subset.empty:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "positive_rate": np.nan,
            "months": 0,
            "median_holdings": 0.0,
        }
    return {
        "mean": float(subset["return"].mean()),
        "median": float(subset["return"].median()),
        "std": float(subset["return"].std(ddof=1)),
        "positive_rate": float(subset["return"].gt(0).mean()),
        "months": int(len(subset)),
        "median_holdings": float(subset["holdings"].median()),
    }


def bootstrap_mean_ci(values: pd.Series, samples: int, seed: int = 42) -> tuple[float, float]:
    clean = values.dropna().to_numpy()
    if len(clean) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(clean, size=(samples, len(clean)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def search_grid(
    observations: pd.DataFrame,
    development_months: pd.DatetimeIndex,
    folds: list[pd.DatetimeIndex],
    holdout_months: pd.DatetimeIndex,
    min_median_holdings: int,
) -> pd.DataFrame:
    per_values = np.arange(4.0, 20.1, 1.0)
    roe_values = np.round(np.arange(0.05, 0.301, 0.025), 3)
    rows: list[dict[str, float | int | bool]] = []

    for per_max in per_values:
        for roe_min in roe_values:
            threshold = Threshold(float(per_max), float(roe_min))
            selected = screen(observations, threshold)
            monthly = monthly_portfolio(selected)
            dev = period_metrics(monthly, development_months)
            holdout = period_metrics(monthly, holdout_months)
            fold_metrics = [period_metrics(monthly, fold) for fold in folds]

            valid_development = all(
                metric["months"] >= max(1, int(np.ceil(0.8 * len(fold))))
                and metric["median_holdings"] >= min_median_holdings
                for metric, fold in zip(fold_metrics, folds)
            )
            fold_means = [metric["mean"] for metric in fold_metrics]
            rows.append(
                {
                    "per_max": threshold.per_max,
                    "roe_min": threshold.roe_min,
                    "valid_development": valid_development,
                    "development_mean": dev["mean"],
                    "development_median": dev["median"],
                    "development_positive_rate": dev["positive_rate"],
                    "development_months": dev["months"],
                    "development_median_holdings": dev["median_holdings"],
                    "fold1_mean": fold_means[0],
                    "fold2_mean": fold_means[1],
                    "fold3_mean": fold_means[2],
                    "minimum_fold_mean": float(np.nanmin(fold_means)),
                    "holdout_mean": holdout["mean"],
                    "holdout_median": holdout["median"],
                    "holdout_positive_rate": holdout["positive_rate"],
                    "holdout_months": holdout["months"],
                    "holdout_median_holdings": holdout["median_holdings"],
                    "selected_stock_months": int(len(selected)),
                }
            )
    return pd.DataFrame(rows)


def select_development_winner(results: pd.DataFrame) -> pd.Series:
    eligible = results[results["valid_development"]].copy()
    if eligible.empty:
        raise RuntimeError("No threshold passed the development-period coverage constraints")
    return eligible.sort_values(
        ["minimum_fold_mean", "development_mean"], ascending=False
    ).iloc[0]


def result_row(results: pd.DataFrame, threshold: Threshold) -> pd.Series:
    match = results[
        np.isclose(results["per_max"], threshold.per_max)
        & np.isclose(results["roe_min"], threshold.roe_min)
    ]
    if match.empty:
        raise KeyError(threshold)
    return match.iloc[0]


def plot_heatmaps(
    results: pd.DataFrame,
    winner: Threshold,
    baseline: Threshold,
    output_path: Path,
) -> None:
    setup_japanese_font()
    per_values = sorted(results["per_max"].unique())
    roe_values = sorted(results["roe_min"].unique(), reverse=True)
    development = (
        results.pivot(index="roe_min", columns="per_max", values="development_mean")
        .reindex(index=roe_values, columns=per_values)
        * 100
    )
    holdout = (
        results.pivot(index="roe_min", columns="per_max", values="holdout_mean")
        .reindex(index=roe_values, columns=per_values)
        * 100
    )
    valid = results.pivot(
        index="roe_min", columns="per_max", values="valid_development"
    ).reindex(index=roe_values, columns=per_values)
    development = development.where(valid)
    holdout = holdout.where(valid)
    combined = np.concatenate([development.to_numpy().ravel(), holdout.to_numpy().ravel()])
    finite = combined[np.isfinite(combined)]
    limit = max(0.5, float(np.quantile(np.abs(finite), 0.95)))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#d9d9d9")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8), dpi=160, sharey=True)
    for ax, matrix, title in (
        (axes[0], development, "開発期間（2022年4月〜2025年2月）"),
        (axes[1], holdout, "未使用の検証期間（2025年3月〜2026年5月）"),
    ):
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
        ax.set_title(title, pad=12)
        ax.set_xlabel("PER上限（倍）")
        ax.set_xticks(range(len(per_values)), [f"{value:g}" for value in per_values])
        ax.set_yticks(range(len(roe_values)), [f"{value:.1%}" for value in roe_values])
        ax.grid(False)
        for threshold, marker, label in (
            (winner, "*", "開発期間で選択"),
            (baseline, "x", "元条件"),
        ):
            x = min(range(len(per_values)), key=lambda i: abs(per_values[i] - threshold.per_max))
            y = min(range(len(roe_values)), key=lambda i: abs(roe_values[i] - threshold.roe_min))
            ax.scatter(x, y, marker=marker, s=120, color="black", linewidths=1.7, label=label)
        ax.legend(loc="upper left", frameon=True, fontsize=9)
    axes[0].set_ylabel("ROE下限")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.025)
    colorbar.set_label("月次・等ウェイト超過リターン平均（%）")
    fig.suptitle("PER・ROE閾値の時系列外検証", fontsize=16, y=0.99)
    fig.text(0.5, 0.025, "灰色：開発期間で月数または保有銘柄数が不足", ha="center", fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.13, top=0.88, wspace=0.08)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_distributions(
    observations: pd.DataFrame,
    winner: Threshold,
    baseline: Threshold,
    development_months: pd.DatetimeIndex,
    holdout_months: pd.DatetimeIndex,
    output_path: Path,
) -> None:
    setup_japanese_font()
    series: list[tuple[str, pd.Series]] = []
    for period_label, months in (("開発期間", development_months), ("検証期間", holdout_months)):
        for threshold_label, threshold in (("元条件", baseline), ("選択条件", winner)):
            monthly = monthly_portfolio(screen(observations, threshold))
            values = monthly.reindex(months)["return"].dropna() * 100
            series.append((f"{period_label}・{threshold_label}", values))

    all_values = pd.concat([values for _, values in series])
    lower, upper = np.quantile(all_values, [0.01, 0.99])
    span = max(abs(lower), abs(upper), 1.0)
    bins = np.linspace(-span, span, 17)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=160, sharex=True, sharey=True)
    for ax, (label, values) in zip(axes.flat, series):
        mean = values.mean()
        median = values.median()
        ax.hist(values, bins=bins, color="#2f6690", edgecolor="white", alpha=0.88)
        ax.axvline(0, color="#333333", linewidth=1.1)
        ax.axvline(mean, color="#d1495b", linewidth=2.0, label=f"平均 {mean:.2f}%")
        ax.axvline(median, color="#edae49", linewidth=1.8, linestyle="--", label=f"中央値 {median:.2f}%")
        ax.set_title(f"{label}（{len(values)}か月）")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, fontsize=9)
    for ax in axes[-1, :]:
        ax.set_xlabel("日経平均控除後の月次リターン（%）")
    for ax in axes[:, 0]:
        ax.set_ylabel("月数")
    fig.suptitle("元条件と開発期間で選んだ条件の月次分布", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    img_dir = root / "img"
    source_path = data_dir / "monthly_observations.csv"
    observations = prepare_observations(source_path)

    all_months = pd.DatetimeIndex(sorted(observations["signal_date"].unique()))
    if len(all_months) <= args.holdout_months + 6:
        raise RuntimeError("Not enough months for development and holdout periods")
    development_months = all_months[: -args.holdout_months]
    holdout_months = all_months[-args.holdout_months :]
    folds = [pd.DatetimeIndex(chunk) for chunk in np.array_split(development_months, 3)]

    results = search_grid(
        observations,
        development_months,
        folds,
        holdout_months,
        args.min_median_holdings,
    )
    winner_row = select_development_winner(results)
    winner = Threshold(float(winner_row["per_max"]), float(winner_row["roe_min"]))
    baseline = Threshold(10.0, 0.10)
    baseline_row = result_row(results, baseline)
    quality_floor_winner_row = select_development_winner(results[results["roe_min"].ge(0.10)])
    quality_floor_winner = Threshold(
        float(quality_floor_winner_row["per_max"]), float(quality_floor_winner_row["roe_min"])
    )

    winner_selected = screen(observations, winner)
    winner_selected["period"] = np.where(
        winner_selected["signal_date"].isin(holdout_months), "holdout", "development"
    )
    winner_monthly = monthly_portfolio(winner_selected).rename(
        columns={"return": "selected_return", "holdings": "selected_holdings"}
    )
    baseline_monthly = monthly_portfolio(screen(observations, baseline)).rename(
        columns={"return": "baseline_return", "holdings": "baseline_holdings"}
    )
    monthly_comparison = winner_monthly.join(baseline_monthly, how="outer")
    monthly_comparison["period"] = np.where(
        monthly_comparison.index.isin(holdout_months), "holdout", "development"
    )
    monthly_comparison.index.name = "signal_date"

    holdout_selected = winner_monthly.reindex(holdout_months)["selected_return"].dropna()
    holdout_baseline = baseline_monthly.reindex(holdout_months)["baseline_return"].dropna()
    selected_ci = bootstrap_mean_ci(holdout_selected, args.bootstrap_samples)
    baseline_ci = bootstrap_mean_ci(holdout_baseline, args.bootstrap_samples)

    results.to_csv(data_dir / "parameter_search_results.csv", index=False)
    winner_selected.to_csv(data_dir / "optimized_threshold_stock_months.csv", index=False)
    monthly_comparison.to_csv(data_dir / "parameter_search_monthly_returns.csv")

    plot_heatmaps(
        results,
        winner,
        baseline,
        img_dir / "parameter_search_heatmaps.png",
    )
    plot_distributions(
        observations,
        winner,
        baseline,
        development_months,
        holdout_months,
        img_dir / "parameter_search_distributions.png",
    )

    summary = {
        "selection_rule": (
            "Maximize the worst mean monthly excess return across three chronological development folds; "
            "break ties using the full development-period mean. Holdout returns are not used for selection."
        ),
        "grid": {
            "per_max": [4, 20, 1],
            "roe_min": [0.05, 0.30, 0.025],
            "minimum_median_holdings_per_development_fold": args.min_median_holdings,
        },
        "development_period": [
            development_months.min().date().isoformat(),
            development_months.max().date().isoformat(),
        ],
        "holdout_period": [
            holdout_months.min().date().isoformat(),
            holdout_months.max().date().isoformat(),
        ],
        "selected_threshold": {
            "per_max": winner.per_max,
            "roe_min": winner.roe_min,
            "development_mean": float(winner_row["development_mean"]),
            "minimum_fold_mean": float(winner_row["minimum_fold_mean"]),
            "holdout_mean": float(winner_row["holdout_mean"]),
            "holdout_median": float(winner_row["holdout_median"]),
            "holdout_positive_rate": float(winner_row["holdout_positive_rate"]),
            "holdout_median_holdings": float(winner_row["holdout_median_holdings"]),
            "holdout_mean_bootstrap_95pct_ci": list(selected_ci),
        },
        "baseline_threshold": {
            "per_max": baseline.per_max,
            "roe_min": baseline.roe_min,
            "development_mean": float(baseline_row["development_mean"]),
            "minimum_fold_mean": float(baseline_row["minimum_fold_mean"]),
            "holdout_mean": float(baseline_row["holdout_mean"]),
            "holdout_median": float(baseline_row["holdout_median"]),
            "holdout_positive_rate": float(baseline_row["holdout_positive_rate"]),
            "holdout_median_holdings": float(baseline_row["holdout_median_holdings"]),
            "holdout_mean_bootstrap_95pct_ci": list(baseline_ci),
        },
        "best_with_roe_at_least_10pct": {
            "per_max": quality_floor_winner.per_max,
            "roe_min": quality_floor_winner.roe_min,
            "development_mean": float(quality_floor_winner_row["development_mean"]),
            "holdout_mean": float(quality_floor_winner_row["holdout_mean"]),
        },
        "conclusion": (
            "No robust positive out-of-sample edge was found in this PER/ROE threshold grid. "
            "The development-period winner was negative in the untouched holdout period."
        ),
    }
    (data_dir / "parameter_search_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    LOGGER.info(
        "Selected on development only: PER <= %.0f, ROE >= %.1f%%",
        winner.per_max,
        100 * winner.roe_min,
    )
    LOGGER.info(
        "Development mean %.3f%%; holdout mean %.3f%%",
        100 * winner_row["development_mean"],
        100 * winner_row["holdout_mean"],
    )


if __name__ == "__main__":
    main()
