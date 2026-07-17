"""PER <= 10 and ROE >= 10% Japanese-stock return histogram.

This script follows the workflow in the referenced Qiita article:

1. Use a TOPIX 500 stock universe.
2. Obtain adjusted closing prices and annual fundamentals with yfinance.
3. At each month-end, screen on the latest fiscal data then available.
4. Hold for one month and subtract the Nikkei 225 return.
5. Save the selected observations and their return histogram.

The article treats a fiscal-period end as the availability date.  This script
does the same by default so that the procedure is comparable.  Use
``--reporting-lag-days`` to impose a more conservative publication lag.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import yfinance as yf


LOGGER = logging.getLogger("value_quality")
BENCHMARK = "^N225"
DEFAULT_UNIVERSE = "topix500_20210430.csv"


@dataclass(frozen=True)
class Config:
    start: str
    end: str
    per_max: float
    roe_min: float
    reporting_lag_days: int
    workers: int
    refresh: bool
    limit: int | None


def parse_args() -> argparse.Namespace:
    today = pd.Timestamp.today().normalize()
    latest_complete_month_start = today.to_period("M").start_time
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2022-04-01", help="Signal-period start (YYYY-MM-DD)")
    parser.add_argument(
        "--end",
        default=latest_complete_month_start.strftime("%Y-%m-%d"),
        help="Price download end, exclusive (default: start of current month)",
    )
    parser.add_argument("--per-max", type=float, default=10.0)
    parser.add_argument("--roe-min", type=float, default=0.10)
    parser.add_argument(
        "--reporting-lag-days",
        type=int,
        default=0,
        help="Days added to fiscal period-end before data may be used; 0 reproduces the article",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--refresh", action="store_true", help="Ignore cached yfinance CSV files")
    parser.add_argument("--limit", type=int, default=None, help="Restrict the universe for a quick test")
    return parser.parse_args()


def first_row(frame: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    for name in candidates:
        if name in frame.index:
            return pd.to_numeric(frame.loc[name], errors="coerce")
    return pd.Series(dtype="float64")


def load_universe(path: Path, limit: int | None = None) -> list[str]:
    raw = pd.read_csv(path, header=None, names=["code"], dtype=str)
    codes = (
        raw["code"]
        .str.extract(r"(\d{4})", expand=False)
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    tickers = [f"{code}.T" for code in codes]
    if limit is not None:
        tickers = tickers[:limit]
    if not tickers:
        raise ValueError(f"No four-digit security codes found in {path}")
    return tickers


def download_prices(
    tickers: list[str], start: str, end: str, cache_path: Path, refresh: bool
) -> pd.DataFrame:
    requested = list(dict.fromkeys([*tickers, BENCHMARK]))
    if cache_path.exists() and not refresh:
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        cached.index = pd.DatetimeIndex(cached.index).tz_localize(None)
        if set(requested).issubset(cached.columns):
            LOGGER.info("Using cached prices: %s", cache_path)
            return cached[requested]

    LOGGER.info("Downloading adjusted close prices for %d symbols", len(requested))
    pieces: list[pd.DataFrame] = []
    chunk_size = 80
    for offset in range(0, len(requested), chunk_size):
        chunk = requested[offset : offset + chunk_size]
        frame = yf.download(
            chunk,
            start=start,
            end=end,
            auto_adjust=True,
            actions=False,
            progress=False,
            group_by="column",
            threads=True,
            timeout=30,
        )
        if frame.empty:
            LOGGER.warning("No price data for chunk beginning %s", chunk[0])
            continue
        if isinstance(frame.columns, pd.MultiIndex):
            close = frame["Close"].copy()
        else:
            close = frame[["Close"]].rename(columns={"Close": chunk[0]})
        pieces.append(close)
        LOGGER.info("Price chunk %d/%d complete", min(offset + chunk_size, len(requested)), len(requested))

    if not pieces:
        raise RuntimeError("yfinance returned no price data")
    prices = pd.concat(pieces, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    prices = prices.sort_index().sort_index(axis=1)
    prices.to_csv(cache_path, index_label="date")
    return prices


def fetch_one_fundamental(ticker: str, attempts: int = 3) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            security = yf.Ticker(ticker)
            income = security.get_income_stmt(freq="yearly")
            balance = security.get_balance_sheet(freq="yearly")
            if income.empty or balance.empty:
                raise ValueError("empty annual statement")

            net_income = first_row(
                income,
                ("NetIncome", "NetIncomeCommonStockholders", "NetIncomeContinuousOperations"),
            )
            shares = first_row(income, ("BasicAverageShares", "DilutedAverageShares"))
            reported_eps = first_row(income, ("BasicEPS", "DilutedEPS"))
            equity = first_row(
                balance,
                ("StockholdersEquity", "CommonStockEquity", "TotalEquityGrossMinorityInterest"),
            )

            dates = net_income.index.union(equity.index).union(shares.index).union(reported_eps.index)
            result = pd.DataFrame(index=pd.DatetimeIndex(dates).tz_localize(None).sort_values())
            result["net_income"] = net_income.reindex(result.index)
            result["stockholders_equity"] = equity.reindex(result.index)
            result["basic_average_shares"] = shares.reindex(result.index)
            calculated_eps = result["net_income"] / result["basic_average_shares"]
            result["eps"] = reported_eps.reindex(result.index).fillna(calculated_eps)
            result["roe"] = result["net_income"] / result["stockholders_equity"]
            result["ticker"] = ticker
            result.index.name = "fiscal_date"
            return result.reset_index()
        except Exception as exc:  # Yahoo occasionally returns transient errors.
            last_error = exc
            if attempt < attempts:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"{ticker}: {last_error}")


def download_fundamentals(
    tickers: list[str], cache_path: Path, workers: int, refresh: bool
) -> tuple[pd.DataFrame, list[str]]:
    if cache_path.exists() and not refresh:
        cached = pd.read_csv(cache_path, parse_dates=["fiscal_date"])
        cached_tickers = set(cached["ticker"].dropna().unique())
        missing = [ticker for ticker in tickers if ticker not in cached_tickers]
        LOGGER.info("Using cached fundamentals: %s", cache_path)
        if missing:
            LOGGER.info("No cached annual statements for %d symbols", len(missing))
        return cached[cached["ticker"].isin(tickers)].copy(), missing

    LOGGER.info("Downloading annual fundamentals for %d symbols", len(tickers))
    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(fetch_one_fundamental, ticker): ticker for ticker in tickers}
        for completed, future in enumerate(as_completed(futures), start=1):
            ticker = futures[future]
            try:
                frames.append(future.result())
            except Exception as exc:
                failures.append(ticker)
                LOGGER.warning("Fundamental download failed: %s", exc)
            if completed % 25 == 0 or completed == len(futures):
                LOGGER.info("Fundamentals %d/%d complete", completed, len(futures))

    if not frames:
        raise RuntimeError("yfinance returned no annual fundamental data")
    fundamentals = pd.concat(frames, ignore_index=True)
    fundamentals = fundamentals.sort_values(["ticker", "fiscal_date"])
    fundamentals.to_csv(cache_path, index=False)
    return fundamentals, failures


def month_end_panel(prices: pd.DataFrame) -> pd.DataFrame:
    panel = prices.resample("ME").last()
    return panel.dropna(how="all")


def build_observations(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    tickers: list[str],
    config: Config,
) -> pd.DataFrame:
    available = [ticker for ticker in tickers if ticker in prices.columns]
    if BENCHMARK not in prices.columns:
        raise RuntimeError(f"Benchmark {BENCHMARK} is missing from downloaded prices")

    monthly = month_end_panel(prices[[*available, BENCHMARK]])
    raw_returns = monthly.pct_change(fill_method=None).shift(-1)
    excess = raw_returns[available].sub(raw_returns[BENCHMARK], axis=0)
    signal_prices = monthly[available]

    start = pd.Timestamp(config.start)
    end = pd.Timestamp(config.end)
    signal_dates = monthly.index[(monthly.index >= start) & (monthly.index < end)]

    f = fundamentals.copy()
    f["available_date"] = pd.to_datetime(f["fiscal_date"]) + pd.to_timedelta(
        config.reporting_lag_days, unit="D"
    )
    f = f.sort_values(["ticker", "available_date"])

    observations: list[pd.DataFrame] = []
    for ticker in available:
        company = f[f["ticker"] == ticker][
            ["available_date", "fiscal_date", "eps", "roe"]
        ].dropna(subset=["available_date"])
        if company.empty:
            continue
        signals = pd.DataFrame({"signal_date": signal_dates})
        aligned = pd.merge_asof(
            signals.sort_values("signal_date"),
            company.sort_values("available_date"),
            left_on="signal_date",
            right_on="available_date",
            direction="backward",
            allow_exact_matches=False,
        )
        aligned["ticker"] = ticker
        aligned["price"] = aligned["signal_date"].map(signal_prices[ticker])
        aligned["stock_return"] = aligned["signal_date"].map(raw_returns[ticker])
        aligned["benchmark_return"] = aligned["signal_date"].map(raw_returns[BENCHMARK])
        aligned["excess_return"] = aligned["signal_date"].map(excess[ticker])
        aligned["per"] = aligned["price"] / aligned["eps"]
        observations.append(aligned)

    if not observations:
        raise RuntimeError("No stock-month observations could be constructed")
    result = pd.concat(observations, ignore_index=True)
    result = result.sort_values(["signal_date", "ticker"]).reset_index(drop=True)
    return result


def select_value_quality(observations: pd.DataFrame, config: Config) -> pd.DataFrame:
    finite = np.isfinite(observations["per"]) & np.isfinite(observations["roe"])
    selected = observations[
        finite
        & observations["per"].gt(0)
        & observations["per"].le(config.per_max)
        & observations["roe"].ge(config.roe_min)
        & observations["excess_return"].notna()
        & observations["excess_return"].le(1.0)
    ].copy()
    if selected.empty:
        raise RuntimeError("The PER/ROE screen produced no observations")
    return selected


def choose_bins(values: pd.Series) -> int:
    clean = values.dropna().to_numpy()
    if len(clean) < 2:
        return 10
    q25, q75 = np.percentile(clean, [25, 75])
    width = 2 * (q75 - q25) / np.cbrt(len(clean))
    if width <= 0:
        return 20
    bins = math.ceil((clean.max() - clean.min()) / width)
    return int(np.clip(bins, 15, 60))


def setup_japanese_font() -> None:
    candidates = ["Yu Gothic", "Meiryo", "Noto Sans CJK JP", "IPAexGothic", "DejaVu Sans"]
    selected = "DejaVu Sans"
    for candidate in candidates:
        try:
            font_manager.findfont(candidate, fallback_to_default=False)
            selected = candidate
            break
        except ValueError:
            continue
    plt.rcParams["font.family"] = selected
    plt.rcParams["axes.unicode_minus"] = False


def plot_histogram(selected: pd.DataFrame, output_path: Path, config: Config) -> None:
    setup_japanese_font()
    values = selected["excess_return"] * 100
    mean = values.mean()
    median = values.median()
    positive_rate = values.gt(0).mean() * 100

    fig, ax = plt.subplots(figsize=(11, 6.8), dpi=160)
    ax.hist(
        values,
        bins=choose_bins(values),
        color="#2f6690",
        edgecolor="white",
        linewidth=0.7,
        alpha=0.9,
    )
    ax.axvline(0, color="#333333", linewidth=1.2, label="0%")
    ax.axvline(mean, color="#d1495b", linewidth=2.0, label=f"平均 {mean:.2f}%")
    ax.axvline(median, color="#edae49", linewidth=2.0, linestyle="--", label=f"中央値 {median:.2f}%")
    ax.set_title(
        f"PER {config.per_max:g}倍以下・ROE {config.roe_min:.0%}以上の翌月超過リターン分布",
        fontsize=16,
        pad=16,
    )
    ax.set_xlabel("日経平均控除後の1か月リターン（%）")
    ax.set_ylabel("銘柄・月の観測数")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    ax.text(
        0.99,
        0.97,
        f"観測数: {len(selected):,}\nプラス比率: {positive_rate:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#cccccc"},
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = Config(
        start=args.start,
        end=args.end,
        per_max=args.per_max,
        roe_min=args.roe_min,
        reporting_lag_days=args.reporting_lag_days,
        workers=args.workers,
        refresh=args.refresh,
        limit=args.limit,
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    img_dir = root / "img"
    data_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    universe_path = data_dir / DEFAULT_UNIVERSE
    if not universe_path.exists():
        raise FileNotFoundError(
            f"Missing {universe_path}. Save a one-column TOPIX 500 security-code CSV there."
        )
    tickers = load_universe(universe_path, config.limit)

    suffix = f"_{config.limit}" if config.limit else ""
    prices_path = data_dir / f"adjusted_close_{config.start}_{config.end}{suffix}.csv"
    fundamentals_path = data_dir / f"annual_fundamentals{suffix}.csv"
    observations_path = data_dir / f"monthly_observations{suffix}.csv"
    selected_path = data_dir / f"selected_value_quality_returns{suffix}.csv"
    failures_path = data_dir / f"fundamental_download_failures{suffix}.txt"
    metadata_path = data_dir / f"run_metadata{suffix}.json"
    image_path = img_dir / f"value_quality_return_histogram{suffix}.png"

    prices = download_prices(tickers, config.start, config.end, prices_path, config.refresh)
    fundamentals, failures = download_fundamentals(
        tickers, fundamentals_path, config.workers, config.refresh
    )
    failures_path.write_text("\n".join(failures), encoding="utf-8")

    observations = build_observations(prices, fundamentals, tickers, config)
    selected = select_value_quality(observations, config)
    observations.to_csv(observations_path, index=False)
    selected.to_csv(selected_path, index=False)
    plot_histogram(selected, image_path, config)

    excess = selected["excess_return"]
    metadata = {
        "created_on": date.today().isoformat(),
        "config": asdict(config),
        "universe_file": universe_path.name,
        "requested_tickers": len(tickers),
        "tickers_with_prices": int(
            sum(ticker in prices.columns and prices[ticker].notna().any() for ticker in tickers)
        ),
        "tickers_with_fundamentals": int(fundamentals["ticker"].nunique()),
        "fundamental_download_failures": len(failures),
        "selected_observations": len(selected),
        "selected_unique_tickers": int(selected["ticker"].nunique()),
        "selected_signal_months": int(selected["signal_date"].nunique()),
        "mean_excess_return": float(excess.mean()),
        "median_excess_return": float(excess.median()),
        "standard_deviation": float(excess.std(ddof=1)),
        "positive_rate": float(excess.gt(0).mean()),
        "minimum_excess_return": float(excess.min()),
        "maximum_excess_return": float(excess.max()),
        "method_note": (
            "Annual fiscal period-end is treated as the data availability date plus reporting_lag_days. "
            "One-month adjusted-close return minus Nikkei 225 one-month adjusted-close return."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info("Selected %d stock-month observations", len(selected))
    LOGGER.info("Mean excess return %.3f%%", 100 * excess.mean())
    LOGGER.info("Histogram saved to %s", image_path)


if __name__ == "__main__":
    main()
