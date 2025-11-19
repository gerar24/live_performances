import json
import math
import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DATA_DIR = os.path.join(ROOT, "src", "data")
PUBLIC_DATA_DIR = os.path.join(ROOT, "public", "data")

CONFIG_PATH = os.path.join(SRC_DATA_DIR, "config.json")
PORTFOLIOS_PATH = os.path.join(SRC_DATA_DIR, "portfolios.json")

EQUITY_OUT = os.path.join(PUBLIC_DATA_DIR, "equity_curve.json")
METRICS_OUT = os.path.join(PUBLIC_DATA_DIR, "metrics.json")
ALLOC_OUT = os.path.join(PUBLIC_DATA_DIR, "allocation_history.json")

TRADING_DAYS_PER_YEAR = 252


def ensure_dirs():
    os.makedirs(PUBLIC_DATA_DIR, exist_ok=True)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_datestr(dt: pd.Timestamp) -> str:
    return dt.strftime("%Y-%m-%d")


def safe_normalize(weights: dict) -> dict:
    total = sum(weights.values())
    if total <= 0:
        return {k: 0.0 for k in weights.keys()}
    return {k: float(v) / total for k, v in weights.items()}


def compute_metrics(daily_returns: pd.Series, equity: pd.Series) -> dict:
    # Annualized metrics assuming daily frequency
    n = len(daily_returns)
    if n <= 1:
        return {"total_return": 0.0, "sharpe": 0.0, "volatility": 0.0, "max_drawdown": 0.0}

    start_val = float(equity.iloc[0]) if n > 0 else 100.0
    end_val = float(equity.iloc[-1]) if n > 0 else 100.0

    # Since inception cumulative return (not annualized)
    try:
        total_return = (end_val / start_val) - 1.0 if start_val > 0 else 0.0
    except Exception:
        total_return = 0.0

    mean_daily = float(daily_returns.mean())
    std_daily = float(daily_returns.std(ddof=0))
    volatility = std_daily * math.sqrt(TRADING_DAYS_PER_YEAR) if std_daily > 0 else 0.0
    sharpe = (mean_daily / std_daily) * math.sqrt(TRADING_DAYS_PER_YEAR) if std_daily > 0 else 0.0

    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    return {
        "total_return": round(total_return, 6),
        "sharpe": round(sharpe, 4),
        "volatility": round(volatility, 6),
        "max_drawdown": round(max_dd, 6),
    }


def compute_sortino(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return 0.0
    downside = daily_returns.copy()
    downside[downside > 0] = 0.0
    dd_std = float(downside.std(ddof=0))
    if dd_std == 0:
        return 0.0
    mean_daily = float(daily_returns.mean())
    sortino = (mean_daily / dd_std) * math.sqrt(TRADING_DAYS_PER_YEAR)
    return round(sortino, 4)


def compute_beta_alpha(asset_returns: pd.Series, market_returns: pd.Series) -> tuple[float, float]:
    # Risk-free assumed ~0 for simplicity
    if asset_returns.empty or market_returns.empty:
        return 0.0, 0.0
    # Align indices
    df = pd.concat([asset_returns, market_returns], axis=1).dropna()
    if df.shape[0] == 0:
        return 0.0, 0.0
    r_a = df.iloc[:, 0]
    r_m = df.iloc[:, 1]
    var_m = float(r_m.var(ddof=0))
    if var_m == 0:
        return 0.0, 0.0
    cov_am = float(r_a.cov(r_m))
    beta = cov_am / var_m
    # alpha removed per user request
    return round(beta, 4), 0.0


def compute_up_down_capture(asset_returns: pd.Series, market_returns: pd.Series) -> tuple[float, float]:
    if asset_returns.empty or market_returns.empty:
        return 0.0, 0.0
    df = pd.concat([asset_returns, market_returns], axis=1).dropna()
    if df.shape[0] == 0:
        return 0.0, 0.0
    r_a = df.iloc[:, 0]
    r_m = df.iloc[:, 1]
    up = df[r_m > 0.0]
    down = df[r_m < 0.0]
    up_cap = 0.0
    down_cap = 0.0
    if up.shape[0] > 0 and float(up.iloc[:, 1].mean()) != 0.0:
        up_cap = float(up.iloc[:, 0].mean()) / float(up.iloc[:, 1].mean())
    if down.shape[0] > 0 and float(down.iloc[:, 1].mean()) != 0.0:
        down_cap = float(down.iloc[:, 0].mean()) / float(down.iloc[:, 1].mean())
    return round(up_cap, 4), round(down_cap, 4)


def download_prices(tickers, start_date):
    if not tickers:
        return pd.DataFrame()
    df = yf.download(
        tickers=sorted(tickers),
        start=start_date,
        end=None,
        progress=False,
        auto_adjust=False,
        threads=False, 
    )
    # Handle multi-index columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close" in df.columns.get_level_values(0)):
            prices = df["Adj Close"].copy()
        else:
            # Fallback to Close
            prices = df["Close"].copy()
    else:
        prices = df.copy()
    # Clean up columns if single ticker returns a Series
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    # If a day's volume is 0, forward-fill the price for that day
    volume_df = None
    if isinstance(df.columns, pd.MultiIndex):
        if ("Volume" in df.columns.get_level_values(0)):
            volume_df = df["Volume"].copy()
    else:
        if "Volume" in df.columns:
            # In single-ticker case, df has single-index columns; we still extract Volume
            volume_df = df[["Volume"]].copy()
    if volume_df is not None and not volume_df.empty:
        volume_df = volume_df.reindex(prices.index)
        for t in prices.columns:
            if t in volume_df.columns:
                vol_t = volume_df[t]
                price_t = prices[t]
                # For zero volume rows, use previous close
                prices[t] = price_t.where(vol_t != 0, price_t.shift(1))
    # Forward-fill to handle missing values
    prices = prices.ffill()
    # Drop columns that are entirely NaN
    prices = prices.dropna(axis=1, how="all")
    return prices


def simulate_portfolio_path(prices: pd.DataFrame, returns: pd.DataFrame, index: pd.DatetimeIndex, rebalances: list) -> tuple[pd.Series, dict]:
    # Build mapping of index position -> target weights and optional price paid
    rb_map = {}
    union_tickers = set()
    for rb in rebalances:
        allocation = rb.get("allocation", {})
        if not allocation:
            continue
        union_tickers.update(allocation.keys())
    union_tickers = [t for t in sorted(union_tickers) if t in returns.columns]
    if len(union_tickers) == 0:
        equity = pd.Series([100.0] * len(index), index=index)
        return equity, {}

    for rb in rebalances:
        date_str = rb.get("date")
        allocation = rb.get("allocation", {})
        if not allocation:
            continue
        norm_alloc = safe_normalize({k: float(v) for k, v in allocation.items() if k in union_tickers})
        price_paid = rb.get("price") or rb.get("prices") or {}
        try:
            rb_date = pd.to_datetime(date_str)
        except Exception:
            continue
        pos = index.searchsorted(rb_date, side="right") - 1
        if pos < 0:
            pos = 0
        if pos >= len(index):
            pos = len(index) - 1
        rb_map[pos] = {"weights": norm_alloc, "prices": price_paid}

    if len(rb_map) == 0:
        equity = pd.Series([100.0] * len(index), index=index)
        return equity, {}

    shares = {t: 0.0 for t in union_tickers}
    equity_values = []
    alloc_history = {t: [] for t in union_tickers}

    for i, dt in enumerate(index):
        # If this is a rebalance day, compute equity at previous close and reallocate using previous close
        if i in rb_map:
            prev_dt = index[i - 1] if i > 0 else None
            # Portfolio value at previous close (or baseline 100 if first day)
            if i > 0:
                total_equity_prev = 0.0
                for t in union_tickers:
                    p_prev = float(prices.at[prev_dt, t]) if t in prices.columns else float('nan')
                    if math.isfinite(p_prev) and p_prev > 0:
                        total_equity_prev += shares[t] * p_prev
                if not math.isfinite(total_equity_prev) or total_equity_prev <= 0.0:
                    total_equity_prev = 100.0
            else:
                total_equity_prev = 100.0

            targets = rb_map[i]["weights"]
            paid = rb_map[i]["prices"]
            for t in union_tickers:
                w = float(targets.get(t, 0.0))
                target_value = total_equity_prev * w
                entry_price = paid.get(t) if isinstance(paid, dict) else None
                if entry_price is None or not isinstance(entry_price, (int, float)) or entry_price <= 0:
                    # Fall back to previous close price when available, else today's close if first day
                    if i > 0:
                        entry_price = float(prices.at[prev_dt, t]) if t in prices.columns else float('nan')
                    else:
                        entry_price = float(prices.at[dt, t]) if t in prices.columns else float('nan')
                if entry_price is None or not math.isfinite(entry_price) or entry_price <= 0:
                    shares[t] = 0.0
                else:
                    shares[t] = target_value / entry_price

        # Value portfolio at today's close
        total_equity_today = 0.0
        for t in union_tickers:
            p = float(prices.at[dt, t]) if t in prices.columns else float('nan')
            if math.isfinite(p) and p > 0:
                total_equity_today += shares[t] * p
        equity_values.append(total_equity_today)

        if total_equity_today > 0.0:
            for t in union_tickers:
                p = float(prices.at[dt, t]) if t in prices.columns else float('nan')
                v = shares[t] * p if math.isfinite(p) and p > 0 else 0.0
                alloc_history[t].append(v / total_equity_today if total_equity_today else 0.0)
        else:
            for t in union_tickers:
                alloc_history[t].append(0.0)

    equity_series = pd.Series(equity_values, index=index)
    return equity_series, alloc_history


def main():
    ensure_dirs()

    config = read_json(CONFIG_PATH)
    portfolios_data = read_json(PORTFOLIOS_PATH)

    benchmark_tickers = list(dict.fromkeys(config.get("benchmarks", [])))
    market_candidate = config.get("market") if isinstance(config, dict) else None
    names_map = config.get("names", {}) if isinstance(config, dict) else {}

    # Collect all tickers from portfolios
    all_tickers = set(benchmark_tickers)
    earliest_date = None
    for rebalances in portfolios_data.values():
        for rb in rebalances:
            allocation = rb.get("allocation", {})
            for t in allocation.keys():
                all_tickers.add(t)
            try:
                d = pd.to_datetime(rb.get("date"))
                if earliest_date is None or d < earliest_date:
                    earliest_date = d
            except Exception:
                pass

    if earliest_date is None:
        earliest_date = pd.Timestamp("2015-01-01")

    prices = download_prices(all_tickers, earliest_date.strftime("%Y-%m-%d"))
    if prices.empty:
        # Produce minimal empty outputs
        with open(EQUITY_OUT, "w", encoding="utf-8") as f:
            json.dump({"dates": [], "portfolios": {}, "benchmarks": {}}, f, ensure_ascii=False, indent=2)
        with open(METRICS_OUT, "w", encoding="utf-8") as f:
            json.dump({"portfolios": {}, "benchmarks": {}}, f, ensure_ascii=False, indent=2)
        with open(ALLOC_OUT, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
        return

    returns = prices.pct_change().fillna(0.0)
    date_index = returns.index
    dates_list = [to_datestr(d) for d in date_index]

    equity_out = {"dates": dates_list, "portfolios": {}, "benchmarks": {}, "names": names_map}
    metrics_out = {"portfolios": {}, "benchmarks": {}}
    allocation_out = {}

    # Benchmarks
    bench_daily_returns = {}
    for t in benchmark_tickers:
        if t not in returns.columns:
            continue
        r = returns[t]
        equity = (1.0 + r).cumprod() * 100.0
        equity_out["benchmarks"][t] = [round(float(x), 4) for x in equity.tolist()]
        base_metrics = compute_metrics(r, equity)
        base_metrics["sortino"] = compute_sortino(r)
        metrics_out["benchmarks"][t] = base_metrics
        bench_daily_returns[t] = r

    # Choose market ticker for beta/alpha
    market_ticker = None
    if market_candidate in bench_daily_returns:
        market_ticker = market_candidate
    elif len(bench_daily_returns) > 0:
        market_ticker = list(bench_daily_returns.keys())[0]
    market_returns = bench_daily_returns.get(market_ticker, pd.Series(dtype=float))

    # Add beta and captures for benchmarks too (relative to market)
    for t, r in bench_daily_returns.items():
        m = metrics_out["benchmarks"].get(t, {})
        if market_returns is not None and not market_returns.empty:
            if t == market_ticker:
                # Market vs itself: define as 1 and hide the suffix
                m["beta"] = 1.0
                m["up_capture"] = 1.0
                m["down_capture"] = 1.0
                m["beta_vs"] = ""
                m["capture_vs"] = ""
            else:
                beta, _ = compute_beta_alpha(r, market_returns)
                upc, downc = compute_up_down_capture(r, market_returns)
                m["beta"] = beta
                m["up_capture"] = upc
                m["down_capture"] = downc
                m["beta_vs"] = market_ticker or ""
                m["capture_vs"] = market_ticker or ""
        else:
            m["beta"] = 0.0
            m["up_capture"] = 0.0
            m["down_capture"] = 0.0
            m["beta_vs"] = ""
            m["capture_vs"] = ""
        metrics_out["benchmarks"][t] = m

    # Portfolios
    for portfolio_name, rebalances in portfolios_data.items():
        if not rebalances:
            flat = [100.0 for _ in dates_list]
            equity_out["portfolios"][portfolio_name] = flat
            metrics_out["portfolios"][portfolio_name] = {
                "total_return": 0.0,
                "sharpe": 0.0,
                "volatility": 0.0,
                "max_drawdown": 0.0,
                "sortino": 0.0,
                "beta": 0.0,
                "beta_vs": market_ticker or "",
                "up_capture": 0.0,
                "down_capture": 0.0,
                "capture_vs": market_ticker or "",
                "correlations": {k: 0.0 for k in bench_daily_returns.keys()},
            }
            allocation_out[portfolio_name] = {"dates": dates_list, "series": {}}
            continue

        equity_series, alloc_history = simulate_portfolio_path(prices, returns, date_index, rebalances)
        daily_returns = equity_series.pct_change().fillna(0.0)

        equity_out["portfolios"][portfolio_name] = [round(float(x), 4) for x in equity_series.tolist()]
        base = compute_metrics(daily_returns, equity_series)
        base["sortino"] = compute_sortino(daily_returns)
        # correlations vs each benchmark
        corrs = {}
        for bt, br in bench_daily_returns.items():
            joined = pd.concat([daily_returns, br], axis=1).dropna()
            if joined.shape[0] == 0:
                corrs[bt] = 0.0
            else:
                corrs[bt] = round(float(joined.corr().iloc[0, 1]), 4)
        base["correlations"] = corrs
        # beta and up/down capture vs market
        if market_returns is not None and not market_returns.empty:
            beta, _ = compute_beta_alpha(daily_returns, market_returns)
            upc, downc = compute_up_down_capture(daily_returns, market_returns)
            base["beta"] = beta
            base["beta_vs"] = market_ticker or ""
            base["up_capture"] = upc
            base["down_capture"] = downc
            base["capture_vs"] = market_ticker or ""
        else:
            base["beta"] = 0.0
            base["beta_vs"] = ""
            base["up_capture"] = 0.0
            base["down_capture"] = 0.0
            base["capture_vs"] = ""

        metrics_out["portfolios"][portfolio_name] = base
        # Include explicit rebalances from input for UI
        cleaned_reb = []
        for rb in rebalances:
            cleaned_reb.append({
                "date": rb.get("date"),
                "notes": rb.get("notes", ""),
                "allocation": rb.get("allocation", {}),
                "price": rb.get("price") or rb.get("prices") or {}
            })
        allocation_out[portfolio_name] = {
            "dates": dates_list,
            "series": {t: [round(float(w), 6) for w in ws] for t, ws in alloc_history.items()},
            "rebalances": cleaned_reb,
        }

    with open(EQUITY_OUT, "w", encoding="utf-8") as f:
        json.dump(equity_out, f, ensure_ascii=False, indent=2)
    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)
    with open(ALLOC_OUT, "w", encoding="utf-8") as f:
        json.dump(allocation_out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
