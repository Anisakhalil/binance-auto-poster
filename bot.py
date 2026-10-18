
import os
import sys
import json
import math
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

OUT = Path("out")
OUT.mkdir(exist_ok=True)

BASE = "https://data-api.binance.vision"
SQUARE_KEY = os.getenv("BINANCE_SQUARE_OPENAPI_KEY", "")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def get_json(path, params=None):
    response = requests.get(
        BASE + path, params=params, timeout=20
    )
    response.raise_for_status()
    return response.json()


def fmt_price(value):
    if value >= 1000:
        return f"{value:,.2f}"
    if value >= 1:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{value:.7f}".rstrip("0").rstrip(".")


def get_market():
    data = get_json("/api/v3/ticker/24hr")
    items = []

    for item in data:
        symbol = item.get("symbol", "")

        if not symbol.endswith("USDT"):
            continue

        if any(x in symbol for x in ("UPUSDT", "DOWNUSDT",
                                      "BULLUSDT", "BEARUSDT")):
            continue

        try:
            price = float(item["lastPrice"])
            change = float(item["priceChangePercent"])
            volume = float(item["quoteVolume"])
            high = float(item["highPrice"])
            low = float(item["lowPrice"])
            trades = int(item.get("count", 0))

            if price > 0 and volume >= 3_000_000:
                items.append({
                    "symbol": symbol,
                    "price": price,
                    "change": change,
                    "volume": volume,
                    "high": high,
                    "low": low,
                    "trades": trades
                })
        except (ValueError, TypeError, KeyError):
            continue

    if not items:
        raise RuntimeError("No usable Binance market data returned.")

    return items


def choose_coin(items, slot):
    by_symbol = {x["symbol"]: x for x in items}

    if slot == 1:
        return (
            by_symbol.get("BTCUSDT")
            or by_symbol.get("ETHUSDT")
            or max(items, key=lambda x: x["volume"])
        )

    stable = {
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "USDCUSDT",
        "FDUSDUSDT", "TUSDUSDT", "USDPUSDT", "DAIUSDT"
    }

    candidates = [
        x for x in items
        if x["symbol"] not in stable
        and x["volume"] >= 10_000_000
        and x["trades"] >= 5000
    ]

    if not candidates:
        candidates = [
            x for x in items
            if x["symbol"] not in stable
        ]

    if not candidates:
        return max(items, key=lambda x: x["volume"])

    if slot == 2:
        # Active, liquid market with meaningful price movement.
        return max(
            candidates,
            key=lambda x: (
                abs(x["change"])
                * math.log10(max(x["volume"], 1))
                * math.log10(max(x["trades"], 10))
            )
        )

    # Technical-analysis slot: favour liquidity and activity,
    # rather than choosing only the biggest price spike.
    return max(
        candidates,
        key=lambda x: (
            0.65 * math.log10(max(x["volume"], 1))
            + 0.25 * math.log10(max(x["trades"], 10))
            + 0.10 * min(abs(x["change"]), 12)
        )
    )


def get_candles(symbol):
    raw = get_json(
        "/api/v3/klines",
        {"symbol": symbol, "interval": "1h", "limit": 72}
    )

    rows = []
    for k in raw:
        rows.append({
            "time": datetime.fromtimestamp(
                k[0] / 1000, timezone.utc
            ),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5])
        })

    if len(rows) < 30:
        raise RuntimeError("Not enough hourly candles returned.")

    return rows


def calculate_rsi(closes, period=14):
    if len(closes) <= period:
        return 50.0

    changes = [
        closes[i] - closes[i - 1]
        for i in range(len(closes) - period, len(closes))
    ]

    gains = [max(x, 0) for x in changes]
    losses = [max(-x, 0) for x in changes]

    average_gain = sum(gains) / period
    average_loss = sum(losses) / period

    if average_loss == 0:
        return 100.0

    rs = average_gain / average_loss
    return 100 - 100 / (1 + rs)


def calculate_ema(values, period=20):
    alpha = 2 / (period + 1)
    result = []

    for i, value in enumerate(values):
        if i == 0:
            result.append(value)
        else:
            result.append(
                alpha * value + (1 - alpha) * result[-1]
            )

    return result


def make_chart(symbol, rows, market):
    closes = [x["close"] for x in rows]
    highs = [x["high"] for x in rows]
    lows = [x["low"] for x in rows]
    volumes = [x["volume"] for x in rows]
    times = [x["time"] for x in rows]

    last = closes[-1]
    recent_high = max(highs[-24:])
    recent_low = min(lows[-24:])
    ema = calculate_ema(closes, 20)
    rsi_value = calculate_rsi(closes)

    fig, (ax, volume_ax) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    fig.patch.set_facecolor("#101820")
    for axis in (ax, volume_ax):
        axis.set_facecolor("#101820")

    for i, row in enumerate(rows):
        rising = row["close"] >= row["open"]
        candle_color = "#00c087" if rising else "#f6465d"

        ax.vlines(
            times[i], row["low"], row["high"],
            color=candle_color, linewidth=1.2
        )

        bottom = min(row["open"], row["close"])
        height = max(
            abs(row["close"] - row["open"]),
            last * 0.00005
        )

        ax.bar(
            times[i], height, bottom=bottom,
            width=0.028, color=candle_color
        )

        volume_ax.bar(
            times[i], volumes[i],
            width=0.028, color=candle_color, alpha=0.8
        )

    ax.plot(
        times, ema, color="#f0b90b",
        linewidth=1.6, label="EMA 20"
    )

    ax.axhline(
        recent_high, color="#00c087",
        linestyle="--", linewidth=1
    )
    ax.axhline(
        recent_low, color="#f6465d",
        linestyle="--", linewidth=1
    )

    ax.set_title(
        f"{symbol} | 1H MARKET ANALYSIS",
        color="white", fontsize=16, loc="left"
    )

    ax.text(
        0.01, 0.94,
        f"Price: {fmt_price(last)} | "
        f"RSI: {rsi_value:.1f} | "
        f"24h: {market['change']:+.2f}%",
        transform=ax.transAxes,
        color="white", fontsize=10, va="top"
    )

    ax.text(
        0.99, 0.94,
        f"24h High: {fmt_price(market['high'])} | "
        f"Low: {fmt_price(market['low'])}",
        transform=ax.transAxes,
        color="#b7c5d3", fontsize=8,
        ha="right", va="top"
    )

    ax.legend(
        loc="upper left", facecolor="#101820",
        labelcolor="white"
    )

    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, pos: fmt_price(value))
    )

    ax.grid(True, color="#33404d", linewidth=0.5, alpha=0.65)
    ax.tick_params(colors="#b7c5d3", labelsize=8)
    volume_ax.tick_params(colors="#b7c5d3", labelsize=8)
    volume_ax.set_ylabel("Volume", color="#b7c5d3")

    volume_ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%d %H:%M", tz=timezone.utc)
    )
    volume_ax.xaxis.set_major_locator(
        mdates.HourLocator(interval=8, tz=timezone.utc)
    )

    for axis in (ax, volume_ax):
        for spine in axis.spines.values():
            spine.set_color("#33404d")

    fig.text(
        0.99, 0.012,
        "Educational analysis | Data: Binance | Not financial advice",
        ha="right", color="#9aa7b4", fontsize=8
    )

    fig.tight_layout(rect=[0, 0.035, 1, 1])

    chart_path = OUT / "chart.png"
    fig.savefig(
        chart_path, dpi=150,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight"
    )
    plt.close(fig)

    return (
        chart_path, last, recent_high, recent_low,
        rsi_value, ema[-1]
    )


def create_post(slot, market, last, high, low, rsi_value, ema):
    symbol = market["symbol"]
    coin = "$" + symbol[:-4]

    if slot == 1:
        heading = "Crypto Market Pulse"
    elif slot == 2:
        heading = f"Trending Coin Watch: {coin}"
    else:
        heading = f"Technical Setup: {coin}"

    direction = "above" if last >= ema else "below"

    facts = (
        f"Coin: {symbol}\n"
        f"Last price: {fmt_price(last)} USDT\n"
        f"24h change: {market['change']:+.2f}%\n"
        f"24h quote volume: ${market['volume']:,.0f}\n"
        f"24h high: {fmt_price(market['high'])}\n"
        f"24h low: {fmt_price(market['low'])}\n"
        f"Recent hourly range low/high: "
        f"{fmt_price(low)} / {fmt_price(high)}\n"
        f"Hourly RSI(14): {rsi_value:.1f}\n"
        f"Price is {direction} EMA20."
    )

    fallback = (
        f"{heading}\n\n"
        f"Here is the current market picture for {coin}.\n\n"
        f"{facts}\n\n"
        f"How to read it: price is {direction} the 20-hour EMA. "
        f"The recent range low at {fmt_price(low)} and range high "
        f"at {fmt_price(high)} are useful levels to watch, not "
        f"automatic buy or sell signals.\n\n"
        f"What matters next? Look for a candle close beyond the "
        f"range and volume confirmation. A wick alone does not "
        f"confirm a breakout. If price returns inside the range, "
        f"the move may have failed.\n\n"
        f"Always decide your invalidation level before taking risk. "
        f"Avoid excessive leverage and never risk money you cannot "
        f"afford to lose.\n\n"
        f"Educational market analysis, not financial advice.\n"
        f"#Crypto #BinanceSquare"
    )

    if not GEMINI_KEY:
        return fallback

    prompt = f"""
Write one original, realistic Binance Square market-analysis post.
Use only the verified data below. Do not invent news, events,
price targets, probabilities, or guaranteed profits.
Write natural English, 130-190 words. Explain the data, the
important levels, confirmation needed, and what could invalidate
the setup. Avoid telling readers to buy or sell. Use 2-3 relevant
hashtags and one short risk disclaimer. No fake urgency or hype.

Topic: {heading}
Verified data:
{facts}

Return only the post text.
"""

    try:
        url = (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{GEMINI_MODEL}:generateContent"
        )

        response = requests.post(
            url,
            params={"key": GEMINI_KEY},
            json={
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 500
                }
            },
            timeout=35
        )
        response.raise_for_status()

        result = response.json()
        generated = result["candidates"][0]["content"]["parts"][0]["text"].strip()

        if 250 <= len(generated) <= 1700:
            return generated

        print("[WARN] Gemini output length invalid; using fallback.")

    except Exception as error:
        print(
            "[WARN] Gemini failed; using market-data template:",
            type(error).__name__, str(error)
        )

    return fallback


def main():
    if not SQUARE_KEY:
        raise SystemExit(
            "Missing BINANCE_SQUARE_OPENAPI_KEY GitHub secret."
        )

    slot_text = os.getenv("SLOT", "").strip()

    if slot_text in {"1", "2", "3"}:
        slot = int(slot_text)
    else:
        hour = datetime.now(timezone.utc).hour
        slot = 1 if hour < 6 else 2 if hour < 14 else 3

    print("UTC time:", datetime.now(timezone.utc).isoformat())
    print("Post slot:", slot)

    market_data = get_market()
    market = choose_coin(market_data, slot)

    print(
        "Selected:", market["symbol"],
        "| 24h change:", f"{market['change']:+.2f}%",
        "| volume:", f"${market['volume']:,.0f}"
    )

    candles = get_candles(market["symbol"])

    chart, last, high, low, rsi_value, ema = make_chart(
        market["symbol"], candles, market
    )

    post = create_post(
        slot, market, last, high, low, rsi_value, ema
    )

    (OUT / "post.txt").write_text(post, encoding="utf-8")

    print("Generated post:")
    print(post)
    print("Chart saved:", chart)

    skill_dir = Path(
        os.getenv(
            "SQUARE_SKILL_DIR",
            "/tmp/binance-skills/skills/binance/square-post"
        )
    )

    script = skill_dir / "scripts" / "post-image.mjs"

    if not script.exists():
        matches = list(
            Path("/tmp/binance-skills").glob("**/post-image.mjs")
        )
        if not matches:
            raise SystemExit(
                "Binance Square image-posting script not found. "
                "Check the skill checkout and its current CLI."
            )
        script = matches[0]

    env = os.environ.copy()
    env["BINANCE_SQUARE_OPENAPI_KEY"] = SQUARE_KEY

    command = [
        "node", str(script),
        "--text", post,
        "--images", str(chart)
    ]

    print("Publishing one image post...")
    result = subprocess.run(
        command, env=env, text=True,
        capture_output=True, timeout=180
    )

    print(result.stdout[-5000:])

    if result.stderr:
        print("STDERR:", result.stderr[-2500:])

    if result.returncode != 0:
        raise SystemExit(
            f"Publishing failed. Exit code: {result.returncode}"
        )

    if "Success!" not in result.stdout and "ID:" not in result.stdout:
        raise SystemExit(
            "No clear success confirmation received. "
            "Check logs and your Square profile before retrying."
        )

    print("Publishing command completed.")


if __name__ == "__main__":
    main()
