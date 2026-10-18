
import os
import math
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUT = Path("out")
OUT.mkdir(exist_ok=True)

API = "https://data-api.binance.vision"
KEY = os.getenv("BINANCE_SQUARE_OPENAPI_KEY", "")
GEMINI = os.getenv("GEMINI_API_KEY", "")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
STABLE = {
    "USDCUSDT", "FDUSDUSDT", "TUSDUSDT",
    "USDPUSDT", "DAIUSDT", "USDEUSDT"
}
MAJORS = {"BTCUSDT", "ETHUSDT", "BNBUSDT"}


def get(path, params=None):
    r = requests.get(API + path, params=params, timeout=25)
    r.raise_for_status()
    return r.json()


def price(v):
    if v >= 100:
        return f"{v:,.2f}"
    if v >= 1:
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return f"{v:.8f}".rstrip("0").rstrip(".")


def market_data():
    rows = []
    for x in get("/api/v3/ticker/24hr"):
        s = x.get("symbol", "")
        if not s.endswith("USDT") or s in STABLE:
            continue
        if any(w in s for w in ("UPUSDT", "DOWNUSDT",
                                 "BULLUSDT", "BEARUSDT")):
            continue
        try:
            row = {
                "symbol": s,
                "price": float(x["lastPrice"]),
                "change": float(x["priceChangePercent"]),
                "volume": float(x["quoteVolume"]),
                "high": float(x["highPrice"]),
                "low": float(x["lowPrice"]),
                "trades": int(x["count"])
            }
            if row["price"] > 0 and row["volume"] >= 5_000_000:
                rows.append(row)
        except (ValueError, KeyError, TypeError):
            pass
    if not rows:
        raise RuntimeError("Binance market data unavailable")
    return rows


def select_coins(rows):
    by = {x["symbol"]: x for x in rows}

    # Post 1: major market coin
    first = by.get("BTCUSDT") or by.get("ETHUSDT")
    if first is None:
        first = max(rows, key=lambda x: x["volume"])

    # Posts 2 and 3: liquid altcoins, excluding the first coin.
    alts = [
        x for x in rows
        if x["symbol"] not in MAJORS
        and x["symbol"] != first["symbol"]
        and x["volume"] >= 10_000_000
        and x["trades"] >= 3000
    ]
    if not alts:
        alts = [
            x for x in rows
            if x["symbol"] not in MAJORS
            and x["symbol"] != first["symbol"]
        ]

    # Momentum ranking for post 2.
    momentum = sorted(
        alts,
        key=lambda x: (
            min(abs(x["change"]), 15)
            * math.log10(max(x["volume"], 1))
        ),
        reverse=True
    )
    second = momentum[0] if momentum else first

    # Post 3: a different liquid coin, ranked by activity.
    remaining = [
        x for x in alts
        if x["symbol"] != second["symbol"]
    ]
    third = max(
        remaining,
        key=lambda x: x["volume"] * math.log10(max(x["trades"], 10)),
        default=first
    )

    return [first, second, third]


def candles(symbol):
    raw = get("/api/v3/klines", {
        "symbol": symbol, "interval": "1h", "limit": 72
    })
    result = []
    for k in raw:
        result.append({
            "time": datetime.fromtimestamp(k[0] / 1000, timezone.utc),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5])
        })
    if len(result) < 30:
        raise RuntimeError("Insufficient candle data")
    return result


def ema(values, period=20):
    a = 2 / (period + 1)
    result = [values[0]]
    for v in values[1:]:
        result.append(a * v + (1 - a) * result[-1])
    return result


def rsi(values, period=14):
    changes = [values[i] - values[i - 1]
               for i in range(1, len(values))]
    if len(changes) < period:
        return 50.0
    gains = [max(v, 0) for v in changes[-period:]]
    losses = [max(-v, 0) for v in changes[-period:]]
    up, down = sum(gains) / period, sum(losses) / period
    if down == 0:
        return 100.0
    return 100 - 100 / (1 + up / down)


def make_chart(row, data, slot):
    symbol = row["symbol"]
    times = [x["time"] for x in data]
    closes = [x["close"] for x in data]
    highs = [x["high"] for x in data]
    lows = [x["low"] for x in data]
    vols = [x["volume"] for x in data]
    line = ema(closes)
    strength = rsi(closes)
    last = closes[-1]
    support = min(lows[-24:])
    resistance = max(highs[-24:])

    fig, (ax, vx) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.patch.set_facecolor("#101820")
    for a in (ax, vx):
        a.set_facecolor("#101820")

    for i, candle in enumerate(data):
        up = candle["close"] >= candle["open"]
        color = "#00c087" if up else "#f6465d"
        ax.vlines(times[i], lows[i], highs[i], color=color, linewidth=1)
        bottom = min(candle["open"], candle["close"])
        height = max(abs(candle["close"] - candle["open"]), last * 0.00005)
        ax.bar(times[i], height, bottom=bottom,
               width=0.028, color=color)
        vx.bar(times[i], vols[i], width=0.028, color=color)

    ax.plot(times, line, color="#f0b90b",
            linewidth=1.7, label="EMA 20")
    ax.axhline(support, color="#00c087",
               linestyle="--", linewidth=1, label="24h range low")
    ax.axhline(resistance, color="#f6465d",
               linestyle="--", linewidth=1, label="24h range high")

    titles = {
        1: "CRYPTO MARKET PULSE",
        2: "ALTCOIN MOMENTUM WATCH",
        3: "TECHNICAL MARKET SETUP"
    }
    ax.set_title(f"{symbol} | {titles[slot]}",
                 color="white", loc="left", fontsize=15)
    ax.text(
        0.01, 0.95,
        f"Price {price(last)} USDT | 24h {row['change']:+.2f}%"
        f" | RSI {strength:.1f}",
        transform=ax.transAxes, color="white", va="top"
    )
    ax.grid(True, color="#34404c", alpha=0.6)
    ax.legend(loc="upper left", facecolor="#101820",
              labelcolor="white", fontsize=8)
    vx.set_ylabel("Volume", color="white")
    vx.xaxis.set_major_formatter(
        mdates.DateFormatter("%d %H:%M", tz=timezone.utc)
    )
    for a in (ax, vx):
        a.tick_params(colors="#b7c5d3")
        for sp in a.spines.values():
            sp.set_color("#34404c")

    fig.text(
        0.99, 0.01,
        "Binance market data | Educational analysis, not financial advice",
        ha="right", color="#b7c5d3", fontsize=8
    )
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    path = OUT / "chart.png"
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)

    return last, support, resistance, strength, line[-1], path


def write_post(row, slot, last, support, resistance, strength, ema20):
    symbol = row["symbol"]
    coin = "$" + symbol[:-4]
    above = last >= ema20

    if slot == 1:
        heading = f"Market Pulse: {coin}"
        insight = (
            "Start with the broader market. A major coin can help "
            "put altcoin volatility into context."
        )
    elif slot == 2:
        heading = f"Altcoin Watch: {coin}"
        insight = (
            "A strong daily move is worth investigating, but momentum "
            "alone does not confirm a sustainable breakout."
        )
    else:
        heading = f"Technical Setup: {coin}"
        insight = (
            "Watch how price reacts around the recent range. A candle "
            "close and volume confirmation are more useful than a wick."
        )

    facts = (
        f"Coin: {symbol}\n"
        f"Price: {price(last)} USDT\n"
        f"24h change: {row['change']:+.2f}%\n"
        f"24h quote volume: ${row['volume']:,.0f}\n"
        f"Recent range low: {price(support)}\n"
        f"Recent range high: {price(resistance)}\n"
        f"Hourly RSI(14): {strength:.1f}\n"
        f"Price vs EMA20: {'above' if above else 'below'}"
    )

    fallback = (
        f"{heading}\n\n{insight}\n\n{facts}\n\n"
        f"How to read this: price is {'above' if above else 'below'} "
        f"the 20-hour EMA. RSI is a momentum measure, not a standalone "
        f"buy or sell signal. The recent range marks areas to watch, "
        f"not guaranteed support or resistance.\n\n"
        f"Possible confirmation: a decisive candle close beyond the "
        f"range with stronger volume. If price falls back into the "
        f"range, the move may fail. Plan risk before any trade and "
        f"avoid excessive leverage.\n\n"
        f"Educational analysis, not financial advice.\n"
        f"#Crypto #BinanceSquare"
    )

    if not GEMINI:
        return fallback

    prompt = f"""
Write a concise, natural Binance Square post of 90-130 words.
Make it engaging, specific, original, and realistic.
Use only the verified facts below. Explain one useful insight,
a possible confirmation, and what could invalidate it.
No guaranteed profit, invented news, fake targets, or commands
to buy or sell. Use 2-3 hashtags and a short risk disclaimer.
Return only the post, no preamble.

Post topic: {heading}
Unique angle: {insight}
Verified Binance data:
{facts}
"""

    try:
        url = (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{MODEL}:generateContent"
        )
        response = requests.post(
            url,
            params={"key": GEMINI},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.85,
                    "maxOutputTokens": 350
                }
            },
            timeout=35
        )
        response.raise_for_status()
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        if 180 <= len(text) <= 1400:
            return text
        print("AI output length invalid; using verified-data template.")
    except Exception as e:
        print("AI writing unavailable:", type(e).__name__, str(e))

    return fallback


def main():
    if not KEY:
        raise RuntimeError("Missing BINANCE_SQUARE_OPENAPI_KEY secret")

    # The workflow sets SLOT=1, 2 or 3 for manual runs.
    # Scheduled runs use the UTC schedule: 01:17, 09:17, 17:17.
    raw_slot = os.getenv("SLOT", "").strip()
    if raw_slot in ("1", "2", "3"):
        slot = int(raw_slot)
    else:
        hour = datetime.now(timezone.utc).hour
        slot = 1 if hour < 6 else 2 if hour < 14 else 3

    rows = market_data()
    chosen = select_coins(rows)
    row = chosen[slot - 1]

    print("Post slot:", slot)
    print("Selected coin:", row["symbol"])
    print("24h change:", row["change"])
    print("24h volume:", row["volume"])

    data = candles(row["symbol"])
    last, support, resistance, strength, ema20, chart = make_chart(
        row, data, slot
    )
    post = write_post(
        row, slot, last, support, resistance, strength, ema20
    )

    (OUT / "post.txt").write_text(post, encoding="utf-8")
    print("Post text:\n", post)
    print("Chart:", chart)

    root = Path("/tmp/binance-skills")
    scripts = list(root.glob("**/post-image.mjs"))
    if not scripts:
        raise RuntimeError(
            "Posting script post-image.mjs not found. "
            "Check the workflow skill download step."
        )

    command = [
        "node", str(scripts[0]),
        "--text", post,
        "--images", str(chart)
    ]
    env = os.environ.copy()
    env["BINANCE_SQUARE_OPENAPI_KEY"] = KEY

    result = subprocess.run(
        command, env=env, capture_output=True,
        text=True, timeout=180
    )
    print(result.stdout[-5000:])
    if result.stderr:
        print("Posting stderr:", result.stderr[-2500:])
    if result.returncode != 0:
        raise RuntimeError("Binance Square publishing command failed")

    if "Success!" not in result.stdout and "ID:" not in result.stdout:
        raise RuntimeError(
            "No explicit publish confirmation found; inspect logs "
            "before retrying to avoid duplicate posts."
        )

    print("Publish success confirmed by command output.")


if __name__ == "__main__":
    main()
