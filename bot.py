
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
STABLE = {
    "USDCUSDT", "FDUSDUSDT", "TUSDUSDT",
    "USDPUSDT", "DAIUSDT", "USDEUSDT"
}
MAJORS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]


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
            continue

    if not rows:
        raise RuntimeError("Binance market data unavailable")
    return rows


def select_coins(rows, slot):
    """Choose 1 major and 2 liquid altcoins; rotate by slot and date."""
    by = {x["symbol"]: x for x in rows}
    today = datetime.now(timezone.utc).timetuple().tm_yday

    major_candidates = [by[s] for s in MAJORS if s in by]
    if major_candidates:
        major = major_candidates[(slot - 1) % len(major_candidates)]
    else:
        major = max(rows, key=lambda x: x["volume"])

    alts = [
        x for x in rows
        if x["symbol"] not in MAJORS
        and x["volume"] >= 10_000_000
        and x["trades"] >= 3000
    ]
    if len(alts) < 2:
        alts = [
            x for x in rows
            if x["symbol"] not in MAJORS
            and x["symbol"] != major["symbol"]
        ]

    # Rank by momentum and liquidity, without claiming this predicts profit.
    alts.sort(
        key=lambda x: (
            min(abs(x["change"]), 20)
            * math.log10(max(x["volume"], 1))
        ),
        reverse=True
    )

    alts = [x for x in alts if x["symbol"] != major["symbol"]]
    if len(alts) < 2:
        raise RuntimeError("Not enough distinct liquid coins found")

    # Change the altcoin pair across the three daily posts.
    start = ((today * 3) + (slot - 1) * 2) % len(alts)
    chosen_alts = [
        alts[start % len(alts)],
        alts[(start + 1) % len(alts)]
    ]

    selected = [major] + chosen_alts
    if len({x["symbol"] for x in selected}) != 3:
        raise RuntimeError("Could not select three distinct coins")
    return selected


def candles(symbol):
    raw = get("/api/v3/klines", {
        "symbol": symbol, "interval": "1h", "limit": 48
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
        raise RuntimeError(f"Insufficient candle data for {symbol}")
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

    recent = changes[-period:]
    up = sum(max(v, 0) for v in recent) / period
    down = sum(max(-v, 0) for v in recent) / period

    if down == 0:
        return 100.0
    return 100 - 100 / (1 + up / down)


def analyze(row):
    data = candles(row["symbol"])
    closes = [x["close"] for x in data]
    highs = [x["high"] for x in data]
    lows = [x["low"] for x in data]

    last = closes[-1]
    ema20 = ema(closes)[-1]
    strength = rsi(closes)
    support = min(lows[-24:])
    resistance = max(highs[-24:])

    return {
        "row": row,
        "data": data,
        "last": last,
        "ema20": ema20,
        "rsi": strength,
        "support": support,
        "resistance": resistance
    }


def make_chart(analyses):
    # One image containing three independent charts.
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.patch.set_facecolor("#101820")

    for ax, item in zip(axes, analyses):
        row = item["row"]
        data = item["data"]
        symbol = row["symbol"]

        times = [x["time"] for x in data]
        closes = [x["close"] for x in data]
        line = ema(closes)
        last = item["last"]

        ax.set_facecolor("#101820")

        for i, candle in enumerate(data):
            up = candle["close"] >= candle["open"]
            color = "#00c087" if up else "#f6465d"
            ax.vlines(
                times[i], candle["low"], candle["high"],
                color=color, linewidth=1
            )
            bottom = min(candle["open"], candle["close"])
            height = max(
                abs(candle["close"] - candle["open"]),
                last * 0.00005
            )
            ax.bar(
                times[i], height, bottom=bottom,
                width=0.025, color=color
            )

        ax.plot(times, line, color="#f0b90b",
                linewidth=1.5, label="EMA 20")
        ax.axhline(
            item["support"], color="#00c087",
            linestyle="--", linewidth=1, label="Recent low"
        )
        ax.axhline(
            item["resistance"], color="#f6465d",
            linestyle="--", linewidth=1, label="Recent high"
        )

        ax.set_title(
            f"{symbol}  |  Price {price(last)} USDT"
            f"  |  24h {row['change']:+.2f}%"
            f"  |  RSI {item['rsi']:.1f}",
            color="white", loc="left", fontsize=11, pad=9
        )
        ax.grid(True, color="#34404c", alpha=0.5)
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%d %H:%M", tz=timezone.utc)
        )
        ax.tick_params(colors="#b7c5d3", labelsize=8)
        ax.legend(
            loc="upper left", facecolor="#101820",
            labelcolor="white", fontsize=7, ncol=3
        )
        for spine in ax.spines.values():
            spine.set_color("#34404c")

    fig.suptitle(
        "3-COIN MARKET ANALYSIS",
        color="white", fontsize=17, fontweight="bold"
    )
    fig.text(
        0.99, 0.012,
        "Binance market data | Educational only, not financial advice",
        ha="right", color="#b7c5d3", fontsize=8
    )
    fig.tight_layout(rect=[0, 0.025, 1, 0.965])

    path = OUT / "chart.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def make_post(analyses, slot):
    titles = {
        1: "Morning Market Pulse",
        2: "Altcoin Momentum Watch",
        3: "Evening Technical Setups"
    }

    lines = [
        f"📊 {titles[slot]}",
        "",
        "Three coins. Three charts. Key levels to watch:",
        ""
    ]

    for item in analyses:
        row = item["row"]
        last = item["last"]
        ema20 = item["ema20"]
        strength = item["rsi"]
        support = item["support"]
        resistance = item["resistance"]

        trend = "above EMA20" if last >= ema20 else "below EMA20"
        if strength >= 70:
            momentum = "RSI elevated"
        elif strength <= 30:
            momentum = "RSI low"
        else:
            momentum = "RSI neutral zone"

        lines.append(
            f"• {row['symbol']} — {row['change']:+.2f}% (24h). "
            f"Price {trend}; {momentum} ({strength:.0f})."
        )
        lines.append(
            f"  Range: {price(support)}–{price(resistance)} USDT."
        )
        lines.append("")

    lines.extend([
        "Watch for candle-close and volume confirmation near key levels. "
        "These indicators can fail; levels are not guaranteed signals.",
        "",
        "Educational analysis only. Manage risk; avoid excessive leverage.",
        "#Crypto #BinanceSquare"
    ])
    return "\n".join(lines)


def main():
    if not KEY:
        raise RuntimeError("Missing BINANCE_SQUARE_OPENAPI_KEY secret")

    raw_slot = os.getenv("SLOT", "").strip()
    if raw_slot in ("1", "2", "3"):
        slot = int(raw_slot)
    else:
        hour = datetime.now(timezone.utc).hour
        slot = 1 if hour < 6 else 2 if hour < 14 else 3

    rows = market_data()
    selected = select_coins(rows, slot)

    print("Post slot:", slot)
    analyses = []
    for row in selected:
        print("Selected coin:", row["symbol"])
        analyses.append(analyze(row))

    chart = make_chart(analyses)
    post = make_post(analyses, slot)

    (OUT / "post.txt").write_text(post, encoding="utf-8")
    print("Post text:\n", post)
    print("Combined chart:", chart)

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
