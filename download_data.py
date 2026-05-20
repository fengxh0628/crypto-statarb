"""Download script: 6 years of 5min OHLCV + funding rate for top 80 coins.

Usage:
    python3 -m cst.download_data

Downloads from Binance Futures (USD-M perpetual contracts).
Handles:
  - Auto-detect top 80 coins by volume
  - Skip coins that weren't listed yet in early periods
  - Resume from where it left off (skip existing files)
  - Rate limiting

Output:
    ./data/
        BTCUSDT_5m.csv
        ETHUSDT_5m.csv
        ...
        BTCUSDT_funding.csv
        ETHUSDT_funding.csv
        ...
"""

from pathlib import Path

from cst.data.download import (
    download_ohlcv,
    download_funding_rate,
    get_all_perpetual_symbols,
)


def main():
    output_dir = "./data"
    timeframe = "5m"
    since = "2020-01-01"
    until = "2026-05-20"

    print("=" * 60)
    print("  Crypto Stat Arb - Data Download")
    print("=" * 60)
    print(f"  Period: {since} to {until}")
    print(f"  Timeframe: {timeframe}")
    print(f"  Output: {output_dir}/")
    print()

    # 获取所有 USDT 永续合约（消除幸存者偏差）
    print("[1] Fetching all USDT perpetual symbols...")
    symbols = get_all_perpetual_symbols()
    print(f"    Got {len(symbols)} symbols")
    print()

    # 检查已有文件，支持断点续传
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    existing_ohlcv = set(f.stem for f in output_path.glob("*_5m.csv"))
    existing_funding = set(f.stem for f in output_path.glob("*_funding.csv"))

    print(f"[2] Downloading OHLCV + Funding Rate...")
    print(f"    Already have: {len(existing_ohlcv)} OHLCV, {len(existing_funding)} funding")
    print()

    success_ohlcv = 0
    success_funding = 0
    failed = []

    for i, symbol in enumerate(symbols):
        sym_clean = symbol.replace("/", "")
        print(f"--- [{i+1}/{len(symbols)}] {symbol} ---")

        # OHLCV
        ohlcv_stem = f"{sym_clean}_{timeframe}"
        if ohlcv_stem in existing_ohlcv:
            print(f"  OHLCV: already exists, skipping")
        else:
            try:
                p = download_ohlcv(symbol, timeframe, since, until, output_dir)
                if p:
                    success_ohlcv += 1
                else:
                    failed.append((symbol, "ohlcv", "no data"))
            except Exception as e:
                print(f"  OHLCV ERROR: {e}")
                failed.append((symbol, "ohlcv", str(e)))

        # Funding rate
        funding_stem = f"{sym_clean}_funding"
        if funding_stem in existing_funding:
            print(f"  Funding: already exists, skipping")
        else:
            try:
                p = download_funding_rate(symbol, since, until, output_dir)
                if p:
                    success_funding += 1
                else:
                    failed.append((symbol, "funding", "no data"))
            except Exception as e:
                print(f"  Funding ERROR: {e}")
                failed.append((symbol, "funding", str(e)))

        print()

    # 汇总
    print("=" * 60)
    print("  Download Complete")
    print("=" * 60)
    print(f"  OHLCV downloaded:   {success_ohlcv} new (+ {len(existing_ohlcv)} existing)")
    print(f"  Funding downloaded: {success_funding} new (+ {len(existing_funding)} existing)")
    if failed:
        print(f"  Failed: {len(failed)}")
        for sym, dtype, err in failed[:10]:
            print(f"    {sym} ({dtype}): {err}")
        if len(failed) > 10:
            print(f"    ... and {len(failed)-10} more")
    print()
    print(f"  Data saved to: {output_dir}/")
    print("  Run backtest with: python3 -m cst.run_real")


if __name__ == "__main__":
    main()
