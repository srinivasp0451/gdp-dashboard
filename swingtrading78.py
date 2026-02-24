"""
Run this FIRST before the dashboard.
python3 diagnose.py

It tells us exactly what yfinance can and cannot do on your machine.
"""
import yfinance as yf
import sys, time

print(f"Python: {sys.version}")
print(f"yfinance: {yf.__version__}")
print("=" * 60)

tests = [
    ("SPY history",    lambda: yf.Ticker("SPY").history(period="2d")["Close"].iloc[-1]),
    ("AAPL history",   lambda: yf.Ticker("AAPL").history(period="2d")["Close"].iloc[-1]),
    ("^NSEI history",  lambda: yf.Ticker("^NSEI").history(period="2d")["Close"].iloc[-1]),
    ("BTC-USD history",lambda: yf.Ticker("BTC-USD").history(period="2d")["Close"].iloc[-1]),
    ("SPY fast_info",  lambda: yf.Ticker("SPY").fast_info.last_price),
    ("SPY options",    lambda: yf.Ticker("SPY").options[:2]),
    ("AAPL options",   lambda: yf.Ticker("AAPL").options[:2]),
    ("SPY chain",      lambda: len(yf.Ticker("SPY").option_chain(yf.Ticker("SPY").options[0]).calls)),
]

results = {}
for name, fn in tests:
    try:
        time.sleep(1.5)
        val = fn()
        print(f"  ✅ {name}: {val}")
        results[name] = True
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        results[name] = False

print()
print("=" * 60)
print("SUMMARY — what the dashboard should use:")
if results.get("SPY options"):
    print("  ✅ Use yfinance options (SPY, AAPL work)")
else:
    print("  ❌ yfinance options BROKEN on this machine")
if results.get("SPY history"):
    print("  ✅ yfinance history works")
else:
    print("  ❌ yfinance history BROKEN")
print()
print("Paste this output and I will fix only what's actually broken.")
