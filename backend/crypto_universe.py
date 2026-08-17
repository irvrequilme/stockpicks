# Liquid crypto pairs, yfinance ticker format.
#
# NOTE: on Yahoo Finance, some tickers collide with unrelated obscure coins
# that share the same 3-4 letter symbol (e.g. plain "ARB-USD" resolves to
# "ARbit", a near-worthless token, not Arbitrum; plain "UNI-USD" resolves to
# "UNICORN Token", not Uniswap). For those, Yahoo disambiguates with a
# CoinMarketCap ID suffix — verified against get_info()['shortName'] below.

CRYPTO_UNIVERSE = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD",
    "DOGE-USD", "AVAX-USD", "LINK-USD", "LTC-USD", "DOT-USD",
    "BCH-USD", "UNI7083-USD", "ATOM-USD", "ETC-USD", "XLM-USD",
    "NEAR-USD", "APT21794-USD", "ARB11841-USD", "OP-USD", "FIL-USD",
    "SUI20947-USD", "INJ-USD",
]

# Search aliases for the dashboard's search bar. Needed because bare short
# codes silently collide with unrelated tickers on Yahoo Finance rather than
# failing cleanly — e.g. "BTC" alone resolves to a Grayscale Bitcoin trust,
# "LINK" to Interlink Electronics (an unrelated small-cap), "ATOM" to
# Atomera Incorporated, "ARB" to an arbitrage ETF. Checking this table first
# means a search for "bitcoin" or "link" reliably lands on the actual coin
# instead of whatever unrelated instrument happens to share that symbol.
CRYPTO_ALIASES = {
    "bitcoin": "BTC-USD", "btc": "BTC-USD",
    "ethereum": "ETH-USD", "eth": "ETH-USD",
    "solana": "SOL-USD", "sol": "SOL-USD",
    "ripple": "XRP-USD", "xrp": "XRP-USD",
    "cardano": "ADA-USD", "ada": "ADA-USD",
    "dogecoin": "DOGE-USD", "doge": "DOGE-USD",
    "avalanche": "AVAX-USD", "avax": "AVAX-USD",
    "chainlink": "LINK-USD", "link": "LINK-USD",
    "litecoin": "LTC-USD", "ltc": "LTC-USD",
    "polkadot": "DOT-USD", "dot": "DOT-USD",
    "bitcoin cash": "BCH-USD", "bch": "BCH-USD",
    "uniswap": "UNI7083-USD", "uni": "UNI7083-USD",
    "cosmos": "ATOM-USD", "atom": "ATOM-USD",
    "ethereum classic": "ETC-USD", "etc": "ETC-USD",
    "stellar": "XLM-USD", "xlm": "XLM-USD",
    "near protocol": "NEAR-USD", "near": "NEAR-USD",
    "aptos": "APT21794-USD", "apt": "APT21794-USD",
    "arbitrum": "ARB11841-USD", "arb": "ARB11841-USD",
    "optimism": "OP-USD", "op": "OP-USD",
    "filecoin": "FIL-USD", "fil": "FIL-USD",
    "sui": "SUI20947-USD",
    "injective": "INJ-USD", "inj": "INJ-USD",
}
