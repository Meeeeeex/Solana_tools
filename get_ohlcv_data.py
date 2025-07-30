# Dependencies to install:

# $ python -m pip install requests

import requests

url = "https://solana-gateway.moralis.io/token/mainnet/pairs/3vLYPfkGHpKk7v8ETkzq3Pqt9wHi6CJ1bgK2vhjL5RRA/ohlcv?timeframe=1min&currency=usd&fromDate=2025-01-25&toDate=2025-02-02&limit=100"

headers = {
  "Accept": "application/json",
  "X-API-Key": ""
}

response = requests.request("GET", url, headers=headers)

print(response.text)
