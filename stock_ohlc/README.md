# Stock OHLC Data Retriever

Retrieves daily OHLC (Open, High, Low, Close) price data for stocks from any global stock exchange using yfinance, going back as far as historical records exist.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the script without arguments to retrieve data for the proof of concept stocks (BHP.AX, RGTI, BATS.L):

```bash
python get_stock_ohlc.py
```

### Custom Tickers

Specify one or more ticker symbols:

```bash
python get_stock_ohlc.py BHP.AX RGTI BATS.L
```

### Ticker Format

- **ASX (Australian Stock Exchange)**: Use `.AX` suffix (e.g., `BHP.AX`)
- **LSE (London Stock Exchange)**: Use `.L` suffix (e.g., `BATS.L`)
- **NASDAQ/NYSE**: Use plain ticker (e.g., `RGTI`, `AAPL`)
- **Other exchanges**: See the script for supported exchange suffixes

## Output

The script generates JSON files in the same directory as the script with the following format:

- Filename: `{TICKER}_{EXCHANGE}_ohlc.json`
- Structure:
  ```json
  {
    "metadata": {
      "ticker_symbol": "BHP.AX",
      "exchange": "ASX",
      "start_date": "1988-01-29",
      "end_date": "2025-11-07",
      "record_count": 9690,
      "retrieved_at": "2025-11-07T13:40:08.848003",
      "data_type": "daily_ohlc"
    },
    "data": {
      "1988-01-29": {
        "Open": 0.39386025071144104,
        "High": 0.39386025071144104,
        "Low": 0.39386025071144104,
        "Close": 0.39386025071144104
      },
      ...
    }
  }
  ```

## Supported Exchanges

The script supports multiple global stock exchanges including:
- ASX (Australian Stock Exchange)
- LSE (London Stock Exchange)
- NASDAQ/NYSE (US exchanges)
- TSE (Tokyo Stock Exchange)
- HKEX (Hong Kong Stock Exchange)
- TSX (Toronto Stock Exchange)
- And many more...

## Error Handling

The script includes comprehensive error handling for:
- Network errors
- Invalid ticker symbols
- Missing or empty data
- Data validation errors

If any error occurs, the script will log detailed error messages and exit with an error code.


