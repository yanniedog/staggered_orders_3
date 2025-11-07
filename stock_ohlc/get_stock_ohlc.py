#!/usr/bin/env python3
"""
Global Stock OHLC Data Retriever

Retrieves daily OHLC (Open, High, Low, Close) price data for stocks from any
global stock exchange using yfinance, going back as far as historical records exist.
Outputs data in JSON format with metadata.
"""

import yfinance as yf
import pandas as pd
import json
import sys
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def extract_exchange_info(ticker_symbol: str) -> Tuple[str, str]:
    """
    Extract exchange information from ticker symbol.
    
    Args:
        ticker_symbol: Stock ticker symbol (e.g., 'BHP.AX', 'RGTI', 'BATS.L')
    
    Returns:
        Tuple of (exchange_name, base_ticker)
    """
    exchange_map = {
        '.AX': 'ASX',
        '.L': 'LSE',
        '.T': 'TSE',  # Tokyo Stock Exchange
        '.HK': 'HKEX',  # Hong Kong Stock Exchange
        '.TO': 'TSX',  # Toronto Stock Exchange
        '.DE': 'XETR',  # Xetra (Germany)
        '.PA': 'EPA',  # Euronext Paris
        '.AS': 'AMS',  # Euronext Amsterdam
        '.BR': 'EBR',  # Euronext Brussels
        '.MC': 'BME',  # Bolsas y Mercados Españoles
        '.SS': 'SSE',  # Shanghai Stock Exchange
        '.SZ': 'SZSE',  # Shenzhen Stock Exchange
        '.SA': 'BVMF',  # B3 (Brazil)
        '.MX': 'BMV',  # Bolsa Mexicana de Valores
        '.JK': 'IDX',  # Indonesia Stock Exchange
    }
    
    for suffix, exchange in exchange_map.items():
        if ticker_symbol.endswith(suffix):
            base_ticker = ticker_symbol[:-len(suffix)]
            return exchange, base_ticker
    
    # Default to US exchanges (NASDAQ/NYSE) if no suffix
    if '.' not in ticker_symbol:
        return 'NASDAQ/NYSE', ticker_symbol
    
    return 'UNKNOWN', ticker_symbol


def fetch_ohlc_data(ticker_symbol: str) -> Optional[pd.DataFrame]:
    """
    Fetch OHLC data for a given ticker symbol using yfinance.
    
    Args:
        ticker_symbol: Stock ticker symbol
    
    Returns:
        DataFrame with OHLC data, or None if retrieval fails
    """
    try:
        logger.info(f"Creating Ticker object for {ticker_symbol}...")
        ticker = yf.Ticker(ticker_symbol)
        
        logger.info(f"Fetching historical data for {ticker_symbol} (period=max)...")
        hist = ticker.history(period="max")
        
        if hist.empty:
            logger.error(f"No historical data found for {ticker_symbol}")
            return None
        
        # Extract only OHLC columns
        if not all(col in hist.columns for col in ['Open', 'High', 'Low', 'Close']):
            missing_cols = [col for col in ['Open', 'High', 'Low', 'Close'] if col not in hist.columns]
            logger.error(f"Missing required columns for {ticker_symbol}: {missing_cols}")
            return None
        
        ohlc_data = hist[['Open', 'High', 'Low', 'Close']].copy()
        
        # Validate data
        if ohlc_data.isnull().all().all():
            logger.error(f"All OHLC data is null for {ticker_symbol}")
            return None
        
        logger.info(f"Successfully retrieved {len(ohlc_data)} records for {ticker_symbol}")
        return ohlc_data
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker_symbol}: {str(e)}", exc_info=True)
        return None


def convert_to_json(ohlc_data: pd.DataFrame, ticker_symbol: str, 
                   exchange: str) -> Dict:
    """
    Convert OHLC DataFrame to JSON format with metadata.
    
    Args:
        ohlc_data: DataFrame with OHLC data indexed by date
        ticker_symbol: Stock ticker symbol
        exchange: Exchange name
    
    Returns:
        Dictionary with metadata and OHLC data
    """
    # Convert DataFrame to dictionary with ISO date strings as keys
    data_dict = {}
    for date, row in ohlc_data.iterrows():
        date_str = date.strftime('%Y-%m-%d') if isinstance(date, pd.Timestamp) else str(date)
        data_dict[date_str] = {
            'Open': float(row['Open']) if pd.notna(row['Open']) else None,
            'High': float(row['High']) if pd.notna(row['High']) else None,
            'Low': float(row['Low']) if pd.notna(row['Low']) else None,
            'Close': float(row['Close']) if pd.notna(row['Close']) else None,
        }
    
    # Get date range
    dates = sorted(data_dict.keys())
    start_date = dates[0] if dates else None
    end_date = dates[-1] if dates else None
    
    # Create metadata
    metadata = {
        'ticker_symbol': ticker_symbol,
        'exchange': exchange,
        'start_date': start_date,
        'end_date': end_date,
        'record_count': len(data_dict),
        'retrieved_at': datetime.now().isoformat(),
        'data_type': 'daily_ohlc'
    }
    
    return {
        'metadata': metadata,
        'data': data_dict
    }


def save_ohlc_to_json(ticker_symbol: str, output_file: Optional[str] = None) -> bool:
    """
    Fetch OHLC data for a ticker and save to JSON file.
    
    Args:
        ticker_symbol: Stock ticker symbol
        output_file: Optional output filename. If None, generates from ticker symbol
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Extract exchange information
        exchange, base_ticker = extract_exchange_info(ticker_symbol)
        logger.info(f"Processing {ticker_symbol} ({exchange})")
        
        # Fetch OHLC data
        ohlc_data = fetch_ohlc_data(ticker_symbol)
        if ohlc_data is None:
            logger.error(f"Failed to retrieve data for {ticker_symbol}")
            return False
        
        # Validate we have data
        if len(ohlc_data) == 0:
            logger.error(f"No data records found for {ticker_symbol}")
            return False
        
        # Convert to JSON format
        json_data = convert_to_json(ohlc_data, ticker_symbol, exchange)
        
        # Generate output filename if not provided
        if output_file is None:
            # Use exchange name in filename for better clarity
            if exchange != 'UNKNOWN':
                # Use exchange name if available (use first part if multiple like 'NASDAQ/NYSE')
                exchange_short = exchange.split('/')[0] if '/' in exchange else exchange
                filename = f"{base_ticker}_{exchange_short}_ohlc.json"
            else:
                safe_ticker = ticker_symbol.replace('.', '_')
                filename = f"{safe_ticker}_ohlc.json"
            
            # Save to the same directory as the script
            script_dir = Path(__file__).parent
            output_file = script_dir / filename
        else:
            # If output_file is provided, ensure it's a Path object
            output_file = Path(output_file)
        
        # Save to file
        logger.info(f"Saving data to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully saved {json_data['metadata']['record_count']} records "
                   f"({json_data['metadata']['start_date']} to {json_data['metadata']['end_date']}) "
                   f"to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {ticker_symbol}: {str(e)}", exc_info=True)
        return False


def main():
    """Main function to handle command-line interface and batch processing."""
    parser = argparse.ArgumentParser(
        description='Retrieve daily OHLC data for stocks from global exchanges',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python get_stock_ohlc.py BHP.AX
  python get_stock_ohlc.py RGTI BATS.L
  python get_stock_ohlc.py --tickers BHP.AX RGTI BATS.L
        """
    )
    parser.add_argument(
        'tickers',
        nargs='*',
        help='Stock ticker symbols to retrieve (e.g., BHP.AX, RGTI, BATS.L)'
    )
    parser.add_argument(
        '--tickers',
        nargs='+',
        dest='tickers_list',
        help='Alternative way to specify ticker symbols'
    )
    
    args = parser.parse_args()
    
    # Combine ticker arguments
    tickers = args.tickers or args.tickers_list or []
    
    # If no tickers provided, run proof of concept
    if not tickers:
        logger.info("No tickers specified, running proof of concept stocks...")
        tickers = ['BHP.AX', 'RGTI', 'BATS.L']
    
    # Process each ticker
    success_count = 0
    fail_count = 0
    failed_tickers = []
    
    for ticker in tickers:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing ticker: {ticker}")
        logger.info(f"{'='*60}")
        
        if save_ohlc_to_json(ticker):
            success_count += 1
        else:
            fail_count += 1
            failed_tickers.append(ticker)
            logger.error(f"Failed to process {ticker}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing complete: {success_count} succeeded, {fail_count} failed")
    if failed_tickers:
        logger.error(f"Failed tickers: {', '.join(failed_tickers)}")
    logger.info(f"{'='*60}")
    
    # Abort on error per user rules
    if fail_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()

