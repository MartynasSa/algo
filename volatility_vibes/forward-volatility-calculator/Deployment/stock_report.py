
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import math
from typing import Optional, Tuple, Dict
from io import StringIO
import warnings
import time
import re
import json
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    print("Warning: beautifulsoup4 not installed. Finviz scraping will be disabled.")
    print("Install with: pip install beautifulsoup4")
warnings.filterwarnings('ignore')

# Configuration
TRADIER_API_TOKEN = "IoKm62KxAubyS2ybtZob7sh39jZE"  # Replace with your Tradier API token
TRADIER_BASE_URL = "https://api.tradier.com/v1"
FRONT_TARGET_DAY = 30  # First leg target DTE (front month should be close to 30 days)
TARGET_DAY = 60  # Second leg target DTE (target should be close to 60 days)
DTE_TOLERANCE = 10  # Allowable deviation from target days (increased for more flexibility)

# Discord Configuration
DISCORD_WEBHOOK_URL = 'https://discordapp.com/api/webhooks/1402749967172636795/ew4ANKYWlZy-aGC6FHPk406ZeVW4_Nc8sObcnwMY4sZmS9110u6svtKZqmhpYay0ycAP'


# In[43]:


def get_sp500_symbols() -> list:
    """Get S&P 500 stock symbols from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # parse the HTML manually with pandas
        tables = pd.read_html(StringIO(response.text))
        df = tables[0]
        symbols = df['Symbol'].tolist()
        # Clean symbols (remove dots for class shares)
        symbols = [s.replace('.', '-') for s in symbols]
        return symbols
    except Exception as e:
        print(f"Error fetching S&P 500 list: {e}")
        # Fallback: return some common symbols
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B']

def get_quote(symbol: str, api_token: str) -> Optional[float]:
    """Get current stock quote."""
    url = f"{TRADIER_BASE_URL}/markets/quotes"
    headers = {'Authorization': f'Bearer {api_token}', 'Accept': 'application/json'}
    params = {'symbols': symbol}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'quotes' in data and 'quote' in data['quotes']:
                quote_data = data['quotes']['quote']
                if isinstance(quote_data, list):
                    quote_data = quote_data[0]
                return float(quote_data.get('last', quote_data.get('bid', 0)))
    except:
        pass
    return None

def get_options_expirations(symbol: str, api_token: str) -> list:
    """Get available expiration dates for a symbol."""
    url = f"{TRADIER_BASE_URL}/markets/options/expirations"
    headers = {'Authorization': f'Bearer {api_token}', 'Accept': 'application/json'}
    params = {'symbol': symbol}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'expirations' in data and 'date' in data['expirations']:
                return data['expirations']['date']
    except:
        pass
    return []

def get_options_chain(symbol: str, expiration: str, api_token: str) -> Optional[dict]:
    """Get options chain for a specific symbol and expiration date."""
    url = f"{TRADIER_BASE_URL}/markets/options/chains"
    headers = {'Authorization': f'Bearer {api_token}', 'Accept': 'application/json'}
    params = {'symbol': symbol, 'expiration': expiration, 'greeks': 'true'}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def get_option_price(opt: dict) -> Optional[float]:
    """Extract option price from option data (use mid price if available, otherwise average of bid/ask)."""
    try:
        # Try mid price first
        mid = opt.get('mid', None)
        if mid is not None:
            return float(mid)

        # Try bid/ask average
        bid = opt.get('bid', None)
        ask = opt.get('ask', None)
        if bid is not None and ask is not None:
            bid_float = float(bid)
            ask_float = float(ask)
            if bid_float > 0 and ask_float > 0:
                return (bid_float + ask_float) / 2.0

        # Try last price
        last = opt.get('last', None)
        if last is not None:
            return float(last)
    except:
        pass
    return None

def get_itm_options_with_price(options_chain: dict, spot_price: float) -> list:
    """Get in-the-money call options with their IV and price. Returns list of (strike, iv, price, option_type)."""
    if 'options' not in options_chain or 'option' not in options_chain['options']:
        return []

    options = options_chain['options']['option']
    if isinstance(options, dict):
        options = [options]

    itm_options = []
    for opt in options:
        try:
            strike = float(opt.get('strike', 0))
            option_type = opt.get('option_type', '').upper()

            # For ITM calls: strike < spot_price
            # For ITM puts: strike > spot_price (but we'll focus on calls)
            if option_type == 'CALL' and strike < spot_price:
                # Get IV
                greeks = opt.get('greeks', {})
                if isinstance(greeks, dict):
                    iv = greeks.get('mid_iv') or greeks.get('bid_iv') or greeks.get('ask_iv') or greeks.get('iv')
                else:
                    iv = opt.get('mid_iv') or opt.get('bid_iv') or opt.get('ask_iv') or opt.get('iv')

                if iv is not None:
                    iv_float = float(iv)
                    if iv_float > 0:
                        # Get option price
                        opt_price = get_option_price(opt)
                        if opt_price is not None and opt_price > 0:
                            itm_options.append((strike, iv_float, opt_price, option_type))
        except:
            continue

    return itm_options

def find_closest_itm_option(options_chain: dict, spot_price: float) -> Optional[Dict]:
    """
    Find the closest ITM call option to the spot price.
    Returns dict with strike, iv, price, or None if not found.
    For ITM calls: strike < spot_price, so we want the highest strike that's still < spot_price.
    """
    if 'options' not in options_chain or 'option' not in options_chain['options']:
        return None

    options = options_chain['options']['option']
    if isinstance(options, dict):
        options = [options]

    best_option = None
    best_strike = -float('inf')  # We want the highest ITM strike (closest to spot but still ITM)

    for opt in options:
        try:
            strike = float(opt.get('strike', 0))
            option_type = opt.get('option_type', '').upper()

            # For ITM calls: strike < spot_price, find the closest (highest) strike below spot
            if option_type == 'CALL' and strike < spot_price and strike > best_strike:
                # Get IV
                greeks = opt.get('greeks', {})
                if isinstance(greeks, dict):
                    iv = greeks.get('mid_iv') or greeks.get('bid_iv') or greeks.get('ask_iv') or greeks.get('iv')
                else:
                    iv = opt.get('mid_iv') or opt.get('bid_iv') or opt.get('ask_iv') or opt.get('iv')

                if iv is not None:
                    iv_float = float(iv)
                    if iv_float > 0:
                        # Get option price
                        opt_price = get_option_price(opt)
                        if opt_price is not None and opt_price > 0:
                            best_option = {
                                'strike': strike,
                                'iv': iv_float,
                                'price': opt_price
                            }
                            best_strike = strike
        except:
            continue

    return best_option

def get_atm_iv(options_chain: dict, spot_price: float) -> Optional[float]:
    """Extract ATM implied volatility from options chain (fallback for backward compatibility)."""
    if 'options' not in options_chain or 'option' not in options_chain['options']:
        return None

    options = options_chain['options']['option']
    if isinstance(options, dict):
        options = [options]

    atm_ivs = []
    for opt in options:
        try:
            strike = float(opt.get('strike', 0))
            if abs(strike - spot_price) / spot_price <= 0.02:  # Within 2% of spot
                greeks = opt.get('greeks', {})
                if isinstance(greeks, dict):
                    iv = greeks.get('mid_iv') or greeks.get('bid_iv') or greeks.get('ask_iv') or greeks.get('iv')
                else:
                    iv = opt.get('mid_iv') or opt.get('bid_iv') or opt.get('ask_iv') or opt.get('iv')

                if iv is not None:
                    iv_float = float(iv)
                    if iv_float > 0:
                        atm_ivs.append(iv_float)
        except:
            continue

    return np.mean(atm_ivs) if atm_ivs else None

def calculate_dte(expiration_date: str) -> int:
    """Calculate days to expiration from today."""
    exp_date = datetime.strptime(expiration_date, '%Y-%m-%d')
    return (exp_date - datetime.now()).days

def calculate_forward_factor(dte1: float, iv1: float, dte2: float, iv2: float) -> Optional[float]:
    """Calculate forward factor. Returns None if invalid."""
    if dte1 < 0 or dte2 < 0 or dte2 <= dte1 or iv1 < 0 or iv2 < 0:
        return None

    T1, T2 = dte1 / 365.0, dte2 / 365.0
    s1, s2 = iv1 / 100.0, iv2 / 100.0
    tv1, tv2 = (s1 ** 2) * T1, (s2 ** 2) * T2

    fwd_var = (tv2 - tv1) / (T2 - T1)
    if fwd_var < 0:
        return None

    fwd_sigma = math.sqrt(fwd_var)
    if fwd_sigma == 0.0:
        return None

    return (s1 - fwd_sigma) / fwd_sigma

def get_earnings_date_finviz(symbol: str) -> Optional[str]:
    """Get earnings date from Finviz by scraping the quote page. Returns date string in YYYY-MM-DD format or None."""
    if not BEAUTIFULSOUP_AVAILABLE:
        return None

    try:
        url = f"https://finviz.com/quote.ashx?t={symbol}&p=d"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find earnings date in the page - it's typically in a table
            # Look for text containing earnings date pattern like "Oct 21 BMO" or "Nov 06 AMC"
            text = soup.get_text()

            # Pattern to match earnings dates like "Oct 21 BMO", "Nov 06 AMC", "Oct 21, 2025"
            # Look for patterns like: Month Day (BMO/AMC) or Month Day, Year
            patterns = [
                r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})\s+(BMO|AMC)',  # Oct 21 BMO
                r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})',  # Oct 21, 2025
                r'Earnings[:\s]+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})',  # Earnings: Oct 21
            ]

            month_map = {
                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
            }

            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        month_str = match.group(1).capitalize()
                        day = int(match.group(2))
                        month = month_map.get(month_str)

                        if month:
                            current_year = datetime.now().year
                            # If pattern includes year, use it; otherwise assume current or next year
                            if len(match.groups()) > 2 and match.group(3).isdigit():
                                year = int(match.group(3))
                            else:
                                year = current_year
                                # If the date is in the past, try next year
                                test_date = datetime(year, month, day).date()
                                if test_date < datetime.now().date():
                                    year = current_year + 1

                            earnings_date = datetime(year, month, day).date()
                            if earnings_date >= datetime.now().date():
                                return earnings_date.strftime('%Y-%m-%d')
                    except (ValueError, IndexError):
                        continue

            # Also try finding in specific table cells
            # Look for table cells containing earnings information
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    for i, cell in enumerate(cells):
                        cell_text = cell.get_text().strip()
                        if 'earnings' in cell_text.lower() and i + 1 < len(cells):
                            next_cell = cells[i + 1].get_text().strip()
                            # Try to parse the next cell which might contain the date
                            for pattern in patterns:
                                match = re.search(pattern, next_cell, re.IGNORECASE)
                                if match:
                                    try:
                                        month_str = match.group(1).capitalize()
                                        day = int(match.group(2))
                                        month = month_map.get(month_str)
                                        if month:
                                            current_year = datetime.now().year
                                            year = current_year
                                            test_date = datetime(year, month, day).date()
                                            if test_date < datetime.now().date():
                                                year = current_year + 1
                                            earnings_date = datetime(year, month, day).date()
                                            if earnings_date >= datetime.now().date():
                                                return earnings_date.strftime('%Y-%m-%d')
                                    except (ValueError, IndexError):
                                        continue
    except Exception:
        pass

    return None

def get_earnings_date(symbol: str) -> Optional[str]:
    """
    Get next earnings announcement date from Finviz.
    Returns date string in YYYY-MM-DD format or None.
    """
    return get_earnings_date_finviz(symbol)

# Track suspicious dates to avoid filtering on unreliable data
_suspicious_dates_cache = set()

def is_suspicious_earnings_date(earnings_date_str: str) -> bool:
    """Check if earnings date seems suspicious (e.g., default placeholder dates)."""
    if not earnings_date_str:
        return True

    # Common suspicious/default dates from FMP
    suspicious_dates = {'2025-11-01', '2024-11-01', '1900-01-01', '2000-01-01'}
    if earnings_date_str in suspicious_dates:
        return True

    # Check if date is in the past (shouldn't happen for upcoming earnings)
    try:
        earnings_date = datetime.strptime(earnings_date_str, '%Y-%m-%d').date()
        if earnings_date < datetime.now().date():
            return True
    except:
        pass

    return False

def has_earnings_before_date(symbol: str, front_exp_date: str) -> bool:
    """
    Check if stock has earnings announcement before the front expiration date.
    Returns True if earnings is before front date (should exclude), False otherwise.
    """
    earnings_date_str = get_earnings_date(symbol)
    if not earnings_date_str:
        return False  # If we can't get earnings date, don't exclude

    try:
        earnings_date = datetime.strptime(earnings_date_str, '%Y-%m-%d').date()
        front_date = datetime.strptime(front_exp_date, '%Y-%m-%d').date()
        return earnings_date < front_date
    except (ValueError, TypeError):
        return False  # If date parsing fails, don't exclude


# In[45]:


def process_stock(symbol: str, api_token: str) -> Optional[Dict]:
    """Process a single stock and return forward factor data."""
    try:
        # Get stock price
        spot_price = get_quote(symbol, api_token)
        if not spot_price or spot_price <= 0:
            return None

        # Get expiration dates
        expirations = get_options_expirations(symbol, api_token)
        if not expirations:
            return None

        # Find appropriate expirations
        exp_dates = [(exp, calculate_dte(exp)) for exp in expirations if calculate_dte(exp) >= 0]
        exp_dates.sort(key=lambda x: x[1])

        if len(exp_dates) < 2:
            return None

        # Find front expiration closest to 30 days (first leg)
        front_exp = min(exp_dates, key=lambda x: abs(x[1] - FRONT_TARGET_DAY))

        # Validate front expiration is within reasonable range (allow up to 20 days deviation)
        if abs(front_exp[1] - FRONT_TARGET_DAY) > 20:
            return None

        # Check earnings EARLY - before expensive options chain calls
        # Get earnings date once and check against front expiration
        earnings_date_str = get_earnings_date(symbol)
        if earnings_date_str:
            # Skip filtering if date seems suspicious/unreliable
            if is_suspicious_earnings_date(earnings_date_str):
                # Don't exclude based on suspicious data
                pass
            else:
                try:
                    earnings_date = datetime.strptime(earnings_date_str, '%Y-%m-%d').date()
                    front_date = datetime.strptime(front_exp[0], '%Y-%m-%d').date()
                    if earnings_date < front_date:
                        return None  # Exclude stock with earnings before front date
                except (ValueError, TypeError):
                    pass  # If date parsing fails, continue processing

        # Find target expiration closest to 60 days (second leg)
        # Only consider expirations after the front expiration
        future_exp_dates = [(exp, dte) for exp, dte in exp_dates if dte > front_exp[1]]
        if len(future_exp_dates) == 0:
            return None

        target_exp = min(future_exp_dates, key=lambda x: abs(x[1] - TARGET_DAY))

        # Validate target expiration is within reasonable range (allow up to 20 days deviation)
        if abs(target_exp[1] - TARGET_DAY) > 20:
            return None

        # Ensure target is after front
        if target_exp[1] <= front_exp[1]:
            return None

        # Get options chains
        front_chain = get_options_chain(symbol, front_exp[0], api_token)
        target_chain = get_options_chain(symbol, target_exp[0], api_token)

        if not front_chain or not target_chain:
            return None

        # Find closest ITM options for each leg
        front_option = find_closest_itm_option(front_chain, spot_price)
        target_option = find_closest_itm_option(target_chain, spot_price)

        if front_option and target_option:
            # Use closest ITM options
            front_iv = front_option['iv']
            front_price = front_option['price']
            front_strike = front_option['strike']

            target_iv = target_option['iv']
            target_price = target_option['price']
            target_strike = target_option['strike']

            # Use front strike as the primary strike (or could use average)
            strike = front_strike
        else:
            # Fallback to ATM IV if no ITM options found
            front_iv = get_atm_iv(front_chain, spot_price)
            target_iv = get_atm_iv(target_chain, spot_price)
            front_price = None
            target_price = None
            strike = None
            front_strike = None
            target_strike = None

        if not front_iv or not target_iv:
            return None

        # Calculate forward factor
        ff = calculate_forward_factor(front_exp[1], front_iv, target_exp[1], target_iv)
        if ff is None:
            return None

        return {
            'Symbol': symbol,
            'Price': spot_price,
            'Strike': strike,
            'Front_Strike': front_strike,
            'Target_Strike': target_strike,
            'Front_Date': front_exp[0],
            'Front_DTE': front_exp[1],
            'Front_IV': front_iv,
            'Front_Price': front_price,
            'Target_Date': target_exp[0],
            'Target_DTE': target_exp[1],
            'Target_IV': target_iv,
            'Target_Price': target_price,
            'Forward_Factor': ff,
            'Forward_Factor_Pct': ff * 100
        }
    except Exception as e:
        return None


# In[46]:


def run_analysis():
    """Main function to run the S&P 500 forward factor analysis."""
    # Get S&P 500 symbols
    print("Fetching S&P 500 stock symbols...")
    sp500_symbols = get_sp500_symbols()
    print(f"Found {len(sp500_symbols)} symbols")
    print(f"\nProcessing stocks... (this may take a while)")
    print("=" * 80)

    # Process all stocks with optimized timing
    results = []
    failed = 0
    error_counts = {}  # Track different types of errors
    start_time = time.time()

    print(f"\n{'='*80}")
    print("Starting processing of all S&P 500 stocks...\n")
    print(f"{'='*80}\n")

    # Process all S&P 500 stocks
    test_symbols = sp500_symbols
    for i, symbol in enumerate(test_symbols, 1):
        try:
            result = process_stock(symbol, TRADIER_API_TOKEN)
            if result:
                results.append(result)
            else:
                failed += 1
                error_counts['no_result'] = error_counts.get('no_result', 0) + 1
        except Exception as e:
            failed += 1
            error_type = type(e).__name__
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            continue

        # Progress reporting every 30 stocks
        if i % 30 == 0 or i == len(test_symbols):
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            print(f"Processed {i}/{len(test_symbols)} stocks... ({len(results)} successful, {failed} failed)")
            print(f"  Rate: {rate:.1f} stocks/sec")

        # Small delay to avoid overwhelming the API
        time.sleep(0.1)

    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Completed in {total_time/60:.1f} minutes")
    print(f"  Successful: {len(results)}")
    print(f"  Failed: {failed}")
    print(f"  Success rate: {len(results)/(len(results)+failed)*100:.1f}%" if (len(results)+failed) > 0 else "  Success rate: 0%")

    if error_counts:
        print(f"\nError breakdown:")
        for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count}")

    # Create DataFrame and sort by Forward Factor (highest first)
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('Forward_Factor', ascending=False)

        # Reorder columns for better display
        display_cols = ['Symbol', 'Price', 'Strike']
        display_cols.extend([
            'Front_Date', 'Front_DTE', 'Front_IV',
            'Target_Date', 'Target_DTE', 'Target_IV',
            'Forward_Factor_Pct'
        ])

        # Only include columns that exist
        available_cols = [col for col in display_cols if col in df.columns]
        df_display = df[available_cols].copy()

        # Rename columns
        column_mapping = {
            'Symbol': 'Symbol',
            'Price': 'Price',
            'Strike': 'Strike',
            'Front_Date': 'Front Date',
            'Front_DTE': 'Front DTE',
            'Front_IV': 'Front IV (%)',
            'Target_Date': 'Target Date',
            'Target_DTE': 'Target DTE',
            'Target_IV': 'Target IV (%)',
            'Forward_Factor_Pct': 'Forward Factor (%)',
            'Forward_Factor': 'Forward Factor'
        }
        df_display.columns = [column_mapping.get(col, col) for col in df_display.columns]

        print("\n" + "=" * 80)
        print("RESULTS - Sorted by Highest Forward Factor")
        print("=" * 80)
        print(f"\nTop 20 Stocks:")
        print(df_display.head(20).to_string(index=False))

        print(f"\n\nAll {len(df_display)} stocks with forward factors:")
        print(df_display.to_string(index=False))
        
        # Send top 20 to Discord after processing completes
        print("\n" + "=" * 80)
        print("Sending top 20 results to Discord...")
        print("=" * 80)

        df_for_discord = pd.DataFrame(results)
        df_for_discord = df_for_discord.sort_values('Forward_Factor', ascending=False)

        total_processed = len(sp500_symbols)
        successful = len(results)
        failed_count = failed

        send_to_discord(DISCORD_WEBHOOK_URL, df_for_discord, total_processed, successful, failed_count)
        
        return {
            'statusCode': 200,
            'body': {
                'success': True,
                'total_processed': total_processed,
                'successful': successful,
                'failed': failed_count,
                'results_count': len(results),
                'top_20_symbols': df_for_discord.head(20)['Symbol'].tolist()
            }
        }
    else:
        print("\nNo results found. Check API token and network connection.")
        return {
            'statusCode': 200,
            'body': {
                'success': False,
                'message': 'No results found. Check API token and network connection.',
                'total_processed': len(sp500_symbols),
                'successful': 0,
                'failed': failed
            }
        }


# In[55]:

# Send top 20 results to Discord
def send_to_discord(webhook_url: str, top_results: pd.DataFrame, total_processed: int, successful: int, failed: int):
    """Send top 20 results to Discord webhook."""
    try:
        # Format the message
        message_parts = []
        message_parts.append("🚀 **S&P 500 Forward Factor Analysis - Top 20 Results**\n")
        message_parts.append(f"📊 **Processing Summary:**\n")
        message_parts.append(f"• Total Processed: {total_processed}\n")
        message_parts.append(f"• Successful: {successful}\n")
        message_parts.append(f"• Failed: {failed}\n")
        message_parts.append(f"• Success Rate: {(successful/total_processed*100):.1f}%\n\n")
        message_parts.append("🏆 **Top 20 Stocks by Forward Factor:**\n")
        message_parts.append("```\n")

        # Format top 20 results
        top_20 = top_results.head(20).reset_index(drop=True)
        for idx, (_, row) in enumerate(top_20.iterrows(), 1):
            symbol = row.get('Symbol', 'N/A')
            front_date = row.get('Front_Date', 'N/A')
            front_dte = row.get('Front_DTE', 'N/A')
            front_iv = row.get('Front_IV', 'N/A')
            target_date = row.get('Target_Date', 'N/A')
            target_dte = row.get('Target_DTE', 'N/A')
            target_iv = row.get('Target_IV', 'N/A')
            ff_pct = row.get('Forward_Factor_Pct', 0)

            # Format values
            front_dte_str = f"{front_dte}" if front_dte != 'N/A' and front_dte is not None else "N/A"
            front_iv_str = f"{front_iv:.4f}" if front_iv != 'N/A' and front_iv is not None else "N/A"
            target_dte_str = f"{target_dte}" if target_dte != 'N/A' and target_dte is not None else "N/A"
            target_iv_str = f"{target_iv:.4f}" if target_iv != 'N/A' and target_iv is not None else "N/A"

            line = f"{idx:2d}. {symbol:5s} | Front: {str(front_date):>10s} DTE:{front_dte_str:>3s} IV:{front_iv_str:>6s} | "
            line += f"Target: {str(target_date):>10s} DTE:{target_dte_str:>3s} IV:{target_iv_str:>6s} | "
            line += f"FF: {ff_pct:>7.2f}%"
            message_parts.append(line + "\n")

        message_parts.append("```\n")
        message_parts.append(f"\n📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Discord has a 2000 character limit per message, so we might need to split
        full_message = ''.join(message_parts)

        if len(full_message) > 2000:
                        # Split into multiple messages if too long
            messages = []
            current_msg = message_parts[0] + message_parts[1] + message_parts[2] + message_parts[3] + message_parts[4] + message_parts[5] + message_parts[6]
            current_msg += "```\n"

            top_20 = top_results.head(20).reset_index(drop=True)
            for idx, (_, row) in enumerate(top_20.iterrows(), 1):
                symbol = row.get('Symbol', 'N/A')
                front_date = row.get('Front_Date', 'N/A')
                front_dte = row.get('Front_DTE', 'N/A')
                front_iv = row.get('Front_IV', 'N/A')
                target_date = row.get('Target_Date', 'N/A')
                target_dte = row.get('Target_DTE', 'N/A')
                target_iv = row.get('Target_IV', 'N/A')
                ff_pct = row.get('Forward_Factor_Pct', 0)

                # Format values (compact for split messages)
                front_dte_str = f"{front_dte}" if front_dte != 'N/A' and front_dte is not None else "N/A"
                front_iv_str = f"{front_iv:.4f}" if front_iv != 'N/A' and front_iv is not None else "N/A"
                target_dte_str = f"{target_dte}" if target_dte != 'N/A' and target_dte is not None else "N/A"
                target_iv_str = f"{target_iv:.4f}" if target_iv != 'N/A' and target_iv is not None else "N/A"

                line = f"{idx:2d}. {symbol:5s} | F:{str(front_date):>10s} {front_dte_str:>3s} {front_iv_str:>6s} | "
                line += f"T:{str(target_date):>10s} {target_dte_str:>3s} {target_iv_str:>6s} | FF:{ff_pct:>6.2f}%"
                new_line = line + "\n"

                if len(current_msg + new_line + "```\n") > 1900:
                    current_msg += "```\n"
                    messages.append(current_msg)
                    current_msg = "```\n" + new_line
                else:
                    current_msg += new_line

            current_msg += "```\n"
            current_msg += f"\n📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            messages.append(current_msg)
        else:
            messages = [full_message]

        # Send each message to Discord
        for msg in messages:
            payload = {
                'content': msg
            }
            response = requests.post(webhook_url, json=payload, timeout=10)
            if response.status_code == 204:
                print(f"✓ Successfully sent message to Discord ({len(msg)} chars)")
            else:
                print(f"✗ Failed to send to Discord: Status {response.status_code}")
                print(f"  Response: {response.text}")

        print(f"\n✓ Discord notification complete!")
        return True
    except Exception as e:
        print(f"\n✗ Error sending to Discord: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    
    Args:
        event: Lambda event object (can be empty dict for scheduled events)
        context: Lambda context object
    
    Returns:
        dict: Response with statusCode and body
    """
    try:
        print(f"Lambda function started at {datetime.now().isoformat()}")
        print(f"Event: {json.dumps(event)}")
        
        # Run the analysis
        result = run_analysis()
        
        # Ensure body is JSON serializable
        if isinstance(result.get('body'), dict):
            return {
                'statusCode': result.get('statusCode', 200),
                'body': json.dumps(result.get('body', {}))
            }
        else:
            return result
            
    except Exception as e:
        error_msg = f"Error in lambda_handler: {type(e).__name__}: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'success': False,
                'error': error_msg
            })
        }


# For local testing (when run directly, not in Lambda)
if __name__ == "__main__":
    result = run_analysis()
    if isinstance(result, dict) and 'body' in result:
        print(f"\nExecution completed with status: {result.get('statusCode')}")
        print(f"Results: {json.dumps(result['body'], indent=2)}")

