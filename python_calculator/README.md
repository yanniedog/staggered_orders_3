# Staggered Order Ladder Calculator

Python-based calculator for generating staggered buy/sell order ladders with profit target optimization.

## Features

- **Exponential Allocation Strategy**: Automatically allocates more budget to lower buy prices and more quantity to higher sell prices
- **Profit Target Optimization**: Calculates order ladders to achieve a specified profit percentage
- **Multiple Output Formats**: Generates Excel spreadsheets and PDF reports with visualizations
- **Command-Line Interface**: Supports both interactive and non-interactive modes

## Installation

Install required dependencies:

```bash
pip install pandas numpy matplotlib openpyxl
```

Or install from the project root requirements.txt:

```bash
pip install -r ../requirements.txt
```

## Usage

### Interactive Mode

Run the script without arguments for interactive mode:

```bash
python staggered_ladder.py
```

You'll be prompted for:
- Budget (required)
- Current price (required)
- Profit target (default: 75%)
- Number of orders (default: 10)
- Buy price range percentage (default: 30%)

### Command-Line Mode

Run with command-line arguments:

```bash
# Minimal input (uses defaults for profit target and number of orders)
python staggered_ladder.py --budget 10000 --price 50

# Customize all parameters
python staggered_ladder.py --budget 10000 --price 50 --profit-target 75 --num-rungs 15 --price-range 30

# Add custom output prefix
python staggered_ladder.py --budget 10000 --price 50 --output-prefix BTC
```

### Command-Line Arguments

- `--budget`: Total budget for buy orders (required if not using interactive mode)
- `--price`: Current price (required if not using interactive mode)
- `--profit-target`: Target profit percentage (default: 75%)
- `--num-rungs`: Number of orders to place (default: 10)
- `--price-range`: Buy price range percentage (default: 30%)
- `--output-prefix`: Optional prefix for output filenames

## Output

The script generates two files in the `output/` directory:

1. **Excel File** (`staggered_ladder_YYYYMMDD_HHMMSS.xlsx`):
   - Summary statistics
   - Detailed buy order table
   - Detailed sell order table
   - All values formatted for easy reading

2. **PDF File** (`staggered_ladder_YYYYMMDD_HHMMSS.pdf`):
   - Summary statistics
   - Order tables
   - Buy ladder visualization (price and quantity)
   - Sell ladder visualization (price and quantity)
   - Combined buy/sell visualization

## How It Works

1. **Buy Orders**: Creates a ladder of buy orders below the current price, with exponential allocation (more quantity at lower prices)

2. **Sell Orders**: Creates a ladder of sell orders above the buy orders, with exponential allocation (more quantity at higher prices)

3. **Profit Target**: Adjusts sell prices to achieve the exact profit target while maintaining volume matching (total buy quantity = total sell quantity)

4. **Price Validation**: Ensures all sell prices exceed buy prices and maintains a gap between the highest buy and lowest sell orders

## Example

```bash
python staggered_ladder.py --budget 13000 --price 148 --profit-target 64 --num-rungs 13
```

This will:
- Use $13,000 budget
- Set current price at $148
- Target 64% profit
- Create 13 buy and 13 sell orders
- Generate Excel and PDF outputs in the `output/` directory

## Notes

- The calculator uses exponential allocation by default (buy more at lower prices, sell more at higher prices)
- Profit mode is always "overall" (entire strategy returns target profit)
- All outputs are saved with timestamps to prevent overwriting
- The output directory is created automatically if it doesn't exist

