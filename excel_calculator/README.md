# Excel Calculator Module

This directory contains all files related to the Excel-based staggered ladder calculator.

## Files

- **`create_excel_calculator.py`** - Main script that generates Excel workbooks with formulas for calculating staggered order ladders. Creates a comprehensive Excel calculator with multiple sheets (Calculator, Helpers, Charts).

- **`visualize_xlsx.py`** - Utility script for visualizing and analyzing Excel files. Can extract charts, formulas, and data from XLSX files.

- **`staggered_ladder_calculator_VBA_CODE.bas`** - VBA code for Excel automation (if needed for advanced features).

- **`staggered_ladder_calculator_with_helpers_structure.json`** - JSON structure definition for the Excel calculator layout.

## Usage

### Creating an Excel Calculator

```bash
python excel_calculator/create_excel_calculator.py [output_filename.xlsx]
```

Default output filename: `staggered_ladder_calculator.xlsx`

### Visualizing Excel Files

```bash
python excel_calculator/visualize_xlsx.py <xlsx_file> [--visualize-charts]
```

## Dependencies

- `openpyxl` - For Excel file manipulation
- `win32com` (optional) - For COM automation on Windows
- `matplotlib` (optional) - For chart visualization in visualize_xlsx.py

## Output Files

Generated Excel files are typically saved in the parent directory. The calculator includes:

1. **Calculator Sheet** - Main interface with input parameters and results
2. **Helpers Sheet** - Protected sheet with all calculation formulas
3. **Charts Sheet** - Visualizations of buy/sell ladders

## Features

- Input validation
- Comprehensive calculation formulas
- Visual charts and graphs
- Protected helper calculations
- Auto-filtering of active rungs
- Validation checks

