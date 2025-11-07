#!/usr/bin/env python3
"""
Simplified Staggered Order Ladder Calculator

Calculates buy and sell ladder orders based on user-specified price rungs
or profit targets. Outputs results to Excel and PDF with visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import sys
import os
import argparse
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.chart import LineChart, Reference
from openpyxl.utils import get_column_letter
from openpyxl.cell.cell import MergedCell

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class StaggeredLadderCalculator:
    """Calculate staggered order ladders for buy and sell orders."""
    
    def __init__(self, budget: float, num_rungs: int = 10):
        """
        Initialize calculator.
        
        Args:
            budget: Total budget for buy orders
            num_rungs: Number of rungs in the ladder
        """
        # Input validation
        if budget <= 0:
            raise ValueError("Budget must be positive")
        if num_rungs < 1:
            raise ValueError("Number of rungs must be at least 1")
        if not isinstance(num_rungs, int):
            raise ValueError("Number of rungs must be an integer")
        
        self.budget = budget
        self.num_rungs = num_rungs
        self.buy_prices = []
        self.buy_quantities = []
        self.sell_prices = []
        self.sell_quantities = []
    
    def _normalize_budget(self):
        """Normalize buy quantities to exactly match budget."""
        total_cost = sum(qty * price for qty, price in zip(self.buy_quantities, self.buy_prices))
        if total_cost > 0:
            scale_factor = self.budget / total_cost
            self.buy_quantities = [qty * scale_factor for qty in self.buy_quantities]
    
    def _validate_prices(self):
        """Validate that sell prices exceed buy prices for all rungs."""
        # Ensure each sell price exceeds its corresponding buy price
        for i in range(self.num_rungs):
            if self.sell_prices[i] <= self.buy_prices[i]:
                raise ValueError(
                    f"Sell price ${self.sell_prices[i]:.2f} at rung {i+1} must exceed "
                    f"buy price ${self.buy_prices[i]:.2f}"
                )
        
        # Ensure top buy order is below bottom sell order
        max_buy_price = max(self.buy_prices)
        min_sell_price = min(self.sell_prices)
        if max_buy_price >= min_sell_price:
            raise ValueError(
                f"Top buy order (${max_buy_price:.2f}) must be below bottom sell order (${min_sell_price:.2f}). "
                f"Gap needed: ${min_sell_price - max_buy_price:.2f}"
            )
        
    def calculate_from_profit_target(self, buy_upper: float, profit_target: float,
                                    buy_price_range_pct: float = 30.0) -> Dict:
        """
        Calculate ladder from upper buy rung and profit target.
        Uses exponential allocation strategy (buy more at lower prices, sell more at higher prices).
        
        Args:
            buy_upper: Upper rung of buy ladder (highest buy price)
            profit_target: Target profit percentage (e.g., 50 for 50%)
            buy_price_range_pct: Percentage range for buy ladder (default 30%)
        
        Returns:
            Dictionary with calculated orders and statistics
        """
        # Input validation
        if buy_upper <= 0:
            raise ValueError("Buy upper must be positive")
        if profit_target < 10 or profit_target > 200:
            raise ValueError("Profit target must be between 10% and 200%")
        if buy_price_range_pct <= 0 or buy_price_range_pct > 100:
            raise ValueError("Buy price range percentage must be between 0 and 100")
        
        profit_multiplier = 1 + (profit_target / 100)
        
        # Overall strategy returns profit_target%
        # Calculate buy prices: ascending order (lowest to highest)
        price_range = buy_upper * (buy_price_range_pct / 100)
        buy_lower = buy_upper - price_range
        if buy_lower <= 0:
            raise ValueError(f"Buy price range results in non-positive buy_lower. "
                           f"Try reducing buy_price_range_pct from {buy_price_range_pct}%")
        
        self.buy_prices = np.linspace(buy_lower, buy_upper, self.num_rungs)
        
        # Calculate consistent price step from buy prices
        if self.num_rungs > 1:
            price_step = (buy_upper - buy_lower) / (self.num_rungs - 1)
        else:
            price_step = buy_upper * 0.1  # Default step if only one rung
        
        # Calculate buy quantities using exponential allocation (buy more at lower prices)
        weights = np.exp(np.linspace(2, 0, self.num_rungs))  # Higher weight for lower prices
        weights = weights / weights.sum()
        self.buy_quantities = [(self.budget * w) / price 
                              for w, price in zip(weights, self.buy_prices)]
        
        # Normalize to exactly match budget
        self._normalize_budget()
        
        # Calculate total cost (should equal budget after normalization)
        total_cost = sum(qty * price for qty, price in zip(self.buy_quantities, self.buy_prices))
        
        # Calculate target revenue
        target_revenue = total_cost * profit_multiplier
        
        # Calculate total buy quantity
        total_buy_qty = sum(self.buy_quantities)
        
        # Allocate sell quantities using exponential allocation (sell more at higher prices)
        weights = np.exp(np.linspace(0, 2, self.num_rungs))  # Higher weight for higher prices
        weights = weights / weights.sum()
        self.sell_quantities = [total_buy_qty * w for w in weights]
        
        # Calculate sell prices with consistent spacing
        # Use the same price_step as buy prices
        max_buy_price = max(self.buy_prices)
        
        # Calculate gap based on profit target to ensure sell prices start high enough
        # For a profit target, we need sell prices to be significantly above buy prices
        # Use profit_target% of max_buy_price as base gap, but ensure at least one price step
        base_gap_pct = profit_target / 100.0  # e.g., 0.5 for 50% profit target
        base_gap_size = max_buy_price * base_gap_pct * 0.5  # Use 50% of profit target as initial gap
        # Round to nearest price_step multiple
        gap_steps = max(1, int(np.ceil(base_gap_size / price_step)))
        gap_size = gap_steps * price_step
        min_sell_price = max_buy_price + gap_size
        
        # Create sell prices with consistent spacing using the same price_step
        self.sell_prices = np.array([min_sell_price + i * price_step for i in range(self.num_rungs)])
        
        # CRITICAL: Ensure sell quantities match buy quantities exactly BEFORE revenue adjustment
        # This ensures volume matching is maintained throughout
        total_sell_qty_initial = sum(self.sell_quantities)
        if total_sell_qty_initial > 0 and abs(total_sell_qty_initial - total_buy_qty) > 0.0001:
            normalize_factor = total_buy_qty / total_sell_qty_initial
            self.sell_quantities = [qty * normalize_factor for qty in self.sell_quantities]
            logger.debug(f"Normalized sell quantities to match buy quantities: {total_buy_qty:.6f}")
        
        # Calculate current revenue with these quantities and prices
        current_revenue = sum(price * qty for price, qty in zip(self.sell_prices, self.sell_quantities))
        
        # Adjust sell PRICES (not quantities) to achieve target revenue while maintaining quantity matching
        # This preserves volume matching while achieving profit target
        if current_revenue > 0 and abs(current_revenue - target_revenue) > 0.01:
            price_scale = target_revenue / current_revenue
            # Scale all sell prices proportionally to achieve target revenue
            # This maintains relative spacing while adjusting absolute levels
            self.sell_prices = self.sell_prices * price_scale
            logger.debug(f"Adjusted sell prices by factor {price_scale:.6f} to achieve target revenue")
            
            # CRITICAL: After scaling, ensure sell prices are still above buy prices
            # If scaling brought prices down too much, shift them up to maintain minimum gap
            max_buy_price_after = max(self.buy_prices)
            min_sell_price_after = min(self.sell_prices)
            required_gap = price_step  # Maintain at least one price step gap
            min_valid_sell_price = max_buy_price_after + required_gap
            
            if min_sell_price_after < min_valid_sell_price:
                # Calculate how much we need to shift prices up
                price_shift = min_valid_sell_price - min_sell_price_after
                self.sell_prices = self.sell_prices + price_shift
                logger.info(f"Shifted sell prices up by ${price_shift:.2f} to maintain gap above buy prices")
                
                # Recalculate revenue with adjusted prices
                adjusted_revenue = sum(price * qty for price, qty in zip(self.sell_prices, self.sell_quantities))
                actual_profit_pct = ((adjusted_revenue - total_cost) / total_cost * 100) if total_cost > 0 else 0
                logger.warning(f"Profit target adjustment: Requested {profit_target:.2f}%, achieved {actual_profit_pct:.2f}% "
                             f"(revenue: ${adjusted_revenue:,.2f} vs target: ${target_revenue:,.2f})")
        
        # CRITICAL: Re-verify quantities still match after any adjustments
        final_total_buy_qty = sum(self.buy_quantities)
        final_total_sell_qty = sum(self.sell_quantities)
        qty_diff = abs(final_total_buy_qty - final_total_sell_qty)
        if qty_diff > 0.0001:
            logger.error(f"QUANTITY MISMATCH DETECTED: buy={final_total_buy_qty:.6f}, sell={final_total_sell_qty:.6f}, diff={qty_diff:.6f}")
            # Force correction: normalize sell quantities to match buy quantities exactly
            if final_total_sell_qty > 0:
                normalize_factor = final_total_buy_qty / final_total_sell_qty
                self.sell_quantities = [qty * normalize_factor for qty in self.sell_quantities]
                final_total_sell_qty_corrected = sum(self.sell_quantities)
                logger.info(f"FORCED CORRECTION: Normalized sell quantities to match buy quantities exactly. "
                          f"New total: {final_total_sell_qty_corrected:.6f} (target: {final_total_buy_qty:.6f})")
            else:
                raise ValueError(f"Cannot normalize sell quantities: total is zero or negative")
        else:
            logger.info(f"Volume matching verified: buy={final_total_buy_qty:.6f}, sell={final_total_sell_qty:.6f}, diff={qty_diff:.9f}")
        
        # Validate that sell prices exceed buy prices and gap exists
        self._validate_prices()
        
        return self._generate_results()
    
    def _generate_results(self) -> Dict:
        """Generate results dictionary with statistics."""
        # Calculate cumulative statistics
        cumulative_buy_cost = []
        cumulative_buy_qty = []
        cumulative_sell_revenue = []
        cumulative_sell_qty = []
        avg_buy_prices = []
        avg_sell_prices = []
        
        # Current price is the highest buy price (baseline)
        current_price = max(self.buy_prices) if len(self.buy_prices) > 0 else 0
        
        # Calculate averages relative to current price
        # For buy orders: orders execute as price drops from highest to lowest
        # buy_prices[0] = lowest price, buy_prices[-1] = highest price (current_price)
        # When price drops to buy_prices[i], orders from buy_prices[-1] down to buy_prices[i] have executed
        # So avg_buy_prices[i] = weighted average of orders from index i to end (inclusive)
        for i in range(self.num_rungs):
            # Include all buy orders from index i to the end (up to current price)
            # This represents: if price drops to buy_prices[i], what orders have executed?
            # Answer: orders at buy_prices[i], buy_prices[i+1], ..., buy_prices[-1]
            total_cost_from_level = 0.0
            total_qty_from_level = 0.0
            
            for j in range(i, self.num_rungs):
                total_cost_from_level += self.buy_quantities[j] * self.buy_prices[j]
                total_qty_from_level += self.buy_quantities[j]
            
            if total_qty_from_level > 0:
                avg_buy_price = total_cost_from_level / total_qty_from_level
            else:
                avg_buy_price = 0.0
            
            # Validate average is within expected range (weighted average should be between min and max)
            prices_in_range = [self.buy_prices[j] for j in range(i, self.num_rungs)]
            if prices_in_range:
                min_price_in_range = min(prices_in_range)
                max_price_in_range = max(prices_in_range)
                # Allow small floating point tolerance
                tolerance = 0.01
                if avg_buy_price < min_price_in_range - tolerance or avg_buy_price > max_price_in_range + tolerance:
                    logger.warning(f"Buy average {avg_buy_price:.2f} at index {i} is outside expected price range "
                                 f"[{min_price_in_range:.2f}, {max_price_in_range:.2f}]. "
                                 f"Prices: {prices_in_range}, Quantities: {[self.buy_quantities[j] for j in range(i, self.num_rungs)]}")
            
            avg_buy_prices.append(avg_buy_price)
        
        # For sell orders: at each level, include all orders from lowest sell price UP TO that level
        # sell_prices[0] = lowest price, sell_prices[-1] = highest price
        # When price rises to sell_prices[i], all orders from sell_prices[0] to sell_prices[i] execute
        # So avg_sell_prices[i] = weighted average of orders from index 0 to i (inclusive)
        for i in range(self.num_rungs):
            # Include all sell orders from index 0 to i (from lowest up to current level)
            # This means if price rises to sell_prices[i], all orders at or below this level would execute
            total_revenue_to_level = 0.0
            total_qty_to_level = 0.0
            
            for j in range(i + 1):  # 0 to i (inclusive)
                total_revenue_to_level += self.sell_quantities[j] * self.sell_prices[j]
                total_qty_to_level += self.sell_quantities[j]
            
            if total_qty_to_level > 0:
                avg_sell_price = total_revenue_to_level / total_qty_to_level
            else:
                avg_sell_price = 0.0
            
            # Validate average is within expected range (weighted average should be between min and max)
            prices_in_range = [self.sell_prices[j] for j in range(i + 1)]
            if prices_in_range:
                min_price_in_range = min(prices_in_range)
                max_price_in_range = max(prices_in_range)
                # Allow small floating point tolerance
                tolerance = 0.01
                if avg_sell_price < min_price_in_range - tolerance or avg_sell_price > max_price_in_range + tolerance:
                    logger.warning(f"Sell average {avg_sell_price:.2f} at index {i} is outside expected price range "
                                 f"[{min_price_in_range:.2f}, {max_price_in_range:.2f}]. "
                                 f"Prices: {prices_in_range}, Quantities: {[self.sell_quantities[j] for j in range(i + 1)]}")
            
            avg_sell_prices.append(avg_sell_price)
        
        # Calculate cumulative values (for display purposes)
        running_buy_cost = 0
        running_buy_qty = 0
        running_sell_revenue = 0
        running_sell_qty = 0
        
        for i in range(self.num_rungs):
            running_buy_cost += self.buy_quantities[i] * self.buy_prices[i]
            running_buy_qty += self.buy_quantities[i]
            cumulative_buy_cost.append(running_buy_cost)
            cumulative_buy_qty.append(running_buy_qty)
            
            running_sell_revenue += self.sell_quantities[i] * self.sell_prices[i]
            running_sell_qty += self.sell_quantities[i]
            cumulative_sell_revenue.append(running_sell_revenue)
            cumulative_sell_qty.append(running_sell_qty)
        
        # CRITICAL: Verify and enforce total buy and sell quantities match exactly
        # This is a final safety check to ensure volume matching is always maintained
        total_buy_qty = sum(self.buy_quantities)
        total_sell_qty = sum(self.sell_quantities)
        qty_diff = abs(total_buy_qty - total_sell_qty)
        qty_tolerance = 0.0001
        
        if qty_diff > qty_tolerance:
            logger.error(f"VOLUME MISMATCH DETECTED IN RESULTS: Total buy quantity ({total_buy_qty:.6f}) != Total sell quantity ({total_sell_qty:.6f}), difference: {qty_diff:.6f}")
            # Force correction by normalizing sell quantities to match buy quantities exactly
            if total_sell_qty > 0:
                normalize_factor = total_buy_qty / total_sell_qty
                self.sell_quantities = [qty * normalize_factor for qty in self.sell_quantities]
                total_sell_qty_corrected = sum(self.sell_quantities)
                qty_diff_after = abs(total_buy_qty - total_sell_qty_corrected)
                logger.info(f"FORCED VOLUME CORRECTION: Normalized sell quantities. "
                          f"New total = {total_sell_qty_corrected:.6f} (target: {total_buy_qty:.6f}), "
                          f"remaining diff: {qty_diff_after:.9f}")
                # Recalculate cumulative sell quantities and revenue after correction
                running_sell_revenue = 0
                running_sell_qty = 0
                cumulative_sell_revenue = []
                cumulative_sell_qty = []
                for i in range(self.num_rungs):
                    running_sell_revenue += self.sell_quantities[i] * self.sell_prices[i]
                    running_sell_qty += self.sell_quantities[i]
                    cumulative_sell_revenue.append(running_sell_revenue)
                    cumulative_sell_qty.append(running_sell_qty)
                # Also recalculate avg_sell_prices since quantities changed
                avg_sell_prices = []
                for i in range(self.num_rungs):
                    total_revenue_to_level = 0.0
                    total_qty_to_level = 0.0
                    for j in range(i + 1):
                        total_revenue_to_level += self.sell_quantities[j] * self.sell_prices[j]
                        total_qty_to_level += self.sell_quantities[j]
                    if total_qty_to_level > 0:
                        avg_sell_price = total_revenue_to_level / total_qty_to_level
                    else:
                        avg_sell_price = 0.0
                    avg_sell_prices.append(avg_sell_price)
            else:
                raise ValueError(f"Cannot normalize sell quantities: total is zero or negative")
        else:
            logger.debug(f"Volume matching verified in results: buy={total_buy_qty:.6f}, sell={total_sell_qty:.6f}, diff={qty_diff:.9f}")
        
        # Verify that highest buy average equals current price exactly
        if len(avg_buy_prices) > 0 and len(self.buy_prices) > 0:
            highest_buy_index = len(avg_buy_prices) - 1
            expected_highest_avg = self.buy_prices[highest_buy_index]  # Should equal current_price
            actual_highest_avg = avg_buy_prices[highest_buy_index]
            if abs(actual_highest_avg - expected_highest_avg) > 0.0001:
                logger.warning(f"Highest buy average mismatch: expected {expected_highest_avg:.2f}, got {actual_highest_avg:.2f}. "
                           f"Buy price: {self.buy_prices[highest_buy_index]:.2f}, Quantity: {self.buy_quantities[highest_buy_index]:.4f}")
                # Force correct value
                avg_buy_prices[highest_buy_index] = self.buy_prices[highest_buy_index]
            else:
                logger.debug(f"Highest buy average verified: {actual_highest_avg:.2f} = {expected_highest_avg:.2f}")
        
        # Verify that first sell average equals first sell price exactly
        if len(avg_sell_prices) > 0 and len(self.sell_prices) > 0:
            expected_first_avg = self.sell_prices[0]
            actual_first_avg = avg_sell_prices[0]
            if abs(actual_first_avg - expected_first_avg) > 0.0001:
                logger.warning(f"First sell average mismatch: expected {expected_first_avg:.2f}, got {actual_first_avg:.2f}. "
                           f"Sell price: {self.sell_prices[0]:.2f}, Quantity: {self.sell_quantities[0]:.4f}")
                # Force correct value
                avg_sell_prices[0] = self.sell_prices[0]
            else:
                logger.debug(f"First sell average verified: {actual_first_avg:.2f} = {expected_first_avg:.2f}")
        
        # Verify buy averages are monotonically increasing (as we go from lowest to highest price)
        # The average should increase as we move closer to current price
        for i in range(1, len(avg_buy_prices)):
            if avg_buy_prices[i] < avg_buy_prices[i-1]:
                logger.warning(f"Buy average decreases at index {i}: {avg_buy_prices[i-1]:.2f} -> {avg_buy_prices[i]:.2f}")
        
        # Verify sell averages are monotonically increasing (as we sell at higher prices)
        for i in range(1, len(avg_sell_prices)):
            if avg_sell_prices[i] < avg_sell_prices[i-1]:
                logger.warning(f"Sell average decreases at index {i}: {avg_sell_prices[i-1]:.2f} -> {avg_sell_prices[i]:.2f}")
        
        total_cost = cumulative_buy_cost[-1]
        total_revenue = cumulative_sell_revenue[-1]
        total_profit = total_revenue - total_cost
        profit_pct = (total_profit / total_cost * 100) if total_cost > 0 else 0
        
        return {
            'buy_prices': self.buy_prices.tolist() if isinstance(self.buy_prices, np.ndarray) else self.buy_prices,
            'buy_quantities': self.buy_quantities,
            'sell_prices': self.sell_prices.tolist() if isinstance(self.sell_prices, np.ndarray) else self.sell_prices,
            'sell_quantities': self.sell_quantities,
            'cumulative_buy_cost': cumulative_buy_cost,
            'cumulative_buy_qty': cumulative_buy_qty,
            'cumulative_sell_revenue': cumulative_sell_revenue,
            'cumulative_sell_qty': cumulative_sell_qty,
            'avg_buy_prices': avg_buy_prices,
            'avg_sell_prices': avg_sell_prices,
            'total_cost': total_cost,
            'total_revenue': total_revenue,
            'total_profit': total_profit,
            'profit_pct': profit_pct,
            'num_rungs': self.num_rungs,
            'budget': self.budget
        }


def generate_excel(results: Dict, filename: str):
    """Generate Excel file with order ladder and statistics."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Order Ladder"
    
    # Header styling
    header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF", size=12)
    border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )
    
    # Title
    ws['A1'] = "Staggered Order Ladder"
    ws['A1'].font = Font(bold=True, size=16)
    ws.merge_cells('A1:F1')
    
    # Summary section
    row = 3
    ws[f'A{row}'] = "Summary"
    ws[f'A{row}'].font = Font(bold=True, size=14)
    row += 1
    
    summary_data = [
        ['Total Budget', f"${results['budget']:,.2f}"],
        ['Number of Rungs', results['num_rungs']],
        ['Total Cost', f"${results['total_cost']:,.2f}"],
        ['Total Revenue', f"${results['total_revenue']:,.2f}"],
        ['Total Profit', f"${results['total_profit']:,.2f}"],
        ['Profit Percentage', f"{results['profit_pct']:.2f}%"]
    ]
    
    for label, value in summary_data:
        ws[f'A{row}'] = label
        ws[f'B{row}'] = value
        ws[f'A{row}'].font = Font(bold=True)
        row += 1
    
    row += 1
    
    # Buy orders header
    ws[f'A{row}'] = "Buy Orders"
    ws[f'A{row}'].font = Font(bold=True, size=14)
    row += 1
    
    headers = ['Buy Price', 'Quantity', 'Cost', 'Cumulative Cost', 'Avg Buy Price']
    for col_idx, header in enumerate(headers, start=1):
        cell = ws.cell(row=row, column=col_idx)
        cell.value = header
        cell.fill = header_fill
        cell.font = header_font
        cell.border = border
        cell.alignment = Alignment(horizontal='center')
    
    row += 1
    
    # Buy orders data (reversed: order 1 = highest price, order 10 = lowest price)
    buy_prices_reversed = list(reversed(results['buy_prices']))
    buy_quantities_reversed = list(reversed(results['buy_quantities']))
    cumulative_buy_cost_reversed = list(reversed(results['cumulative_buy_cost']))
    avg_buy_prices_reversed = list(reversed(results['avg_buy_prices']))
    
    for i in range(results['num_rungs']):
        ws.cell(row=row, column=1).value = buy_prices_reversed[i]
        ws.cell(row=row, column=2).value = buy_quantities_reversed[i]
        ws.cell(row=row, column=3).value = buy_prices_reversed[i] * buy_quantities_reversed[i]
        ws.cell(row=row, column=4).value = cumulative_buy_cost_reversed[i]
        ws.cell(row=row, column=5).value = avg_buy_prices_reversed[i]
        
        for col in range(1, 6):
            ws.cell(row=row, column=col).border = border
            ws.cell(row=row, column=col).number_format = '#,##0.00'
        
        row += 1
    
    row += 1
    
    # Sell orders header
    ws[f'A{row}'] = "Sell Orders"
    ws[f'A{row}'].font = Font(bold=True, size=14)
    row += 1
    
    headers = ['Sell Price', 'Quantity', 'Revenue', 'Cumulative Revenue', 'Avg Sell Price']
    for col_idx, header in enumerate(headers, start=1):
        cell = ws.cell(row=row, column=col_idx)
        cell.value = header
        cell.fill = header_fill
        cell.font = header_font
        cell.border = border
        cell.alignment = Alignment(horizontal='center')
    
    row += 1
    
    # Sell orders data
    for i in range(results['num_rungs']):
        ws.cell(row=row, column=1).value = results['sell_prices'][i]
        ws.cell(row=row, column=2).value = results['sell_quantities'][i]
        ws.cell(row=row, column=3).value = results['sell_prices'][i] * results['sell_quantities'][i]
        ws.cell(row=row, column=4).value = results['cumulative_sell_revenue'][i]
        ws.cell(row=row, column=5).value = results['avg_sell_prices'][i]
        
        for col in range(1, 6):
            ws.cell(row=row, column=col).border = border
            ws.cell(row=row, column=col).number_format = '#,##0.00'
        
        row += 1
    
    # Auto-adjust column widths
    for col_idx, column in enumerate(ws.columns, start=1):
        max_length = 0
        # Skip if first cell is a MergedCell
        if isinstance(column[0], MergedCell):
            continue
        
        column_letter = get_column_letter(col_idx)
        for cell in column:
            try:
                if not isinstance(cell, MergedCell) and cell.value is not None:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
            except:
                pass
        if max_length > 0:
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column_letter].width = adjusted_width
    
    wb.save(filename)
    logger.info(f"Excel file saved: {filename}")


def generate_pdf(results: Dict, filename: str):
    """Generate PDF file with order ladder, statistics, and graphs."""
    with PdfPages(filename) as pdf:
        # Create figure for summary and orders
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.95, 'Staggered Order Ladder', 
                ha='center', fontsize=16, fontweight='bold')
        
        # Summary text
        summary_text = f"""
Summary:
  Total Budget: ${results['budget']:,.2f}
  Number of Rungs: {results['num_rungs']}
  Total Cost: ${results['total_cost']:,.2f}
  Total Revenue: ${results['total_revenue']:,.2f}
  Total Profit: ${results['total_profit']:,.2f}
  Profit Percentage: {results['profit_pct']:.2f}%
        """
        
        fig.text(0.1, 0.85, summary_text, fontsize=10, family='monospace',
                verticalalignment='top')
        
        # Create table for buy orders
        ax1 = fig.add_subplot(211)
        ax1.axis('tight')
        ax1.axis('off')
        
        buy_data = []
        # Reverse order: Buy order 1 = highest price, Buy order 10 = lowest price
        buy_prices_reversed = list(reversed(results['buy_prices']))
        buy_quantities_reversed = list(reversed(results['buy_quantities']))
        cumulative_buy_cost_reversed = list(reversed(results['cumulative_buy_cost']))
        avg_buy_prices_reversed = list(reversed(results['avg_buy_prices']))
        
        for i in range(results['num_rungs']):
            buy_data.append([
                f"${buy_prices_reversed[i]:,.2f}",
                f"{buy_quantities_reversed[i]:,.4f}",
                f"${buy_prices_reversed[i] * buy_quantities_reversed[i]:,.2f}",
                f"${cumulative_buy_cost_reversed[i]:,.2f}",
                f"${avg_buy_prices_reversed[i]:,.2f}"
            ])
        
        buy_table = ax1.table(cellText=buy_data,
                              colLabels=['Buy Price', 'Quantity', 'Cost', 
                                        'Cumulative Cost', 'Avg Buy Price'],
                              cellLoc='center',
                              loc='center')
        buy_table.auto_set_font_size(False)
        buy_table.set_fontsize(8)
        buy_table.scale(1, 1.5)
        ax1.set_title('Buy Orders', fontweight='bold', pad=20)
        
        # Create table for sell orders
        ax2 = fig.add_subplot(212)
        ax2.axis('tight')
        ax2.axis('off')
        
        sell_data = []
        for i in range(results['num_rungs']):
            sell_data.append([
                f"${results['sell_prices'][i]:,.2f}",
                f"{results['sell_quantities'][i]:,.4f}",
                f"${results['sell_prices'][i] * results['sell_quantities'][i]:,.2f}",
                f"${results['cumulative_sell_revenue'][i]:,.2f}",
                f"${results['avg_sell_prices'][i]:,.2f}"
            ])
        
        sell_table = ax2.table(cellText=sell_data,
                              colLabels=['Sell Price', 'Quantity', 'Revenue',
                                        'Cumulative Revenue', 'Avg Sell Price'],
                              cellLoc='center',
                              loc='center')
        sell_table.auto_set_font_size(False)
        sell_table.set_fontsize(8)
        sell_table.scale(1, 1.5)
        ax2.set_title('Sell Orders', fontweight='bold', pad=20)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Graph: Buy price and quantity as price levels fill
        fig, ax1 = plt.subplots(figsize=(11, 8.5))
        # Reverse order: Buy order 1 = highest price, Buy order 10 = lowest price
        buy_price_levels = list(reversed(results['buy_prices']))  # Reverse: highest to lowest
        buy_quantities = list(reversed(results['buy_quantities']))
        avg_buy_prices = list(reversed(results['avg_buy_prices']))
        
        # Plot with highest price (Buy order 1) on left, lowest price (Buy order 10) on right
        x_positions = list(range(len(buy_price_levels)))
        
        # Create second y-axis for quantity (right side) - create early for area chart
        ax2 = ax1.twinx()
        
        # Plot quantity as area chart (less competing with lines, more subtle)
        ax2.fill_between(x_positions, 0, buy_quantities, alpha=0.2, color='#81C784', 
                        label='Quantity', step='mid')
        # Add subtle line on top of area for clarity
        ax2.plot(x_positions, buy_quantities, marker='^', markersize=5, 
                linewidth=1.5, color='#66BB6A', linestyle=':', alpha=0.7)
        ax2.set_ylabel('Quantity', fontsize=12, fontweight='bold', color='#66BB6A')
        ax2.tick_params(axis='y', labelcolor='#66BB6A')
        
        # Plot individual buy prices on left y-axis (ax1) - make more prominent
        line1 = ax1.plot(x_positions, buy_price_levels, marker='o', linewidth=2.5, 
                markersize=9, color='#1B5E20', label='Individual Buy Price', 
                linestyle='-', zorder=3)
        line2 = ax1.plot(x_positions, avg_buy_prices, marker='s', linewidth=2.5, 
                markersize=7, color='#2E7D32', label='Average Buy Price', 
                linestyle='--', zorder=3)
        ax1.set_ylabel('Price ($)', fontsize=12, fontweight='bold', color='#2E7D32')
        ax1.tick_params(axis='y', labelcolor='#2E7D32')
        ax1.set_title('Buy Ladder: Price and Quantity by Order Rung', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='both', linestyle='--', linewidth=0.5)
        
        # Add value labels on price points (every other point to avoid clutter)
        for i in range(0, len(x_positions), 2):
            ax1.annotate(f'${buy_price_levels[i]:.0f}', 
                        xy=(x_positions[i], buy_price_levels[i]),
                        xytext=(0, 10), textcoords='offset points',
                        fontsize=8, ha='center', color='#1B5E20', alpha=0.7)
        
        # Add quantity value labels on peaks (top 3 quantities)
        sorted_indices = sorted(range(len(buy_quantities)), 
                               key=lambda i: buy_quantities[i], reverse=True)[:3]
        for i in sorted_indices:
            ax2.annotate(f'{buy_quantities[i]:.1f}', 
                        xy=(x_positions[i], buy_quantities[i]),
                        xytext=(0, 5), textcoords='offset points',
                        fontsize=8, ha='center', color='#66BB6A', 
                        fontweight='bold', alpha=0.8)
        
        # Set x-axis labels to show buy orders (1 = highest price, 10 = lowest price)
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels([f'Buy order {i+1}' for i in range(len(buy_price_levels))], 
                           rotation=0, ha='center', fontsize=10)
        ax1.set_xlabel('Buy Order', fontsize=12, fontweight='bold')
        
        # Combine legends with better positioning
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10,
                  framealpha=0.9, edgecolor='gray', fancybox=True)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Graph: Sell price and quantity as price levels fill
        fig, ax1 = plt.subplots(figsize=(11, 8.5))
        sell_price_levels = results['sell_prices']  # Already in ascending order (lowest to highest)
        sell_quantities = results['sell_quantities']
        avg_sell_prices = results['avg_sell_prices']  # Cumulative averages as orders fill
        
        # Plot in ascending order: lowest price on left, highest on right
        # This shows how average price increases as you fill orders from lowest to highest
        x_positions = list(range(len(sell_price_levels)))
        
        # Create second y-axis for quantity (right side) - create early for area chart
        ax2 = ax1.twinx()
        
        # Plot quantity as area chart (less competing with lines, more subtle)
        ax2.fill_between(x_positions, 0, sell_quantities, alpha=0.2, color='#EF5350', 
                        label='Quantity', step='mid')
        # Add subtle line on top of area for clarity
        ax2.plot(x_positions, sell_quantities, marker='^', markersize=5, 
                linewidth=1.5, color='#E53935', linestyle=':', alpha=0.7)
        ax2.set_ylabel('Quantity', fontsize=12, fontweight='bold', color='#E53935')
        ax2.tick_params(axis='y', labelcolor='#E53935')
        
        # Plot individual sell prices on left y-axis (ax1) - make more prominent
        line1 = ax1.plot(x_positions, sell_price_levels, marker='o', linewidth=2.5, 
                markersize=9, color='#B71C1C', label='Individual Sell Price', 
                linestyle='-', zorder=3)
        line2 = ax1.plot(x_positions, avg_sell_prices, marker='s', linewidth=2.5, 
                markersize=7, color='#C62828', label='Average Sell Price', 
                linestyle='--', zorder=3)
        ax1.set_ylabel('Price ($)', fontsize=12, fontweight='bold', color='#C62828')
        ax1.tick_params(axis='y', labelcolor='#C62828')
        ax1.set_title('Sell Ladder: Price and Quantity by Order Rung', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='both', linestyle='--', linewidth=0.5)
        
        # Add value labels on price points (every other point to avoid clutter)
        for i in range(0, len(x_positions), 2):
            ax1.annotate(f'${sell_price_levels[i]:.0f}', 
                        xy=(x_positions[i], sell_price_levels[i]),
                        xytext=(0, 10), textcoords='offset points',
                        fontsize=8, ha='center', color='#B71C1C', alpha=0.7)
        
        # Add quantity value labels on peaks (top 3 quantities)
        sorted_indices = sorted(range(len(sell_quantities)), 
                               key=lambda i: sell_quantities[i], reverse=True)[:3]
        for i in sorted_indices:
            ax2.annotate(f'{sell_quantities[i]:.1f}', 
                        xy=(x_positions[i], sell_quantities[i]),
                        xytext=(0, 5), textcoords='offset points',
                        fontsize=8, ha='center', color='#E53935', 
                        fontweight='bold', alpha=0.8)
        
        # Set x-axis labels to show sell orders
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels([f'Sell order {i+1}' for i in range(len(sell_price_levels))], 
                           rotation=0, ha='center', fontsize=10)
        ax1.set_xlabel('Sell Order', fontsize=12, fontweight='bold')
        
        # Combine legends with better positioning
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10,
                  framealpha=0.9, edgecolor='gray', fancybox=True)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Combined graph: Buy orders on left, Sell orders on right
        # X-axis: Price (buy prices negative, sell prices positive)
        # Y-axis: Average price
        # Point size: Quantity
        fig, ax = plt.subplots(figsize=(11, 8.5))
        
        # Prepare buy order data
        # buy_prices is in ascending order (lowest to highest)
        # buy_prices[0] = lowest price (executes LAST when price drops)
        # buy_prices[-1] = highest price (executes FIRST when price drops)
        # avg_buy_prices[0] = average when price drops to lowest price (ALL orders have executed)
        # avg_buy_prices[-1] = average when price drops to highest price (only highest order executed)
        buy_prices = results['buy_prices']  # Ascending order: [lowest, ..., highest]
        buy_quantities = results['buy_quantities']
        avg_buy_prices = results['avg_buy_prices']
        
        # Verify highest buy average equals highest buy price (only that order has executed)
        if len(buy_prices) > 0 and len(avg_buy_prices) > 0:
            highest_index = len(buy_prices) - 1
            if abs(avg_buy_prices[highest_index] - buy_prices[highest_index]) > 0.01:
                logger.warning(f"Graph: Highest buy average {avg_buy_prices[highest_index]:.2f} != highest buy price {buy_prices[highest_index]:.2f}")
                # Don't force correction - let the calculated value stand
        
        # Prepare sell order data
        # sell_prices[0] = lowest price (executes first)
        # avg_sell_prices[0] = average after first order executes = sell_prices[0]
        sell_prices = results['sell_prices']  # Already in ascending order (lowest to highest)
        sell_quantities = results['sell_quantities']
        avg_sell_prices = results['avg_sell_prices']
        
        # Verify first sell average equals first sell price
        if len(sell_prices) > 0 and len(avg_sell_prices) > 0:
            if abs(avg_sell_prices[0] - sell_prices[0]) > 0.01:
                logger.warning(f"Graph: First sell average {avg_sell_prices[0]:.2f} != first sell price {sell_prices[0]:.2f}, correcting...")
                avg_sell_prices = list(avg_sell_prices)  # Make a copy to modify
                avg_sell_prices[0] = sell_prices[0]
        
        # Normalize quantities for point sizing (scale to reasonable size range)
        all_quantities = buy_quantities + sell_quantities
        min_qty = min(all_quantities)
        max_qty = max(all_quantities)
        qty_range = max_qty - min_qty if max_qty > min_qty else 1
        
        # Scale point sizes: min_size to max_size based on quantity
        min_point_size = 50
        max_point_size = 500
        def scale_size(qty):
            if qty_range == 0:
                return (min_point_size + max_point_size) / 2
            normalized = (qty - min_qty) / qty_range
            return min_point_size + normalized * (max_point_size - min_point_size)
        
        buy_point_sizes = [scale_size(qty) for qty in buy_quantities]
        sell_point_sizes = [scale_size(qty) for qty in sell_quantities]
        
        # Calculate x-axis positions based on actual prices
        # Use actual price values scaled so buys are on left (negative) and sells on right (positive)
        min_buy_price = min(buy_prices)
        max_buy_price = max(buy_prices)
        min_sell_price = min(sell_prices)
        max_sell_price = max(sell_prices)
        
        # Find the midpoint between max buy and min sell to use as the y-axis (x=0) separator
        price_midpoint = (max_buy_price + min_sell_price) / 2
        
        # Scale prices so they're positioned relative to the midpoint
        # Buy prices become negative (left of y-axis), sell prices become positive (right of y-axis)
        # This maintains the actual price spacing while separating buys and sells
        buy_x_positions = [price - price_midpoint for price in buy_prices]
        sell_x_positions = [price - price_midpoint for price in sell_prices]
        
        # Calculate volumes (quantity * price) for each order
        buy_volumes = [qty * price for qty, price in zip(buy_quantities, buy_prices)]
        sell_volumes = [qty * price for qty, price in zip(sell_quantities, sell_prices)]
        
        # Get y-axis limits for price to position volume bars at bottom
        y_min_price = min(min(avg_buy_prices), min(avg_sell_prices))
        y_max_price = max(max(avg_buy_prices), max(avg_sell_prices))
        y_range_price = y_max_price - y_min_price
        
        # Calculate bar width based on x-axis spacing
        if len(buy_x_positions) > 1:
            buy_bar_width = (buy_x_positions[1] - buy_x_positions[0]) * 0.8
        else:
            buy_bar_width = abs(buy_x_positions[0]) * 0.1 if buy_x_positions else 1
        
        if len(sell_x_positions) > 1:
            sell_bar_width = (sell_x_positions[1] - sell_x_positions[0]) * 0.8
        else:
            sell_bar_width = sell_x_positions[0] * 0.1 if sell_x_positions else 1
        
        # Normalize volumes to fit in bottom portion of graph (bottom 20% of y-axis)
        all_volumes = buy_volumes + sell_volumes
        max_volume = max(all_volumes) if all_volumes else 1
        volume_bar_height = y_range_price * 0.15  # Use 15% of y-axis range for volume bars
        volume_scale = volume_bar_height / max_volume if max_volume > 0 else 1
        
        # Position volume bars at bottom of graph
        volume_bottom = y_min_price - y_range_price * 0.05  # Position slightly below minimum price
        
        # Plot volume bars for buy orders
        buy_volume_heights = [vol * volume_scale for vol in buy_volumes]
        ax.bar(buy_x_positions, buy_volume_heights, width=buy_bar_width,
               bottom=volume_bottom, alpha=0.5, color='#81C784', 
               edgecolor='#2E7D32', linewidth=1, label='Buy Volume', zorder=2)
        
        # Plot volume bars for sell orders
        sell_volume_heights = [vol * volume_scale for vol in sell_volumes]
        ax.bar(sell_x_positions, sell_volume_heights, width=sell_bar_width,
               bottom=volume_bottom, alpha=0.5, color='#EF5350', 
               edgecolor='#C62828', linewidth=1, label='Sell Volume', zorder=2)
        
        # Plot buy orders on left side
        ax.scatter(buy_x_positions, avg_buy_prices, s=buy_point_sizes, 
                  alpha=0.6, color='#2E7D32', edgecolors='#1B5E20', 
                  linewidths=1.5, label='Buy Orders', zorder=3)
        
        # Plot sell orders on right side
        ax.scatter(sell_x_positions, avg_sell_prices, s=sell_point_sizes, 
                  alpha=0.6, color='#C62828', edgecolors='#B71C1C', 
                  linewidths=1.5, label='Sell Orders', zorder=3)
        
        # Add vertical line at x=0 (y-axis) to separate buy and sell sides
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=2, alpha=0.7, zorder=1)
        
        # Set x-axis limits
        all_x_positions = buy_x_positions + sell_x_positions
        x_min = min(all_x_positions)
        x_max = max(all_x_positions)
        x_range = x_max - x_min
        x_padding = x_range * 0.05
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        
        # Adjust y-axis limits to include volume bars
        max_volume_bar_height = max(max(buy_volume_heights) if buy_volume_heights else 0,
                                   max(sell_volume_heights) if sell_volume_heights else 0)
        # Top of tallest volume bar
        volume_bar_top = volume_bottom + max_volume_bar_height
        # Ensure y-axis includes both price range and volume bars
        y_min_with_volume = min(volume_bottom, y_min_price) - y_range_price * 0.02
        y_max_with_volume = max(volume_bar_top, y_max_price) + y_range_price * 0.02
        ax.set_ylim(y_min_with_volume, y_max_with_volume)
        
        # Format x-axis labels to show actual price values
        x_ticks = []
        x_labels = []
        
        # Add buy price ticks (show actual prices)
        for i, price in enumerate(buy_prices):
            x_ticks.append(buy_x_positions[i])
            x_labels.append(f'${price:.2f}')
        
        # Add sell price ticks (show actual prices)
        for i, price in enumerate(sell_prices):
            x_ticks.append(sell_x_positions[i])
            x_labels.append(f'${price:.2f}')
        
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        
        # Set labels and title
        ax.set_xlabel('Order Price ($)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Price ($)', fontsize=12, fontweight='bold')
        ax.set_title('Combined Buy/Sell Ladder: Average Price vs Order Price\n(Point size = Order Quantity)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
        
        # Add legend with quantity scale information
        legend = ax.legend(loc='upper left', fontsize=10, framealpha=0.9, 
                          edgecolor='gray', fancybox=True)
        
        # Add text annotation explaining point sizes
        min_qty_formatted = f"{min_qty:.2f}" if min_qty < 1000 else f"{min_qty:.0f}"
        max_qty_formatted = f"{max_qty:.2f}" if max_qty < 1000 else f"{max_qty:.0f}"
        size_info = f"Point size scale:\nMin: {min_qty_formatted}\nMax: {max_qty_formatted}"
        ax.text(0.02, 0.98, size_info, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Add labels for buy and sell sides
        ax.text(0.02, 0.02, 'BUY ORDERS', transform=ax.transAxes, 
               fontsize=11, fontweight='bold', color='#2E7D32',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        ax.text(0.98, 0.02, 'SELL ORDERS', transform=ax.transAxes, 
               fontsize=11, fontweight='bold', color='#C62828',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
               ha='right')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    logger.info(f"PDF file saved: {filename}")


def get_smart_input(args: Optional[argparse.Namespace] = None) -> Dict:
    """
    Get input for ladder calculation with intelligent defaults.
    
    Args:
        args: Optional argparse.Namespace with command-line arguments
    
    Returns:
        Dictionary with calculation results
    """
    print("\n" + "="*60)
    print("Staggered Order Ladder Calculator")
    print("="*60 + "\n")
    
    # Get budget (required)
    if args and args.budget:
        budget = args.budget
        print(f"Budget: ${budget:,.2f}")
    else:
        while True:
            try:
                budget_str = input("Enter total budget for buy orders: $").strip()
                if not budget_str:
                    print("Budget is required.")
                    continue
                budget = float(budget_str)
                if budget <= 0:
                    print("Budget must be positive.")
                    continue
                break
            except ValueError:
                print("Please enter a valid number.")
    
    # Get current price (required for profit target method)
    if args and args.price:
        current_price = args.price
        print(f"Current Price: ${current_price:,.2f}")
    else:
        while True:
            try:
                price_str = input("Enter current price: $").strip()
                if not price_str:
                    print("Price is required.")
                    continue
                current_price = float(price_str)
                if current_price <= 0:
                    print("Price must be positive.")
                    continue
                break
            except ValueError:
                print("Please enter a valid number.")
    
    # Get profit target (with default 75%)
    if args and args.profit_target:
        profit_target = args.profit_target
        print(f"Profit Target: {profit_target}%")
    else:
        while True:
            try:
                profit_str = input("Enter profit target percentage (default 75%): ").strip()
                if not profit_str:
                    profit_target = 75.0
                    print(f"Using default profit target: {profit_target}%")
                    break
                profit_target = float(profit_str)
                if profit_target < 10 or profit_target > 200:
                    print("Profit target must be between 10% and 200%.")
                    continue
                break
            except ValueError:
                print("Please enter a valid number.")
    
    # Get number of orders (with default 10)
    if args and args.num_rungs:
        num_rungs = args.num_rungs
        print(f"Number of Orders: {num_rungs}")
    else:
        while True:
            try:
                orders_str = input("Enter number of orders to place (default 10): ").strip()
                if not orders_str:
                    num_rungs = 10
                    print(f"Using default number of orders: {num_rungs}")
                    break
                num_rungs = int(orders_str)
                if num_rungs < 1:
                    print("Number of orders must be at least 1.")
                    continue
                break
            except ValueError:
                print("Please enter a valid integer.")
    
    # Smart defaults for other parameters
    buy_price_range_pct = args.price_range if (args and args.price_range) else 30.0
    
    print(f"\nConfiguration:")
    print(f"  Number of orders: {num_rungs}")
    print(f"  Profit target: {profit_target}%")
    print(f"  Allocation strategy: exponential (buy more at lower prices, sell more at higher prices)")
    print(f"  Buy price range: {buy_price_range_pct}%")
    print(f"  Profit mode: overall")
    
    # Create calculator
    calculator = StaggeredLadderCalculator(budget, num_rungs)
    
    # Use profit target method with exponential allocation and overall profit mode
    # Use current_price as the upper buy price (buy_upper)
    results = calculator.calculate_from_profit_target(
        current_price, profit_target, buy_price_range_pct
    )
    
    return results


def main():
    """Main function with command-line argument support."""
    parser = argparse.ArgumentParser(
        description='Staggered Order Ladder Calculator - Uses exponential allocation and overall profit mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Minimal input (only budget and price required)
  python staggered_ladder.py --budget 10000 --price 50
  
  # Customize profit target and number of orders
  python staggered_ladder.py --budget 10000 --price 50 --profit-target 75 --num-rungs 15

Defaults:
  - Profit target: 75%
  - Number of orders: 10
  - Allocation: exponential (buy more at lower prices, sell more at higher prices)
  - Buy price range: 30%
  - Profit mode: overall
        """
    )
    
    parser.add_argument(
        '--budget',
        type=float,
        help='Total budget for buy orders (required if not using interactive mode)'
    )
    parser.add_argument(
        '--price',
        type=float,
        help='Current price (required if not using interactive mode)'
    )
    parser.add_argument(
        '--profit-target',
        type=float,
        default=75.0,
        help='Target profit percentage (default: 75%%)'
    )
    parser.add_argument(
        '--num-rungs',
        type=int,
        default=10,
        help='Number of orders to place (default: 10)'
    )
    parser.add_argument(
        '--price-range',
        type=float,
        default=30.0,
        help='Buy price range percentage (default: 30%%)'
    )
    parser.add_argument(
        '--output-prefix',
        type=str,
        default=None,
        help='Optional prefix for output filenames'
    )
    
    args = parser.parse_args()
    
    try:
        results = get_smart_input(args)
        
        # Create output directory relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{args.output_prefix}_" if args.output_prefix else ""
        excel_filename = os.path.join(output_dir, f"{prefix}staggered_ladder_{timestamp}.xlsx")
        pdf_filename = os.path.join(output_dir, f"{prefix}staggered_ladder_{timestamp}.pdf")
        
        # Generate outputs
        print("\nGenerating outputs...")
        generate_excel(results, excel_filename)
        generate_pdf(results, pdf_filename)
        
        print(f"\n[OK] Excel file generated: {excel_filename}")
        print(f"[OK] PDF file generated: {pdf_filename}")
        
        # Calculate total volumes (quantities)
        total_buy_volume = sum(results['buy_quantities'])
        total_sell_volume = sum(results['sell_quantities'])
        
        print(f"\nSummary:")
        print(f"  Total Cost: ${results['total_cost']:,.2f}")
        print(f"  Total Revenue: ${results['total_revenue']:,.2f}")
        print(f"  Total Profit: ${results['total_profit']:,.2f}")
        print(f"  Profit Percentage: {results['profit_pct']:.2f}%")
        print(f"  Total Buy Volume: {total_buy_volume:,.4f}")
        print(f"  Total Sell Volume: {total_sell_volume:,.4f}")
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()

