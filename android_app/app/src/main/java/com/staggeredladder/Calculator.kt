package com.staggeredladder

import com.staggeredladder.models.CalculationResult
import com.staggeredladder.models.OrderRung
import kotlin.math.*

/**
 * Calculate staggered order ladders for buy and sell orders.
 * Ported from Python staggered_ladder.py
 */
class Calculator {
    
    /**
     * Calculate ladder from upper buy rung and profit target.
     * Uses exponential allocation strategy (buy more at lower prices, sell more at higher prices).
     */
    fun calculateFromProfitTarget(
        budget: Double,
        buyUpper: Double,
        profitTarget: Double,
        numRungs: Int,
        buyPriceRangePct: Double = 30.0
    ): CalculationResult {
        // Input validation
        require(budget > 0) { "Budget must be positive" }
        require(buyUpper > 0) { "Buy upper must be positive" }
        require(profitTarget >= 10 && profitTarget <= 200) { "Profit target must be between 10% and 200%" }
        require(buyPriceRangePct > 0 && buyPriceRangePct <= 100) { "Buy price range percentage must be between 0 and 100" }
        require(numRungs >= 1) { "Number of rungs must be at least 1" }
        
        val profitMultiplier = 1 + (profitTarget / 100)
        
        // Calculate buy prices: ascending order (lowest to highest)
        val priceRange = buyUpper * (buyPriceRangePct / 100)
        val buyLower = buyUpper - priceRange
        require(buyLower > 0) { 
            "Buy price range results in non-positive buy_lower. Try reducing buy_price_range_pct from ${buyPriceRangePct}%" 
        }
        
        val buyPrices = linspace(buyLower, buyUpper, numRungs)
        
        // Calculate consistent price step from buy prices
        val priceStep = if (numRungs > 1) {
            (buyUpper - buyLower) / (numRungs - 1)
        } else {
            buyUpper * 0.1
        }
        
        // Calculate buy quantities using exponential allocation (buy more at lower prices)
        val buyWeights = calculateExponentialWeights(numRungs, 2.0, 0.0)
        val buyQuantities = buyPrices.mapIndexed { i, price ->
            (budget * buyWeights[i]) / price
        }
        
        // Normalize to exactly match budget
        val normalizedBuyQuantities = normalizeBudget(buyPrices, buyQuantities, budget)
        
        // Calculate total cost (should equal budget after normalization)
        val totalCost = normalizedBuyQuantities.zip(buyPrices).sumOf { (qty, price) -> qty * price }
        
        // Calculate target revenue
        val targetRevenue = totalCost * profitMultiplier
        
        // Calculate total buy quantity
        val totalBuyQty = normalizedBuyQuantities.sum()
        
        // Allocate sell quantities using exponential allocation (sell more at higher prices)
        val sellWeights = calculateExponentialWeights(numRungs, 0.0, 2.0)
        val sellQuantities = sellWeights.map { totalBuyQty * it }
        
        // Ensure sell quantities match buy quantities exactly
        val normalizedSellQuantities = normalizeQuantities(sellQuantities, totalBuyQty)
        
        // Calculate gap and minimum sell price
        val maxBuyPrice = buyPrices.maxOrNull() ?: buyUpper
        val baseGapPct = profitTarget / 100.0
        val baseGapSize = maxBuyPrice * baseGapPct * 0.5
        val gapSteps = max(1, ceil(baseGapSize / priceStep).toInt())
        val gapSize = gapSteps * priceStep
        val minSellPrice = maxBuyPrice + gapSize
        
        // Create sell prices with consistent spacing
        val sellPrices = (0 until numRungs).map { minSellPrice + it * priceStep }
        
        // Calculate current revenue
        var currentRevenue = sellPrices.zip(normalizedSellQuantities).sumOf { (price, qty) -> price * qty }
        
        // Adjust sell prices to achieve target revenue
        var adjustedSellPrices = sellPrices.toMutableList()
        if (currentRevenue > 0 && abs(currentRevenue - targetRevenue) > 0.01) {
            val priceScale = targetRevenue / currentRevenue
            adjustedSellPrices = sellPrices.map { it * priceScale }.toMutableList()
            
            // Ensure gap is maintained after scaling
            val maxBuyPriceAfter = buyPrices.maxOrNull() ?: buyUpper
            val minSellPriceAfter = adjustedSellPrices.minOrNull() ?: minSellPrice
            val requiredGap = priceStep
            val minValidSellPrice = maxBuyPriceAfter + requiredGap
            
            if (minSellPriceAfter < minValidSellPrice) {
                val priceShift = minValidSellPrice - minSellPriceAfter
                adjustedSellPrices = adjustedSellPrices.map { it + priceShift }.toMutableList()
                currentRevenue = adjustedSellPrices.zip(normalizedSellQuantities).sumOf { (price, qty) -> price * qty }
            }
        }
        
        // Final quantity matching check
        val finalTotalBuyQty = normalizedBuyQuantities.sum()
        val finalTotalSellQty = normalizedSellQuantities.sum()
        val qtyDiff = abs(finalTotalBuyQty - finalTotalSellQty)
        
        val finalSellQuantities = if (qtyDiff > 0.0001) {
            normalizeQuantities(normalizedSellQuantities, finalTotalBuyQty)
        } else {
            normalizedSellQuantities
        }
        
        // Validate that sell prices exceed buy prices
        validatePrices(buyPrices, adjustedSellPrices)
        
        // Generate results
        return generateResults(
            budget,
            numRungs,
            buyPrices,
            normalizedBuyQuantities,
            adjustedSellPrices,
            finalSellQuantities
        )
    }
    
    /**
     * Calculate exponential weights for allocation.
     * For buy: startVal=2.0, endVal=0.0 (higher weight for lower prices)
     * For sell: startVal=0.0, endVal=2.0 (higher weight for higher prices)
     */
    private fun calculateExponentialWeights(count: Int, startVal: Double, endVal: Double): List<Double> {
        if (count == 1) {
            return listOf(1.0)
        }
        
        val weights = (0 until count).map { i ->
            val ratio = i / (count - 1).toDouble()
            val linearVal = startVal + (endVal - startVal) * ratio
            exp(linearVal)
        }
        
        // Normalize weights to sum to 1.0
        val sumWeights = weights.sum()
        return if (sumWeights > 0) {
            weights.map { it / sumWeights }
        } else {
            weights
        }
    }
    
    /**
     * Create evenly spaced values between start and end (inclusive).
     */
    private fun linspace(start: Double, end: Double, count: Int): List<Double> {
        if (count == 1) {
            return listOf(end)
        }
        return (0 until count).map { i ->
            start + (end - start) * i / (count - 1).toDouble()
        }
    }
    
    /**
     * Normalize buy quantities to exactly match budget.
     */
    private fun normalizeBudget(
        prices: List<Double>,
        quantities: List<Double>,
        budget: Double
    ): List<Double> {
        val totalCost = quantities.zip(prices).sumOf { (qty, price) -> qty * price }
        return if (totalCost > 0) {
            val scaleFactor = budget / totalCost
            quantities.map { it * scaleFactor }
        } else {
            quantities
        }
    }
    
    /**
     * Normalize quantities to match target exactly.
     */
    private fun normalizeQuantities(quantities: List<Double>, target: Double): List<Double> {
        val total = quantities.sum()
        return if (total > 0 && abs(total - target) > 0.0001) {
            val normalizeFactor = target / total
            quantities.map { it * normalizeFactor }
        } else {
            quantities
        }
    }
    
    /**
     * Validate that sell prices exceed buy prices for all rungs.
     */
    private fun validatePrices(buyPrices: List<Double>, sellPrices: List<Double>) {
        for (i in buyPrices.indices) {
            require(sellPrices[i] > buyPrices[i]) {
                "Sell price ${String.format("%.2f", sellPrices[i])} at rung ${i + 1} must exceed buy price ${String.format("%.2f", buyPrices[i])}"
            }
        }
        
        val maxBuyPrice = buyPrices.maxOrNull() ?: 0.0
        val minSellPrice = sellPrices.minOrNull() ?: 0.0
        require(maxBuyPrice < minSellPrice) {
            "Top buy order (${String.format("%.2f", maxBuyPrice)}) must be below bottom sell order (${String.format("%.2f", minSellPrice)}). Gap needed: ${String.format("%.2f", minSellPrice - maxBuyPrice)}"
        }
    }
    
    /**
     * Generate results dictionary with statistics.
     */
    private fun generateResults(
        budget: Double,
        numRungs: Int,
        buyPrices: List<Double>,
        buyQuantities: List<Double>,
        sellPrices: List<Double>,
        sellQuantities: List<Double>
    ): CalculationResult {
        // Calculate average buy prices (from index i to end)
        val avgBuyPrices = (0 until numRungs).map { i ->
            var totalCostFromLevel = 0.0
            var totalQtyFromLevel = 0.0
            
            for (j in i until numRungs) {
                totalCostFromLevel += buyQuantities[j] * buyPrices[j]
                totalQtyFromLevel += buyQuantities[j]
            }
            
            if (totalQtyFromLevel > 0) {
                totalCostFromLevel / totalQtyFromLevel
            } else {
                0.0
            }
        }
        
        // Verify highest buy average equals highest buy price
        val highestIndex = numRungs - 1
        val expectedHighestAvg = buyPrices[highestIndex]
        val actualHighestAvg = avgBuyPrices[highestIndex]
        val correctedAvgBuyPrices = avgBuyPrices.toMutableList()
        if (abs(actualHighestAvg - expectedHighestAvg) > 0.0001) {
            correctedAvgBuyPrices[highestIndex] = expectedHighestAvg
        }
        
        // Calculate average sell prices (from index 0 to i)
        val avgSellPrices = (0 until numRungs).map { i ->
            var totalRevenueToLevel = 0.0
            var totalQtyToLevel = 0.0
            
            for (j in 0..i) {
                totalRevenueToLevel += sellQuantities[j] * sellPrices[j]
                totalQtyToLevel += sellQuantities[j]
            }
            
            if (totalQtyToLevel > 0) {
                totalRevenueToLevel / totalQtyToLevel
            } else {
                0.0
            }
        }
        
        // Verify first sell average equals first sell price
        val expectedFirstAvg = sellPrices[0]
        val actualFirstAvg = avgSellPrices[0]
        val correctedAvgSellPrices = avgSellPrices.toMutableList()
        if (abs(actualFirstAvg - expectedFirstAvg) > 0.0001) {
            correctedAvgSellPrices[0] = expectedFirstAvg
        }
        
        // Calculate cumulative values
        val cumulativeBuyCost = mutableListOf<Double>()
        val cumulativeBuyQty = mutableListOf<Double>()
        val cumulativeSellRevenue = mutableListOf<Double>()
        val cumulativeSellQty = mutableListOf<Double>()
        
        var runningBuyCost = 0.0
        var runningBuyQty = 0.0
        var runningSellRevenue = 0.0
        var runningSellQty = 0.0
        
        for (i in 0 until numRungs) {
            runningBuyCost += buyQuantities[i] * buyPrices[i]
            runningBuyQty += buyQuantities[i]
            cumulativeBuyCost.add(runningBuyCost)
            cumulativeBuyQty.add(runningBuyQty)
            
            runningSellRevenue += sellQuantities[i] * sellPrices[i]
            runningSellQty += sellQuantities[i]
            cumulativeSellRevenue.add(runningSellRevenue)
            cumulativeSellQty.add(runningSellQty)
        }
        
        // Final quantity matching verification
        val totalBuyQty = buyQuantities.sum()
        val totalSellQty = sellQuantities.sum()
        val qtyDiff = abs(totalBuyQty - totalSellQty)
        
        val finalSellQuantities = if (qtyDiff > 0.0001 && totalSellQty > 0) {
            val normalizeFactor = totalBuyQty / totalSellQty
            sellQuantities.map { it * normalizeFactor }
        } else {
            sellQuantities
        }
        
        // Recalculate sell statistics if quantities were corrected
        val finalAvgSellPrices = if (qtyDiff > 0.0001) {
            val recalculated = (0 until numRungs).map { i ->
                var totalRevenueToLevel = 0.0
                var totalQtyToLevel = 0.0
                
                for (j in 0..i) {
                    totalRevenueToLevel += finalSellQuantities[j] * sellPrices[j]
                    totalQtyToLevel += finalSellQuantities[j]
                }
                
                if (totalQtyToLevel > 0) {
                    totalRevenueToLevel / totalQtyToLevel
                } else {
                    0.0
                }
            }.toMutableList()
            
            // Verify first sell average equals first sell price
            if (abs(recalculated[0] - sellPrices[0]) > 0.0001) {
                recalculated[0] = sellPrices[0]
            }
            recalculated
        } else {
            correctedAvgSellPrices
        }
        
        val finalCumulativeSellRevenue = if (qtyDiff > 0.0001) {
            var runningRev = 0.0
            (0 until numRungs).map { i ->
                runningRev += finalSellQuantities[i] * sellPrices[i]
                runningRev
            }
        } else {
            cumulativeSellRevenue
        }
        
        val finalCumulativeSellQty = if (qtyDiff > 0.0001) {
            var runningQty = 0.0
            (0 until numRungs).map { i ->
                runningQty += finalSellQuantities[i]
                runningQty
            }
        } else {
            cumulativeSellQty
        }
        
        val totalCost = cumulativeBuyCost.lastOrNull() ?: 0.0
        val totalRevenue = finalCumulativeSellRevenue.lastOrNull() ?: 0.0
        val totalProfit = totalRevenue - totalCost
        val profitPct = if (totalCost > 0) (totalProfit / totalCost * 100) else 0.0
        
        // Create OrderRung objects
        val buyOrders = (0 until numRungs).map { i ->
            OrderRung(
                rungNumber = i + 1,
                price = buyPrices[i],
                quantity = buyQuantities[i],
                costOrRevenue = buyQuantities[i] * buyPrices[i],
                cumulativeCostOrRevenue = cumulativeBuyCost[i],
                cumulativeQuantity = cumulativeBuyQty[i],
                averagePrice = correctedAvgBuyPrices[i]
            )
        }
        
        val sellOrders = (0 until numRungs).map { i ->
            OrderRung(
                rungNumber = i + 1,
                price = sellPrices[i],
                quantity = finalSellQuantities[i],
                costOrRevenue = finalSellQuantities[i] * sellPrices[i],
                cumulativeCostOrRevenue = finalCumulativeSellRevenue[i],
                cumulativeQuantity = finalCumulativeSellQty[i],
                averagePrice = finalAvgSellPrices[i]
            )
        }
        
        return CalculationResult(
            budget = budget,
            numRungs = numRungs,
            buyOrders = buyOrders,
            sellOrders = sellOrders,
            totalCost = totalCost,
            totalRevenue = totalRevenue,
            totalProfit = totalProfit,
            profitPercentage = profitPct,
            totalBuyQuantity = totalBuyQty,
            totalSellQuantity = finalSellQuantities.sum()
        )
    }
}

