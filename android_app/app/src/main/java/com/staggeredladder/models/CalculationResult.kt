package com.staggeredladder.models

/**
 * Container for all calculation results
 */
data class CalculationResult(
    val budget: Double,
    val numRungs: Int,
    val buyOrders: List<OrderRung>,
    val sellOrders: List<OrderRung>,
    val totalCost: Double,
    val totalRevenue: Double,
    val totalProfit: Double,
    val profitPercentage: Double,
    val totalBuyQuantity: Double,
    val totalSellQuantity: Double
)


