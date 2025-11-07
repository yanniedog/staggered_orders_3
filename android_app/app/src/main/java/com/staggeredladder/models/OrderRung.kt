package com.staggeredladder.models

/**
 * Represents a single order rung (buy or sell)
 */
data class OrderRung(
    val rungNumber: Int,
    val price: Double,
    val quantity: Double,
    val costOrRevenue: Double,
    val cumulativeCostOrRevenue: Double,
    val cumulativeQuantity: Double,
    val averagePrice: Double
)

