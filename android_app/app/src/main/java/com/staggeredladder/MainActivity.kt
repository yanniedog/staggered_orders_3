package com.staggeredladder

import android.os.Bundle
import android.text.TextUtils
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.staggeredladder.models.CalculationResult
import com.staggeredladder.models.OrderRung
import java.text.DecimalFormat

class MainActivity : AppCompatActivity() {
    
    private lateinit var editBudget: com.google.android.material.textfield.TextInputEditText
    private lateinit var editCurrentPrice: com.google.android.material.textfield.TextInputEditText
    private lateinit var editProfitTarget: com.google.android.material.textfield.TextInputEditText
    private lateinit var editNumRungs: com.google.android.material.textfield.TextInputEditText
    private lateinit var editBuyPriceRange: com.google.android.material.textfield.TextInputEditText
    private lateinit var btnCalculate: com.google.android.material.button.MaterialButton
    
    private lateinit var cardSummary: androidx.cardview.widget.CardView
    private lateinit var textTotalCost: TextView
    private lateinit var textTotalRevenue: TextView
    private lateinit var textTotalProfit: TextView
    private lateinit var textProfitPercentage: TextView
    
    private lateinit var textBuyOrdersTitle: TextView
    private lateinit var textSellOrdersTitle: TextView
    private lateinit var recyclerViewBuyOrders: RecyclerView
    private lateinit var recyclerViewSellOrders: RecyclerView
    
    private val calculator = Calculator()
    private val currencyFormat = DecimalFormat("#,##0.00")
    private val quantityFormat = DecimalFormat("#,##0.0000")
    private val percentageFormat = DecimalFormat("#,##0.00")
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        initializeViews()
        setupRecyclerViews()
        
        btnCalculate.setOnClickListener {
            calculateAndDisplay()
        }
    }
    
    private fun initializeViews() {
        editBudget = findViewById(R.id.editBudget)
        editCurrentPrice = findViewById(R.id.editCurrentPrice)
        editProfitTarget = findViewById(R.id.editProfitTarget)
        editNumRungs = findViewById(R.id.editNumRungs)
        editBuyPriceRange = findViewById(R.id.editBuyPriceRange)
        btnCalculate = findViewById(R.id.btnCalculate)
        
        cardSummary = findViewById(R.id.cardSummary)
        textTotalCost = findViewById(R.id.textTotalCost)
        textTotalRevenue = findViewById(R.id.textTotalRevenue)
        textTotalProfit = findViewById(R.id.textTotalProfit)
        textProfitPercentage = findViewById(R.id.textProfitPercentage)
        
        textBuyOrdersTitle = findViewById(R.id.textBuyOrdersTitle)
        textSellOrdersTitle = findViewById(R.id.textSellOrdersTitle)
        recyclerViewBuyOrders = findViewById(R.id.recyclerViewBuyOrders)
        recyclerViewSellOrders = findViewById(R.id.recyclerViewSellOrders)
    }
    
    private fun setupRecyclerViews() {
        recyclerViewBuyOrders.layoutManager = LinearLayoutManager(this)
        recyclerViewSellOrders.layoutManager = LinearLayoutManager(this)
    }
    
    private fun calculateAndDisplay() {
        try {
            // Get and validate inputs
            val budget = getDoubleValue(editBudget, "Budget")
            val currentPrice = getDoubleValue(editCurrentPrice, "Current Price")
            val profitTarget = getDoubleValue(editProfitTarget, "Profit Target")
            val numRungs = getIntValue(editNumRungs, "Number of Rungs")
            val buyPriceRange = getDoubleValue(editBuyPriceRange, "Buy Price Range")
            
            // Validate ranges
            if (budget <= 0) {
                showError("Budget must be positive")
                return
            }
            if (currentPrice <= 0) {
                showError("Current price must be positive")
                return
            }
            if (profitTarget < 10 || profitTarget > 200) {
                showError("Profit target must be between 10% and 200%")
                return
            }
            if (numRungs < 1 || numRungs > 20) {
                showError("Number of rungs must be between 1 and 20")
                return
            }
            if (buyPriceRange <= 0 || buyPriceRange > 100) {
                showError("Buy price range must be between 0% and 100%")
                return
            }
            
            // Perform calculation
            val result = calculator.calculateFromProfitTarget(
                budget = budget,
                buyUpper = currentPrice,
                profitTarget = profitTarget,
                numRungs = numRungs,
                buyPriceRangePct = buyPriceRange
            )
            
            // Display results
            displayResults(result)
            
        } catch (e: IllegalArgumentException) {
            showError(e.message ?: "Invalid input")
        } catch (e: Exception) {
            showError("Calculation error: ${e.message}")
            e.printStackTrace()
        }
    }
    
    private fun getDoubleValue(editText: com.google.android.material.textfield.TextInputEditText, fieldName: String): Double {
        val text = editText.text?.toString()?.trim()
        if (TextUtils.isEmpty(text)) {
            throw IllegalArgumentException("$fieldName is required")
        }
        return try {
            text.toDouble()
        } catch (e: NumberFormatException) {
            throw IllegalArgumentException("$fieldName must be a valid number")
        }
    }
    
    private fun getIntValue(editText: com.google.android.material.textfield.TextInputEditText, fieldName: String): Int {
        val text = editText.text?.toString()?.trim()
        if (TextUtils.isEmpty(text)) {
            throw IllegalArgumentException("$fieldName is required")
        }
        return try {
            text.toInt()
        } catch (e: NumberFormatException) {
            throw IllegalArgumentException("$fieldName must be a valid integer")
        }
    }
    
    private fun showError(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
    }
    
    private fun displayResults(result: CalculationResult) {
        // Display summary
        textTotalCost.text = "$${currencyFormat.format(result.totalCost)}"
        textTotalRevenue.text = "$${currencyFormat.format(result.totalRevenue)}"
        textTotalProfit.text = "$${currencyFormat.format(result.totalProfit)}"
        textProfitPercentage.text = "${percentageFormat.format(result.profitPercentage)}%"
        
        cardSummary.visibility = View.VISIBLE
        
        // Display buy orders
        recyclerViewBuyOrders.adapter = OrderAdapter(result.buyOrders, true)
        textBuyOrdersTitle.visibility = View.VISIBLE
        recyclerViewBuyOrders.visibility = View.VISIBLE
        
        // Display sell orders
        recyclerViewSellOrders.adapter = OrderAdapter(result.sellOrders, false)
        textSellOrdersTitle.visibility = View.VISIBLE
        recyclerViewSellOrders.visibility = View.VISIBLE
    }
    
    private inner class OrderAdapter(
        private val orders: List<OrderRung>,
        private val isBuy: Boolean
    ) : RecyclerView.Adapter<OrderAdapter.ViewHolder>() {
        
        inner class ViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
            val textRungNumber: TextView = itemView.findViewById(R.id.textRungNumber)
            val textPrice: TextView = itemView.findViewById(R.id.textPrice)
            val textQuantity: TextView = itemView.findViewById(R.id.textQuantity)
            val textCostRevenue: TextView = itemView.findViewById(R.id.textCostRevenue)
            val textCumulative: TextView = itemView.findViewById(R.id.textCumulative)
            val textAveragePrice: TextView = itemView.findViewById(R.id.textAveragePrice)
        }
        
        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
            val view = LayoutInflater.from(parent.context)
                .inflate(R.layout.item_order_row, parent, false)
            return ViewHolder(view)
        }
        
        override fun onBindViewHolder(holder: ViewHolder, position: Int) {
            if (position == 0) {
                // Header row
                holder.textRungNumber.text = "Rung"
                holder.textPrice.text = "Price"
                holder.textQuantity.text = "Qty"
                holder.textCostRevenue.text = if (isBuy) "Cost" else "Revenue"
                holder.textCumulative.text = "Cumulative"
                holder.textAveragePrice.text = "Avg Price"
                
                // Style header
                val whiteColor = ContextCompat.getColor(holder.itemView.context, android.R.color.white)
                holder.textRungNumber.setTextColor(whiteColor)
                holder.textPrice.setTextColor(whiteColor)
                holder.textQuantity.setTextColor(whiteColor)
                holder.textCostRevenue.setTextColor(whiteColor)
                holder.textCumulative.setTextColor(whiteColor)
                holder.textAveragePrice.setTextColor(whiteColor)
                
                val headerBgColor = if (isBuy) {
                    ContextCompat.getColor(holder.itemView.context, android.R.color.holo_green_dark)
                } else {
                    ContextCompat.getColor(holder.itemView.context, android.R.color.holo_red_dark)
                }
                holder.itemView.setBackgroundColor(headerBgColor)
            } else {
                val order = orders[position - 1]
                holder.textRungNumber.text = order.rungNumber.toString()
                holder.textPrice.text = "$${currencyFormat.format(order.price)}"
                holder.textQuantity.text = quantityFormat.format(order.quantity)
                holder.textCostRevenue.text = "$${currencyFormat.format(order.costOrRevenue)}"
                holder.textCumulative.text = "$${currencyFormat.format(order.cumulativeCostOrRevenue)}"
                holder.textAveragePrice.text = "$${currencyFormat.format(order.averagePrice)}"
                
                // Reset colors for data rows
                val blackColor = ContextCompat.getColor(holder.itemView.context, android.R.color.black)
                holder.textRungNumber.setTextColor(blackColor)
                holder.textPrice.setTextColor(blackColor)
                holder.textQuantity.setTextColor(blackColor)
                holder.textCostRevenue.setTextColor(blackColor)
                holder.textCumulative.setTextColor(blackColor)
                holder.textAveragePrice.setTextColor(blackColor)
                
                val whiteColor = ContextCompat.getColor(holder.itemView.context, android.R.color.white)
                holder.itemView.setBackgroundColor(whiteColor)
            }
        }
        
        override fun getItemCount(): Int = orders.size + 1 // +1 for header
    }
}

