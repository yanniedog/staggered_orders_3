# Staggered Ladder Calculator - Android App

A native Android application for calculating staggered buy/sell order ladders for trading.

## Features

- Calculate buy and sell order ladders based on budget and profit targets
- Exponential allocation strategy (buy more at lower prices, sell more at higher prices)
- Volume matching ensures buy and sell quantities match exactly
- Clean, Material Design 3 interface
- Real-time calculation and validation
- Scrollable tables for viewing all orders

## Requirements

- Android 7.0 (Nougat) or higher (API level 24+)
- Android Studio Hedgehog (2023.1.1) or later for building

## Quick Start

1. Open the `android_app` folder in Android Studio
2. Wait for Gradle sync to complete
3. Connect your Android phone via USB
4. Enable USB debugging on your phone
5. Click Run to build and install

See [INSTALLATION_INSTRUCTIONS.md](INSTALLATION_INSTRUCTIONS.md) for detailed installation steps.

## Project Structure

```
android_app/
├── app/
│   ├── src/
│   │   └── main/
│   │       ├── java/com/staggeredladder/
│   │       │   ├── MainActivity.kt
│   │       │   ├── Calculator.kt
│   │       │   └── models/
│   │       │       ├── OrderRung.kt
│   │       │       └── CalculationResult.kt
│   │       ├── res/
│   │       │   ├── layout/
│   │       │   │   ├── activity_main.xml
│   │       │   │   └── item_order_row.xml
│   │       │   └── values/
│   │       └── AndroidManifest.xml
│   └── build.gradle.kts
├── build.gradle.kts
├── settings.gradle.kts
└── gradle.properties
```

## License

This project is part of the staggered_orders_3 repository.

