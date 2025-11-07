# Staggered Ladder Calculator - Android App Installation Guide

This guide provides step-by-step instructions for installing the Staggered Ladder Calculator Android app on your Android phone.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Method 1: Install via Android Studio (Recommended)](#method-1-install-via-android-studio-recommended)
3. [Method 2: Install via APK File](#method-2-install-via-apk-file)
4. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before you begin, ensure you have the following:

### Required Software:
1. **Android Studio** (latest version recommended)
   - Download from: https://developer.android.com/studio
   - Minimum version: Android Studio Hedgehog (2023.1.1) or later
   - Includes Android SDK, Android Emulator, and all necessary tools

2. **Java Development Kit (JDK)**
   - JDK 8 or higher (usually included with Android Studio)
   - Verify installation: Open terminal/command prompt and type `java -version`

### Required Hardware:
- **Android Phone** running Android 7.0 (Nougat) or higher (API level 24+)
- **USB Cable** to connect your phone to your computer
- **Computer** running Windows, macOS, or Linux

---

## Method 1: Install via Android Studio (Recommended)

This method allows you to build and install the app directly from Android Studio.

### Step 1: Download/Clone the Project

1. Navigate to the project directory:
   ```bash
   cd android_app
   ```

2. If you're using Git, you can clone the repository. Otherwise, ensure all project files are in the `android_app` directory.

### Step 2: Open Project in Android Studio

1. **Launch Android Studio**

2. **Open the Project:**
   - Click "Open" on the welcome screen
   - Navigate to the `android_app` folder
   - Select the folder and click "OK"
   - Android Studio will start syncing Gradle files (this may take a few minutes on first launch)

3. **Wait for Gradle Sync:**
   - Android Studio will automatically download dependencies
   - You'll see "Gradle sync finished" in the status bar when complete
   - If prompted, accept any license agreements

### Step 3: Enable Developer Options on Your Android Phone

1. **Open Settings** on your Android phone

2. **Navigate to About Phone:**
   - Scroll down and tap "About Phone" (or "About Device")
   - Location varies by manufacturer (Samsung: Settings > About Phone, OnePlus: Settings > About Phone, etc.)

3. **Find Build Number:**
   - Scroll down to find "Build Number" or "Build Version"
   - Tap it **7 times** rapidly
   - You'll see a message saying "You are now a developer!" or similar

4. **Go Back to Settings:**
   - Press the back button
   - You should now see "Developer Options" in the Settings menu

### Step 4: Enable USB Debugging

1. **Open Developer Options:**
   - In Settings, tap "Developer Options" (or "Developer Settings")
   - If you don't see it, go back and tap Build Number 7 more times

2. **Enable USB Debugging:**
   - Toggle "USB Debugging" to ON
   - You may see a warning dialog - tap "OK" or "Allow"
   - Some phones also have "Install via USB" - enable this too

3. **Enable Stay Awake (Optional but Recommended):**
   - Toggle "Stay Awake" to ON
   - This keeps your screen on while charging via USB

### Step 5: Connect Your Phone to Computer

1. **Connect via USB:**
   - Use a USB cable to connect your phone to your computer
   - Use a data cable (not just a charging cable)

2. **Authorize USB Debugging:**
   - On your phone, you'll see a popup: "Allow USB debugging?"
   - Check "Always allow from this computer" (optional but recommended)
   - Tap "OK" or "Allow"

3. **Verify Connection:**
   - In Android Studio, open the terminal (View > Tool Windows > Terminal)
   - Type: `adb devices`
   - You should see your device listed (e.g., "ABC123XYZ    device")
   - If you see "unauthorized", check your phone for the authorization prompt

### Step 6: Build and Install the App

1. **Select Your Device:**
   - In Android Studio, look at the top toolbar
   - Click the device dropdown (next to the Run button)
   - Select your connected phone from the list

2. **Build the Project:**
   - Click "Build" > "Make Project" (or press Ctrl+F9 / Cmd+F9)
   - Wait for the build to complete (check the Build output window)

3. **Run the App:**
   - Click the green "Run" button (or press Shift+F10 / Ctrl+R)
   - Android Studio will:
     - Build the APK
     - Install it on your phone
     - Launch the app automatically

4. **First Launch:**
   - The app will open on your phone
   - You may see a security prompt - tap "Install" or "Allow"
   - The app is now installed and ready to use!

---

## Method 2: Install via APK File

This method generates an APK file that you can install directly on your phone without Android Studio.

### Step 1: Generate APK File

1. **Open Project in Android Studio** (follow Steps 1-2 from Method 1)

2. **Build APK:**
   - Click "Build" > "Build Bundle(s) / APK(s)" > "Build APK(s)"
   - Wait for the build to complete
   - You'll see a notification: "APK(s) generated successfully"

3. **Locate APK File:**
   - Click "locate" in the notification, OR
   - Navigate to: `android_app/app/build/outputs/apk/debug/app-debug.apk`
   - Copy this file to your phone (via USB, email, cloud storage, etc.)

### Step 2: Enable Unknown Sources on Your Phone

1. **Open Settings** on your Android phone

2. **Navigate to Security (or Apps):**
   - Settings > Security (or Settings > Apps > Special Access)
   - Look for "Install Unknown Apps" or "Unknown Sources"

3. **Enable Installation:**
   - Tap on the app you'll use to install (e.g., "Files", "Chrome", "Email")
   - Toggle "Allow from this source" to ON
   - Different Android versions have different locations:
     - **Android 8.0+**: Settings > Apps > Special Access > Install Unknown Apps
     - **Android 7.1 and below**: Settings > Security > Unknown Sources

### Step 3: Install APK on Your Phone

1. **Transfer APK to Phone:**
   - Copy `app-debug.apk` to your phone via:
     - USB file transfer
     - Email attachment
     - Cloud storage (Google Drive, Dropbox, etc.)
     - Bluetooth

2. **Open APK File:**
   - On your phone, use a file manager app (Files, My Files, etc.)
   - Navigate to where you saved the APK
   - Tap on `app-debug.apk`

3. **Install:**
   - You'll see an installation screen
   - Tap "Install"
   - Wait for installation to complete
   - Tap "Open" to launch the app, or find it in your app drawer

---

## Troubleshooting

### Problem: "Device not found" or "No devices detected"

**Solutions:**
- Ensure USB debugging is enabled (see Step 4 above)
- Try a different USB cable (use a data cable, not just charging)
- Try a different USB port on your computer
- On your phone, revoke USB debugging authorizations and reconnect
- Install/update USB drivers for your phone manufacturer
- Restart ADB: In Android Studio terminal, type:
  ```bash
  adb kill-server
  adb start-server
  adb devices
  ```

### Problem: "Gradle sync failed"

**Solutions:**
- Check your internet connection (Gradle downloads dependencies)
- In Android Studio: File > Invalidate Caches / Restart
- Check if you have the latest Android SDK installed:
  - Tools > SDK Manager > SDK Platforms (install latest)
  - Tools > SDK Manager > SDK Tools (ensure all are updated)

### Problem: "Build failed" or compilation errors

**Solutions:**
- Ensure you're using Android Studio Hedgehog or later
- Check that all files are in the correct directories
- Clean and rebuild: Build > Clean Project, then Build > Rebuild Project
- Check the Build output window for specific error messages

### Problem: "App won't install" or "Installation blocked"

**Solutions:**
- Ensure "Unknown Sources" or "Install Unknown Apps" is enabled
- Free up storage space on your phone
- Uninstall any previous version of the app first
- Check if your phone has any security apps blocking installation

### Problem: "App crashes on launch"

**Solutions:**
- Ensure your phone is running Android 7.0 (API 24) or higher
- Check Android Studio's Logcat for error messages
- Try uninstalling and reinstalling the app
- Clear app data: Settings > Apps > Staggered Ladder > Storage > Clear Data

### Problem: "USB debugging authorization keeps appearing"

**Solutions:**
- When the prompt appears, check "Always allow from this computer"
- Revoke all USB debugging authorizations and reconnect:
  - Settings > Developer Options > Revoke USB debugging authorizations

---

## Using the App

1. **Launch the App:**
   - Find "Staggered Ladder" in your app drawer
   - Tap to open

2. **Enter Parameters:**
   - **Total Budget ($)**: Your total investment budget
   - **Current Price ($)**: Current market price (highest buy rung)
   - **Profit Target (%)**: Target profit percentage (10-200%)
   - **Number of Rungs**: Number of buy/sell orders (1-20)
   - **Buy Price Range (%)**: Price range for buy ladder (0-100%)

3. **Calculate:**
   - Tap "Calculate" button
   - Results will appear below

4. **View Results:**
   - Summary shows total cost, revenue, profit, and profit percentage
   - Buy Orders table shows all buy order details
   - Sell Orders table shows all sell order details
   - Scroll to see all orders

---

## Additional Notes

- **Minimum Android Version**: Android 7.0 (Nougat) - API level 24
- **Target Android Version**: Android 14 - API level 34
- **App Permissions**: None required (no internet, storage, or other permissions needed)
- **Data Storage**: All calculations are done locally - no data is sent anywhere

---

## Support

If you encounter any issues not covered in this guide:

1. Check the Android Studio Logcat for error messages
2. Verify all prerequisites are met
3. Ensure your Android version is compatible (7.0+)
4. Try uninstalling and reinstalling the app

---

## Building a Release APK (Optional)

To create a release APK for distribution:

1. **Generate Signed Bundle/APK:**
   - Build > Generate Signed Bundle / APK
   - Select "APK"
   - Create a new keystore (or use existing)
   - Follow the signing wizard
   - Build variant: release

2. **Find Release APK:**
   - Location: `android_app/app/release/app-release.apk`
   - This APK is signed and ready for distribution

---

**Congratulations!** You've successfully installed the Staggered Ladder Calculator app on your Android phone. Enjoy calculating your staggered order ladders!

