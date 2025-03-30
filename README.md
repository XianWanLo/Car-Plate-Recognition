# Car-Plate-Recognition

![Project Banner](https://via.placeholder.com/1000x300?text=Car-Plate-Recognition)

## 📌 Overview
This project focuses on license plate recognition using image processing techniques. The system is designed to handle images of varying difficulty levels (**EASY, MEDIUM, and HARD**) by implementing different preprocessing and recognition strategies.

## 📷 Difficulty Levels & Methodology

### 🟢 EASY
✅ Clear license plates without background interference.
- **Techniques Used:**
  - License plate color recognition
  - Morphological image processing
  - Character segmentation
  - Character recognition

### 🟡 MEDIUM
✅ Vehicles and surrounding objects present, blue license plates.
- **Techniques Used:**
  - License plate localization
  - Morphological image processing
  - Character segmentation
  - Character recognition

### 🔴 HARD
✅ Vehicles, objects, varying plate colors, and large inclination angles.
- **Techniques Used:**
  - License plate localization
  - License plate color recognition
  - Angle correction
  - Character segmentation
  - Character recognition

## 🛠 Algorithm Breakdown
The algorithm consists of three key components:

### 1️⃣ License Plate Localization & Image Processing
- Different localization techniques based on difficulty level.
- **EASY** does not require localization, while **MEDIUM** and **HARD** use advanced techniques.

### 2️⃣ Character Segmentation
- A consistent segmentation algorithm isolates characters across all difficulty levels.

### 3️⃣ Character Recognition
- A uniform recognition method extracts the plate number from segmented characters.

## 📝 License
This project is open-source and available under the [MIT License](LICENSE).

---
✨ *Feel free to contribute and star this repository if you find it useful!* ⭐

