# 🩺 Chest X-ray Diagnosis Web App using Liquid Neural Networks

A Flask-based web application that analyzes chest X-ray images using **Liquid Neural Networks (LNNs)** for detecting **lung opacity** and **pneumonia**, and combines results using an **ensemble method** for improved diagnostic accuracy. Designed for **bulk prediction** and **Excel report generation** with confidence-based ranking.

---

## 📌 Project Highlights

- 🧠 **Two Expert Models**:
  - **Model 1**: Detects **Lung Opacity**
  - **Model 2**: Detects **Pneumonia**
- ⚖️ **Ensemble Method**: Combines both model outputs to generate final predictions
- 🗂️ **Bulk Upload Support**: Upload **1000+ chest X-ray images** at once
- 📊 **Excel Report Output**: Generates an `.xlsx` file with:
  - File name
  - Predictions (for both models)
  - Final ensemble prediction
  - Confidence scores
  - Ranking from high to low risk

---

## 🧠 Theory & Architecture

### 🔬 Liquid Neural Networks (LNNs)
Liquid Neural Networks dynamically adapt their internal parameters and are highly effective for sequential and medical imaging tasks. This project uses two separate fine-tuned LNNs:
- One for detecting **lung opacity**
- One for detecting **pneumonia**

### 🔗 Ensemble Method
Combines the output probabilities or votes of both models to produce a single, robust final prediction with a confidence score.

---

## 🖼️ Screenshots

### 🧾 Web Interface
![Web Interface](https://github.com/AniketOvhal18/Chest-X-Ray-disease-detection-using-Liquid-Neural-Network-LNN-/blob/main/Screenshot%202025-07-22%20145728.png)

### 🖼️ Upload Page
![Upload Page](https://github.com/AniketOvhal18/Chest-X-Ray-disease-detection-using-Liquid-Neural-Network-LNN-/blob/main/Screenshot%202025-07-22%20145834.png)

### 🔍 Ensemble Prediction Visual
![Ensemble Graph](https://github.com/AniketOvhal18/Chest-X-Ray-disease-detection-using-Liquid-Neural-Network-LNN-/blob/main/Screenshot%202025-07-22%20145950.png)

### 📊 Sample Excel Output
![Excel Output](https://github.com/AniketOvhal18/Chest-X-Ray-disease-detection-using-Liquid-Neural-Network-LNN-/blob/main/Screenshot%202025-07-22%20150207.png)


> 🔁 Add your own screenshots in the `/screenshots` folder and update paths accordingly.

---

## 🗂️ Dataset Used

- [NIH Chest X-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [Kaggle Pneumonia Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)


---

