# StepCNN-GNSS: An AI Model Enhancing Step Detection and Site Velocity Estimation from Global GNSS Data
Guoquan Wang et al.

3/15/2025

gwang@uh.edu

**Abstract**

Estimating long-term site velocities from Global Navigation Satellite System (GNSS) time series is vital for tectonic motion and hazard analysis. However, this estimation is hindered by step discontinuities caused by earthquakes, equipment changes, and other unexplained sources. Applying linear regression to entire time series often yields erroneous velocities. We propose a two-stage hybrid framework for automated step detection in GNSS-derived displacement time series. The framework integrates: (1) analytical methods, including a sliding-window algorithm for abrupt-step detection and a weighted cubic polynomial fit for gradual-step detection via curvature analysis; and (2) the artificial intelligence (AI) model StepCNN-GNSS, which quantitatively assesses step-detection quality and iteratively optimizes analytical parameters. This dual approach ensures robust step detection. The AI model is trained on a Convolutional Neural Network (CNN) using approximately 2,000 labeled plots classified as “good” if the detected steps are suitable for long-term site velocity estimation or “bad” if unsuitable. Site velocities are estimated from the longest step-free segment (minimum 4 years) of the time series, independently for each direction. The AI model can be directly applied to regional or global GNSS network data analysis, advancing automated GNSS time series analysis for interdisciplinary applications. Additionally, the training methodology and training datasets can be adapted to develop AI models suited for specific studies, such as for studying co-seismic and post-seismic displacements on regional or global scales. This study delivers reliable long-term site velocities (IGS14) for approximately 13,000 permanent GNSS stations worldwide, offering a foundational dataset for researchers in geodesy, tectonophysics, and hazard mitigation. 

---

**Programs and Datasets**

This paper's programs and datasets, including a large training dataset (***data.tgz***) and the CNN model (***StepCNN-GNSS.keras***), exceed GitHub's 25 MB file size limit. These two files, essential for replicating the study, are permanently archived on the author’s research website:

http://easd.geosc.uh.edu/gwang/publications.php

---

**StepCNN-GNSS Model Usage Guide**

Compatibility Requirements

✅ Supported:

Python: 3.8.x, 3.9.x, 3.10.x

TensorFlow: 2.15.0

❌ Not Supported:

Python ≥ 3.12 (incompatible with TensorFlow 2.15.0)

TensorFlow > 2.15.0 (may cause loading errors)


Quick Start For Installing Python and TensorFlow

Install Python 3.10 (Recommended)

Download and install Python 3.10 from python.org.

During installation:

Check ✅ "Add Python to PATH".

Install TensorFlow 2.15.0

pip install tensorflow==2.15.0


For Users with Higher Python/TensorFlow Versions
If your system has Python ≥ 3.12 or TensorFlow > 2.15.0, you may need to retrain the CNN model on your computer. Please read the details in Train_StepCNN-GNSS.py

---

<u>***Instructions for Understanding and Retraining the CNN Model***</u>

Place the following files in your working directory:

./Train_StepCNN-GNSS.py

./data/train/good/*

./data/train/bad/*

The training datasets (good and bad samples/plots) are included in ***data.tgz***. You may extract the contents and explore the Python file ***Train_StepCNN-GNSS.py*** to understand the CNN training method. I have added detailed comments within the code. To train the model, simply run the Python script (***Train_StepCNN-GNSS.py***) on your computer.

---

<u>***Instructions for Using the CNN Model for Step Detection and Velocity Estimation***</u>

***Taiwan_IGS14.tgz*** contains displacement time series (*.col) for numerous GPS stations in Taiwan, which can be used as sample data (displacement time series). Place the following files in your working directory:

./GNSS_StepDetection_VelocityEstimation.py

./StepCNN-GNSS.keras

./*.col (sample data)

You may read the Python script ***GNSS_StepDetection_VelocityEstimation.py***, which is written based on the methods described in the paper. I have included detailed comments within the code for clarity. You can run the script with the sample GNSS time series (*.col) on your computer. Make sure that the CNN model (***StepCNN-GNSS.keras***) is under your working directory.

---

***Examples of "good" plots in training data, ./data/train/good/***
![final_AACR_CAB18_neu_cm_candidate_E](https://github.com/user-attachments/assets/d001b28a-ba00-4019-bbdc-2f5f06271df8)
![final_BIMO_IGS14_neu_cm_candidate_N](https://github.com/user-attachments/assets/acb58841-309e-4e6c-93da-e5bce412b841)
![final_BDRL_candidate_U](https://github.com/user-attachments/assets/5346d480-14c9-4897-98a1-e01e5108652f)
![final_BIRC_candidate_N](https://github.com/user-attachments/assets/de0873ad-dd89-4ca7-94fd-ec9d4ec64cc5)
![final_CHI7_IGS14_neu_cm_candidate_N](https://github.com/user-attachments/assets/83d5769e-84f6-448a-b59d-37832d52cc2b)
![final_COTD_candidate_U](https://github.com/user-attachments/assets/8b898ab7-833f-438c-84d4-c699edc90e0c)
![final_FUQE_candidate_N](https://github.com/user-attachments/assets/9850ec38-0047-42a0-946d-5d612c1ff3dc)

***Examples of "bad" plots in training data, ./data/train/bad/***
![final_AGRD_candidate_N](https://github.com/user-attachments/assets/10916597-4015-4c56-abd7-3d73d5805c91)
![final_AMS2_candidate_U](https://github.com/user-attachments/assets/baaa588e-2e82-4e29-b48a-825104955c98)
![final_AZPE_candidate_N](https://github.com/user-attachments/assets/41fe47e0-e14c-4d8c-8f37-2a760b0e0fff)
![final_BATG_candidate_U](https://github.com/user-attachments/assets/c53ef0f4-1440-4a5a-a786-01b876c90f5a)
![final_CAN3_candidate_U](https://github.com/user-attachments/assets/d71b59a6-a217-484f-97f7-60809618ef14)
![final_CLO1_IGS14_neu_cm_candidate_N](https://github.com/user-attachments/assets/33a7ae52-69e2-4d3e-a838-8baa9808acfa)
![final_G040_candidate_E](https://github.com/user-attachments/assets/d2daa353-4ee9-4a41-bd0c-63f6dbb3d52d)
![final_G045_candidate_E](https://github.com/user-attachments/assets/bcf7a071-22cc-41a7-b175-edc035743d66)
![final_HAV2_IGS14_neu_cm_candidate_N](https://github.com/user-attachments/assets/4ea86316-af68-45c6-a833-c78a4f011ef6)
![final_S063_IGS14_neu_cm_candidate_N](https://github.com/user-attachments/assets/d59a1fc9-0aed-4c4f-845d-d95d365269a2)


