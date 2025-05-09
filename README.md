# StepCNN-GNSS: An AI Model Enhancing Step Detection and Site Velocity Estimation from Global GNSS Data
Guoquan Wang et al.

gwang@uh.edu
Abstract

Estimating long-term site velocities from Global Navigation Satellite System (GNSS) time series is vital for tectonic motion and hazard analysis. However, this estimation is hindered by step discontinuities caused by earthquakes, equipment changes, and other unexplained sources. Applying linear regression to entire time series often yields erroneous velocities. We propose a two-stage hybrid framework for automated step detection in GNSS-derived displacement time series. The framework integrates: (1) analytical methods, including a sliding-window algorithm for abrupt-step detection and a weighted cubic polynomial fit for gradual-step detection via curvature analysis; and (2) the artificial intelligence (AI) model StepCNN-GNSS, which quantitatively assesses step-detection quality and iteratively optimizes analytical parameters. This dual approach ensures robust step detection. The AI model is trained on a Convolutional Neural Network (CNN) using approximately 2,000 labeled plots classified as “good” if the detected steps are suitable for long-term site velocity estimation or “bad” if unsuitable. Site velocities are estimated from the longest step-free segment (minimum 4 years) of the time series, independently for each direction. The AI model can be directly applied to regional or global GNSS network data analysis, advancing automated GNSS time series analysis for interdisciplinary applications. Additionally, the training methodology and training datasets can be adapted to develop AI models suited for specific studies, such as for studying co-seismic and post-seismic displacements on regional or global scales. This study delivers reliable long-term site velocities (IGS14) for approximately 13,000 permanent GNSS stations worldwide, offering a foundational dataset for researchers in geodesy, tectonophysics, and hazard mitigation. 


Programs and Datasets

This paper's programs and datasets, including a large training dataset (data.tgz) and the CNN model (StepCNN-GNSS.keras), exceed GitHub's 25 MB file size limit. These two files, essential for replicating the study, are permanently archived on the author’s research website:

http://easd.geosc.uh.edu/gwang/publications.php

# Instructions for Understanding and Retraining the CNN Model

Place the following files in your working directory:

./Train_StepCNN-GNSS.py

./train/data/good/*

./train/data/bad/*

The training datasets (good and bad samples/plots) are included in data.tgz. You may extract the contents and explore the Python file Train_StepCNN-GNSS.py to understand the CNN training method. I have added detailed comments within the code. To train the model, simply run the Python script (Train_StepCNN-GNSS.py) on your computer.


# Instructions for Using the CNN Model for Step Detection and Velocity Estimation

Taiwan_IGS14.tgz contains displacement time series (*.col) for numerous GPS stations in Taiwan, which can be used as sample data. Place the following files in your working directory:

./GNSS_StepDetection_VelocityEstimation.py

./StepCNN-GNSS.keras

./*.col (sample data)

You may read the Python script GNSS_StepDetection_VelocityEstimation.py, which is written based on the methods described in the paper. I have included detailed comments within the code for clarity. You can run the script with the sample GNSS time series (*.col) on your computer. Make sure that the CNN model (StepCNN-GNSS.keras) is under your working directory.
