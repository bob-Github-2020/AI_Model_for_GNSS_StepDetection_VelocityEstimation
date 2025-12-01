ChangePointCNN-GNSS: An AI Model for Assessing Change Points and Optimizing Site Velocity Estimation from Global GNSS Data

Guoquan Wang et al.

gwang@uh.edu

You may find this paper at:

https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025JH000910


**Abstract**

Estimating long-term site velocities from Global Navigation Satellite System (GNSS)-derived daily displacement time series is vital for studying secular tectonic motions and establishing regional and global geodetic reference frames. However, this estimation is complicated by displacements caused by earthquakes, equipment changes, hydraulic head changes, and other sources, which introduce change points in GNSS time series. This study introduces a two-stage hybrid framework for automated change-point detection in GNSS time series. The framework integrates: (1) analytical methods, including a sliding-window algorithm for instant change-point detection and a cubic polynomial fit for transitional change-point detection; and (2) an artificial intelligence (AI) model, ChangePointCNN-GNSS, which evaluates the suitability of candidate change points for site velocity estimation and iteratively optimizes analytical parameters. Unlike prior data-driven approaches, our framework leverages an image-driven method, employing a Convolutional Neural Network (CNN) to visually assess and select the most suitable change-point configuration for reliable site velocity estimation. Site velocities are computed from the longest change-point-free segment (minimum 4 years), processed independently for each station and direction. This integrated approach ensures robust site velocity estimation across large GNSS networks. The CNN is trained using approximately 6,000 time series plots with marked change points. Each plot is labeled as “good” if the detected change points are suitable for reliable site velocity estimation or “bad” if unsuitable. This study delivers long-term site velocities (IGS20) for approximately 14,600 permanent GNSS stations worldwide, with a 95% confidence interval below 1 mm/year, offering a foundational dataset for researchers in geodesy, tectonophysics, and hazard mitigation.

---

**Programs and Datasets**

    **Train_ChangePointCNN-GNSS_VGG.py**: The Python script used to train the ChangePointCNN-GNSS model from scratch using the provided dataset.

    *GNSS_CPD_VelocityEstimation_VGG.py*: The main implementation script for applying the trained model to new GNSS data. This script performs automated change-point detection and calculates long-term site velocities.
    
    ***Taiwan_IGS14.tgz***: Sample files for testing GNSS_CPD_VelocityEstimation_VGG.py

    ***ChangePointCNN_VGG_V7.keras***: The pre-trained Convolutional Neural Network (CNN) model, ready for immediate use in the detection framework.

    ***data.tgz***: A compressed archive containing the training dataset of approximately 6,000 labeled time series plots used to train the CNN model.
   
    IGS20_Velocities_at_Global_GNSS.txt or .xls: The resulting velocity dataset for approximately 14,600 global GNSS stations, estimated using this framework in the IGS20 reference frame.

    IGS20_Velocities_CNN_MIDAS.txt or .xls: A comparative dataset presenting site velocities for global GNSS stations estimated using both the ChangePointCNN method and the MIDAS method.
 
This paper's programs and datasets, including a large training dataset (***data.tgz***) and a trained-CNN model (***ChangePointCNN_VGG_V7.keras***), exceed GitHub's 25 MB file size limit. These files, essential for replicating the study, are permanently archived at:

   ***https://doi.org/10.5281/zenodo.17180354***

---

**Quick Start For Installing Python and TensorFlow**

<u>**Install Python 3.10.x or higher (Recommended)**</u>

    Download and install Python 3.x from python.org.

    During installation:

    Check ✅ "Add Python to PATH".

For installing specific version of TensorFlow, you may use (for example):

    <u>**Install TensorFlow 2.15.0**</u>

    ***pip install tensorflow==2.15.0***

---

The trained-CNN model (***ChangePointCNN_VGG_V7.keras***) is version-specific to TensorFlow. I strongly suggest you train the model on your computer with the trainning program (***Train_ChangePointCNN-GNSS_VGG.py***) and the dataset (***data.tgz***). Please read the detailed instruction inside the trainning program. 

<u>***Instructions for Understanding and Retraining the CNN Model***</u>

Place the following files in your working directory:

./Train_ChangePointCNN-GNSS_VGG.py

./data/train/good/*

./data/train/bad/*

The training datasets (good and bad samples/plots) are included in ***data.tgz***. You may extract the contents and explore the Python file ***Train_ChangePointCNN-GNSS_VGG.py*** to understand the CNN training method. I have added detailed comments within the code. To train the model, simply run the Python script (***Train_ChangePointCNN-GNSS_VGG.py***) on your computer.

---

<u>***Instructions for Using the CNN Model for Step Detection and Velocity Estimation***</u>

***Taiwan_IGS14.tgz*** contains displacement time series (*.col) for numerous GPS stations in Taiwan, which can be used as sample data (displacement time series). Place the following files in your working directory:

./GNSS_CPD_VelocityEstimation_VGG.py

./ChangePointCNN_VGG_V7.keras  (or another name trained on your computer)

./*.col (sample data)

You may read the Python script ***GNSS_CPD_VelocityEstimation_VGG.py***, which is written based on the methods described in the paper. I have included detailed comments within the code for clarity. You can run the script with the sample GNSS time series (*.col) on your computer. Make sure that the CNN model is under your working directory, or you may specify the location of the model in the program.

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


