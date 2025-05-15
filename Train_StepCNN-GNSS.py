#!/usr/bin/python3
"""
Title: StepCNN-GNSS: Step Detection CNN Training for GNSS Velocity Estimation
Author: Guoquan Wang, et al., gwang@uh.edu
Last updated: May 15, 2025
Description:
    This script trains a Convolutional Neural Network (CNN) based on the VGG16 architecture to classify GNSS step-detection plots as "good" (suitable for velocity estimation) or "bad" (unsuitable due to steps or noise). The model is trained in two phases: first, with the VGG16 base layers frozen to train custom top layers, and second, fine-tuning the top VGG16 layers for improved performance. The trained model outputs a probability score (0–1) indicating the suitability of a plot for velocity estimation, which is used in the GNSS_StepDetection_VelocityEstimation.py pipeline to select optimal step-detection configurations for ~13,000 global GNSS stations.

Dependencies:
    - Python 3.9.x, 3.10.x (You may try higher versions on your computer)
    - TensorFlow 2.15.0 (You may try higher versions on your computer)
    - Please specify the versions of Python and TensorFlow for the final model, StepCNN-GNSS.py
    - NumPy
    - scikit-learn
    - A dataset of labeled step-detection plots (224x224 pixels) in the format: data/train/[good|bad]/*.png
    
Usage:
    1. Organize your dataset in the following structure:
       ./data/train/good/*.png  # plot with good/correct step detection
       ./data/train/bad/*.png   # Plots with bad/incorrect step detection
    2. Adjust hyperparameters (e.g., batch_size, epochs, learning rates) as needed.
    3. Run the script: `python3 Train_StepCNN-GNSS.py`
    4. The trained model will be saved as 'StepCNN-GNSS.keras'.

Output:
    - A trained model file: StepCNN-GNSS.keras
    - Training summary with validation accuracy and loss for both phases

Notes:
     - The model is designed for binary classification but outputs a probability score, allowing users to rank candidate plots and select the best configuration for velocity estimation.
    - For more details, see the associated manuscript: [Insert manuscript reference or link].
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
import numpy as np
import os

# Directory and Hyperparameters
# Define the path to the training dataset, which should contain subdirectories 'good' and 'bad' with labeled step-detection plots.
train_dir = "data/train"
# Set image dimensions to 224x224 pixels, matching VGG16's expected input size for pre-trained ImageNet weights.
img_height, img_width = 224, 224
# Batch size of 32 balances memory usage and training stability; adjust based on GPU memory (e.g., reduce to 16 for smaller GPUs).
batch_size = 32
# Phase 1 trains the top layers for 10 epochs, sufficient to learn initial features without overfitting.
epochs_phase1 = 10
# Phase 2 fine-tunes the model for up to 30 epochs, increased from 20 to allow more iterations with new data, with early stopping to prevent overfitting.
epochs_phase2 = 30
# Use 20% of the data for validation to monitor generalization performance during training.
validation_split = 0.2

# Data Augmentation
# Configure ImageDataGenerator for data augmentation and preprocessing.
# Rescale pixel values to [0, 1] as required by VGG16's pre-trained weights.
# Horizontal flips are enabled to increase dataset variety while avoiding excessive distortion (rotation and zoom were removed in v6 to reduce overfitting).
# Validation split ensures 20% of the data is reserved for validation.
datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    validation_split=validation_split
)

# Load training data using the generator, shuffling to ensure random batch sampling.
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',  # Binary classification: "good" (1) vs. "bad" (0)
    subset='training',
    shuffle=True
)

# Load validation data, also shuffled for unbiased evaluation.
val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

# Compute Class Weights
# Calculate class weights to handle potential class imbalance (e.g., more "bad" plots than "good").
# This ensures the model doesn't bias toward the majority class during training.
train_labels = train_generator.classes
cw = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = dict(enumerate(cw))
print("Class weights:", class_weights)

# Model Definition with VGG16
# Load VGG16 pre-trained on ImageNet, excluding the top fully connected layers, to use as a feature extractor.
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(img_height, img_width, 3)
)

# Phase 1: Train Top Layers with Frozen Base
# Freeze the VGG16 base layers to prevent their weights from being updated, focusing training on the custom top layers.
base_model.trainable = False

# Build the model by adding custom layers on top of VGG16.
model = models.Sequential([
    base_model,
    layers.Flatten(),  # Flatten the 7x7x512 feature maps from VGG16 into a 1D vector
    layers.Dense(256, activation='relu'),  # First dense layer for feature learning
    layers.Dropout(0.5),  # Dropout to reduce overfitting, set to 0.5 based on empirical testing
    layers.Dense(128, activation='relu'),  # Second dense layer for further abstraction
    layers.Dropout(0.3),  # Additional dropout to further mitigate overfitting
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification (probability score 0–1)
])

# Compile the model with Adam optimizer and a learning rate of 1e-4, suitable for initial training of top layers.
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',  # Standard loss for binary classification
    metrics=['accuracy']
)

# Train Phase 1: Top Layers Only
print("Phase 1: Training top layers with frozen VGG16")
history_phase1 = model.fit(
    train_generator,
    epochs=epochs_phase1,
    validation_data=val_generator,
    class_weight=class_weights  # Apply class weights to handle imbalance
)

# Phase 2: Fine-Tune VGG16 Layers
# Unfreeze the VGG16 base layers to allow fine-tuning, but keep the bottom layers frozen to preserve low-level features.
base_model.trainable = True
# Freeze all layers except the top 8, increased from 4 in earlier versions to allow more fine-tuning while avoiding overfitting to low-level features.
for layer in base_model.layers[:-8]:
    layer.trainable = False

# Recompile the model with a lower learning rate (2e-5) to make small, careful updates during fine-tuning.
model.compile(
    optimizer=Adam(learning_rate=2e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Define early stopping to halt training if validation accuracy doesn’t improve for 5 epochs, restoring the best weights.
print("Phase 2: Fine-tuning with unfrozen VGG16 layers")
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

# Train Phase 2: Fine-tuning
history_phase2 = model.fit(
    train_generator,
    epochs=epochs_phase2,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[early_stopping]
)

# Save the Trained Model
# Save the model in Keras format for use in the GNSS_StepDetection_VelocityEstimation.py pipeline.
model.save("StepCNN-GNSS.keras")
print("StepCNN-GNSS.keras")

# Print Training Summary
# Display the final validation accuracy for both phases and the full history of validation metrics for Phase 2.
# This helps users assess the model’s performance and convergence behavior.
print("Phase 1 - Final val_accuracy:", history_phase1.history['val_accuracy'][-1])
print("Phase 2 - Final val_accuracy:", history_phase2.history['val_accuracy'][-1])
print("Phase 2 - Validation accuracy history:", history_phase2.history['val_accuracy'])
print("Phase 2 - Validation loss history:", history_phase2.history['val_loss'])
