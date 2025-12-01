#!/usr/bin/python3
"""
Title: ChangePointCNN-GNSS: Step Detection CNN Training for GNSS Velocity Estimation
Author: Guoquan Wang, et al., gwang@uh.edu
Last updated: December 1, 2025

You may find the detailed method in the original publication:
https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025JH000910

Description:
    This script trains a Convolutional Neural Network (CNN) based on the VGG16 architecture to classify GNSS step-detection plots as "good" (suitable for velocity estimation) or "bad" (unsuitable due to steps or noise). The model is trained in two phases: first, with the VGG16 base layers frozen to train custom top layers, and second, fine-tuning the top VGG16 layers for improved performance. The trained model outputs a probability score (0–1) indicating the suitability of a plot for velocity estimation, which is used in the GNSS_StepDetection_VelocityEstimation.py pipeline to select optimal step-detection configurations for ~13,000 global GNSS stations. A test function evaluates the model on a test dataset (10% of the data), reporting accuracy, loss, and average probability scores for "good" and "bad" classes.

Dependencies:
    - Python 3.8+
    - TensorFlow 2.15.0 or higher (version-sensitive).
    - NumPy
    - scikit-learn
    - Many other Python packages

Important Versioning Notes:
    1. Model Compatibility:
        - The trained model is version-specific to TensorFlow
        - Always specify the exact TensorFlow version used for training (e.g., 2.15.0, 2.19.0).
          You may use "pip show tensorflow" to check the version on your computer.
      
    2. Usage Recommendation:
        - For optimal results, train the model on your computer
        - Version mismatches may cause unexpected behavior
 
Usage:
    1. Organize your dataset in the following structure:
       ./data/train/good/*.png  # Step detection plots suitable for velocity estimation (80% of data)
       ./data/train/bad/*.png   # Step detection plots unsuitable for velocity estimation
       ./data/test/good/*.png   # Test plots suitable for velocity estimation (10% of data)
       ./data/test/bad/*.png    # Test plots unsuitable for velocity estimation
       You may download the training data at (data.tgz): 
       https://doi.org/10.5281/zenodo.17180354

    2. Adjust hyperparameters (e.g., batch_size, epochs, learning rates) as needed.
    3. Run the script: `python3 Train_ChangePointCNN-GNSS_VGG.py`
    4. The trained model will be saved as 'ChangePointCNN-GNSS_VGG_V1.keras'. (You may change the version)

Output:
    - A trained model file: ChangePointCNN-GNSS_VGG_V1.keras
    - Training summary with validation accuracy and loss for both phases
    - Test accuracy, test loss, and average probability scores for "good" and "bad" classes

Notes:
    - The model is designed for binary classification but outputs a probability score, allowing users to rank candidate plots and select the best configuration for velocity estimation.
    - For how to use this model, please read the line 1395:  cnn_model = tf.keras.models.load_model("ChangePointCNN-GNSS_VGG_V1.keras")
      in GNSS_CPD_VelocityEstimation_VGG.py 
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
train_dir = "data/train"
test_dir = "data/test"
img_height, img_width = 224, 224
batch_size = 32
epochs_phase1 = 10
epochs_phase2 = 30
validation_split = 0.2

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    validation_split=validation_split
)

# Test data generator (no augmentation, only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    shuffle=True
)

# Load validation data
val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

# Load test data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Compute Class Weights
train_labels = train_generator.classes
cw = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = dict(enumerate(cw))
print("Class weights:", class_weights)

# Model Definition with VGG16
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(img_height, img_width, 3)
)

# Phase 1: Train Top Layers with Frozen Base
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train Phase 1: Top Layers Only
print("Phase 1: Training top layers with frozen VGG16")
history_phase1 = model.fit(
    train_generator,
    epochs=epochs_phase1,
    validation_data=val_generator,
    class_weight=class_weights
)

# Phase 2: Fine-Tune VGG16 Layers
base_model.trainable = True
for layer in base_model.layers[:-8]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=2e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Phase 2: Fine-tuning with unfrozen VGG16 layers")
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

history_phase2 = model.fit(
    train_generator,
    epochs=epochs_phase2,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[early_stopping]
)

# Save the Trained Model (check the version)
model.save("ChangePointCNN-GNSS_VGG_V1.keras")
print("Model saved as ChangePointCNN-GNSS_VGG_V1.keras")

# Test Function with Accuracy, Loss, and Average Probability Scores
def test_model(model_path, test_dir, batch_size=32):
    """
    Evaluate the trained model on the test dataset, reporting accuracy, loss, and average probability scores.
    
    Parameters:
    - model_path: Path to the saved model file (e.g., 'ChangePointCNN-GNSS_VGG_V1.keras').
    - test_dir: Directory containing test data (subfolders 'good' and 'bad').
    - batch_size: Batch size for test data loading.
    
    Returns:
    - test_accuracy: Accuracy on the test set.
    - test_loss: Loss on the test set.
    - avg_prob_good: Average probability score for 'good' class.
    - avg_prob_bad: Average probability score for 'bad' class.
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Create test data generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False  # Ensure consistent order for predictions
    )
    
    # Get class indices
    class_indices = test_generator.class_indices  # e.g., {'bad': 0, 'good': 1}
    good_class_idx = class_indices['good']
    bad_class_idx = class_indices['bad']
    
    # Predict probabilities for all test images
    predictions = model.predict(test_generator)
    true_labels = test_generator.labels
    
    # Separate probabilities by class
    good_probs = predictions[true_labels == good_class_idx]
    bad_probs = predictions[true_labels == bad_class_idx]
    
    # Calculate average probabilities
    avg_prob_good = np.mean(good_probs) if len(good_probs) > 0 else 0.0
    avg_prob_bad = np.mean(bad_probs) if len(bad_probs) > 0 else 0.0
    
    # Evaluate accuracy and loss
    test_loss, test_accuracy = model.evaluate(test_generator)
    
    # Print results
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Average probability score for 'good' class: {avg_prob_good:.4f}")
    print(f"Average probability score for 'bad' class: {avg_prob_bad:.4f}")
    print(f"Number of 'good' images: {len(good_probs)}")
    print(f"Number of 'bad' images: {len(bad_probs)}")
    
    return test_accuracy, test_loss, avg_prob_good, avg_prob_bad

# Run the test function
print("Evaluating model on test dataset")
test_accuracy, test_loss, avg_prob_good, avg_prob_bad = test_model(
    "ChangePointCNN-GNSS_VGG_V7.keras", test_dir, batch_size
)

# Print Training Summary
print("Phase 1 - Final val_accuracy:", history_phase1.history['val_accuracy'][-1])
print("Phase 2 - Final val_accuracy:", history_phase2.history['val_accuracy'][-1])
print("Phase 2 - Validation accuracy history:", history_phase2.history['val_accuracy'])
print("Phase 2 - Validation loss history:", history_phase2.history['val_loss'])
print("Final Test Accuracy:", test_accuracy)
print("Final Test Loss:", test_loss)
