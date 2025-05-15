#!/usr/bin/python3
"""
Title: GNSS Step Detection and Velocity Estimation with CNN (Main Program)
Author: Guoquan Wang, et al., gwang@uh.edu
Date: May 15 2025
Description:
    This script processes GNSS time series data to estimate site velocities by detecting and handling steps (abrupt and slow) in the North (N), East (E), and Up (U) directions. It integrates a pre-trained Convolutional Neural Network (CNN) model (based on VGG16) to optimize small-step detection parameters, ensuring robust velocity estimates. The pipeline includes outlier removal, step detection (using sliding windows and cubic fitting for slow steps), velocity calculation, and visualization of results. The script is designed for ~13,000 global GNSS stations and supports studies of tectonic motion and hazard assessment.

Dependencies:
    - Python 3.9+
    - TensorFlow 2.15.0 (or higher)
    - Please check the specific versions of Python and TensofrFlow used for trainning the CNN model (StepCNN-GNSS.keras)
    - NumPy
    - Pandas
    - Matplotlib
    - SciPy
    - ruptures (for change point detection)
    - joblib (optional, for XGBoost if used)
    - Pillow (PIL)
    - Input data: GNSS time series files in the format *_IGS14_NEU_cm.col (e.g., P646_IGS14_NEU_cm.col)
    - Pre-trained CNN model: step_detection_CNN_VGG16_224by224_2phases_v5.keras

Usage:
    1. Place GNSS time series files (*cm.col) in the working directory.
    2. Ensure the pre-trained CNN model (StepCNN-GNSS.keras) is in the working directory.
    3. Run the script: `python3 GNSS_StepDetection_VelocityEstimation.py`
    4. Outputs per station:
       - Velocity file: <station>_Velocity.txt (e.g., P646_Velocity.txt)
       - AI parameters file: <station>_AI_parameters.csv (e.g., P646_AI_parameters.csv)
       - Plots: <station>.png (time series with velocity trend), <station>_detrended.png (detrended series with steps)

Output Format (Velocity File):
    - Columns: Station Vel_N(mm/yr) Vel_E(mm/yr) Vel_U(mm/yr) Dur_N Dur_E Dur_U S_begin_N S_end_N S_begin_E S_end_E S_begin_U S_end_U W_begin W_end W_Du
    - Velocities are in mm/yr; durations are in years; S_begin/end are segment start/end years; W_begin/end/Du are whole-series bounds and duration.

Notes:
    - The script uses a hybrid step detection approach: sliding windows for abrupt steps and cubic fitting for slow steps, optimized by the CNN.
    - Memory management is implemented with garbage collection (gc.collect()) and TensorFlow session clearing to handle large datasets.
    - Outlier removal uses iterative detrending and sigma-based filtering to ensure data quality.
    - For more details, see the associated manuscript: [Insert manuscript reference or link].
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import os
import glob
import ruptures as rpt
import joblib  # Optional: for loading XGBoost models if needed
import tensorflow as tf
import gc
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from matplotlib import font_manager

# Configure Matplotlib for consistent plotting (e.g., for publication in JGR)
# Use DejaVu Sans font and ensure PDF output is compatible with publication standards.
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['pdf.fonttype'] = 42

# ============================================================================
# 1. Data Loading Function (for NEU Data in cm)
# ============================================================================
def load_data_neu(file_path):
    """
    Load GNSS time series data from a file in NEU format (North, East, Up in cm).

    Args:
        file_path (str): Path to the input file (e.g., P646_IGS14_NEU_cm.col).

    Returns:
        tuple: (original_data, cleaned_data)
            - original_data: DataFrame with raw data (Decimal_Year, N, E, U).
            - cleaned_data: DataFrame with outliers removed using sigma-based detrending.
    """
    # Define expected columns in the input file.
    columns = ["Decimal_Year", "N_cm", "E_cm", "U_cm", "Sigma_N_cm", "Sigma_E_cm", "Sigma_U_cm"]
    # Read the file, skipping comment lines starting with '#', using whitespace as delimiter.
    data = pd.read_csv(file_path, delim_whitespace=True, names=columns, comment='#')
    # Select only the columns we need for processing.
    data = data[["Decimal_Year", "N_cm", "E_cm", "U_cm"]]
    # Convert all columns to numeric, coercing errors to NaN.
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    # Rename columns for simplicity (remove '_cm' suffix).
    data = data.rename(columns={"N_cm": "N", "E_cm": "E", "U_cm": "U"})
    # Keep a copy of the original data for comparison.
    original_data = data.copy()
    # Remove outliers in each direction using sigma-based detrending.
    data = remove_outliers_sigma_detrend(data, "N")
    data = remove_outliers_sigma_detrend(data, "E")
    data = remove_outliers_sigma_detrend(data, "U")
    # Reset index to ensure consecutive indices after outlier removal.
    data = data.reset_index(drop=True)
    return original_data, data

# ============================================================================
# 2. Outlier Removal Function (Using Detrending)
# ============================================================================
def remove_outliers_sigma_detrend(data, column, n_sigma=2.5, window=350):
    """
    Remove outliers from a GNSS time series column using iterative detrending and sigma-based filtering.

    The process first applies an absolute threshold to remove extreme outliers, then iteratively
    detrends the data and removes points beyond n_sigma standard deviations from the local mean.

    Args:
        data (pd.DataFrame): DataFrame with 'Decimal_Year' and the target column (e.g., 'N', 'E', 'U').
        column (str): Column name to clean (e.g., 'N', 'E', 'U').
        n_sigma (float): Number of standard deviations for outlier threshold (default: 2.5).
        window (int): Rolling window size for local statistics (default: 350, ~1 year of daily data).

    Returns:
        pd.DataFrame: DataFrame with outliers set to NaN.
    """
    max_iterations = 3  # Number of detrending iterations to ensure robust outlier removal.
    abs_threshold = 800.0  # Absolute threshold in cm (8m) for extreme outliers.

    data_cleaned = data.copy()
    t = data_cleaned["Decimal_Year"].values
    y = data_cleaned[column].values

    # Step 1: Apply absolute threshold to remove extreme outliers (e.g., >8m displacement).
    abs_outlier_mask = (y < -abs_threshold) | (y > abs_threshold)
    data_cleaned.loc[abs_outlier_mask, column] = np.nan
    y = data_cleaned[column].values  # Update after absolute thresholding.

    # Step 2: Iterative detrending and sigma-based outlier removal.
    for iteration in range(max_iterations):
        valid = ~np.isnan(y)
        if valid.sum() < 2:  # Stop if too few points remain for detrending.
            break

        # Linear detrending to remove the overall trend.
        slope, intercept, _, _, _ = linregress(t[valid], y[valid])
        trend = slope * t + intercept
        detrended = y - trend
        detrended_series = pd.Series(detrended, index=data_cleaned.index)

        # Compute rolling mean and standard deviation for local statistics.
        mean = detrended_series.rolling(window=window, center=True, min_periods=1).mean()
        std = detrended_series.rolling(window=window, center=True, min_periods=1).std()

        # Define sigma-based bounds for outlier detection.
        lower_bound = mean - n_sigma * std
        upper_bound = mean + n_sigma * std

        # Identify and remove outliers beyond the sigma threshold.
        sigma_outlier_mask = (detrended_series < lower_bound) | (detrended_series > upper_bound)
        data_cleaned.loc[sigma_outlier_mask, column] = np.nan
        y = data_cleaned[column].values  # Update for the next iteration.

    return data_cleaned

# ============================================================================
# 3. Step Detection (Sliding Window)
# ============================================================================
def detect_change_points_sliding_window(
    data,
    window_size=30,
    N_threshold=0.4,
    E_threshold=0.4,
    U_threshold=1.0,
    min_distance=200
):
    """
    Detect abrupt change points in GNSS time series using a sliding window approach.

    The function first detrends and centers the data in each direction (N, E, U), then applies
    a sliding window to detect candidate change points based on mean differences exceeding
    direction-specific thresholds. Finally, it filters change points to ensure they are separated
    by a minimum time distance.

    Args:
        data (pd.DataFrame): DataFrame with 'Decimal_Year', 'N', 'E', 'U' columns.
        window_size (int): Size of the sliding window in data points (default: 30).
        N_threshold (float): Threshold for North direction in cm (default: 0.4).
        E_threshold (float): Threshold for East direction in cm (default: 0.4).
        U_threshold (float): Threshold for Up direction in cm (default: 1.0).
        min_distance (int): Minimum distance between change points in days (default: 200).

    Returns:
        dict: Dictionary with change points for each direction (e.g., {'N': [idx1, idx2], ...}).
    """
    thresholds = {"N": N_threshold, "E": E_threshold, "U": U_threshold}
    change_points = {}
    data_detrended = data.copy()

    # Step 1: Detrend and center the data in each direction.
    for direction in ["N", "E", "U"]:
        x = data["Decimal_Year"]
        y = data[direction]
        valid_mask = y.notna()
        if valid_mask.sum() < 2:  # Skip if too few valid points.
            continue
        slope, intercept, _, _, _ = linregress(x[valid_mask], y[valid_mask])
        detrended = y - (slope * x + intercept)
        detrended -= detrended.mean()  # Center the data.
        data_detrended[direction] = detrended

    # Step 2: Perform sliding-window detection on detrended data.
    candidate_cp = {}
    for direction in ["N", "E", "U"]:
        threshold = thresholds[direction]
        valid_mask = data_detrended[direction].notna()
        valid_indices = data_detrended.loc[valid_mask].index
        ts = data_detrended.loc[valid_mask, direction].values
        n = len(ts)
        if n < 2 * window_size:  # Ensure enough data for sliding window.
            candidate_cp[direction] = [valid_indices[0], valid_indices[-1]] if len(valid_indices) > 1 else []
            continue

        raw_cps_local = [0]
        i = window_size
        while i < n - window_size:
            window_before = ts[i - window_size:i]
            window_after = ts[i:i + window_size]
            mean_before = window_before.mean()
            mean_after = window_after.mean()
            if abs(mean_after - mean_before) > threshold:
                raw_cps_local.append(i + window_size)
                i += window_size  # Jump ahead to avoid overlapping detections.
            else:
                i += 1
        raw_cps_local.append(n - 1)
        # Convert local indices to global indices.
        cp_global = [valid_indices[loc_idx] for loc_idx in raw_cps_local if loc_idx < n]
        candidate_cp[direction] = cp_global

    # Step 3: Filter change points that are too close together.
    filtered_cp = {}
    min_gap_years = min_distance / 365.25  # Convert days to years.
    for direction in ["N", "E", "U"]:
        cp_list = candidate_cp.get(direction, [])
        if not cp_list:
            filtered_cp[direction] = []
            continue
        cp_years = [data.loc[idx, "Decimal_Year"] for idx in cp_list]
        filtered = [cp_list[0]]
        last_year = data.loc[cp_list[0], "Decimal_Year"]
        for idx in cp_list[1:]:
            year_val = data.loc[idx, "Decimal_Year"]
            if (year_val - last_year) >= min_gap_years:
                filtered.append(idx)
                last_year = year_val
        filtered_cp[direction] = filtered

    change_points = filtered_cp
    return change_points

# ============================================================================
# 4. Fitting Functions for Step Detection
# ============================================================================
def linear_func(x, a, b):
    """Linear model: y = a*x + b."""
    return a * x + b

def quadratic_func(x, a, b, c):
    """Quadratic model: y = a*x^2 + b*x + c."""
    return a * x**2 + b * x + c

def exp_func(t, u0, v, delta, tau):
    """
    Exponential post-seismic model: f(t) = u0 + v*t + delta * exp(-t/tau).
    Used for modeling post-seismic relaxation (not used in this script but included for completeness).
    """
    return u0 + v * t + delta * np.exp(-t / tau)

def cubic_func(x, a, b, c, d):
    """Cubic model: f(x) = a*x^3 + b*x^2 + c*x + d."""
    return a * x**3 + b * x**2 + c * x + d

# ============================================================================
# 5. Optimal Cubic Fit for Slow-Step Detection
# ============================================================================
def optimal_cubic_fit(x_seg, y_seg, magnitude=0.2, p0=None):
    """
    Fit a cubic model to a segment of data, optimizing weights to emphasize either the start or end.

    The function first performs a standard cubic fit, then evaluates the mean squared error (MSE)
    in the first and last thirds of the segment. It selects a weight coefficient (positive or negative)
    to emphasize the region with higher error, re-fits the cubic model with weights, and returns the
    best weight, parameters, and MSE.

    Args:
        x_seg (array-like): Independent variable (e.g., Decimal_Year) for the segment.
        y_seg (array-like): Dependent variable (e.g., detrended displacement) for the segment.
        magnitude (float): Absolute value of the weight coefficient to test (default: 0.2).
        p0 (list, optional): Initial guess for cubic parameters [a, b, c, d]. Defaults to [0, 0, 0, mean(y_seg)].

    Returns:
        tuple: (best_weight, best_cubic_params, best_mse)
            - best_weight (float): Chosen weight coefficient.
            - best_cubic_params (ndarray): Best-fit cubic parameters [a, b, c, d].
            - best_mse (float): Mean squared error of the weighted fit.
    """
    # Set default initial guess if none provided.
    if p0 is None:
        p0 = [0, 0, 0, np.mean(y_seg)] if not np.isnan(np.mean(y_seg)) else [0, 0, 0, 0]
    # Convert inputs to numpy arrays.
    x_seg = np.asarray(x_seg)
    y_seg = np.asarray(y_seg)
    # Skip if too few points or NaN values are present.
    if len(x_seg) < 4 or np.any(np.isnan(x_seg)) or np.any(np.isnan(y_seg)):
        print(f"Skipping fit: too few points ({len(x_seg)}) or NaN values")
        return 0, p0, np.inf

    # Step 1: Standard cubic fit (no weighting).
    try:
        cubic_params_std, _ = curve_fit(cubic_func, x_seg, y_seg, p0=p0, maxfev=20000)
        cubic_pred_std = cubic_func(x_seg, *cubic_params_std)
        mse_std = np.mean((y_seg - cubic_pred_std)**2)
    except RuntimeError:
        print("curve_fit failed to converge")
        mse_std = np.inf
        cubic_params_std = p0

    # Step 2: Divide the segment into first and last thirds to evaluate fit quality.
    x_min, x_max = x_seg.min(), x_seg.max()
    time_span = x_max - x_min
    t_first_cut = x_min + time_span / 3.0
    t_last_cut = x_min + 2.0 * time_span / 3.0
    first_mask = (x_seg <= t_first_cut)
    last_mask = (x_seg >= t_last_cut)

    mse_first = np.mean((y_seg[first_mask] - cubic_pred_std[first_mask])**2) if np.any(first_mask) else mse_std
    mse_last = np.mean((y_seg[last_mask] - cubic_pred_std[last_mask])**2) if np.any(last_mask) else mse_std

    # Step 3: Choose weight based on which region has higher error.
    chosen_sign = 1 if mse_first > mse_last else -1  # Positive weight for early emphasis, negative for late.
    coeff = chosen_sign * magnitude
    weights = np.exp(-coeff * (x_seg - x_min))

    # Step 4: Re-fit with the chosen weights.
    try:
        best_cubic_params, _ = curve_fit(
            cubic_func,
            x_seg,
            y_seg,
            p0=p0,
            sigma=1/weights,
            absolute_sigma=True,
            maxfev=20000
        )
        cubic_pred_weighted = cubic_func(x_seg, *best_cubic_params)
        best_mse = np.mean((y_seg - cubic_pred_weighted)**2)
    except Exception as e:
        best_mse = np.inf
        best_cubic_params = cubic_params_std  # Fallback to standard fit.

    best_weight = chosen_sign * magnitude
    return best_weight, best_cubic_params, best_mse

# ============================================================================
# 6. Hybrid Step Detection (Abrupt and Slow Steps)
# ============================================================================
def detect_abrupt_slow_steps(
    data,
    window_size=30,
    N_threshold=0.4,
    E_threshold=0.4,
    U_threshold=1.0,
    min_distance=100,
    N_curve_threshold=0.3,
    E_curve_threshold=0.3,
    U_curve_threshold=0.5,
    improvement_ratio=0.2
):
    """
    Detect both abrupt and slow steps in GNSS time series using a hybrid approach.

    The function first detects abrupt steps using a sliding window, then identifies the longest
    step-free segment and applies a cubic fit to detect slow steps (curves) within that segment.
    Slow steps are identified as turning points in the cubic fit if they exceed a threshold amplitude
    and improve the fit over a linear model by a specified ratio.

    Args:
        data (pd.DataFrame): DataFrame with 'Decimal_Year', 'N', 'E', 'U' columns.
        window_size (int): Sliding window size for abrupt step detection (default: 30).
        N_threshold (float): Threshold for North abrupt steps in cm (default: 0.4).
        E_threshold (float): Threshold for East abrupt steps in cm (default: 0.4).
        U_threshold (float): Threshold for Up abrupt steps in cm (default: 1.0).
        min_distance (int): Minimum distance between steps in days (default: 100).
        N_curve_threshold (float): Minimum amplitude for North slow steps in cm (default: 0.3).
        E_curve_threshold (float): Minimum amplitude for East slow steps in cm (default: 0.3).
        U_curve_threshold (float): Minimum amplitude for Up slow steps in cm (default: 0.5).
        improvement_ratio (float): Required fractional improvement of cubic fit over linear (default: 0.2).

    Returns:
        dict: Dictionary with change points for each direction (e.g., {'N': [idx1, idx2], ...}).
    """
    min_gap_years = min_distance / 365.25  # Convert days to years.
    directions = ["N", "E", "U"]
    thresholds = {"N": N_threshold, "E": E_threshold, "U": U_threshold}
    curve_thresholds = {"N": N_curve_threshold, "E": E_curve_threshold, "U": U_curve_threshold}
    final_cp = {}

    for direction in directions:
        # Step 1: Detrend and center the data.
        valid_mask = data[direction].notna()
        if valid_mask.sum() < 2:
            final_cp[direction] = []
            continue

        x_valid = data.loc[valid_mask, "Decimal_Year"].values
        y_valid = data.loc[valid_mask, direction].values
        slope, intercept, _, _, _ = linregress(x_valid, y_valid)
        detrended = y_valid - (slope * x_valid + intercept)
        detrended -= detrended.mean()

        # Step 2: Detect abrupt steps using sliding window.
        valid_indices = data.loc[valid_mask].index
        n = len(detrended)
        if n < 2 * window_size:
            candidate = [valid_indices[0], valid_indices[-1]] if len(valid_indices) > 1 else []
        else:
            raw_candidates = [0]
            i = window_size
            while i < n - window_size:
                window_before = detrended[i - window_size:i]
                window_after = detrended[i:i + window_size]
                if abs(window_after.mean() - window_before.mean()) > thresholds[direction]:
                    raw_candidates.append(i + window_size)
                    i += window_size
                else:
                    i += 1
            raw_candidates.append(n - 1)
            candidate = [valid_indices[idx] for idx in raw_candidates if idx < n]

        # Step 3: Filter candidate steps by time difference.
        filtered = []
        if candidate:
            filtered.append(candidate[0])
            last_year = data.loc[candidate[0], "Decimal_Year"]
            for idx in candidate[1:]:
                current_year = data.loc[idx, "Decimal_Year"]
                if (current_year - last_year) >= min_gap_years:
                    filtered.append(idx)
                    last_year = current_year

        # Step 4: Add change points for large gaps in the time series.
        valid_years = data.loc[valid_indices, "Decimal_Year"].values
        additional = []
        for j in range(len(valid_years) - 1):
            if valid_years[j + 1] - valid_years[j] >= 1.5:  # Minimum gap of 1.5 years.
                additional.append(valid_indices[j])
                additional.append(valid_indices[j + 1])
        candidate_cp = sorted(set(filtered + additional))

        # Step 5: Identify the longest step-free segment.
        boundaries = sorted(set([valid_indices[0]] + candidate_cp + [valid_indices[-1]]))
        longest_duration = 0
        seg_start, seg_end = boundaries[0], boundaries[-1]
        for i in range(len(boundaries) - 1):
            t1 = data.loc[boundaries[i], "Decimal_Year"]
            t2 = data.loc[boundaries[i + 1], "Decimal_Year"]
            duration = t2 - t1
            if duration > longest_duration:
                longest_duration = duration
                seg_start, seg_end = boundaries[i], boundaries[i + 1]

        # Step 6: Detect slow steps in the longest segment using cubic fit.
        candidate_updated = candidate_cp.copy()
        if seg_end - seg_start >= 1100:  # Require at least 4 years (~1100 days) for cubic fit.
            seg_mask = (data.index >= seg_start) & (data.index <= seg_end) & valid_mask
            seg_data = data.loc[seg_mask]
            x_seg = data.loc[seg_mask, "Decimal_Year"].values
            y_seg = detrended[seg_mask[valid_mask].values]

            if len(x_seg) >= 5:
                try:
                    lin_params, _ = curve_fit(linear_func, x_seg, y_seg, p0=[0, np.mean(y_seg)])
                    lin_pred = linear_func(x_seg, *lin_params)
                    lin_mse = np.mean((y_seg - lin_pred)**2)
                except:
                    lin_mse = np.inf

                best_weight_coef, cubic_params, cubic_mse = optimal_cubic_fit(x_seg, y_seg, magnitude=0.15, p0=None)
                cubic_pred = cubic_func(x_seg, *cubic_params)

                if not np.isfinite(cubic_mse) or lin_mse < 1e-8:
                    mse_ratio = 0
                else:
                    mse_ratio = abs((lin_mse - cubic_mse) / lin_mse)

                if mse_ratio >= improvement_ratio:
                    a, b, c, d = cubic_params
                    # Find turning points by solving the derivative: 3ax^2 + 2bx + c = 0.
                    discriminant = (2 * b)**2 - 4 * 3 * a * c
                    if discriminant >= 0:
                        turning_x1 = (-2 * b + np.sqrt(discriminant)) / (2 * 3 * a)
                        turning_x2 = (-2 * b - np.sqrt(discriminant)) / (2 * 3 * a)
                        turning_points = [turning_x1, turning_x2]
                    else:
                        turning_points = []

                    # Filter turning points within the segment range.
                    valid_turning_points = [x for x in turning_points if x_seg.min() <= x <= x_seg.max()]
                    baseline = np.mean(y_seg)
                    for turning_x in valid_turning_points:
                        turning_y = cubic_func(turning_x, *cubic_params)
                        amplitude = abs(turning_y - baseline)
                        if amplitude >= curve_thresholds[direction]:
                            seg_years = seg_data["Decimal_Year"].values
                            idx_nearest = np.argmin(np.abs(seg_years - turning_x))
                            slow_step_idx = seg_data.index[idx_nearest]
                            candidate_updated.append(slow_step_idx)

        # Step 7: Finalize change points.
        candidate_updated = sorted(set(candidate_updated))
        final_cp[direction] = candidate_updated

    return final_cp

# ============================================================================
# 7. Correct Steps (Not Used in This Pipeline)
# ============================================================================
def correct_step(data, station_name, change_points, step_threshold=None):
    """
    Correct steps in the GNSS time series by adjusting data after each change point.

    Note: This function is not used in the current pipeline, as the method prioritizes
    step-free segments over step correction. It is included for potential future use.

    Args:
        data (pd.DataFrame): DataFrame with 'Decimal_Year', 'N', 'E', 'U' columns.
        station_name (str): Name of the station (for plotting).
        change_points (dict): Dictionary of change points for each direction.
        step_threshold (float, optional): Minimum step size to correct (in cm).

    Returns:
        pd.DataFrame: Corrected DataFrame with steps adjusted.
    """
    corrected = data.copy()
    original_data = data.copy()
    corrected_times = {"N": [], "E": [], "U": []}
    for direction in ["N", "E", "U"]:
        cplist = change_points.get(direction, [])
        if len(cplist) <= 2:
            continue
        valid_mask = corrected[direction].notna()
        valid_indices = corrected.loc[valid_mask].index
        ts = corrected.loc[valid_mask, direction].values
        for local_cp_global in cplist:
            if local_cp_global == valid_indices[0] or local_cp_global == valid_indices[-1]:
                continue
            if step_threshold is not None:
                local_idx = np.where(valid_indices == local_cp_global)[0]
                if len(local_idx) == 0:
                    continue
                local_idx = local_idx[0]
                window_size = 20
                start_local = max(local_idx - window_size, 0)
                end_local = min(local_idx + window_size, len(ts))
                if (local_idx - start_local) < 3 or (end_local - local_idx) < 3:
                    continue
                left_slice = ts[start_local:local_idx - 1]
                right_slice = ts[local_idx + 1:end_local]
                if len(left_slice) == 0 or len(right_slice) == 0:
                    continue
                left_mean = np.mean(left_slice)
                right_mean = np.mean(right_slice)
                step_val = right_mean - left_mean
                if abs(step_val) < step_threshold:
                    continue
            else:
                local_idx = np.where(valid_indices == local_cp_global)[0]
                if len(local_idx) == 0:
                    continue
                local_idx = local_idx[0]
                if local_idx == 0 or local_idx >= len(ts):
                    continue
                step_val = ts[local_idx] - ts[local_idx - 1]
            mask_after = corrected.index >= local_cp_global
            corrected.loc[mask_after, direction] -= step_val
            corrected_times[direction].append(corrected.loc[local_cp_global, "Decimal_Year"])
    return corrected

# ============================================================================
# 8. Change Point Detection Using RBF (Optional, Not Used)
# ============================================================================
def detect_change_points(data, model="rbf", penalty=400, method="pelt", jump=5):
    """
    Detect change points using the ruptures library with an RBF kernel.

    Note: This function is not used in the current pipeline but is included for potential
    future use or comparison with the sliding window method.

    Args:
        data (pd.DataFrame): DataFrame with 'Decimal_Year', 'N', 'E', 'U' columns.
        model (str): Model type for ruptures (default: 'rbf').
        penalty (float): Penalty value for change point detection (default: 400).
        method (str): Detection method ('pelt', 'binseg', 'window'; default: 'pelt').
        jump (int): Jump parameter for detection (default: 5).

    Returns:
        dict: Dictionary of change points for each direction.
    """
    change_points = {}
    for direction in ["N", "E", "U"]:
        valid_mask = data[direction].notna()
        valid_indices = data.loc[valid_mask].index
        ts = data.loc[valid_mask, direction].values.reshape(-1, 1)
        if method == "binseg":
            algo = rpt.Binseg(model=model).fit(ts)
        elif method == "window":
            algo = rpt.Window(model=model).fit(ts)
        else:
            algo = rpt.KernelCPD(kernel="rbf", min_size=200).fit(ts)
        cp_indices = algo.predict(pen=penalty)
        cp_indices = [0] + cp_indices + [len(ts) - 1]
        original_cps = [valid_indices[i] for i in cp_indices if i < len(valid_indices)]
        change_points[direction] = sorted(set(original_cps))
    return change_points

# ============================================================================
# 9. Calculate Velocities and Identify Longest Segments
# ============================================================================
def calculate_segment_velocity(data, change_points, min_years=1.0, min_samples=200):
    """
    Calculate velocities for the longest step-free segment in each direction.

    The function identifies segments between change points, selects the longest segment
    meeting minimum duration and sample requirements, and computes the velocity using
    linear regression.

    Args:
        data (pd.DataFrame): DataFrame with 'Decimal_Year', 'N', 'E', 'U' columns.
        change_points (dict): Dictionary of change points for each direction.
        min_years (float): Minimum segment duration in years (default: 1.0).
        min_samples (int): Minimum number of samples in a segment (default: 200).

    Returns:
        tuple: (velocities, intercepts, longest_segments)
            - velocities (dict): Velocities in cm/yr for each direction.
            - intercepts (dict): Intercepts of the linear fit for each direction.
            - longest_segments (dict): Longest segment indices and duration for each direction.
    """
    velocities = {}
    intercepts = {}
    longest_segments = {}
    for direction in ["N", "E", "U"]:
        segments = []
        indices = sorted(set([0] + change_points[direction] + [len(data) - 1]))
        for i in range(len(indices) - 1):
            start, end = indices[i], indices[i + 1]
            if start == end:
                continue
            segment = data.iloc[start:end + 1]
            seg_valid = segment.dropna(subset=["Decimal_Year", direction])
            if len(seg_valid) < 3:
                continue
            duration = seg_valid["Decimal_Year"].iloc[-1] - seg_valid["Decimal_Year"].iloc[0]
            if duration >= min_years and len(seg_valid) >= min_samples:
                segments.append((start, end, duration))
        if segments:
            longest_segment = max(segments, key=lambda x: x[2])
        else:
            longest_segment = (0, len(data) - 1, data["Decimal_Year"].iloc[-1] - data["Decimal_Year"].iloc[0])
        segment_data = data.iloc[longest_segment[0]:longest_segment[1] + 1].dropna(subset=["Decimal_Year", direction])
        if len(segment_data) < 30:
            slope, intercept = 0, 0
        else:
            slope, intercept, _, _, _ = linregress(segment_data["Decimal_Year"], segment_data[direction])
        velocities[direction] = slope
        intercepts[direction] = intercept
        longest_segments[direction] = longest_segment
    return velocities, intercepts, longest_segments

# ============================================================================
# 10. Plot Time Series with Velocity Trend
# ============================================================================
def plot_time_series_with_longest_segment(data, velocities, intercepts, longest_segments, file_path):
    """
    Plot the original GNSS time series with the velocity trend of the longest step-free segment.

    The plot shows the raw displacement in NS, EW, and UD directions, with the velocity trend
    (red dashed line) overlaid on the longest segment used for velocity calculation.

    Args:
        data (pd.DataFrame): DataFrame with 'Decimal_Year', 'N', 'E', 'U' columns.
        velocities (dict): Velocities for each direction in cm/yr.
        intercepts (dict): Intercepts of the linear fit for each direction.
        longest_segments (dict): Longest segment indices and duration for each direction.
        file_path (str): Path to the input file (used to extract station name).
    """
    fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True)
    plt.subplots_adjust(hspace=0.5, top=0.92)
    directions = ["N", "E", "U"]
    direction_labels = {"N": "NS", "E": "EW", "U": "UD"}
    y_ranges = {"N": (-2, 2), "E": (-2, 2), "U": (-3, 3)}
    # Remove mean from each direction for better visualization.
    mean_shifts = {direction: data[direction].mean() for direction in directions}
    data_mean_removed = data.copy()
    for direction in directions:
        data_mean_removed[direction] -= mean_shifts[direction]

    for i, direction in enumerate(directions):
        ax = axes[i]
        valid = data_mean_removed[direction].notna()
        ax.plot(data_mean_removed.loc[valid, "Decimal_Year"], data_mean_removed.loc[valid, direction], 'bo', markersize=2,
                label=f"{direction_labels[direction]} displacement")
        longest_segment = longest_segments.get(direction)
        if longest_segment is not None:
            seg_start, seg_end, _ = longest_segment
            segment_data = data.iloc[seg_start:seg_end + 1].dropna(subset=["Decimal_Year", direction])
            if len(segment_data) > 1:
                slope = velocities.get(direction, 0)
                intercept = intercepts.get(direction, 0)
                trend_line = slope * segment_data["Decimal_Year"] + intercept - mean_shifts[direction]
                ax.plot(segment_data["Decimal_Year"], trend_line, "r--", linewidth=3,
                        label=f"Velocity: {slope*10:.1f} mm/yr")
        ax.set_ylabel(f"{direction_labels[direction]} (cm)")
        ax.legend()
    axes[-1].set_xlabel("Decimal Year")
    station_name = os.path.basename(file_path).split('_')[0]
    plt.suptitle(f"{station_name}", fontsize=12, y=0.935)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{station_name}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    del fig, axes
    gc.collect()

# ============================================================================
# 11. Plot Detrended Time Series with Steps
# ============================================================================
def plot_time_series_with_velocity(data, change_points, velocities, intercepts, longest_segments, file_path):
    """
    Plot the detrended GNSS time series with detected steps and the longest segment used for velocity.

    The plot shows the detrended displacement in NS, EW, and UD directions, with detected steps
    (red vertical lines) and the longest step-free segment (red dots) used for velocity calculation.

    Args:
        data (pd.DataFrame): DataFrame with 'Decimal_Year', 'N', 'E', 'U' columns.
        change_points (dict): Dictionary of change points for each direction.
        velocities (dict): Velocities for each direction in cm/yr.
        intercepts (dict): Intercepts of the linear fit for each direction.
        longest_segments (dict): Longest segment indices and duration for each direction.
        file_path (str): Path to the input file (used to extract station name).
    """
    fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True)
    plt.subplots_adjust(hspace=0.5, top=0.92)
    directions = ["N", "E", "U"]
    direction_labels = {"N": "NS", "E": "EW", "U": "UD"}
    y_ranges = {"N": (-3, 2), "E": (-2.5, 2.5), "U": (-3, 3)}

    for i, direction in enumerate(directions):
        ax = axes[i]
        slope = velocities.get(direction, 0)
        intercept = intercepts.get(direction, 0)
        de_trended = data[direction] - (slope * data["Decimal_Year"] + intercept)
        valid = de_trended.notna()
        ax.plot(data.loc[valid, "Decimal_Year"], de_trended[valid], 'bo', markersize=2,
                label=f"Detrended {direction_labels[direction]} displacement")

        # Recalculate the longest step-free segment.
        cp_indices = sorted(set([0] + change_points[direction] + [len(data) - 1]))
        segments = []
        for j in range(len(cp_indices) - 1):
            start, end = cp_indices[j], cp_indices[j + 1]
            if start == end:
                continue
            segment_data = data.iloc[start:end + 1].dropna(subset=["Decimal_Year", direction])
            if len(segment_data) < 30:
                continue
            duration = segment_data["Decimal_Year"].iloc[-1] - segment_data["Decimal_Year"].iloc[0]
            if duration >= 1.0:
                segments.append((start, end, duration))

        if segments:
            longest_segment = max(segments, key=lambda x: x[2])
            seg_start, seg_end = longest_segment[0], longest_segment[1]
        else:
            seg_start, seg_end = 0, len(data) - 1

        # Plot the detrended longest segment.
        segment_data = data.iloc[seg_start:seg_end + 1]
        segment_de_trended = de_trended.iloc[seg_start:seg_end + 1]
        valid_seg = segment_de_trended.notna()
        if valid_seg.sum() >= 2:
            ax.plot(segment_data.loc[valid_seg, "Decimal_Year"], segment_de_trended[valid_seg],
                    'r.', markersize=1, label=f"Segment for calculating velocity")

        # Plot detected change points.
        for j, cp_idx in enumerate(change_points.get(direction, [])):
            if cp_idx in data.index:
                ax.axvline(data.loc[cp_idx, "Decimal_Year"], color='r', linestyle='--',
                           label="Detected steps" if j == 0 else "")

        ax.set_ylabel(f"{direction_labels[direction]} (cm)")
        y_vals = de_trended[valid]
        ymin = np.min(y_vals)
        ymax = np.max(y_vals)
        if direction in ["N", "E"]:
            if ymin < -2 or ymax > 2:
                ax.set_ylim(ymin, ymax)
            else:
                ax.set_ylim(-2, 2)
        elif direction == "U":
            if ymin < -3 or ymax > 3:
                ax.set_ylim(ymin, ymax)
            else:
                ax.set_ylim(-3, 3)

    axes[-1].set_xlabel("Decimal Year")
    ax.legend()
    station_name = os.path.basename(file_path).split('_')[0]
    plt.suptitle(f"Detrended {station_name}-NEU: Step Detection", fontsize=12, y=0.935)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{station_name}_detrended.png", bbox_inches='tight', pad_inches=0.1)
    plt.close()
    del fig, axes
    gc.collect()

# ============================================================================
# 12. Save Velocities to File
# ============================================================================
def save_velocities(file_path, velocities, longest_segments, data):
    """
    Save calculated velocities and segment information to a text file.

    The output file includes velocities in mm/yr, segment durations, and start/end times
    for each direction, as well as the whole time series bounds.

    Args:
        file_path (str): Path to the input file (used to extract station name).
        velocities (dict): Velocities for each direction in cm/yr.
        longest_segments (dict): Longest segment indices and duration for each direction.
        data (pd.DataFrame): Original DataFrame with 'Decimal_Year' column.
    """
    station_name = os.path.splitext(os.path.basename(file_path))[0]
    station = os.path.basename(file_path)[:4]
    output_file = f"{station_name}_Velocity.txt"

    # Get whole time series bounds.
    valid_times = data["Decimal_Year"].dropna()
    whole_begin = valid_times.iloc[0] if not valid_times.empty else 'NA'
    whole_end = valid_times.iloc[-1] if not valid_times.empty else 'NA'
    whole_duration = (whole_end - whole_begin) if whole_begin != 'NA' and whole_end != 'NA' else 'NA'

    with open(output_file, 'w') as f:
        # Write header with all fields.
        f.write("Station Vel_N(mm/yr) Vel_E(mm/yr) Vel_U(mm/yr) Dur_N Dur_E Dur_U S_begin_N S_end_N S_begin_E S_end_E S_begin_U S_end_U W_begin W_end W_Du\n")

        # Convert velocities to mm/yr (from cm/yr).
        velN = velocities.get("N", 'NA') * 10
        velE = velocities.get("E", 'NA') * 10
        velU = velocities.get("U", 'NA') * 10
        durN = (longest_segments["N"][2] if "N" in longest_segments else 'NA')
        durE = (longest_segments["E"][2] if "E" in longest_segments else 'NA')
        durU = (longest_segments["U"][2] if "U" in longest_segments else 'NA')

        # Segment start and end times.
        seg_begin_N = (data["Decimal_Year"].iloc[longest_segments["N"][0]] if "N" in longest_segments else 'NA')
        seg_end_N = (data["Decimal_Year"].iloc[longest_segments["N"][1]] if "N" in longest_segments else 'NA')
        seg_begin_E = (data["Decimal_Year"].iloc[longest_segments["E"][0]] if "E" in longest_segments else 'NA')
        seg_end_E = (data["Decimal_Year"].iloc[longest_segments["E"][1]] if "E" in longest_segments else 'NA')
        seg_begin_U = (data["Decimal_Year"].iloc[longest_segments["U"][0]] if "U" in longest_segments else 'NA')
        seg_end_U = (data["Decimal_Year"].iloc[longest_segments["U"][1]] if "U" in longest_segments else 'NA')

        # Write the data line.
        f.write(f"{station} "
                f"{velN:.1f} {velE:.1f} {velU:.1f} "
                f"{durN:.2f} {durE:.2f} {durU:.2f} "
                f"{seg_begin_N:.4f} {seg_end_N:.3f} "
                f"{seg_begin_E:.4f} {seg_end_E:.3f} "
                f"{seg_begin_U:.4f} {seg_end_U:.3f} "
                f"{whole_begin:.4f} {whole_end:.3f} {whole_duration:.2f}\n")

# ============================================================================
# 13. Save Velocities (Short Format, Not Used)
# ============================================================================
def save_velocities_short(file_path, velocities, longest_segments):
    """
    Save velocities in a shorter format (not used in the current pipeline).

    Args:
        file_path (str): Path to the input file.
        velocities (dict): Velocities for each direction in cm/yr.
        longest_segments (dict): Longest segment indices and duration for each direction.
    """
    station_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = f"{station_name}_Velocity.txt"
    with open(output_file, 'w') as f:
        f.write("Station Vel_N(cm/yr) Vel_E(cm/yr) Vel_U(cm/yr) Duration_N(years) Duration_E(years) Duration_U(years)\n")
        velN = velocities.get("N", 'NA')
        velE = velocities.get("E", 'NA')
        velU = velocities.get("U", 'NA')
        durN = (longest_segments["N"][2] if "N" in longest_segments else 'NA')
        durE = (longest_segments["E"][2] if "E" in longest_segments else 'NA')
        durU = (longest_segments["U"][2] if "U" in longest_segments else 'NA')
        f.write(f"{station_name} {velN:.2f} {velE:.2f} {velU:.2f} {durN:.2f} {durE:.2f} {durU:.2f}\n")

# ============================================================================
# 14. Plot Original vs. Cleaned Time Series (Not Used)
# ============================================================================
def plot_original_vs_cleaned(original_data, cleaned_data, file_path):
    """
    Plot the original vs. cleaned time series to visualize outlier removal.

    Note: This function is not used in the current pipeline but is included for debugging.

    Args:
        original_data (pd.DataFrame): Original DataFrame with raw data.
        cleaned_data (pd.DataFrame): DataFrame with outliers removed.
        file_path (str): Path to the input file.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    directions = ["N", "E", "U"]
    for i, direction in enumerate(directions):
        ax = axes[i]
        ax.plot(original_data["Decimal_Year"], original_data[direction], 'r.', alpha=0.5,
                label="Original Data (Outliers Included)")
        outlier_mask = original_data[direction].notna() & cleaned_data[direction].isna()
        ax.plot(original_data["Decimal_Year"][outlier_mask], original_data[direction][outlier_mask],
                'kx', markersize=6, label="Removed Outliers")
        ax.set_ylabel(f"{direction} (cm)")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Decimal Year")
    station_name = os.path.basename(file_path).split('_')[0]
    plt.suptitle(f"Original vs. Cleaned Time Series: {station_name}", y=0.96)
    plt.tight_layout()
    plt.savefig(f"{station_name}_NEU_Outliers.png", bbox_inches='tight', pad_inches=0.1)
    plt.close()

# ============================================================================
# 15. Convert Change Points to Decimal Years
# ============================================================================
def get_change_points_year(data, change_points):
    """
    Convert change point indices to their corresponding decimal years.

    Args:
        data (pd.DataFrame): DataFrame with 'Decimal_Year' column.
        change_points (dict): Dictionary of change point indices for each direction.

    Returns:
        dict: Dictionary of change point years for each direction.
    """
    cp_years = {}
    for direction, cp_ids in change_points.items():
        cp_years[direction] = []
        for cp_id in cp_ids:
            try:
                year = data.loc[cp_id, "Decimal_Year"]
                cp_years[direction].append(year)
            except KeyError:
                print(f"Warning: Index {cp_id} not found in data for direction {direction}.")
        cp_years[direction] = sorted(cp_years[direction])
    return cp_years

# ============================================================================
# 16. Write AI Parameters File
# ============================================================================
def write_ai_parameters(
    station_name,
    input_file,
    big_window_size,
    big_N_threshold,
    big_E_threshold,
    big_U_threshold,
    big_min_distance,
    step_threshold,
    small_window_size,
    small_N_threshold,
    small_E_threshold,
    small_U_threshold,
    small_min_distance,
    N_curve_threshold,
    E_curve_threshold,
    U_curve_threshold,
    improvement_ratio,
    small_change_points_year,
    scores,
    label="good"
):
    """
    Write step detection parameters and CNN scores to a CSV file for AI training or analysis.

    Args:
        station_name (str): Name of the station.
        input_file (str): Path to the input file.
        big_window_size (int): Window size for large step detection.
        big_N_threshold (float): Threshold for North large steps.
        big_E_threshold (float): Threshold for East large steps.
        big_U_threshold (float): Threshold for Up large steps.
        big_min_distance (int): Minimum distance for large steps.
        step_threshold (float): Step correction threshold (not used).
        small_window_size (int): Window size for small step detection.
        small_N_threshold (float): Threshold for North small steps.
        small_E_threshold (float): Threshold for East small steps.
        small_U_threshold (float): Threshold for Up small steps.
        small_min_distance (int): Minimum distance for small steps.
        N_curve_threshold (float): Curve threshold for North slow steps.
        E_curve_threshold (float): Curve threshold for East slow steps.
        U_curve_threshold (float): Curve threshold for Up slow steps.
        improvement_ratio (float): Improvement ratio for cubic fit.
        small_change_points_year (dict): Change point years for small steps.
        scores (dict): CNN scores for each direction.
        label (str): Label for the configuration (default: "good").
    """
    cp_years_N = ",".join([str(val) for val in small_change_points_year.get("N", [])])
    cp_years_E = ",".join([str(val) for val in small_change_points_year.get("E", [])])
    cp_years_U = ",".join([str(val) for val in small_change_points_year.get("U", [])])

    params = {
        "input_file": input_file,
        "small_window_size": small_window_size,
        "small_N_threshold": small_N_threshold,
        "small_E_threshold": small_E_threshold,
        "small_U_threshold": small_U_threshold,
        "small_min_distance": small_min_distance,
        "N_curve_threshold": N_curve_threshold,
        "E_curve_threshold": E_curve_threshold,
        "U_curve_threshold": U_curve_threshold,
        "small_change_points_year_N": cp_years_N,
        "small_change_points_year_E": cp_years_E,
        "small_change_points_year_U": cp_years_U,
        "Scores": scores,
        "label": label
    }
    df_params = pd.DataFrame([params])
    output_filename = f"{station_name}_AI_parameters.csv"
    df_params.to_csv(output_filename, index=False, header=False)
    print(f"AI parameters saved to {output_filename}")

# ============================================================================
# 17. Use CNN to Predict Plot Quality
# ============================================================================
def cnn_predict_plot(cnn_model, plot_path, target_size=(224, 224)):
    """
    Predict the quality of a step detection plot using a trained CNN model.

    The CNN outputs a probability score (0–1) indicating the likelihood of the plot being
    "good" (suitable for velocity estimation).

    Args:
        cnn_model: Trained TensorFlow/Keras model.
        plot_path (str): Path to the plot image (PNG).
        target_size (tuple): Image size for resizing (default: (224, 224)).

    Returns:
        float: Probability of "good" classification, or None if an error occurs.
    """
    try:
        img = Image.open(plot_path).convert('RGB')
        img = img.resize(target_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = cnn_model.predict(img_array, verbose=0)
        return prediction[0][0]
    except Exception as e:
        print(f"Error processing {plot_path}: {e}")
        return None
    finally:
        try:
            del img, img_array
        except NameError:
            pass
        gc.collect()

# ============================================================================
# 18. Generate Candidate Plot for CNN Evaluation
# ============================================================================
def plot_candidate_for_direction(data, candidate_change_points, temp_plot, file_path, direction):
    """
    Generate a candidate plot for a single direction (N, E, or U) for CNN evaluation.

    The plot shows the detrended time series with detected change points, matching the format
    used during CNN training.

    Args:
        data (pd.DataFrame): DataFrame with 'Decimal_Year', 'N', 'E', 'U' columns.
        candidate_change_points (dict): Candidate change points for each direction.
        temp_plot (str): Path to save the temporary plot.
        file_path (str): Path to the input file (for station name).
        direction (str): Direction to plot ('N', 'E', or 'U').
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    df_valid = pd.DataFrame({'x': data["Decimal_Year"], 'y': data[direction]}).dropna()
    if len(df_valid) < 2:
        print(f"[DEBUG] Not enough valid data for direction={direction}. Aborting plot.")
        plt.close()
        return

    x_valid = df_valid['x'].astype(float)
    y_valid = df_valid['y'].astype(float)
    slope, intercept, _, _, _ = linregress(x_valid, y_valid)
    x_full = data["Decimal_Year"]
    y_full = data[direction]
    de_trended = y_full - (slope * x_full + intercept)
    valid = de_trended.notna()
    if valid.sum() == 0:
        print(f"No valid detrended data for direction {direction}.")
        plt.close()
        return

    ax.plot(x_full[valid], de_trended[valid], 'bo', markersize=2, label=f"{direction} Detrended")
    for j, cp_idx in enumerate(candidate_change_points.get(direction, [])):
        if cp_idx in data.index:
            cp_year = data.loc[cp_idx, "Decimal_Year"]
            ax.axvline(cp_year, color='r', linestyle='--', linewidth=4,
                       label="Detected Change" if j == 0 else "")

    ax.set_ylabel(f"{direction} (cm)")
    y_vals = de_trended[valid]
    y_min = np.min(y_vals)
    y_max = np.max(y_vals)
    if direction in ["N", "E"]:
        if y_min < -2 or y_max > 2:
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(-2, 2)
    elif direction == "U":
        if y_min < -3 or y_max > 3:
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(-3, 3)

    FDir = {"N": "NS", "E": "EW", "U": "UD"}.get(direction, direction)
    ax.set_xlim(x_valid.min(), x_valid.max())
    ax.set_xlabel("Decimal Year", fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    station_name = os.path.splitext(os.path.basename(file_path))[0]
    ax.set_title(f"Step Detection: {FDir} Direction", fontsize=14)
    plt.tight_layout()
    plt.savefig(temp_plot, bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    plt.close('all')
    gc.collect()

# ============================================================================
# 19. Select Optimal Small-Step Parameters Using CNN
# ============================================================================
def select_optimal_small_params_sequential(step_corrected_data, cnn_model, station_name, threshold_good=0.85):
    """
    Select optimal small-step detection parameters using the CNN model.

    The function sequentially optimizes parameters for each direction (NS, EW, UD) by testing
    candidate thresholds and curve amplitudes, selecting the first configuration that meets
    the CNN's "good" threshold or the best-scoring configuration if none meet the threshold.

    Args:
        step_corrected_data (pd.DataFrame): DataFrame with cleaned data.
        cnn_model: Trained TensorFlow/Keras model.
        station_name (str): Name of the station.
        threshold_good (float): CNN probability threshold for "good" (default: 0.85).

    Returns:
        tuple: (best_params, scores, final_change_points)
            - best_params (dict): Optimal parameters for small-step detection.
            - scores (dict): CNN scores for each direction.
            - final_change_points (dict): Final change points with optimal parameters.
    """
    candidate_C_N = [0.3, 0.5, 1.0]
    candidate_C_E = [0.3, 0.5, 1.0]
    candidate_C_U = [0.4, 0.6, 1.2]
    candidate_N = [0.3, 0.4, 0.5, 0.25, 0.2, 0.15, 0.7, 1.0, 1.4, 2.0, 10]
    candidate_E = [0.3, 0.4, 0.25, 0.2, 0.15, 0.5, 0.7, 1.0, 1.4, 2.0, 10]
    candidate_U = [1.0, 0.8, 0.7, 0.6, 0.5, 1.2, 1.4, 1.8, 2.5, 20]

    small_window_size = 30
    small_min_distance = 60
    improvement_ratio = 0.2

    best_params = {}
    scores = {}

    # NS Direction: Optimize N threshold and curve amplitude.
    best_score_N = -1
    best_N = None
    best_C_N = None
    for c_val in candidate_C_N:
        for n_val in candidate_N:
            candidate_change_points = detect_abrupt_slow_steps(
                step_corrected_data,
                window_size=small_window_size,
                N_threshold=n_val,
                E_threshold=candidate_E[0],
                U_threshold=candidate_U[0],
                min_distance=small_min_distance,
                N_curve_threshold=c_val,
                E_curve_threshold=candidate_C_E[0],
                U_curve_threshold=candidate_C_U[0],
                improvement_ratio=improvement_ratio
            )
            temp_plot = f"temp_{station_name}_candidate_N.png"
            plot_candidate_for_direction(step_corrected_data, candidate_change_points, temp_plot, station_name, "N")
            prob_good = cnn_predict_plot(cnn_model, temp_plot)
            tf.keras.backend.clear_session()
            if prob_good > best_score_N:
                best_score_N = prob_good
                best_N = n_val
                best_C_N = c_val
            if prob_good >= threshold_good:
                best_params["small_N_threshold"] = best_N
                best_params["small_C_N_threshold"] = best_C_N
                scores["small_N_threshold"] = best_score_N
                break
        else:
            continue
        break
    if "small_N_threshold" not in best_params:
        best_params["small_N_threshold"] = best_N
        best_params["small_C_N_threshold"] = best_C_N
        scores["small_N_threshold"] = best_score_N

    # EW Direction: Optimize E threshold and curve amplitude.
    best_score_E = -1
    best_E = None
    best_C_E = None
    for c_val in candidate_C_E:
        for e_val in candidate_E:
            candidate_change_points = detect_abrupt_slow_steps(
                step_corrected_data,
                window_size=small_window_size,
                N_threshold=best_N,
                E_threshold=e_val,
                U_threshold=candidate_U[0],
                min_distance=small_min_distance,
                N_curve_threshold=best_C_N,
                E_curve_threshold=c_val,
                U_curve_threshold=candidate_C_U[0],
                improvement_ratio=improvement_ratio
            )
            temp_plot = f"temp_{station_name}_candidate_E.png"
            plot_candidate_for_direction(step_corrected_data, candidate_change_points, temp_plot, station_name, "E")
            prob_good = cnn_predict_plot(cnn_model, temp_plot)
            tf.keras.backend.clear_session()
            if prob_good > best_score_E:
                best_score_E = prob_good
                best_E = e_val
                best_C_E = c_val
            if prob_good >= threshold_good:
                best_params["small_E_threshold"] = best_E
                best_params["small_C_E_threshold"] = best_C_E
                scores["small_E_threshold"] = best_score_E
                break
        else:
            continue
        break
    if "small_E_threshold" not in best_params:
        best_params["small_E_threshold"] = best_E
        best_params["small_C_E_threshold"] = best_C_E
        scores["small_E_threshold"] = best_score_E

    # UD Direction: Optimize U threshold and curve amplitude.
    best_score_U = -1
    best_U = None
    best_C_U = None
    for c_val in candidate_C_U:
        for u_val in candidate_U:
            candidate_change_points = detect_abrupt_slow_steps(
                step_corrected_data,
                window_size=small_window_size,
                N_threshold=best_N,
                E_threshold=best_E,
                U_threshold=u_val,
                min_distance=small_min_distance,
                N_curve_threshold=best_C_N,
                E_curve_threshold=best_C_E,
                U_curve_threshold=c_val,
                improvement_ratio=improvement_ratio
            )
            temp_plot = f"temp_{station_name}_candidate_U.png"
            plot_candidate_for_direction(step_corrected_data, candidate_change_points, temp_plot, station_name, "U")
            prob_good = cnn_predict_plot(cnn_model, temp_plot)
            tf.keras.backend.clear_session()
            if prob_good > best_score_U:
                best_score_U = prob_good
                best_U = u_val
                best_C_U = c_val
            if prob_good >= threshold_good:
                best_params["small_U_threshold"] = best_U
                best_params["small_C_U_threshold"] = best_C_U
                scores["small_U_threshold"] = best_score_U
                break
        else:
            continue
        break
    best_params["small_U_threshold"] = best_U
    best_params["small_C_U_threshold"] = best_C_U
    scores["small_U_threshold"] = best_score_U

    best_params["small_window_size"] = small_window_size
    best_params["small_min_distance"] = small_min_distance

    # Generate final change points with optimal parameters.
    final_change_points = detect_abrupt_slow_steps(
        step_corrected_data,
        window_size=small_window_size,
        N_threshold=best_N,
        E_threshold=best_E,
        U_threshold=best_U,
        min_distance=small_min_distance,
        N_curve_threshold=best_C_N,
        E_curve_threshold=best_C_E,
        U_curve_threshold=best_C_U,
        improvement_ratio=improvement_ratio
    )

    # Generate final plots for all directions.
    final_plot_n = f"final_{station_name}_candidate_N.png"
    final_plot_e = f"final_{station_name}_candidate_E.png"
    final_plot_u = f"final_{station_name}_candidate_U.png"
    plot_candidate_for_direction(step_corrected_data, final_change_points, final_plot_n, station_name, "N")
    plot_candidate_for_direction(step_corrected_data, final_change_points, final_plot_e, station_name, "E")
    plot_candidate_for_direction(step_corrected_data, final_change_points, final_plot_u, station_name, "U")

    best_params["score"] = np.mean([best_score_N, best_score_E, best_score_U])
    gc.collect()
    return best_params, scores, final_change_points

# ============================================================================
# 20. Main Processing Function
# ============================================================================
def process_file(file_path, cnn_model, big_params, step_threshold):
    """
    Process a single GNSS time series file to estimate site velocities.

    The function loads the data, removes outliers, uses the CNN to optimize small-step
    detection parameters, calculates velocities, and generates plots and output files.

    Args:
        file_path (str): Path to the input file (e.g., P646_IGS14_NEU_cm.col).
        cnn_model: Trained TensorFlow/Keras model for step detection.
        big_params (dict): Parameters for large step detection (used in output).
        step_threshold (float): Step correction threshold (not used).
    """
    # Step 1: Load data and remove outliers.
    original_data, cleaned_data = load_data_neu(file_path)
    station_name = os.path.basename(file_path).split('_')[0]

    # Step 2: Optimize small-step detection parameters using the CNN.
    optimal_small, cnn_scores, change_points = select_optimal_small_params_sequential(
        cleaned_data, cnn_model, station_name, threshold_good=0.9
    )

    # Step 3: Extract curve thresholds for slow steps.
    N_curve_threshold = optimal_small["small_C_N_threshold"]
    E_curve_threshold = optimal_small["small_C_E_threshold"]
    U_curve_threshold = optimal_small["small_C_U_threshold"]

    # Step 4: Convert change point indices to decimal years.
    small_change_points_year = get_change_points_year(cleaned_data, change_points)

    # Step 5: Write AI parameters file for training or analysis.
    write_ai_parameters(
        station_name=station_name,
        input_file=file_path,
        big_window_size=big_params["window_size"],
        big_N_threshold=big_params["N_threshold"],
        big_E_threshold=big_params["E_threshold"],
        big_U_threshold=big_params["U_threshold"],
        big_min_distance=big_params["min_distance"],
        step_threshold=step_threshold,
        small_window_size=optimal_small["small_window_size"],
        small_N_threshold=optimal_small["small_N_threshold"],
        small_E_threshold=optimal_small["small_E_threshold"],
        small_U_threshold=optimal_small["small_U_threshold"],
        small_min_distance=optimal_small["small_min_distance"],
        N_curve_threshold=N_curve_threshold,
        E_curve_threshold=E_curve_threshold,
        U_curve_threshold=U_curve_threshold,
        improvement_ratio=0.2,
        small_change_points_year=small_change_points_year,
        scores=cnn_scores,
        label="good"
    )

    # Step 6: Calculate velocities and generate plots.
    velocities, intercepts, longest_segments = calculate_segment_velocity(cleaned_data, change_points)
    plot_time_series_with_velocity(cleaned_data, change_points, velocities, intercepts, longest_segments, file_path)
    plot_time_series_with_longest_segment(cleaned_data, velocities, intercepts, longest_segments, file_path)
    save_velocities(file_path, velocities, longest_segments, original_data)

    # Step 7: Clean up memory.
    del original_data, cleaned_data, velocities, intercepts, longest_segments, change_points
    del optimal_small, cnn_scores, small_change_points_year
    plt.close('all')
    tf.keras.backend.clear_session()
    gc.collect()

# ============================================================================
# Main Block
# ============================================================================
if __name__ == "__main__":
    # Load the pre-trained CNN model.
    print("Loading CNN model...")
    cnn_model = tf.keras.models.load_model("StepCNN-GNSS.keras")

    # Define fixed parameters for large step detection (included in output for reference).
    big_params = {
        "window_size": 20,
        "N_threshold": 0.5,
        "E_threshold": 0.5,
        "U_threshold": 2.0,
        "min_distance": 30
    }
    step_threshold = 20  # Not used in the current pipeline.

    # Find all GNSS time series files in the current directory.
    file_paths = glob.glob("*cm.col")
    if not file_paths:
        print("No NEU .col files found in the current directory.")
        exit(1)

    # Process each file sequentially.
    total_files = len(file_paths)
    for i, file_path in enumerate(file_paths, 1):
        print(f"Processing {file_path} ({i}/{total_files})...")
        process_file(file_path, cnn_model, big_params, step_threshold)
        gc.collect()

    # Final cleanup.
    del cnn_model
    gc.collect()
    print("Processing complete.")
