# -*- coding: utf-8 -*-
"""
Module 1: Data Preprocessing Code
Corresponding to Section 2.2 - Data Preprocessing

Functions:
1. Data Reading: Support reading CSV format data (containing patient ID, glucose value, timestamp fields)
2. Unit Conversion: Convert glucose values to unified unit
3. Time Formatting: Standardize timestamp format
4. Normalization: Normalize glucose values
5. Time Step Construction: Generate time series steps grouped by patient
6. Output: Preprocessed data with time steps and normalization parameter file

Dependencies: pandas, numpy, json
"""

import pandas as pd
import numpy as np
import json
import os
from config import (
    GLUCOSE_CONVERSION_FACTOR, 
    WINDOW_SIZE,
    DEFAULT_INPUT_DATA_PATH,
    DEFAULT_OUTPUT_DATA_PATH,
    DEFAULT_NORMALIZATION_PARAMS_PATH
)


def time2decimal(time_str):
    """
    Time formatting: Standardize timestamp format
    
    Parameters:
        time_str: Time string
    
    Returns:
        float: Standardized time value
    """
    time_str = str(time_str)
    if ':' in time_str:
        h, m = time_str.split(':')
        decimal_time = float(h) + float(m) / 60.0
    else:
        decimal_time = float(time_str)
    return decimal_time


def create_time_steps(data, window_size):
    """
    Time step construction: Generate time series steps
    
    Parameters:
        data: DataFrame containing timestamp and glucose value columns
        window_size: Window size
    
    Returns:
        list: List of time steps
    """
    time_steps = []
    for i in range(0, len(data) - window_size + 1, window_size):
        window_data = data.iloc[i:i+window_size]
        glucose_seq = window_data['glucose_normalized'].values.tolist()
        time_seq = window_data['time_decimal'].values.tolist()
        time_steps.append({
            'glucose_seq': glucose_seq,
            'time_seq': time_seq
        })
    return time_steps


def data_preprocessing(input_path, output_data_path, output_params_path):
    """
    Main function for data preprocessing
    
    Parameters:
        input_path: Input CSV file path
        output_data_path: Output CSV file path for preprocessed data
        output_params_path: Output JSON file path for normalization parameters
    
    Returns:
        pd.DataFrame: Preprocessed data
    """
    print("=" * 60)
    print("Module 1: Data Preprocessing (Corresponding to Section 2.2)")
    print("=" * 60)
    
    # ==================== Step 1: Data Reading ====================
    print("\nStep 1: Reading raw data...")
    df = pd.read_csv(input_path)
    print(f"Reading completed, total {len(df)} records")
    
    # Check required columns
    required_columns = ['patient_id', 'glucose_value', 'time']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Data file must contain the following columns: {required_columns}. Missing columns: {missing_columns}")
    
    # ==================== Step 2: Unit Conversion ====================
    print("\nStep 2: Unit conversion...")
    df['glucose_mgdL'] = df['glucose_value'] * GLUCOSE_CONVERSION_FACTOR
    print(f"Conversion completed, glucose value range: {df['glucose_mgdL'].min():.2f} ~ {df['glucose_mgdL'].max():.2f}")
    
    # ==================== Step 3: Time Formatting ====================
    print("\nStep 3: Time formatting...")
    df['time_decimal'] = df['time'].apply(time2decimal)
    print(f"Formatting completed, time range: {df['time_decimal'].min():.2f} ~ {df['time_decimal'].max():.2f}")
    
    # ==================== Step 4: Normalization ====================
    print("\nStep 4: Normalization processing...")
    X_min = df['glucose_mgdL'].min()
    X_max = df['glucose_mgdL'].max()
    df['glucose_normalized'] = (df['glucose_mgdL'] - X_min) / (X_max - X_min)
    print(f"Normalization completed")
    
    # Save normalization parameters
    normalization_params = {
        'X_min': float(X_min),
        'X_max': float(X_max)
    }
    with open(output_params_path, 'w', encoding='utf-8') as f:
        json.dump(normalization_params, f, indent=4, ensure_ascii=False)
    print(f"Normalization parameters saved to: {output_params_path}")
    
    # ==================== Step 5: Time Step Construction ====================
    print(f"\nStep 5: Time step construction...")
    all_time_steps = []
    
    for patient_id, patient_data in df.groupby('patient_id'):
        patient_data = patient_data.sort_values('time_decimal').reset_index(drop=True)
        time_steps = create_time_steps(patient_data, WINDOW_SIZE)
        
        for step_idx, step_data in enumerate(time_steps):
            all_time_steps.append({
                'patient_id': patient_id,
                'time_step_id': step_idx,
                'glucose_seq': step_data['glucose_seq'],
                'time_seq': step_data['time_seq']
            })
    
    # Convert to DataFrame
    df_time_steps = pd.DataFrame(all_time_steps)
    print(f"Time step construction completed, generated {len(df_time_steps)} time steps")
    print(f"Number of patients involved: {df_time_steps['patient_id'].nunique()}")
    
    # ==================== Step 6: Output Results ====================
    print(f"\nStep 6: Saving preprocessing results...")
    # Save preprocessed data with time steps
    df_time_steps.to_csv(output_data_path, index=False, encoding='utf-8-sig')
    print(f"Preprocessed data saved to: {output_data_path}")
    
    print("\n" + "=" * 60)
    print("Data preprocessing completed!")
    print("=" * 60)
    
    return df_time_steps


# ==================== Usage Example ====================
if __name__ == "__main__":
    # Specify test input and output paths
    input_path = DEFAULT_INPUT_DATA_PATH  # Modify to actual input file path
    output_data_path = DEFAULT_OUTPUT_DATA_PATH
    output_params_path = DEFAULT_NORMALIZATION_PARAMS_PATH
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_data_path) if os.path.dirname(output_data_path) else '.', exist_ok=True)
    os.makedirs(os.path.dirname(output_params_path) if os.path.dirname(output_params_path) else '.', exist_ok=True)
    
    # Execute preprocessing
    preprocessed_data = data_preprocessing(
        input_path=input_path,
        output_data_path=output_data_path,
        output_params_path=output_params_path
    )
    
    print(f"\nPreprocessing results preview (first 5 rows):")
    print(preprocessed_data.head())

