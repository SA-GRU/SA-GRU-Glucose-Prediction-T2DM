# -*- coding: utf-8 -*-
"""
Module 2: Data Splitting Example
Corresponding to Section 2.5 - Data Splitting Method

Functions:
1. Read preprocessed data with time steps from Module 1
2. Stratified splitting by patient ID (core: all time steps of the same patient belong to the same set, no cross-set)
3. Implement multiple repetition validation splits:
   - Each split allocates patients by ratio (split patients first, then assign all their time steps)
   - Generate train/val/test set data for each split (containing time step information)
4. Output:
   - Multiple repetition split summary file (repetition_splits.csv)
   - Separate train/val/test set data for each split (CSV format)

Dependencies: pandas, numpy, random
"""

import pandas as pd
import numpy as np
import random
import os
from config import (
    SPLIT_RATIO_TRAIN,
    SPLIT_RATIO_VAL,
    SPLIT_RATIO_TEST,
    NUM_REPETITIONS,
    RANDOM_SEED_BASE,
    DEFAULT_OUTPUT_DATA_PATH,
    DEFAULT_SPLIT_SUMMARY_PATH,
    DEFAULT_SPLIT_DATA_DIR
)


def data_split_10repetitions(input_data_path, output_summary_path, output_split_dir):
    """
    Main function for data splitting: Implement multiple repetition validation splits (corresponding to Section 2.5)
    
    Parameters:
        input_data_path: Preprocessed data path from Module 1 (CSV format, containing patient_id, time_step_id, glucose_seq, time_seq columns)
        output_summary_path: Output summary file path for multiple repetition splits (CSV format)
        output_split_dir: Directory path for output split data
    
    Returns:
        pd.DataFrame: Split summary information
    """
    print("=" * 60)
    print("Module 2: Data Splitting Example (Corresponding to Section 2.5)")
    print("=" * 60)
    
    # ==================== Step 1: Read Preprocessed Data ====================
    print("\nStep 1: Reading preprocessed data from Module 1...")
    df = pd.read_csv(input_data_path)
    print(f"Reading completed, total {len(df)} time steps")
    print(f"Number of patients involved: {df['patient_id'].nunique()}")
    
    # Check required columns
    required_columns = ['patient_id', 'time_step_id', 'glucose_seq', 'time_seq']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Data file must contain the following columns: {required_columns}. Missing columns: {missing_columns}")
    
    # ==================== Step 2: Get All Unique Patient IDs ====================
    print("\nStep 2: Extracting all unique patient IDs...")
    unique_patients = df['patient_id'].unique().tolist()
    num_patients = len(unique_patients)
    print(f"Total {num_patients} unique patients")
    
    # ==================== Step 3: Multiple Repetition Validation Splits ====================
    print(f"\nStep 3: Executing {NUM_REPETITIONS} repetition validation splits...")
    
    # Create output directory
    os.makedirs(output_split_dir, exist_ok=True)
    
    # Store summary information for all splits
    all_splits_summary = []
    
    for repetition_id in range(1, NUM_REPETITIONS + 1):
        print(f"\n--- Repetition {repetition_id} ---")
        
        # Set random seed (use different seed each time to ensure different splits)
        random_seed = RANDOM_SEED_BASE + repetition_id
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Randomly shuffle patient list
        shuffled_patients = unique_patients.copy()
        np.random.shuffle(shuffled_patients)
        
        # Allocate patients by ratio
        num_train = int(num_patients * SPLIT_RATIO_TRAIN)
        num_val = int(num_patients * SPLIT_RATIO_VAL)
        num_test = num_patients - num_train - num_val  # Ensure all patients are allocated
        
        train_patients = shuffled_patients[:num_train]
        val_patients = shuffled_patients[num_train:num_train + num_val]
        test_patients = shuffled_patients[num_train + num_val:]
        
        print(f"Train set patients: {len(train_patients)} ({len(train_patients)/num_patients*100:.1f}%)")
        print(f"Validation set patients: {len(val_patients)} ({len(val_patients)/num_patients*100:.1f}%)")
        print(f"Test set patients: {len(test_patients)} ({len(test_patients)/num_patients*100:.1f}%)")
        
        # Extract corresponding time step data based on patient allocation
        train_data = df[df['patient_id'].isin(train_patients)].copy()
        val_data = df[df['patient_id'].isin(val_patients)].copy()
        test_data = df[df['patient_id'].isin(test_patients)].copy()
        
        print(f"Train set time steps: {len(train_data)}")
        print(f"Validation set time steps: {len(val_data)}")
        print(f"Test set time steps: {len(test_data)}")
        
        # Validation: Ensure all time steps of the same patient are in the same set
        train_patients_actual = set(train_data['patient_id'].unique())
        val_patients_actual = set(val_data['patient_id'].unique())
        test_patients_actual = set(test_data['patient_id'].unique())
        
        # Check if any patient crosses sets
        overlap_train_val = train_patients_actual & val_patients_actual
        overlap_train_test = train_patients_actual & test_patients_actual
        overlap_val_test = val_patients_actual & test_patients_actual
        
        if overlap_train_val or overlap_train_test or overlap_val_test:
            raise ValueError(f"Repetition {repetition_id} has patient cross-set issue! "
                           f"Train-Val overlap: {overlap_train_val}, "
                           f"Train-Test overlap: {overlap_train_test}, "
                           f"Val-Test overlap: {overlap_val_test}")
        
        # Save train/val/test set data for this split
        train_data.to_csv(
            os.path.join(output_split_dir, f'repetition_{repetition_id}_train.csv'),
            index=False,
            encoding='utf-8-sig'
        )
        val_data.to_csv(
            os.path.join(output_split_dir, f'repetition_{repetition_id}_val.csv'),
            index=False,
            encoding='utf-8-sig'
        )
        test_data.to_csv(
            os.path.join(output_split_dir, f'repetition_{repetition_id}_test.csv'),
            index=False,
            encoding='utf-8-sig'
        )
        
        # Record summary information for this split
        for patient_id in train_patients:
            patient_time_steps = train_data[train_data['patient_id'] == patient_id]['time_step_id'].tolist()
            for time_step_id in patient_time_steps:
                all_splits_summary.append({
                    'repetition_id': repetition_id,
                    'patient_id': patient_id,
                    'set_type': 'train',
                    'time_step_id': time_step_id
                })
        
        for patient_id in val_patients:
            patient_time_steps = val_data[val_data['patient_id'] == patient_id]['time_step_id'].tolist()
            for time_step_id in patient_time_steps:
                all_splits_summary.append({
                    'repetition_id': repetition_id,
                    'patient_id': patient_id,
                    'set_type': 'val',
                    'time_step_id': time_step_id
                })
        
        for patient_id in test_patients:
            patient_time_steps = test_data[test_data['patient_id'] == patient_id]['time_step_id'].tolist()
            for time_step_id in patient_time_steps:
                all_splits_summary.append({
                    'repetition_id': repetition_id,
                    'patient_id': patient_id,
                    'set_type': 'test',
                    'time_step_id': time_step_id
                })
    
    # ==================== Step 4: Save Split Summary File ====================
    print(f"\nStep 4: Saving split summary file...")
    df_summary = pd.DataFrame(all_splits_summary)
    df_summary.to_csv(output_summary_path, index=False, encoding='utf-8-sig')
    print(f"Split summary file saved to: {output_summary_path}")
    print(f"Summary information: total {len(df_summary)} records, covering {NUM_REPETITIONS} repetitions")
    
    # Statistics
    print("\nSplit statistics:")
    for rep_id in range(1, NUM_REPETITIONS + 1):
        rep_data = df_summary[df_summary['repetition_id'] == rep_id]
        train_count = len(rep_data[rep_data['set_type'] == 'train'])
        val_count = len(rep_data[rep_data['set_type'] == 'val'])
        test_count = len(rep_data[rep_data['set_type'] == 'test'])
        print(f"Repetition {rep_id}: train={train_count}, val={val_count}, test={test_count}")
    
    print("\n" + "=" * 60)
    print("Data splitting completed!")
    print("=" * 60)
    
    return df_summary


# ==================== Usage Example ====================
if __name__ == "__main__":
    # Link to Module 1 output data path
    input_data_path = DEFAULT_OUTPUT_DATA_PATH  # Preprocessed data from Module 1
    output_summary_path = DEFAULT_SPLIT_SUMMARY_PATH
    output_split_dir = DEFAULT_SPLIT_DATA_DIR
    
    # Execute data splitting
    split_summary = data_split_10repetitions(
        input_data_path=input_data_path,
        output_summary_path=output_summary_path,
        output_split_dir=output_split_dir
    )
    
    print(f"\nSplit summary preview (first 10 rows):")
    print(split_summary.head(10))

