# -*- coding: utf-8 -*-
"""
Module 3: Pseudocode and Flowchart
Corresponding to the complete workflow in the paper

This module contains:
1. Simplified pseudocode (covering core workflow framework)
2. Flowchart description (key node visualization)

Workflow coverage:
- Raw data → Module 1 preprocessing → Module 2 data splitting → Model training → Performance evaluation
"""

# ==================== Pseudocode ====================
"""
================================================================================
PSEUDOCODE: Workflow Framework
================================================================================
Corresponding to the complete workflow, covering Module 1, Module 2, and subsequent model training and evaluation steps

ALGORITHM: Blood Glucose Prediction Workflow
Corresponding to: Section 2.2 (Data Preprocessing), Section 2.3-2.4 (Model Architecture), 
                  Section 2.5 (Data Splitting), Section 2.6 (Training & Evaluation)

INPUT: 
    - Raw data CSV file containing: patient_id, glucose_value, time
    - Configuration parameters

OUTPUT:
    - Preprocessed data with time steps
    - Multiple repetition split results
    - Trained model
    - Performance metrics

BEGIN
    // ========== STEP 1: Data Preprocessing ==========
    FUNCTION data_preprocessing(input_path, output_data_path, output_params_path):
        // Step 1.1: Read raw data
        raw_data ← READ_CSV(input_path)
        
        // Step 1.2: Unit conversion
        FOR each row in raw_data:
            glucose_converted ← CONVERT_UNIT(glucose_value)
        
        // Step 1.3: Time formatting
        FOR each row in raw_data:
            time_formatted ← FORMAT_TIME(time)
        
        // Step 1.4: Normalization
        X_min ← MIN(glucose_converted)
        X_max ← MAX(glucose_converted)
        FOR each row in raw_data:
            glucose_normalized ← NORMALIZE(glucose_converted, X_min, X_max)
        
        // Save normalization parameters
        SAVE_JSON({X_min, X_max}, output_params_path)
        
        // Step 1.5: Time step construction
        preprocessed_data ← []
        FOR each patient_id in UNIQUE(raw_data.patient_id):
            patient_data ← FILTER(raw_data, patient_id)
            
            // Generate time steps using sliding window
            time_steps ← CREATE_TIME_STEPS(patient_data)
            APPEND(preprocessed_data, time_steps)
        
        SAVE_CSV(preprocessed_data, output_data_path)
        RETURN preprocessed_data
    
    // ========== STEP 2: Data Splitting ==========
    FUNCTION data_split_repetitions(input_data_path, output_summary_path, output_split_dir):
        preprocessed_data ← READ_CSV(input_data_path)
        unique_patients ← UNIQUE(preprocessed_data.patient_id)
        
        // Multiple repetition splits
        FOR repetition_id = 1 TO NUM_REPETITIONS:
            // Set random seed
            SET_RANDOM_SEED(seed_base + repetition_id)
            
            // Shuffle and split patients
            shuffled_patients ← SHUFFLE(unique_patients)
            train_patients, val_patients, test_patients ← SPLIT_PATIENTS(shuffled_patients)
            
            // Extract time steps for each set
            train_data ← FILTER(preprocessed_data, patient_id IN train_patients)
            val_data ← FILTER(preprocessed_data, patient_id IN val_patients)
            test_data ← FILTER(preprocessed_data, patient_id IN test_patients)
            
            // Save split data
            SAVE_CSV(train_data, output_split_dir + "/repetition_" + repetition_id + "_train.csv")
            SAVE_CSV(val_data, output_split_dir + "/repetition_" + repetition_id + "_val.csv")
            SAVE_CSV(test_data, output_split_dir + "/repetition_" + repetition_id + "_test.csv")
        
        SAVE_CSV(split_summary, output_summary_path)
        RETURN split_summary
    
    // ========== STEP 3: Model Training ==========
    FUNCTION train_model(train_data, val_data, config):
        // Initialize model architecture
        model ← INITIALIZE_MODEL(config)
        
        // Initialize optimizer
        optimizer ← INITIALIZE_OPTIMIZER(model.parameters(), config)
        loss_function ← INITIALIZE_LOSS_FUNCTION()
        
        // Training loop
        FOR epoch = 1 TO NUM_EPOCHS:
            // Training phase
            model.TRAIN()
            FOR each batch in train_data:
                input_seq ← PREPARE_INPUT(batch)
                target ← PREPARE_TARGET(batch)
                
                prediction ← model(input_seq)
                loss ← loss_function(prediction, target)
                
                optimizer.ZERO_GRAD()
                loss.BACKWARD()
                optimizer.STEP()
            
            // Validation phase
            model.EVAL()
            val_loss ← EVALUATE(val_data, model)
            
            // Save best model
            IF val_loss < best_val_loss:
                best_val_loss ← val_loss
                SAVE_MODEL(model, "best_model.pth")
    
    // ========== STEP 4: Model Evaluation ==========
    FUNCTION evaluate_model(test_data, model, normalization_params):
        // Load best model
        model ← LOAD_MODEL("best_model.pth")
        model.EVAL()
        
        predictions ← []
        labels ← []
        
        FOR each batch in test_data:
            input_seq ← PREPARE_INPUT(batch)
            target ← PREPARE_TARGET(batch)
            
            prediction ← model(input_seq)
            
            // Denormalize
            prediction_denorm ← DENORMALIZE(prediction, normalization_params)
            label_denorm ← DENORMALIZE(target, normalization_params)
            
            APPEND(predictions, prediction_denorm)
            APPEND(labels, label_denorm)
        
        // Calculate performance metrics
        metrics ← CALCULATE_METRICS(labels, predictions)
        RETURN metrics
    
    // ========== MAIN WORKFLOW ==========
    BEGIN MAIN:
        // Step 1: Preprocessing
        preprocessed_data ← data_preprocessing(...)
        
        // Step 2: Data splitting
        split_summary ← data_split_repetitions(...)
        
        // Step 3-4: Model training and evaluation
        FOR repetition_id = 1 TO NUM_REPETITIONS:
            train_data ← LOAD_CSV("split_data/repetition_" + repetition_id + "_train.csv")
            val_data ← LOAD_CSV("split_data/repetition_" + repetition_id + "_val.csv")
            test_data ← LOAD_CSV("split_data/repetition_" + repetition_id + "_test.csv")
            
            model ← train_model(train_data, val_data, config)
            metrics ← evaluate_model(test_data, model, normalization_params)
            
            SAVE_METRICS(metrics, "results/repetition_" + repetition_id + "_metrics.csv")
        
        // Aggregate results
        final_metrics ← AGGREGATE_METRICS(all_repetition_metrics)
        PRINT(final_metrics)
    
END
"""

# ==================== Flowchart Description ====================
"""
================================================================================
FLOWCHART DESCRIPTION: Workflow Visualization
================================================================================
Text-based flowchart covering key nodes of the complete workflow

┌─────────────────────────────────────────────────────────────────────────┐
│                    START: Raw Data Input                                 │
│                    (patient_id, glucose_value, time)                     │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE 1: Data Preprocessing                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Step 1.1: Read CSV data                                           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                             │                                            │
│                             ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Step 1.2: Unit Conversion                                        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                             │                                            │
│                             ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Step 1.3: Time Formatting                                        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                             │                                            │
│                             ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Step 1.4: Normalization                                          │  │
│  │         → Save: normalization_params.json                     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                             │                                            │
│                             ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Step 1.5: Time Step Construction                                 │  │
│  │         → Group by patient_id                                    │  │
│  │         → Generate time steps                                     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                             │                                            │
│                             ▼                                            │
│                    preprocessed_data.csv                                 │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE 2: Data Splitting                                               │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Step 2.1: Read preprocessed data                                 │  │
│  │         → Extract unique patient_id list                         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                             │                                            │
│                             ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Step 2.2: Multiple Repetitions Loop                              │  │
│  │         FOR repetition_id = 1 TO NUM_REPETITIONS:               │  │
│  │             - Shuffle patients                                  │  │
│  │             - Split patients into train/val/test                │  │
│  │             - Assign all time steps of each patient to same set  │  │
│  │             - Save: repetition_N_train/val/test.csv             │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                             │                                            │
│                             ▼                                            │
│                    repetition_splits.csv                                 │
│                    split_data/ (multiple sets)                          │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE 3: Model Training                                               │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Step 3.1: Model Architecture Initialization                       │  │
│  │         → Initialize model                                       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                             │                                            │
│                             ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Step 3.2: Training Loop                                         │  │
│  │         FOR epoch = 1 TO NUM_EPOCHS:                            │  │
│  │             - Forward pass                                      │  │
│  │             - Loss calculation                                  │  │
│  │             - Backward pass                                     │  │
│  │             - Validation                                        │  │
│  │             - Save best model                                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                             │                                            │
│                             ▼                                            │
│                    best_model.pth                                       │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE 4: Model Evaluation                                             │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Step 4.1: Load Best Model                                        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                             │                                            │
│                             ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Step 4.2: Test Set Prediction                                    │  │
│  │         → Denormalize predictions and labels                     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                             │                                            │
│                             ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Step 4.3: Performance Metrics Calculation                        │  │
│  │         → Calculate evaluation metrics                            │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                             │                                            │
│                             ▼                                            │
│                    metrics.csv                                           │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  AGGREGATION: Multiple Repetitions Results                              │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Step 5: Aggregate Results                                       │  │
│  │         → Calculate statistics across repetitions                │  │
│  │         → Final performance report                              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                             │                                            │
│                             ▼                                            │
│                    final_results.csv                                     │
│                            END                                           │
└─────────────────────────────────────────────────────────────────────────┘

Key Node Descriptions:
1. Data Input: Raw CSV file
2. Module 1 Output: Preprocessed data with time steps
3. Module 2 Output: Multiple split train/val/test sets
4. Model Input: Preprocessed time series data
5. Model Training: Training loop and validation
6. Performance Evaluation: Calculate evaluation metrics
7. Result Aggregation: Statistical results across multiple repetitions
"""

# ==================== Module Interconnection ====================
"""
================================================================================
MODULE INTERCONNECTION: Module Call Relationships
================================================================================

Module 1 (Preprocessing) → Module 2 (Data Splitting) → Module 3 (Model Training) → Module 4 (Performance Evaluation)

Input-Output Relationships:
┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  Module 1   │      │  Module 2   │      │  Module 3   │      │  Module 4   │
│ Preprocessing│─────▶│Data Splitting│─────▶│Model Training│─────▶│  Evaluation │
└─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
     │                     │                     │                     │
     │                     │                     │                     │
     ▼                     ▼                     ▼                     ▼
raw_data.csv    preprocessed_data.csv    split_data/*.csv    best_model.pth
     │                     │                     │                     │
     │                     │                     │                     │
     ▼                     ▼                     ▼                     ▼
normalization_  repetition_splits.csv    train/val/test      metrics.csv
params.json

Data Flow:
1. Module 1 Input: Raw data
   Module 1 Output: Preprocessed data, normalization parameters

2. Module 2 Input: Preprocessed data from Module 1
   Module 2 Output: Multiple split train/val/test sets, split summary

3. Module 3 Input: Train/val sets from Module 2
   Module 3 Output: Trained model

4. Module 4 Input: Test set from Module 2 + Model from Module 3 + Normalization parameters from Module 1
   Module 4 Output: Performance metrics
"""

if __name__ == "__main__":
    print("=" * 60)
    print("Module 3: Pseudocode and Flowchart")
    print("=" * 60)
    print("\nThis module contains:")
    print("1. Simplified pseudocode (covering core workflow framework)")
    print("2. Flowchart description (key node visualization)")
    print("\nPlease refer to the file content for detailed pseudocode and flowchart descriptions.")
